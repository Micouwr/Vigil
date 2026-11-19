import json
import os
import uuid
import yaml
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from risk_controls.pii_filters import redact_pii, detect_pii_types, REDACTION_TOKENS, PII_PATTERNS
from datetime import datetime, timezone

# --- Configuration Loading ---

def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).parent.parent

def _validate_config(config: dict):
    """Basic validation of scope.yaml: required keys present, thresholds sane.

    Raises ValueError on a config that cannot support safe routing. Called by
    load_config, so a broken config fails loudly at import instead of silently
    degrading governance at runtime.
    """
    if not isinstance(config, dict):
        raise ValueError("scope.yaml did not load as a mapping")
    bot_config = config.get("bot_config")
    if not isinstance(bot_config, dict):
        raise ValueError("scope.yaml is missing the 'bot_config' section")
    if not bot_config.get("model_name"):
        raise ValueError("scope.yaml bot_config.model_name must be a non-empty string")
    ticket_types = config.get("ticket_types")
    if not isinstance(ticket_types, list) or not ticket_types:
        raise ValueError("scope.yaml ticket_types must be a non-empty list")
    rules = config.get("escalation_rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("scope.yaml escalation_rules must be a non-empty list")

def load_config(config_path: Path | None = None):
    """Loads the main configuration from the YAML file."""
    config_path = config_path or get_project_root() / "governance" / "config" / "scope.yaml"
    with config_path.open("r") as f:
        loaded = yaml.safe_load(f)
    _validate_config(loaded)
    return loaded

# Load configuration and environment variables
config = load_config()
bot_config = config.get("bot_config", {})
TICKET_TYPES = config.get("ticket_types", [])
MODEL_NAME = bot_config.get("model_name", "gemini-3.8-flash")

load_dotenv(get_project_root() / ".env")

# --- Gemini API Initialization ---

try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except (TypeError, ValueError) as e:
    print(f"Error: Gemini API key not configured. Please set GEMINI_API_KEY in a .env file. Details: {e}")
    client = None

# --- Core Functions ---

def prepare_prompt(ticket_text: str) -> str:
    """Loads the prompt template and formats it with categories and the ticket text."""
    prompt_path = get_project_root() / "prompts" / "classification_prompt.txt"
    with prompt_path.open("r") as f:
        prompt_template = f.read()

    categories_str = ", ".join(TICKET_TYPES)
    return f"{prompt_template.format(categories=categories_str)}\n\nTicket: {ticket_text}"

def _strip_json_fence(response_text: str) -> str:
    """Removes an optional Markdown fenced-code wrapper around a JSON response."""
    stripped = response_text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()

def invoke_llm(prompt: str) -> dict:
    """Invokes the Gemini model and returns the parsed JSON response.

    Only the provider call itself is wrapped in the broad handler: any
    exception from the provider call is a provider failure and is reported
    as a ConnectionError. Response handling sits outside that handler, so a
    programming error in _strip_json_fence propagates instead of being
    relabelled as an API failure. Malformed JSON from the model is a
    controlled model-output validation failure and raises ValueError.
    """
    if not client:
        raise ConnectionError("Gemini client is not initialized. Check API key.")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                http_options=types.HttpOptions(timeout=30000)
            )
        )
    except Exception as e:
        raise ConnectionError(f"API call to Gemini failed: {e}") from e
    # Extract text from response
    response_text = _strip_json_fence(response.text)
    # Assuming response text is a JSON string
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned malformed JSON: {response_text[:200]}") from e

def process_llm_response(result: dict) -> tuple[str, float]:
    """Extracts and validates category and confidence from the LLM response.

    A non-numeric confidence is a model-output validation failure and
    raises ValueError, the same as a confidence outside 0.0 to 1.0.
    """
    category = result.get("category", "unknown")
    confidence = result.get("confidence", 0.0)

    if category not in TICKET_TYPES:
        category = "unknown"
        confidence = 0.0 # Confidence is unreliable if category is invalid

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        raise ValueError(f"LLM returned non-numeric confidence: {confidence!r}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"LLM returned confidence outside 0.0 to 1.0: {confidence}")

    return category, confidence

# --- Escalation Rules ---
# The rules below are declared in governance/config/scope.yaml as plain strings,
# e.g. "if confidence_score < 0.7: escalate_to: human_agent". The parser here
# turns them into evaluators so every classification result carries a routing
# decision instead of the rules living only in documentation.

def _compare(left, op: str, right) -> bool:
    if op == "==":
        return left == right
    if op == "<":
        return left < right
    if op == ">":
        return left > right
    if op == "<=":
        return left <= right
    if op == ">=":
        return left >= right
    return False

def _parse_condition(condition_str: str):
    """Parses a rule condition into a callable evaluated against a result dict."""
    condition_str = condition_str.strip()
    for op in ("==", "<=", ">=", "<", ">"):
        if op in condition_str:
            field, _, value = condition_str.partition(op)
            field = field.strip()
            value = value.strip()
            if field == "confidence_score":
                threshold = float(value)
                return lambda result, f=field, t=threshold, o=op: _compare(float(result.get(f, 0.0)), o, t)
            return lambda result, f=field, v=value, o=op: _compare(str(result.get(f, "")), o, v)
    # Bare flag, e.g. "contains_pii": matches when the flag is truthy.
    field = condition_str
    return lambda result, f=field: bool(result.get(f, False))

def _parse_action(action_str: str) -> str:
    """Parses a rule action such as 'escalate_to: human_agent' into a route name."""
    _, _, route = action_str.strip().partition(":")
    route = route.strip()
    if not route:
        raise ValueError(f"Unparseable escalation action: {action_str!r}")
    return route

def _parse_escalation_rule(rule_str: str):
    """Parses one 'if <condition>: <action>' or 'else: <action>' rule string."""
    text = rule_str.strip()
    if text.startswith("else:"):
        return (lambda result: True), _parse_action(text[len("else:"):]), text
    if text.startswith("if "):
        condition_part, _, action_part = text[len("if "):].partition(":")
        if not action_part:
            raise ValueError(f"Unparseable escalation rule: {rule_str!r}")
        return _parse_condition(condition_part), _parse_action(action_part), text
    raise ValueError(f"Unparseable escalation rule: {rule_str!r}")

def _load_escalation_rules():
    """Parses the escalation_rules list from scope.yaml.

    A malformed rule raises ValueError, failing loudly at import: a silently
    dropped rule would degrade governance invisibly, and a dropped confidence
    rule could let a low-confidence ticket reach automated response.
    """
    parsed = []
    for rule_str in config.get("escalation_rules", []):
        parsed.append(_parse_escalation_rule(str(rule_str)))
    return parsed

ESCALATION_RULES = _load_escalation_rules()

def evaluate_routing(result: dict) -> dict:
    """
    Applies the escalation rules from governance/config/scope.yaml in order and
    returns the routing decision for a classification result, e.g.
    {"route": "human_agent", "reason": "if confidence_score < 0.7: escalate_to: human_agent"}.
    """
    for condition, route, rule_text in ESCALATION_RULES:
        if condition(result):
            return {"route": route, "reason": rule_text}
    # Fail safe: if no rule matches, escalate to a human.
    return {"route": "human_agent", "reason": "no matching rule; defaulted to human review"}

# --- Logging: metadata validation, preparation, persistence ---

def _validate_metadata(subject, sender, source_ticket_id):
    """Validates and normalizes optional logging metadata.

    A value of None means absent. Any supplied value, meaning any value
    that is not None, is validated, including falsy non-strings such as 0
    or False, which raise TypeError and are never treated as absent. A
    supplied value that is not a string raises TypeError. A supplied string
    is stripped of leading and trailing whitespace, and a string that is
    empty or whitespace-only after stripping becomes None. A non-blank
    source ticket ID must be a string of at most 200 characters; a string
    longer than 200 characters raises ValueError. Returns the normalized
    (subject, sender, source_ticket_id) tuple. All of these are caller
    errors raised before any ID is assigned.
    """
    def _normalize(value, name):
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string when supplied")
        stripped = value.strip()
        return stripped if stripped else None

    subject = _normalize(subject, "subject")
    sender = _normalize(sender, "sender")
    source_ticket_id = _normalize(source_ticket_id, "source_ticket_id")
    if source_ticket_id is not None and len(source_ticket_id) > 200:
        raise ValueError("source_ticket_id must be at most 200 characters")
    return subject, sender, source_ticket_id

def _validate_decision_id(result):
    """Validates the decision ID in a direct caller's result dict.

    Returns the canonical UUIDv4 string. A missing or non-string
    decision_id raises TypeError, and an empty or wrongly formatted
    decision_id raises ValueError, both before any write. The check parses
    with uuid.UUID(s) without a version argument, asserts version 4, and
    asserts the supplied string equals str(uuid.UUID(s)), so uppercase or
    unhyphenated forms are rejected.
    """
    decision_id = result.get("decision_id") if isinstance(result, dict) else None
    if decision_id is None:
        raise TypeError("result decision_id is missing")
    if not isinstance(decision_id, str):
        raise TypeError("result decision_id must be a string")
    if not decision_id:
        raise ValueError("result decision_id must not be empty")
    try:
        parsed = uuid.UUID(decision_id)
    except ValueError:
        raise ValueError("result decision_id is not a valid UUID")
    if parsed.version != 4:
        raise ValueError("result decision_id must be a UUIDv4")
    if decision_id != str(parsed):
        raise ValueError("result decision_id must use the canonical UUID string form")
    return decision_id

def _day_path(attempt_timestamp):
    """Returns the per-day log file path for the captured attempt timestamp."""
    return get_project_root() / "logs" / f"{attempt_timestamp.date().isoformat()}.jsonl"

def _extract_sender_domain(sender):
    """Extracts the first email domain from the original sender value.

    Uses the existing email pattern, requires a dot plus a 2-letter-or-longer
    ending, and returns the domain lowercased, or None when no email is
    present. The first email wins when more than one is present.
    """
    for label, pattern in PII_PATTERNS:
        if label != "email":
            continue
        match = pattern.search(sender)
        if match is None:
            return None
        return match.group(0).split("@", 1)[1].lower()
    return None

def _prepare_primary_entry(ticket_text, result, decision_id, attempt_timestamp,
                           subject=None, sender=None, source_ticket_id=None,
                           failure_kind=None, pii_types=None):
    """Prepares one primary decision-record entry (persistence step 1).

    Receives already-validated metadata and performs no validation itself.
    Normal mode detects PII types and redacts the ticket text, redacts
    subject and sender, and extracts sender_domain from the original sender
    value before redaction. Failure mode withholds the ticket text and never
    calls detect_pii_types or redact_pii on it, and prepares metadata
    independently. The stored nested result carries decision_id but never
    the persistence fields logging_failed, record_status,
    record_failure_category, or persistence_error_type. Returns a complete
    safe prepared entry, or raises: a preparation failure means no entry is
    persisted, partial or raw.
    """
    stored_result = {
        key: value for key, value in result.items()
        if key not in ("logging_failed", "record_status",
                       "record_failure_category", "persistence_error_type")
    }
    stored_result["decision_id"] = decision_id

    entry = {
        "decision_id": decision_id,
        "ticket": None,
        "result": stored_result,
        "timestamp": attempt_timestamp.isoformat(),
    }

    if failure_kind is not None:
        entry["ticket"] = f"[Ticket withheld: PII {failure_kind} failed]"
        if failure_kind == "detection":
            entry["detection_failed"] = True
        else:
            entry["redaction_failed"] = True
            if pii_types:
                entry["pii_types"] = pii_types
    else:
        derived_types = detect_pii_types(ticket_text)
        entry["ticket"] = redact_pii(ticket_text)
        if derived_types:
            entry["pii_types"] = derived_types

    if subject is not None:
        entry["subject"] = redact_pii(subject)
    if sender is not None:
        entry["sender"] = redact_pii(sender)
        domain = _extract_sender_domain(sender)
        if domain is not None:
            entry["sender_domain"] = domain
    if source_ticket_id is not None:
        entry["source_ticket_id"] = source_ticket_id
    return entry

def _persist_entry(entry, log_path):
    """Persistence steps 2 and 3: serialize, then write.

    Returns (recorded, failure_category, error_type): recorded True means
    the entry was written; otherwise the failure category is storage_failed
    and error_type is the exception type name only. Never raises for a
    persistence failure.
    """
    try:
        log_entry(log_path, entry)
    except Exception as e:
        return False, "storage_failed", type(e).__name__
    return True, None, None

def _apply_persistence_outcome(result, recorded, failure_category, error_type):
    """Sets the returned-result persistence fields from a persistence attempt."""
    if recorded:
        result["logging_failed"] = False
        result["record_status"] = "recorded"
    else:
        result["logging_failed"] = True
        result["record_status"] = "record_failed"
        result["record_failure_category"] = failure_category
        if error_type is not None:
            result["persistence_error_type"] = error_type

def _log_fallback_validated(ticket_text, result, decision_id, attempt_timestamp,
                            subject=None, sender=None, source_ticket_id=None):
    """Logs one primary decision record for already-validated metadata and ID.

    Performs the logging step for already-validated metadata and is what
    classify_ticket calls. Returns (recorded, failure_category, error_type):
    recorded True means the primary record was written; otherwise the
    failure category is privacy_preparation_failed for a step 1 preparation
    failure or storage_failed for a step 2 or step 3 failure, with
    persistence_error_type as the exception type name for storage failures
    and absent for preparation failures. Never raises for a persistence
    failure.
    """
    try:
        entry = _prepare_primary_entry(
            ticket_text, result, decision_id, attempt_timestamp,
            subject=subject, sender=sender, source_ticket_id=source_ticket_id,
        )
    except Exception:
        return False, "privacy_preparation_failed", None
    return _persist_entry(entry, _day_path(attempt_timestamp))

def classify_ticket(ticket_text: str, subject=None, sender=None, source_ticket_id=None) -> dict:
    """
    Classifies a support ticket using the Gemini 3 Flash model with governance layers.

    PII detection and redaction run before the prompt is built. The ticket
    text sent to the model is redacted, and the PII flag from pre-redaction
    detection still forces human routing. If detection or redaction fails,
    the ticket fails closed to human review with no model call.

    Every non-empty attempt that passes metadata validation receives one
    internally generated UUIDv4 decision ID and one UTC timestamp, captured
    at ID assignment. Every controlled result that reaches the logging step
    receives one primary decision-record attempt in a per-day file. The
    returned result carries decision_id, logging_failed, and record_status;
    the stored entry never carries the persistence fields.

    Metadata (subject, sender, source_ticket_id) is validated before ID
    assignment, is never included in the prompt, and is never parsed from
    ticket text. A source ticket ID is an optional external reference under
    an unenforced caller contract: it is stored verbatim up to 200
    characters and must not carry credentials, personal information, or
    other sensitive values.

    Complies with ISO/IEC 42001:2023 requirements for AI system operation and monitoring.
    """
    if not ticket_text or ticket_text.isspace():
        empty_result = {
            "ticket_type": "unknown",
            "confidence_score": 0.0,
            "contains_pii": False,
            "model": MODEL_NAME,
            "api_failed": False
        }
        empty_result["routing_decision"] = evaluate_routing(empty_result)
        return empty_result

    # Validate metadata, then assign the ID and capture one timestamp.
    subject, sender, source_ticket_id = _validate_metadata(subject, sender, source_ticket_id)
    decision_id = str(uuid.uuid4())
    attempt_timestamp = datetime.now(timezone.utc)

    category = "unknown"
    confidence = 0.0
    api_failed = False
    model_skip = False

    # PII preprocessing runs before any model call. Detection reads the
    # original text; redaction produces a separate variable. ticket_text
    # is never overwritten, so logging still receives the original text.
    pii_types = []
    failure_kind = None
    try:
        pii_types = detect_pii_types(ticket_text)
    except Exception:
        failure_kind = "detection"
    if failure_kind is None:
        try:
            redacted_text = redact_pii(ticket_text)
        except Exception:
            failure_kind = "redaction"
    if failure_kind is not None:
        failed_result = {
            "ticket_type": "unknown",
            "confidence_score": 0.0,
            "contains_pii": True,
            "model": MODEL_NAME,
            "api_failed": False,
            "decision_id": decision_id,
            "preprocessing_failed": True,
        }
        failed_result["routing_decision"] = evaluate_routing(failed_result)
        # Path A: the failure entry goes to the per-day file with the same
        # decision ID, and never reaches the unconditional log call below.
        try:
            entry = _prepare_primary_entry(
                ticket_text, failed_result, decision_id, attempt_timestamp,
                subject=subject, sender=sender, source_ticket_id=source_ticket_id,
                failure_kind=failure_kind, pii_types=pii_types,
            )
        except Exception:
            _apply_persistence_outcome(failed_result, False, "privacy_preparation_failed", None)
            return failed_result
        recorded, failure_category, error_type = _persist_entry(entry, _day_path(attempt_timestamp))
        _apply_persistence_outcome(failed_result, recorded, failure_category, error_type)
        return failed_result
    pii_flag = len(pii_types) > 0

    # Meaningful-text check: strip the redaction tokens and whitespace.
    # A ticket with nothing left skips the model call and keeps the
    # step 2 flag, so literal token text alone does not count as PII.
    remaining = redacted_text
    for token in REDACTION_TOKENS.values():
        remaining = remaining.replace(token, "")
    if remaining.strip():
        try:
            prompt = prepare_prompt(redacted_text)
            llm_result = invoke_llm(prompt)
            category, confidence = process_llm_response(llm_result)
        except (ConnectionError, ValueError) as e:
            api_failed = True
            try:
                log_llm_error(ticket_text, str(e), decision_id=decision_id)
            except Exception:
                # Diagnostic failure must not block the primary record and
                # must not set the primary persistence outcome.
                pass
    else:
        model_skip = True

    # Routing reads the raw confidence; the returned score is rounded below.
    final_result = {
        "ticket_type": category,
        "confidence_score": confidence,
        "contains_pii": pii_flag,
        "model": MODEL_NAME,
        "api_failed": api_failed,
        "decision_id": decision_id,
    }
    if model_skip:
        final_result["model_skip"] = True
    final_result["routing_decision"] = evaluate_routing(final_result)
    final_result["confidence_score"] = round(confidence, 2)

    # Unconditional logging (Path C): every controlled result that reaches
    # this step receives one primary decision-record attempt.
    recorded, failure_category, error_type = _log_fallback_validated(
        ticket_text, final_result, decision_id, attempt_timestamp,
        subject=subject, sender=sender, source_ticket_id=source_ticket_id,
    )
    _apply_persistence_outcome(final_result, recorded, failure_category, error_type)
    return final_result

# --- Governance and Auditing ---

def log_entry(log_path: Path, entry: dict):
    """Appends a JSON entry to a specified log file.

    Serializes the entry before opening the file, so a serialization
    failure creates no file and no partial line. Then opens the file in
    append mode, writes the serialized line plus a newline in a single
    append operation, flushes, and closes. On POSIX systems a newly created
    log file is created with mode 0600 and a newly created parent directory
    with mode 0700, set at creation so the result does not depend on the
    process umask. On Windows, POSIX mode bits do not map to ACLs and a
    created file inherits the parent directory ACL.
    """
    serialized = json.dumps(entry)
    parent = log_path.parent
    if not parent.exists():
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(serialized + "\n")
        f.flush()

def log_llm_error(ticket_text: str, error_msg: str, decision_id=None):
    """Governance: Log when the AI model fails or returns an invalid response.

    The ticket preview is always redacted with redact_pii, and pii_types
    are derived from the original text and included when non-empty. The
    error message is always redacted. Redaction covers 4 PII types only,
    and applying the existing redaction function consistently does not
    guarantee all personal or sensitive information is removed. Redaction
    runs on the full text before truncating, so a PII value cut at the
    preview boundary cannot leak as a fragment. When decision_id is supplied
    it is included so the diagnostic entry can be correlated to the primary
    decision record; a direct call without decision_id omits the key.
    """
    entry = {
        "error": redact_pii(error_msg),
        "ticket_preview": redact_pii(ticket_text)[:100],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    pii_types = detect_pii_types(ticket_text)
    if pii_types:
        entry["pii_types"] = pii_types
    if decision_id is not None:
        entry["decision_id"] = decision_id
    log_path = get_project_root() / "governance" / "llm_error_log.jsonl"
    log_entry(log_path, entry)

def log_fallback(ticket_text: str, result: dict, subject=None, sender=None, source_ticket_id=None):
    """Governance: Log a controlled decision for human review and audit.

    Writes one primary decision-record attempt to the per-day file for the
    captured attempt timestamp. The stored ticket text is always redacted
    with redact_pii, and pii_types are derived from the original text and
    included when non-empty. Redaction covers 4 PII types only, and
    applying the existing redaction function consistently does not
    guarantee all personal or sensitive information is removed.

    A direct caller must supply a result containing a valid decision ID, a
    non-empty string in UUIDv4 format. A missing or non-string decision_id
    raises TypeError, and an empty or wrongly formatted decision_id raises
    ValueError, both before any write. Metadata is validated the same way
    as in classify_ticket. A source ticket ID is an optional external
    reference under an unenforced caller contract: it is stored verbatim up
    to 200 characters and must not carry credentials, personal
    information, or other sensitive values. This function does not generate
    decision IDs and does not set persistence fields on the result.
    """
    subject, sender, source_ticket_id = _validate_metadata(subject, sender, source_ticket_id)
    decision_id = _validate_decision_id(result)
    attempt_timestamp = datetime.now(timezone.utc)
    _log_fallback_validated(ticket_text, result, decision_id, attempt_timestamp,
                            subject=subject, sender=sender, source_ticket_id=source_ticket_id)
