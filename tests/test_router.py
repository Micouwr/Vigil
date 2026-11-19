# tests/test_router.py
import contextlib
import json
import os
import re
import shutil
import uuid
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock
from bot_engine.router import (
    classify_ticket,
    evaluate_routing,
    _load_escalation_rules,
    load_config,
    process_llm_response,
    invoke_llm,
    prepare_prompt,
    TICKET_TYPES,
)

# Define a consistent set of ticket types for testing purposes
MOCK_TICKET_TYPES = [
    "access_request",
    "billing_question",
    "technical_support",
    "unknown"
]

def _read_day_entries(tmp_path):
    """Reads all entries from the tmp_path day files, in file order."""
    entries = []
    logs_dir = tmp_path / "logs"
    for day_file in sorted(logs_dir.glob("*.jsonl")):
        entries.extend(
            json.loads(line) for line in day_file.read_text().splitlines() if line.strip()
        )
    return entries

@contextlib.contextmanager
def _tmp_project_root(tmp_path):
    """Yields a tmp project root holding the prompt template, with get_project_root patched."""
    (tmp_path / "prompts").mkdir(exist_ok=True)
    shutil.copy(
        Path(__file__).parent.parent / "prompts" / "classification_prompt.txt",
        tmp_path / "prompts" / "classification_prompt.txt",
    )
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        yield tmp_path

def _mock_llm(mock_client, category, confidence):
    """Configures the mocked LLM client to return one classification."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({"category": category, "confidence": confidence})
    mock_client.models.generate_content.return_value = mock_response

@pytest.fixture(autouse=True)
def _isolate_project_root(tmp_path):
    """Routes every test's logging, prompt, and config reads to a tmp project root.

    The router logs unconditionally, so a test that classifies a ticket
    would otherwise write a real day file into the repo. This fixture
    copies the prompt template and the scope config into a tmp root and
    points get_project_root there for the whole test. Tests that patch
    get_project_root to their own tmp_path keep working: the inner patch
    wins while it is active.
    """
    project_root = Path(__file__).parent.parent
    (tmp_path / "prompts").mkdir(exist_ok=True)
    shutil.copy(
        project_root / "prompts" / "classification_prompt.txt",
        tmp_path / "prompts" / "classification_prompt.txt",
    )
    scope_dir = tmp_path / "governance" / "config"
    scope_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        project_root / "governance" / "config" / "scope.yaml",
        scope_dir / "scope.yaml",
    )
    with patch("bot_engine.router.get_project_root", return_value=tmp_path):
        yield

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_classify_password_reset(mock_client):
    """Test classification for a password reset request."""
    # Arrange: Configure the mock LLM response
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "access_request",
        "confidence": 0.92
    })
    mock_client.models.generate_content.return_value = mock_response

    ticket_text = "Please reset my password ASAP."

    # Act: Call the function under test
    result = classify_ticket(ticket_text)

    # Assert: Verify the outcome
    assert result["ticket_type"] == "access_request"
    assert result["confidence_score"] == 0.92
    assert result["contains_pii"] is False

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_pii_phone_number(mock_client):
    """Test that PII is correctly detected."""
    # Arrange
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "technical_support",
        "confidence": 0.85
    })
    mock_client.models.generate_content.return_value = mock_response

    ticket_text = "My phone number is 555-123-4567."

    # Act
    result = classify_ticket(ticket_text)

    # Assert
    assert result["contains_pii"] is True
    assert result["ticket_type"] == "technical_support"

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_classify_billing_question(mock_client):
    """Test classification for a standard billing question."""
    # Arrange
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "billing_question",
        "confidence": 0.88
    })
    mock_client.models.generate_content.return_value = mock_response

    ticket_text = "Can you explain this invoice?"

    # Act
    result = classify_ticket(ticket_text)

    # Assert
    assert result["ticket_type"] == "billing_question"
    assert result["confidence_score"] == 0.88
    assert result["contains_pii"] is False

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_classify_unknown_with_pii(mock_client):
    """Test a low-confidence classification with PII."""
    # Arrange
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "unknown",
        "confidence": 0.40
    })
    mock_client.models.generate_content.return_value = mock_response

    ticket_text = "My ZIP is 40202 and my SSN is 123-45-6789."

    # Act
    result = classify_ticket(ticket_text)

    # Assert
    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.40
    assert result["contains_pii"] is True

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_llm_failure_fallback(mock_client):
    """Test the fallback mechanism when the LLM call fails."""
    # Arrange: Simulate an exception from the LLM
    mock_client.models.generate_content.side_effect = Exception("API Error")

    ticket_text = "My request is urgent."

    # Act
    result = classify_ticket(ticket_text)

    # Assert: Verify that the function falls back to 'unknown'
    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    assert result["contains_pii"] is False

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router._log_fallback_validated', return_value=(True, None, None))
@patch('bot_engine.router.client')
def test_low_confidence_triggers_fallback_log(mock_client, mock_log_fallback):
    """Test that low confidence scores trigger fallback logging."""
    # Arrange
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "technical_support",
        "confidence": 0.49
    })
    mock_client.models.generate_content.return_value = mock_response

    ticket_text = "The app is crashing."

    # Act
    classify_ticket(ticket_text)

    # Assert
    mock_log_fallback.assert_called_once()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router._log_fallback_validated', return_value=(True, None, None))
@patch('bot_engine.router.client')
def test_high_confidence_triggers_fallback_log(mock_client, mock_log_fallback):
    """Test that high confidence scores trigger logging: every controlled decision is logged."""
    # Arrange
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "access_request",
        "confidence": 0.90
    })
    mock_client.models.generate_content.return_value = mock_response

    ticket_text = "I need to reset my 2FA."

    # Act
    classify_ticket(ticket_text)

    # Assert
    mock_log_fallback.assert_called_once()

# --- Routing decision tests (escalation rules from governance/config/scope.yaml) ---

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_routing_escalates_below_07_confidence(mock_client):
    """A 0.65 score passes the 0.5 fallback-logging threshold but must still escalate per scope.yaml."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "technical_support",
        "confidence": 0.65
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("The app is crashing.")

    assert result["routing_decision"]["route"] == "human_agent"
    assert "confidence_score < 0.7" in result["routing_decision"]["reason"]

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES + ["compliance_flag"])
@patch('bot_engine.router.client')
def test_routing_escalates_compliance_flag(mock_client):
    """A compliance_flag ticket escalates even at high confidence."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "compliance_flag",
        "confidence": 0.95
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("Delete all customer records immediately.")

    assert result["routing_decision"]["route"] == "human_agent"
    assert "compliance_flag" in result["routing_decision"]["reason"]

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_routing_escalates_pii(mock_client):
    """A PII-flagged ticket escalates even at high confidence."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "technical_support",
        "confidence": 0.90
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("My phone number is 555-123-4567.")

    assert result["contains_pii"] is True
    assert result["routing_decision"]["route"] == "human_agent"
    assert "contains_pii" in result["routing_decision"]["reason"]

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_routing_automated_response(mock_client):
    """A high-confidence, non-PII ticket routes to automated response."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "access_request",
        "confidence": 0.92
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("Please reset my password ASAP.")

    assert result["routing_decision"]["route"] == "automated_response"

def test_routing_empty_ticket_escalates():
    """An empty ticket carries an escalation decision, not just unknown fields."""
    decision = evaluate_routing({
        "ticket_type": "unknown",
        "confidence_score": 0.0,
        "contains_pii": False,
    })
    assert decision["route"] == "human_agent"

def test_routing_first_matching_rule_wins():
    """Rules apply in order: a compliance_flag at low confidence cites the compliance rule."""
    decision = evaluate_routing({
        "ticket_type": "compliance_flag",
        "confidence_score": 0.40,
        "contains_pii": False,
    })
    assert decision["route"] == "human_agent"
    assert "compliance_flag" in decision["reason"]

def test_routing_escalates_confident_unknown():
    """A confident 'unknown' classification must still go to a human, not automated response."""
    decision = evaluate_routing({
        "ticket_type": "unknown",
        "confidence_score": 0.95,
        "contains_pii": False,
    })
    assert decision["route"] == "human_agent"
    assert "ticket_type == unknown" in decision["reason"]

# --- Error-log redaction (A1) and config validation (A2) ---

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_llm_failure_redacts_pii_in_error_log(mock_client, tmp_path):
    """When the LLM fails on a PII ticket, the error log must not hold raw PII."""
    mock_client.models.generate_content.side_effect = Exception("API Error")

    # Route logs and the prompt template at a scratch root so the test
    # never writes into the real repo.
    (tmp_path / "prompts").mkdir(exist_ok=True)
    shutil.copy(
        Path(__file__).parent.parent / "prompts" / "classification_prompt.txt",
        tmp_path / "prompts" / "classification_prompt.txt",
    )
    ticket_text = "My SSN is 123-45-6789, please verify my account."
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        result = classify_ticket(ticket_text)

    assert result["ticket_type"] == "unknown"

    error_log = tmp_path / "governance" / "llm_error_log.jsonl"
    error_entry = json.loads(error_log.read_text().splitlines()[-1])
    assert "123-45-6789" not in error_entry["ticket_preview"]
    assert "[REDACTED_SSN]" in error_entry["ticket_preview"]
    assert error_entry["pii_types"] == ["ssn"]

    # The fallback log written for the same ticket is redacted too.
    fallback_entry = _read_day_entries(tmp_path)[-1]
    assert "123-45-6789" not in fallback_entry["ticket"]
    assert "[REDACTED_SSN]" in fallback_entry["ticket"]


def test_load_escalation_rules_raises_on_bad_rule():
    """A malformed escalation rule fails loudly, naming the rule, not silence."""
    bad_config = {
        "escalation_rules": [
            "if confidence_score < 0.7: escalate_to: human_agent",
            "this is not a rule",
        ]
    }
    with patch('bot_engine.router.config', bad_config):
        with pytest.raises(ValueError) as excinfo:
            _load_escalation_rules()
    assert "this is not a rule" in str(excinfo.value)


def test_load_escalation_rules_loads_all_five_in_order():
    """The shipped scope.yaml rules load as five parsers, in file order."""
    parsed = _load_escalation_rules()
    assert len(parsed) == 5
    assert [rule_text for _, _, rule_text in parsed] == [
        "if ticket_type == compliance_flag: escalate_to: human_agent",
        "if ticket_type == unknown: escalate_to: human_agent",
        "if confidence_score < 0.7: escalate_to: human_agent",
        "if contains_pii: escalate_to: human_agent",
        "else: route_to: automated_response",
    ]


def test_load_config_accepts_current_scope():
    """The shipped scope.yaml passes validation unchanged."""
    config = load_config()
    assert config["bot_config"]["model_name"] == "gemini-3.8-flash"


def test_load_config_accepts_missing_threshold(tmp_path):
    """A scope.yaml without a confidence threshold loads, since the key is removed."""
    scope_yaml = tmp_path / "scope.yaml"
    scope_yaml.write_text(
        "bot_config:\n"
        "  model_name: gemini-3.8-flash\n"
        "ticket_types: [access_request]\n"
        "escalation_rules: ['else: route_to: automated_response']\n"
    )
    config = load_config(scope_yaml)
    assert config["bot_config"]["model_name"] == "gemini-3.8-flash"


def test_load_config_accepts_legacy_threshold_key(tmp_path):
    """A scope.yaml that still carries a legacy threshold key loads, since validation ignores extra keys."""
    scope_yaml = tmp_path / "scope.yaml"
    scope_yaml.write_text(
        "bot_config:\n"
        "  confidence_threshold: 0.5\n"
        "  model_name: gemini-3.8-flash\n"
        "ticket_types: [access_request]\n"
        "escalation_rules: ['else: route_to: automated_response']\n"
    )
    config = load_config(scope_yaml)
    assert config["bot_config"]["model_name"] == "gemini-3.8-flash"

# --- Code-fix spec: unrounded confidence, validation, fences, timeout, redaction ---

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_routing_uses_unrounded_confidence_below_gate(mock_client):
    """0.696 rounds to 0.7 for display but must still escalate on the raw value."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "technical_support",
        "confidence": 0.696
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("The app is crashing.")

    assert result["routing_decision"]["route"] == "human_agent"
    assert "confidence_score < 0.7" in result["routing_decision"]["reason"]
    assert result["confidence_score"] == 0.7

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_routing_uses_unrounded_confidence_above_gate(mock_client):
    """0.704 rounds to 0.7 for display but must still automate on the raw value."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "technical_support",
        "confidence": 0.704
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("The app is crashing.")

    assert result["routing_decision"]["route"] == "automated_response"
    assert result["confidence_score"] == 0.7

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.log_llm_error')
@patch('bot_engine.router.log_fallback')
@patch('bot_engine.router.client')
def test_confidence_above_one_rejected(mock_client, mock_log_fallback, mock_log_error):
    """A confidence of 95.0 is rejected: unknown, 0.0, and an error log entry."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "access_request",
        "confidence": 95.0
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("Please reset my password ASAP.")

    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    mock_log_error.assert_called_once()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.log_llm_error')
@patch('bot_engine.router.log_fallback')
@patch('bot_engine.router.client')
def test_confidence_below_zero_rejected(mock_client, mock_log_fallback, mock_log_error):
    """A confidence of -0.1 is rejected: unknown, and an error log entry."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "access_request",
        "confidence": -0.1
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("Please reset my password ASAP.")

    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    mock_log_error.assert_called_once()

def test_confidence_boundaries_pass_validation():
    """0.0 and 1.0 are valid confidences and pass through unchanged."""
    assert process_llm_response({"category": "access_request", "confidence": 0.0}) == ("access_request", 0.0)
    assert process_llm_response({"category": "access_request", "confidence": 1.0}) == ("access_request", 1.0)

def test_missing_confidence_defaults_to_zero():
    """A response without a confidence key defaults to 0.0."""
    assert process_llm_response({"category": "access_request"}) == ("access_request", 0.0)

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.log_llm_error')
@patch('bot_engine.router.log_fallback')
@patch('bot_engine.router.client')
def test_string_confidence_fails_closed(mock_client, mock_log_fallback, mock_log_error):
    """A non-numeric confidence string fails closed: unknown, and an error log entry."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "access_request",
        "confidence": "high"
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("Please reset my password ASAP.")

    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    mock_log_error.assert_called_once()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.log_llm_error')
@patch('bot_engine.router.log_fallback')
@patch('bot_engine.router.client')
def test_none_confidence_fails_closed(mock_client, mock_log_fallback, mock_log_error):
    """A null confidence fails closed: unknown, and an error log entry."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "access_request",
        "confidence": None
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("Please reset my password ASAP.")

    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    mock_log_error.assert_called_once()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_fenced_json_response_parses(mock_client):
    """A response wrapped in a ```json fence parses and classifies normally."""
    mock_response = MagicMock()
    mock_response.text = '```json\n{"category": "access_request", "confidence": 0.92}\n```'
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("Please reset my password ASAP.")

    assert result["ticket_type"] == "access_request"
    assert result["confidence_score"] == 0.92

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_fenced_json_with_surrounding_whitespace_parses(mock_client):
    """A fenced response with leading and trailing whitespace parses."""
    mock_response = MagicMock()
    mock_response.text = '\n\n  ```\n{"category": "billing_question", "confidence": 0.88}\n```  \n\n'
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("Can you explain this invoice?")

    assert result["ticket_type"] == "billing_question"
    assert result["confidence_score"] == 0.88

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.log_llm_error')
@patch('bot_engine.router.log_fallback')
@patch('bot_engine.router.client')
def test_malformed_json_still_fails_closed(mock_client, mock_log_fallback, mock_log_error):
    """A genuinely malformed response still returns unknown with an error log entry."""
    mock_response = MagicMock()
    mock_response.text = 'not json at all'
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("Please reset my password ASAP.")

    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    mock_log_error.assert_called_once()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.log_llm_error')
@patch('bot_engine.router.log_fallback')
@patch('bot_engine.router.client')
def test_api_failure_sets_api_failed_flag(mock_client, mock_log_fallback, mock_log_error):
    """A raising LLM call produces api_failed True, unknown, and an error log entry."""
    mock_client.models.generate_content.side_effect = Exception("API Error")

    result = classify_ticket("My request is urgent.")

    assert result["api_failed"] is True
    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    mock_log_error.assert_called_once()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_successful_call_sets_api_failed_false(mock_client):
    """A successful LLM call produces api_failed False."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "access_request",
        "confidence": 0.92
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("Please reset my password ASAP.")

    assert result["api_failed"] is False

@patch('bot_engine.router.client')
def test_generate_call_uses_30_second_timeout(mock_client):
    """The generate call carries http_options with a 30-second timeout."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "access_request",
        "confidence": 0.92
    })
    mock_client.models.generate_content.return_value = mock_response

    invoke_llm("Classify this ticket.")

    config = mock_client.models.generate_content.call_args.kwargs["config"]
    assert config.http_options.timeout == 30000

def test_error_log_redacts_error_field(tmp_path):
    """An error string carrying PII is redacted in the written error log entry."""
    from bot_engine.router import log_llm_error
    ticket_text = "Please verify my account."
    error_msg = "LLM returned malformed JSON: {\"ssn\": \"123-45-6789\"}"
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        log_llm_error(ticket_text, error_msg)

    error_log = tmp_path / "governance" / "llm_error_log.jsonl"
    entry = json.loads(error_log.read_text().splitlines()[-1])
    assert "123-45-6789" not in entry["error"]
    assert "[REDACTED_SSN]" in entry["error"]

@patch('bot_engine.router.client')
def test_malformed_json_error_bounded_to_200_chars(mock_client):
    """The malformed-JSON error message carries at most 200 chars of response text."""
    mock_response = MagicMock()
    mock_response.text = "x" * 500
    mock_client.models.generate_content.return_value = mock_response

    with pytest.raises(ValueError) as excinfo:
        invoke_llm("Classify this ticket.")

    assert str(excinfo.value) == "LLM returned malformed JSON: " + "x" * 200

def test_invalid_category_maps_to_unknown():
    """A category string outside the configured types maps to unknown with 0.0."""
    assert process_llm_response({"category": "not_a_category", "confidence": 0.9}) == ("unknown", 0.0)

@patch('bot_engine.router.log_llm_error')
@patch('bot_engine.router.log_fallback')
def test_empty_ticket_returns_unknown_without_logging(mock_log_fallback, mock_log_error):
    """Empty and whitespace-only tickets return unknown with no log write."""
    for ticket_text in ("", "   "):
        result = classify_ticket(ticket_text)
        assert result["ticket_type"] == "unknown"
        assert result["confidence_score"] == 0.0
        assert result["api_failed"] is False
        assert result["routing_decision"]["route"] == "human_agent"
    mock_log_fallback.assert_not_called()
    mock_log_error.assert_not_called()

def test_prepare_prompt_formats_ticket_and_categories():
    """prepare_prompt loads the template and formats in the ticket and categories."""
    ticket_text = "My printer is jammed."
    prompt = prepare_prompt(ticket_text)
    assert ticket_text in prompt
    assert ", ".join(TICKET_TYPES) in prompt
    assert "JSON object" in prompt


# --- PII ordering: redact before submission (approved spec, 2026-11-18) ---

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_pii_ordering_email_redacted_before_model(mock_client):
    """An email is redacted in the text sent to the model, and the flag still forces human routing."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "technical_support",
        "confidence": 0.90
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("Email me at j.smith@example.com about my ticket.")

    contents = mock_client.models.generate_content.call_args.kwargs["contents"]
    assert "[REDACTED_EMAIL]" in contents
    assert "j.smith@example.com" not in contents
    assert result["contains_pii"] is True
    assert result["routing_decision"]["route"] == "human_agent"

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_pii_ordering_phone_redacted_before_model(mock_client):
    """A phone number is redacted in the text sent to the model, and the flag still forces human routing."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "technical_support",
        "confidence": 0.90
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("My phone number is 555-123-4567, call me back.")

    contents = mock_client.models.generate_content.call_args.kwargs["contents"]
    assert "[REDACTED_PHONE]" in contents
    assert "555-123-4567" not in contents
    assert result["contains_pii"] is True
    assert result["routing_decision"]["route"] == "human_agent"

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_pii_ordering_card_redacted_before_model(mock_client):
    """A Luhn-valid card number is redacted in the text sent to the model."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "billing_question",
        "confidence": 0.90
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("My card 4242424242424242 was charged twice, please refund.")

    contents = mock_client.models.generate_content.call_args.kwargs["contents"]
    assert "[REDACTED_CARD]" in contents
    assert "4242424242424242" not in contents
    assert result["contains_pii"] is True
    assert result["routing_decision"]["route"] == "human_agent"

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_pii_ordering_ssn_redacted_before_model(mock_client):
    """A dashed SSN is redacted in the text sent to the model."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "technical_support",
        "confidence": 0.90
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("My SSN is 123-45-6789, please verify my account.")

    contents = mock_client.models.generate_content.call_args.kwargs["contents"]
    assert "[REDACTED_SSN]" in contents
    assert "123-45-6789" not in contents
    assert result["contains_pii"] is True
    assert result["routing_decision"]["route"] == "human_agent"

# LIMITATION test, not a pass: a card-like number that fails the Luhn check
# is treated as not PII, is not redacted, and reaches the model raw.
@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_luhn_fail_card_reaches_model_limitation(mock_client):
    """A Luhn-failing card number is not detected and is sent to the model raw."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "technical_support",
        "confidence": 0.92
    })
    mock_client.models.generate_content.return_value = mock_response

    result = classify_ticket("Order number 4111111111111112, please check the status.")

    contents = mock_client.models.generate_content.call_args.kwargs["contents"]
    assert "4111111111111112" in contents
    assert "[REDACTED_CARD]" not in contents
    assert result["contains_pii"] is False
    assert result["routing_decision"]["route"] == "automated_response"

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_pii_ordering_no_pii_prompt_unchanged(mock_client):
    """A ticket with no PII sends the same prompt as the original text would."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "access_request",
        "confidence": 0.92
    })
    mock_client.models.generate_content.return_value = mock_response

    ticket_text = "Please reset my password ASAP."
    result = classify_ticket(ticket_text)

    contents = mock_client.models.generate_content.call_args.kwargs["contents"]
    assert contents == prepare_prompt(ticket_text)
    assert result["contains_pii"] is False
    assert result["ticket_type"] == "access_request"
    assert result["routing_decision"]["route"] == "automated_response"

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_pii_ordering_only_pii_skips_model(mock_client, tmp_path):
    """A ticket with only PII values skips the model call and logs the correct types."""
    (tmp_path / "prompts").mkdir(exist_ok=True)
    shutil.copy(
        Path(__file__).parent.parent / "prompts" / "classification_prompt.txt",
        tmp_path / "prompts" / "classification_prompt.txt",
    )
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        result = classify_ticket("123-45-6789   ")

    mock_client.models.generate_content.assert_not_called()
    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    assert result["contains_pii"] is True
    assert result["routing_decision"]["route"] == "human_agent"

    entry = _read_day_entries(tmp_path)[-1]
    assert entry["pii_types"] == ["ssn"]
    assert "123-45-6789" not in entry["ticket"]
    assert "[REDACTED_SSN]" in entry["ticket"]

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_pii_ordering_literal_tokens_only_keep_flag_false(mock_client):
    """Literal token text with no real PII skips the model but keeps contains_pii False."""
    result = classify_ticket("[REDACTED_SSN]   ")

    mock_client.models.generate_content.assert_not_called()
    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    assert result["contains_pii"] is False
    assert result["routing_decision"]["route"] == "human_agent"

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.detect_pii_types', side_effect=RuntimeError("detection boom"))
@patch('bot_engine.router.client')
def test_pii_ordering_detection_failure_fails_closed(mock_client, mock_detect, tmp_path):
    """A detection failure fails closed with a detection_failed entry and no model call."""
    (tmp_path / "prompts").mkdir(exist_ok=True)
    shutil.copy(
        Path(__file__).parent.parent / "prompts" / "classification_prompt.txt",
        tmp_path / "prompts" / "classification_prompt.txt",
    )
    ticket_text = "My SSN is 123-45-6789, please verify my account."
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        result = classify_ticket(ticket_text)

    mock_client.models.generate_content.assert_not_called()
    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    assert result["contains_pii"] is True
    assert result["api_failed"] is False
    assert result["routing_decision"]["route"] == "human_agent"

    entry = _read_day_entries(tmp_path)[-1]
    assert "123-45-6789" not in json.dumps(entry)
    assert "PII detection failed" in entry["ticket"]
    assert "pii_types" not in entry
    assert entry["detection_failed"] is True
    assert "redaction_failed" not in entry
    assert not (tmp_path / "governance" / "llm_error_log.jsonl").exists()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.redact_pii', side_effect=RuntimeError("redaction boom"))
@patch('bot_engine.router.client')
def test_pii_ordering_redaction_failure_preserves_types(mock_client, mock_redact, tmp_path):
    """A redaction failure preserves the detected types with a redaction_failed entry."""
    (tmp_path / "prompts").mkdir(exist_ok=True)
    shutil.copy(
        Path(__file__).parent.parent / "prompts" / "classification_prompt.txt",
        tmp_path / "prompts" / "classification_prompt.txt",
    )
    ticket_text = "Email j.smith@example.com about my ticket."
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        result = classify_ticket(ticket_text)

    mock_client.models.generate_content.assert_not_called()
    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    assert result["contains_pii"] is True
    assert result["api_failed"] is False
    assert result["routing_decision"]["route"] == "human_agent"

    entry = _read_day_entries(tmp_path)[-1]
    assert "j.smith@example.com" not in json.dumps(entry)
    assert "PII redaction failed" in entry["ticket"]
    assert entry["pii_types"] == ["email"]
    assert entry["redaction_failed"] is True
    assert "detection_failed" not in entry
    assert not (tmp_path / "governance" / "llm_error_log.jsonl").exists()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_pii_ordering_gemini_failure_with_pii_ticket(mock_client, tmp_path):
    """When Gemini fails on a PII ticket, both logs are redacted with correct types."""
    mock_client.models.generate_content.side_effect = Exception("API Error")
    (tmp_path / "prompts").mkdir(exist_ok=True)
    shutil.copy(
        Path(__file__).parent.parent / "prompts" / "classification_prompt.txt",
        tmp_path / "prompts" / "classification_prompt.txt",
    )
    ticket_text = "My SSN is 123-45-6789, please verify my account."
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        result = classify_ticket(ticket_text)

    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    assert result["contains_pii"] is True
    assert result["api_failed"] is True
    assert result["routing_decision"]["route"] == "human_agent"

    contents = mock_client.models.generate_content.call_args.kwargs["contents"]
    assert "[REDACTED_SSN]" in contents
    assert "123-45-6789" not in contents

    error_log = tmp_path / "governance" / "llm_error_log.jsonl"
    error_entry = json.loads(error_log.read_text().splitlines()[-1])
    assert "123-45-6789" not in json.dumps(error_entry)
    assert "[REDACTED_SSN]" in error_entry["ticket_preview"]
    assert error_entry["pii_types"] == ["ssn"]

    fallback_entry = _read_day_entries(tmp_path)[-1]
    assert "123-45-6789" not in json.dumps(fallback_entry)
    assert "[REDACTED_SSN]" in fallback_entry["ticket"]
    assert fallback_entry["pii_types"] == ["ssn"]

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_pii_ordering_malformed_json_echo_redacted_in_error_log(mock_client, tmp_path):
    """PII echoed in a malformed model response is redacted in the error log."""
    mock_response = MagicMock()
    mock_response.text = "not json: SSN 123-45-6789, email j.smith@example.com"
    mock_client.models.generate_content.return_value = mock_response
    (tmp_path / "prompts").mkdir(exist_ok=True)
    shutil.copy(
        Path(__file__).parent.parent / "prompts" / "classification_prompt.txt",
        tmp_path / "prompts" / "classification_prompt.txt",
    )
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        result = classify_ticket("Please verify my account.")

    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    assert result["api_failed"] is True

    error_log = tmp_path / "governance" / "llm_error_log.jsonl"
    error_entry = json.loads(error_log.read_text().splitlines()[-1])
    assert "123-45-6789" not in json.dumps(error_entry)
    assert "j.smith@example.com" not in json.dumps(error_entry)
    assert "[REDACTED_SSN]" in error_entry["error"]
    assert "[REDACTED_EMAIL]" in error_entry["error"]

def test_pii_ordering_direct_log_fallback_preserves_types(tmp_path):
    """A direct log_fallback call with original text records redacted text and correct types."""
    from bot_engine.router import log_fallback
    ticket_text = "My SSN is 123-45-6789, please verify my account."
    result = {"ticket_type": "unknown", "confidence_score": 0.0, "contains_pii": True,
              "decision_id": str(uuid.uuid4())}
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        log_fallback(ticket_text, result)

    entry = _read_day_entries(tmp_path)[-1]
    assert "123-45-6789" not in json.dumps(entry)
    assert "[REDACTED_SSN]" in entry["ticket"]
    assert entry["pii_types"] == ["ssn"]

def test_pii_ordering_direct_log_llm_error_preserves_types(tmp_path):
    """A direct log_llm_error call with original text records redacted fields and correct types."""
    from bot_engine.router import log_llm_error
    ticket_text = "Contact j.smith@example.com about my account."
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        log_llm_error(ticket_text, "API failed for j.smith@example.com")

    entry = json.loads((tmp_path / "governance" / "llm_error_log.jsonl").read_text().splitlines()[-1])
    assert "j.smith@example.com" not in json.dumps(entry)
    assert "[REDACTED_EMAIL]" in entry["ticket_preview"]
    assert "[REDACTED_EMAIL]" in entry["error"]
    assert entry["pii_types"] == ["email"]

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_pii_ordering_classify_path_preserves_types(mock_client, tmp_path):
    """A low-confidence PII ticket keeps correct types in the fallback log on the real path."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "technical_support",
        "confidence": 0.40
    })
    mock_client.models.generate_content.return_value = mock_response
    (tmp_path / "prompts").mkdir(exist_ok=True)
    shutil.copy(
        Path(__file__).parent.parent / "prompts" / "classification_prompt.txt",
        tmp_path / "prompts" / "classification_prompt.txt",
    )
    ticket_text = "My phone number is 555-123-4567, the app crashes."
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        result = classify_ticket(ticket_text)

    contents = mock_client.models.generate_content.call_args.kwargs["contents"]
    assert "[REDACTED_PHONE]" in contents
    assert "555-123-4567" not in contents
    assert result["contains_pii"] is True
    assert result["routing_decision"]["route"] == "human_agent"

    entry = _read_day_entries(tmp_path)[-1]
    assert "555-123-4567" not in json.dumps(entry)
    assert "[REDACTED_PHONE]" in entry["ticket"]
    assert entry["pii_types"] == ["phone"]

# --- Full-decision logging: Fix 1 and Fix 2 (spec Revision 25) ---

UUIDV4_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)

def _read_error_entries(tmp_path):
    """Reads all entries from the tmp_path diagnostic error log, or an empty list."""
    error_log = tmp_path / "governance" / "llm_error_log.jsonl"
    if not error_log.exists():
        return []
    return [json.loads(line) for line in error_log.read_text().splitlines() if line.strip()]

def _fdopen_with_fault(fault):
    """Returns an fdopen replacement whose file faults on the named method.

    The write fault writes the data first, then raises, simulating a
    partial write. The flush fault raises instead of flushing. The close
    fault runs the real close, then raises, simulating a close error.
    """
    real_fdopen = os.fdopen
    def faulty_fdopen(fd, mode):
        f = real_fdopen(fd, mode)
        if fault == "write":
            real_write = f.write
            def write(data):
                real_write(data)
                raise OSError("disk full")
            f.write = write
        elif fault == "flush":
            def flush():
                raise OSError("flush failed")
            f.flush = flush
        elif fault == "close":
            real_close = f.close
            def close():
                real_close()
                raise OSError("close failed")
            f.close = close
        return f
    return faulty_fdopen

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_every_decision_logged_automated_099(mock_client, tmp_path):
    """A 0.99 ticket with no PII routes automated and writes one entry to today's day file."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        result = classify_ticket("Please reset my password.")
    assert result["routing_decision"]["route"] == "automated_response"
    assert result["logging_failed"] is False
    assert result["record_status"] == "recorded"
    assert "record_failure_category" not in result
    assert "persistence_error_type" not in result
    day_files = sorted((tmp_path / "logs").glob("*.jsonl"))
    assert len(day_files) == 1
    entries = _read_day_entries(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert day_files[0].name == entry["timestamp"][:10] + ".jsonl"
    assert entry["decision_id"] == result["decision_id"]
    assert entry["result"]["decision_id"] == result["decision_id"]
    assert entry["result"]["routing_decision"] == result["routing_decision"]

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_every_decision_logged_human_06(mock_client, tmp_path):
    """A 0.6 ticket with a known category routes to human and writes one entry."""
    _mock_llm(mock_client, "technical_support", 0.6)
    with _tmp_project_root(tmp_path):
        result = classify_ticket("The app crashes on launch.")
    assert result["routing_decision"]["route"] == "human_agent"
    assert result["record_status"] == "recorded"
    assert len(_read_day_entries(tmp_path)) == 1

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_every_decision_logged_pii_099(mock_client, tmp_path):
    """A 0.99 ticket with PII routes to human and writes one redacted entry with pii_types."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        result = classify_ticket("My SSN is 123-45-6789, please verify my account.")
    assert result["routing_decision"]["route"] == "human_agent"
    assert result["contains_pii"] is True
    assert result["record_status"] == "recorded"
    entries = _read_day_entries(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert "123-45-6789" not in json.dumps(entry)
    assert "[REDACTED_SSN]" in entry["ticket"]
    assert entry["pii_types"] == ["ssn"]

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_every_decision_logged_unknown_099(mock_client, tmp_path):
    """An unknown category at 0.99 routes to human and writes one entry."""
    _mock_llm(mock_client, "not_a_category", 0.99)
    with _tmp_project_root(tmp_path):
        result = classify_ticket("Something odd happened.")
    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    assert result["routing_decision"]["route"] == "human_agent"
    assert len(_read_day_entries(tmp_path)) == 1

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_api_failure_writes_decision_and_error_entries(mock_client, tmp_path):
    """An API failure returns one decision entry with api_failed True and one error log entry."""
    mock_client.models.generate_content.side_effect = ConnectionError("connection reset")
    with _tmp_project_root(tmp_path):
        result = classify_ticket("Please reset my password.")
    assert result["api_failed"] is True
    assert result["routing_decision"]["route"] == "human_agent"
    assert result["record_status"] == "recorded"
    entries = _read_day_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["result"]["api_failed"] is True
    error_entries = _read_error_entries(tmp_path)
    assert len(error_entries) == 1
    assert error_entries[0]["decision_id"] == result["decision_id"]

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_model_call_exception_boundary(mock_client, tmp_path):
    """A programming error in prompt preparation or response processing propagates as classify_failed."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.prepare_prompt', side_effect=TypeError("prompt boom")):
            with pytest.raises(TypeError):
                classify_ticket("Please reset my password.")
        assert not (tmp_path / "logs").exists()
        assert not (tmp_path / "governance" / "llm_error_log.jsonl").exists()
        with patch('bot_engine.router.process_llm_response', side_effect=TypeError("response boom")):
            with pytest.raises(TypeError):
                classify_ticket("Please reset my password.")
        assert not (tmp_path / "logs").exists()
        assert not (tmp_path / "governance" / "llm_error_log.jsonl").exists()
        # A transport error from the provider call still returns an api_failed result.
        mock_client.models.generate_content.side_effect = RuntimeError("socket closed")
        result = classify_ticket("Please reset my password.")
        assert result["api_failed"] is True
        assert result["record_status"] == "recorded"
        assert len(_read_day_entries(tmp_path)) == 1
        assert len(_read_error_entries(tmp_path)) == 1

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_pii_only_ticket_skips_model_with_decision_id(mock_client, tmp_path):
    """A ticket of only PII tokens skips the model, returns unknown 0.0 with model_skip True and a decision ID."""
    with _tmp_project_root(tmp_path):
        result = classify_ticket("123-45-6789")
    mock_client.models.generate_content.assert_not_called()
    assert result["ticket_type"] == "unknown"
    assert result["confidence_score"] == 0.0
    assert result["model_skip"] is True
    assert UUIDV4_PATTERN.fullmatch(result["decision_id"])
    assert result["record_status"] == "recorded"
    entries = _read_day_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["result"]["model_skip"] is True
    assert entries[0]["decision_id"] == result["decision_id"]

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.detect_pii_types', side_effect=RuntimeError("detection boom"))
@patch('bot_engine.router.client')
def test_preprocessing_detection_failure_single_entry(mock_client, mock_detect, tmp_path):
    """A detection failure writes exactly one entry with the ID and marker, and no second entry."""
    with _tmp_project_root(tmp_path):
        result = classify_ticket("Please reset my password.")
    mock_client.models.generate_content.assert_not_called()
    assert mock_detect.call_count == 1
    assert result["preprocessing_failed"] is True
    assert result["contains_pii"] is True
    assert result["routing_decision"]["route"] == "human_agent"
    assert result["record_status"] == "recorded"
    entries = _read_day_entries(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["decision_id"] == result["decision_id"]
    assert entry["detection_failed"] is True
    assert "pii_types" not in entry
    assert "redaction_failed" not in entry
    assert "Please reset my password." not in json.dumps(entry)

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.redact_pii', side_effect=RuntimeError("redaction boom"))
@patch('bot_engine.router.client')
def test_preprocessing_redaction_failure_single_entry(mock_client, mock_redact, tmp_path):
    """A redaction failure preserves the detected types, writes exactly one entry, and redacts once."""
    with _tmp_project_root(tmp_path):
        result = classify_ticket("Email j.smith@example.com about my ticket.")
    mock_client.models.generate_content.assert_not_called()
    assert mock_redact.call_count == 1
    assert result["preprocessing_failed"] is True
    assert result["contains_pii"] is True
    entries = _read_day_entries(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["redaction_failed"] is True
    assert entry["pii_types"] == ["email"]
    assert "detection_failed" not in entry
    assert "j.smith@example.com" not in json.dumps(entry)

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.detect_pii_types', side_effect=RuntimeError("detection boom"))
@patch('bot_engine.router.client')
def test_preprocessing_failure_with_metadata(mock_client, mock_detect, tmp_path):
    """A preprocessing failure with metadata stores the metadata redacted, with sender_domain."""
    with _tmp_project_root(tmp_path):
        result = classify_ticket(
            "Please reset my password.",
            subject="Reset for jane@example.com",
            sender="Jane Smith <jane@example.com>",
            source_ticket_id="INC-42",
        )
    assert result["record_status"] == "recorded"
    entries = _read_day_entries(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["subject"] == "Reset for [REDACTED_EMAIL]"
    assert entry["sender"] == "Jane Smith <[REDACTED_EMAIL]>"
    assert entry["sender_domain"] == "example.com"
    assert entry["source_ticket_id"] == "INC-42"
    assert "jane@example.com" not in json.dumps(entry)

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_preparation_failure_writes_no_entry(mock_client, tmp_path):
    """If metadata preparation fails after subject redaction succeeds, no entry is written at all."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router._extract_sender_domain', side_effect=RuntimeError("domain boom")):
            result = classify_ticket(
                "Please reset my password.",
                subject="A subject",
                sender="Jane <jane@example.com>",
            )
    assert result["logging_failed"] is True
    assert result["record_status"] == "record_failed"
    assert result["record_failure_category"] == "privacy_preparation_failed"
    assert "persistence_error_type" not in result
    assert not (tmp_path / "logs").exists()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_decision_id_uuidv4_unique_and_unseeded(mock_client, tmp_path):
    """Decision IDs are UUIDv4, differ across attempts, and a pre-seeded day file does not affect the next ID."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        seeded_id = "11111111-1111-4111-8111-111111111111"
        day_file = logs_dir / (datetime.now(timezone.utc).date().isoformat() + ".jsonl")
        day_file.write_text(json.dumps({"decision_id": seeded_id}) + "\n")
        result1 = classify_ticket("Please reset my password.")
        result2 = classify_ticket("Please reset my password.")
    assert UUIDV4_PATTERN.fullmatch(result1["decision_id"])
    assert UUIDV4_PATTERN.fullmatch(result2["decision_id"])
    assert result1["decision_id"] != result2["decision_id"]
    assert result1["decision_id"] != seeded_id
    assert result2["decision_id"] != seeded_id
    entries = _read_day_entries(tmp_path)
    assert len(entries) == 3
    assert entries[0]["decision_id"] == seeded_id

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_metadata_stored_redacted_and_excluded_from_prompt(mock_client, tmp_path):
    """Subject and sender are stored redacted, sender_domain is derived, source ID is verbatim, and none reach the prompt."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        result = classify_ticket(
            "Please reset my password.",
            subject="Reset request for jane@example.com",
            sender="Jane",
            source_ticket_id="INC-12345",
        )
    assert result["record_status"] == "recorded"
    entry = _read_day_entries(tmp_path)[0]
    assert entry["subject"] == "Reset request for [REDACTED_EMAIL]"
    assert entry["sender"] == "Jane"
    assert "sender_domain" not in entry
    assert entry["source_ticket_id"] == "INC-12345"
    assert "jane@example.com" not in json.dumps(entry)
    contents = mock_client.models.generate_content.call_args.kwargs["contents"]
    assert "Reset request for jane@example.com" not in contents
    assert "jane@example.com" not in contents
    assert "INC-12345" not in contents

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_metadata_keys_absent_when_blank(mock_client, tmp_path):
    """When no metadata is supplied, the entry carries no subject, sender, sender_domain, or source_ticket_id keys."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        result = classify_ticket("Please reset my password.")
    entry = _read_day_entries(tmp_path)[0]
    for key in ("subject", "sender", "sender_domain", "source_ticket_id"):
        assert key not in entry
    assert result["record_status"] == "recorded"

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_sender_examples(mock_client, tmp_path):
    """Sender examples: name with email, bare email, multiple emails, and malformed addresses."""
    _mock_llm(mock_client, "access_request", 0.99)
    cases = [
        ("John Smith <john@example.com>", "John Smith <[REDACTED_EMAIL]>", "example.com"),
        ("john@example.com", "[REDACTED_EMAIL]", "example.com"),
        ("a@x.com and b@y.com", "[REDACTED_EMAIL] and [REDACTED_EMAIL]", "x.com"),
        ("john@", "john@", None),
        ("john@example", "john@example", None),
    ]
    with _tmp_project_root(tmp_path):
        for sender, expected_stored, expected_domain in cases:
            result = classify_ticket("Please reset my password.", sender=sender)
            assert result["record_status"] == "recorded"
    entries = _read_day_entries(tmp_path)
    assert len(entries) == len(cases)
    for entry, (sender, expected_stored, expected_domain) in zip(entries, cases):
        assert entry["sender"] == expected_stored
        if expected_domain is None:
            assert "sender_domain" not in entry
        else:
            assert entry["sender_domain"] == expected_domain

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_metadata_validation_raises_before_id_assignment(mock_client, tmp_path):
    """Invalid metadata raises before any ID is assigned and before any write."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        with pytest.raises(TypeError):
            classify_ticket("Please reset my password.", subject=0)
        with pytest.raises(TypeError):
            classify_ticket("Please reset my password.", sender=False)
        with pytest.raises(ValueError):
            classify_ticket("Please reset my password.", source_ticket_id="x" * 201)
    assert not (tmp_path / "logs").exists()
    mock_client.models.generate_content.assert_not_called()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_persistence_failure_keeps_result_and_next_id_differs(mock_client, tmp_path):
    """A log write raising OSError returns the result with storage_failed, and the next attempt gets a different ID."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.log_entry', side_effect=OSError("disk full")):
            result1 = classify_ticket("Please reset my password.")
        assert result1["logging_failed"] is True
        assert result1["record_status"] == "record_failed"
        assert result1["record_failure_category"] == "storage_failed"
        assert result1["persistence_error_type"] == "OSError"
        assert result1["ticket_type"] == "access_request"
        assert result1["routing_decision"]["route"] == "automated_response"
        assert not (tmp_path / "logs").exists()
        result2 = classify_ticket("Please reset my password.")
        assert result2["decision_id"] != result1["decision_id"]
        assert result2["record_status"] == "recorded"
        assert len(_read_day_entries(tmp_path)) == 1

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_unexpected_classification_exception_writes_no_record(mock_client, tmp_path):
    """An unexpected classification exception before a usable result returns no result and writes no primary record."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        good = classify_ticket("Please reset my password.")
        assert good["record_status"] == "recorded"
        assert len(_read_day_entries(tmp_path)) == 1
        with patch('bot_engine.router.evaluate_routing', side_effect=RuntimeError("routing boom")):
            with pytest.raises(RuntimeError):
                classify_ticket("Please reset my password.")
        assert len(_read_day_entries(tmp_path)) == 1

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.detect_pii_types', side_effect=RuntimeError("detection boom"))
@patch('bot_engine.router.client')
def test_persistence_failure_on_preprocessing_failure_path(mock_client, mock_detect, tmp_path):
    """A log write raising OSError on the preprocessing-failure path keeps the failure visible with storage_failed."""
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.log_entry', side_effect=OSError("disk full")):
            result = classify_ticket("Please reset my password.")
    assert result["preprocessing_failed"] is True
    assert result["contains_pii"] is True
    assert result["logging_failed"] is True
    assert result["record_status"] == "record_failed"
    assert result["record_failure_category"] == "storage_failed"
    assert result["persistence_error_type"] == "OSError"
    assert not (tmp_path / "logs").exists()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_unexpected_persistence_exception(mock_client, tmp_path):
    """A log write raising RuntimeError is still a storage_failed record_failed, with no expected/unexpected field."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.log_entry', side_effect=RuntimeError("weird boom")):
            result = classify_ticket("Please reset my password.")
    assert result["logging_failed"] is True
    assert result["record_status"] == "record_failed"
    assert result["record_failure_category"] == "storage_failed"
    assert result["persistence_error_type"] == "RuntimeError"
    assert result["ticket_type"] == "access_request"
    assert "expected" not in result
    assert "unexpected" not in result

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_diagnostic_log_failure_does_not_block_primary(mock_client, tmp_path):
    """A diagnostic log_llm_error raising OSError does not block the primary record or set its outcome."""
    mock_client.models.generate_content.side_effect = ConnectionError("connection reset")
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.log_llm_error', side_effect=OSError("diagnostic disk full")):
            result = classify_ticket("Please reset my password.")
    assert result["api_failed"] is True
    assert result["logging_failed"] is False
    assert result["record_status"] == "recorded"
    assert len(_read_day_entries(tmp_path)) == 1
    assert len(_read_error_entries(tmp_path)) == 0

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_diagnostic_preparation_failure_does_not_block_primary(mock_client, tmp_path):
    """A redact_pii failure inside log_llm_error does not block the primary record."""
    from bot_engine.router import redact_pii as real_redact
    mock_client.models.generate_content.side_effect = ConnectionError("connection reset")
    def scoped_redact(text):
        if "API call to Gemini failed" in text:
            raise RuntimeError("redaction boom")
        return real_redact(text)
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.redact_pii', side_effect=scoped_redact):
            result = classify_ticket("Please reset my password.")
    assert result["api_failed"] is True
    assert result["logging_failed"] is False
    assert result["record_status"] == "recorded"
    assert len(_read_day_entries(tmp_path)) == 1
    assert len(_read_error_entries(tmp_path)) == 0

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_error_log_carries_decision_id(mock_client, tmp_path):
    """An API failure writes an error log entry whose decision_id matches the primary decision_id."""
    mock_client.models.generate_content.side_effect = ConnectionError("connection reset")
    with _tmp_project_root(tmp_path):
        result = classify_ticket("Please reset my password.")
    error_entries = _read_error_entries(tmp_path)
    assert len(error_entries) == 1
    assert error_entries[0]["decision_id"] == result["decision_id"]
    entries = _read_day_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["decision_id"] == result["decision_id"]

def test_direct_error_log_without_decision_id_omits_key(tmp_path):
    """A direct log_llm_error call without a decision_id omits the decision_id key."""
    from bot_engine.router import log_llm_error
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        log_llm_error("Please verify my account.", "API Error")
    error_entries = _read_error_entries(tmp_path)
    assert len(error_entries) == 1
    assert "decision_id" not in error_entries[0]

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_unexpected_diagnostic_exception_does_not_block_primary(mock_client, tmp_path):
    """A log_llm_error raising RuntimeError does not block the primary record and the result is returned."""
    mock_client.models.generate_content.side_effect = ConnectionError("connection reset")
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.log_llm_error', side_effect=RuntimeError("diagnostic boom")):
            result = classify_ticket("Please reset my password.")
    assert result["api_failed"] is True
    assert result["record_status"] == "recorded"
    assert len(_read_day_entries(tmp_path)) == 1

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_serialization_type_error_is_storage_failure(mock_client, tmp_path):
    """A mocked json.dumps raising TypeError in step 2 is a storage_failed record_failed, not a classification failure."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.json.dumps', side_effect=TypeError("not serializable")):
            result = classify_ticket("Please reset my password.")
    assert result["ticket_type"] == "access_request"
    assert result["logging_failed"] is True
    assert result["record_status"] == "record_failed"
    assert result["record_failure_category"] == "storage_failed"
    assert result["persistence_error_type"] == "TypeError"
    assert not (tmp_path / "logs").exists()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_subject_preparation_failure(mock_client, tmp_path):
    """A redact_pii failure while redacting the subject is privacy_preparation_failed with no entry and no raw text."""
    from bot_engine.router import redact_pii as real_redact
    _mock_llm(mock_client, "access_request", 0.99)
    def scoped_redact(text):
        if text == "A subject":
            raise RuntimeError("redaction boom")
        return real_redact(text)
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.redact_pii', side_effect=scoped_redact):
            result = classify_ticket("Please reset my password.", subject="A subject")
    assert result["logging_failed"] is True
    assert result["record_status"] == "record_failed"
    assert result["record_failure_category"] == "privacy_preparation_failed"
    assert "persistence_error_type" not in result
    assert not (tmp_path / "logs").exists()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_sender_preparation_failure(mock_client, tmp_path):
    """A redact_pii failure while redacting the sender is privacy_preparation_failed with no entry and no raw text."""
    from bot_engine.router import redact_pii as real_redact
    _mock_llm(mock_client, "access_request", 0.99)
    def scoped_redact(text):
        if text == "Jane <jane@example.com>":
            raise RuntimeError("redaction boom")
        return real_redact(text)
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.redact_pii', side_effect=scoped_redact):
            result = classify_ticket("Please reset my password.", sender="Jane <jane@example.com>")
    assert result["logging_failed"] is True
    assert result["record_status"] == "record_failed"
    assert result["record_failure_category"] == "privacy_preparation_failed"
    assert "persistence_error_type" not in result
    assert not (tmp_path / "logs").exists()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_log_preparation_detection_failure(mock_client, tmp_path):
    """A detect_pii_types failure during log preparation is privacy_preparation_failed with no entry."""
    from bot_engine.router import detect_pii_types as real_detect
    _mock_llm(mock_client, "access_request", 0.99)
    calls = {"n": 0}
    def scoped_detect(text):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("detection boom")
        return real_detect(text)
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.detect_pii_types', side_effect=scoped_detect):
            result = classify_ticket("Please reset my password.")
    assert result["logging_failed"] is True
    assert result["record_status"] == "record_failed"
    assert result["record_failure_category"] == "privacy_preparation_failed"
    assert "persistence_error_type" not in result
    assert not (tmp_path / "logs").exists()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_partial_write_is_record_failed(mock_client, tmp_path):
    """A write raising after a partial write is record_failed, and the result does not claim the record is absent."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.os.fdopen', side_effect=_fdopen_with_fault("write")):
            result = classify_ticket("Please reset my password.")
    assert result["logging_failed"] is True
    assert result["record_status"] == "record_failed"
    assert result["record_failure_category"] == "storage_failed"
    assert result["persistence_error_type"] == "OSError"
    entries = _read_day_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["decision_id"] == result["decision_id"]

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_flush_failure_is_record_failed(mock_client, tmp_path):
    """A mocked flush raising OSError is a storage_failed record_failed."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.os.fdopen', side_effect=_fdopen_with_fault("flush")):
            result = classify_ticket("Please reset my password.")
    assert result["logging_failed"] is True
    assert result["record_status"] == "record_failed"
    assert result["record_failure_category"] == "storage_failed"
    assert result["persistence_error_type"] == "OSError"

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_close_failure_is_record_failed(mock_client, tmp_path):
    """A mocked close raising OSError is a storage_failed record_failed."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.os.fdopen', side_effect=_fdopen_with_fault("close")):
            result = classify_ticket("Please reset my password.")
    assert result["logging_failed"] is True
    assert result["record_status"] == "record_failed"
    assert result["record_failure_category"] == "storage_failed"
    assert result["persistence_error_type"] == "OSError"

def test_log_entry_serializes_before_opening(tmp_path):
    """A mocked json.dumps raising TypeError creates no file and raises as an expected storage failure."""
    from bot_engine.router import log_entry
    log_path = tmp_path / "logs" / "2026-11-19.jsonl"
    with patch('bot_engine.router.json.dumps', side_effect=TypeError("not serializable")):
        with pytest.raises(TypeError):
            log_entry(log_path, {"decision_id": "x"})
    assert not log_path.exists()
    assert not (tmp_path / "logs").exists()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_stored_entry_has_no_persistence_fields(mock_client, tmp_path):
    """The stored entry and its nested result carry no persistence fields, while the returned result does."""
    _mock_llm(mock_client, "access_request", 0.99)
    with _tmp_project_root(tmp_path):
        result = classify_ticket("Please reset my password.")
    assert result["logging_failed"] is False
    assert result["record_status"] == "recorded"
    assert "record_failure_category" not in result
    assert "persistence_error_type" not in result
    entry = _read_day_entries(tmp_path)[0]
    for field in ("logging_failed", "record_status", "record_failure_category", "persistence_error_type"):
        assert field not in entry
        assert field not in entry["result"]
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.log_entry', side_effect=OSError("disk full")):
            failed = classify_ticket("Please reset my password.")
    assert failed["logging_failed"] is True
    assert failed["record_status"] == "record_failed"
    assert failed["record_failure_category"] == "storage_failed"
    assert failed["persistence_error_type"] == "OSError"
    assert len(_read_day_entries(tmp_path)) == 1

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_stored_entry_matches_returned_result(mock_client, tmp_path):
    """The stored routing_decision and decision_id match the returned result, top-level and nested."""
    _mock_llm(mock_client, "technical_support", 0.6)
    with _tmp_project_root(tmp_path):
        result = classify_ticket("The app crashes on launch.")
    entry = _read_day_entries(tmp_path)[0]
    assert entry["result"]["routing_decision"] == result["routing_decision"]
    assert entry["decision_id"] == result["decision_id"]
    assert entry["result"]["decision_id"] == result["decision_id"]
    assert entry["decision_id"] == entry["result"]["decision_id"]

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_empty_input_returns_unknown_without_logging_or_id(mock_client, tmp_path):
    """Empty input still returns unknown with no log entry and no decision_id key."""
    with _tmp_project_root(tmp_path):
        result = classify_ticket("   ")
    assert result["ticket_type"] == "unknown"
    assert "decision_id" not in result
    assert not (tmp_path / "logs").exists()
    mock_client.models.generate_content.assert_not_called()

@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes do not apply on Windows")
def test_log_file_and_directory_permissions(tmp_path):
    """A day file created through log_entry under a wide umask has mode 0600 and its logs directory 0700 on POSIX."""
    import stat
    from bot_engine.router import log_entry
    old_umask = os.umask(0o022)
    try:
        log_entry(tmp_path / "logs" / "2026-11-19.jsonl", {"decision_id": "x"})
    finally:
        os.umask(old_umask)
    day_file = tmp_path / "logs" / "2026-11-19.jsonl"
    assert stat.S_IMODE(day_file.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "logs").stat().st_mode) == 0o700

@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes do not apply on Windows")
def test_diagnostic_log_permissions(tmp_path):
    """An error log file created through log_llm_error has mode 0600 on POSIX."""
    import stat
    from bot_engine.router import log_llm_error
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        log_llm_error("Please verify my account.", "API Error")
    error_log = tmp_path / "governance" / "llm_error_log.jsonl"
    assert stat.S_IMODE(error_log.stat().st_mode) == 0o600

def test_direct_log_fallback_ignores_caller_timestamp(tmp_path):
    """A direct log_fallback call uses its own captured timestamp, never the caller's result timestamp."""
    from bot_engine.router import log_fallback
    result = {"ticket_type": "unknown", "confidence_score": 0.0, "contains_pii": False,
              "decision_id": str(uuid.uuid4()), "timestamp": "2000-01-01T00:00:00+00:00"}
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        log_fallback("Please reset my password.", result)
    entry = _read_day_entries(tmp_path)[0]
    assert entry["timestamp"] != "2000-01-01T00:00:00+00:00"
    assert entry["timestamp"][:4] != "2000"
    day_files = sorted((tmp_path / "logs").glob("*.jsonl"))
    assert len(day_files) == 1
    assert day_files[0].name == entry["timestamp"][:10] + ".jsonl"

# --- Fix 2: always-redact storage ---

def test_direct_log_fallback_redacts_unflagged_pii(tmp_path):
    """A direct log_fallback call with contains_pii False and an SSN in the text stores a redacted text and pii_types ['ssn']."""
    from bot_engine.router import log_fallback
    ticket_text = "My SSN is 123-45-6789, please verify my account."
    result = {"ticket_type": "unknown", "confidence_score": 0.0, "contains_pii": False,
              "decision_id": str(uuid.uuid4())}
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        log_fallback(ticket_text, result)
    entry = _read_day_entries(tmp_path)[0]
    assert "123-45-6789" not in json.dumps(entry)
    assert "[REDACTED_SSN]" in entry["ticket"]
    assert entry["pii_types"] == ["ssn"]

def test_direct_log_fallback_non_pii_text_stored_verbatim(tmp_path):
    """A direct log_fallback call with non-PII text stores the text verbatim."""
    from bot_engine.router import log_fallback
    ticket_text = "Please reset my password."
    result = {"ticket_type": "access_request", "confidence_score": 0.99, "contains_pii": False,
              "decision_id": str(uuid.uuid4())}
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        log_fallback(ticket_text, result)
    entry = _read_day_entries(tmp_path)[0]
    assert entry["ticket"] == ticket_text
    assert "pii_types" not in entry

def test_direct_log_fallback_decision_id_validation(tmp_path):
    """A direct log_fallback call validates the decision ID before any write."""
    from bot_engine.router import log_fallback
    base = {"ticket_type": "unknown", "confidence_score": 0.0, "contains_pii": False}
    cases = [
        ({}, pytest.raises(TypeError)),
        ({"decision_id": None}, pytest.raises(TypeError)),
        ({"decision_id": 123}, pytest.raises(TypeError)),
        ({"decision_id": ""}, pytest.raises(ValueError)),
        ({"decision_id": "not-a-uuid"}, pytest.raises(ValueError)),
        ({"decision_id": "11111111-1111-3111-8111-111111111111"}, pytest.raises(ValueError)),
        ({"decision_id": "11111111-1111-4111-8111-11111111111"}, pytest.raises(ValueError)),
        ({"decision_id": "AAAAAAAA-1111-4111-8111-111111111111"}, pytest.raises(ValueError)),
    ]
    for extra, expectation in cases:
        result = dict(base)
        result.update(extra)
        with patch('bot_engine.router.get_project_root', return_value=tmp_path):
            with expectation:
                log_fallback("Please reset my password.", result)
        assert not (tmp_path / "logs").exists()
    valid_id = str(uuid.uuid4())
    result = dict(base)
    result["decision_id"] = valid_id
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        log_fallback("Please reset my password.", result)
    entries = _read_day_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["decision_id"] == valid_id
    assert entries[0]["result"]["decision_id"] == valid_id

def test_direct_log_fallback_metadata_validation(tmp_path):
    """A direct log_fallback call validates metadata: falsy non-strings raise TypeError, blank values become None."""
    from bot_engine.router import log_fallback
    base = {"ticket_type": "unknown", "confidence_score": 0.0, "contains_pii": False,
            "decision_id": str(uuid.uuid4())}
    for kwargs in ({"subject": 0}, {"subject": False}, {"sender": 0}, {"source_ticket_id": False}):
        with patch('bot_engine.router.get_project_root', return_value=tmp_path):
            with pytest.raises(TypeError):
                log_fallback("Please reset my password.", dict(base), **kwargs)
    assert not (tmp_path / "logs").exists()
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        log_fallback("Please reset my password.", dict(base), subject="  A subject  ", sender="   ",
                     source_ticket_id="  INC-1  ")
    entry = _read_day_entries(tmp_path)[0]
    assert entry["subject"] == "A subject"
    assert "sender" not in entry
    assert "sender_domain" not in entry
    assert entry["source_ticket_id"] == "INC-1"

def test_error_log_always_redacts_preview(tmp_path):
    """An unflagged error log entry still stores a redacted preview when the text contains PII."""
    from bot_engine.router import log_llm_error
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        log_llm_error("My SSN is 123-45-6789, please verify my account.", "API Error")
    entry = _read_error_entries(tmp_path)[0]
    assert "123-45-6789" not in json.dumps(entry)
    assert "[REDACTED_SSN]" in entry["ticket_preview"]
    assert entry["pii_types"] == ["ssn"]

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.client')
def test_redaction_failure_during_log_preparation(mock_client, tmp_path):
    """A redaction failure during log preparation returns record_failed with privacy_preparation_failed and no raw entry."""
    from bot_engine.router import redact_pii as real_redact
    _mock_llm(mock_client, "access_request", 0.99)
    calls = {"n": 0}
    def scoped_redact(text):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_redact(text)
        raise RuntimeError("redaction boom")
    with _tmp_project_root(tmp_path):
        with patch('bot_engine.router.redact_pii', side_effect=scoped_redact):
            result = classify_ticket("Please reset my password.")
    assert result["logging_failed"] is True
    assert result["record_status"] == "record_failed"
    assert result["record_failure_category"] == "privacy_preparation_failed"
    assert "persistence_error_type" not in result
    assert not (tmp_path / "logs").exists()
