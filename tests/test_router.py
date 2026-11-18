# tests/test_router.py
import json
import shutil
import pytest
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
@patch('bot_engine.router.CONFIDENCE_THRESHOLD', 0.5)
@patch('bot_engine.router.log_fallback')
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
@patch('bot_engine.router.CONFIDENCE_THRESHOLD', 0.5)
@patch('bot_engine.router.log_fallback')
@patch('bot_engine.router.client')
def test_high_confidence_does_not_trigger_fallback_log(mock_client, mock_log_fallback):
    """Test that high confidence scores do not trigger fallback logging."""
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
    mock_log_fallback.assert_not_called()

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
    (tmp_path / "prompts").mkdir()
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
    fallback_log = tmp_path / "fallback_log.jsonl"
    fallback_entry = json.loads(fallback_log.read_text().splitlines()[-1])
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
    assert config["bot_config"]["confidence_threshold"] == 0.5
    assert config["bot_config"]["model_name"] == "gemini-3.8-flash"


def test_load_config_rejects_missing_threshold(tmp_path):
    """A scope.yaml without a usable confidence threshold fails loudly."""
    bad_yaml = tmp_path / "scope.yaml"
    bad_yaml.write_text(
        "bot_config:\n"
        "  model_name: gemini-3.8-flash\n"
        "ticket_types: [access_request]\n"
        "escalation_rules: ['else: route_to: automated_response']\n"
    )
    try:
        load_config(bad_yaml)
    except ValueError as e:
        assert "confidence_threshold" in str(e)
    else:
        raise AssertionError("load_config accepted a config with no confidence threshold")

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
@patch('bot_engine.router.CONFIDENCE_THRESHOLD', 0.5)
@patch('bot_engine.router.log_fallback')
@patch('bot_engine.router.client')
def test_audit_log_uses_unrounded_confidence_below_threshold(mock_client, mock_log_fallback):
    """0.496 rounds to 0.5 but must still write a fallback log entry."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "technical_support",
        "confidence": 0.496
    })
    mock_client.models.generate_content.return_value = mock_response

    classify_ticket("The app is crashing.")

    mock_log_fallback.assert_called_once()

@patch('bot_engine.router.TICKET_TYPES', MOCK_TICKET_TYPES)
@patch('bot_engine.router.CONFIDENCE_THRESHOLD', 0.5)
@patch('bot_engine.router.log_fallback')
@patch('bot_engine.router.client')
def test_audit_log_uses_unrounded_confidence_above_threshold(mock_client, mock_log_fallback):
    """0.504 rounds to 0.5 but must not write a fallback log entry."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "category": "technical_support",
        "confidence": 0.504
    })
    mock_client.models.generate_content.return_value = mock_response

    classify_ticket("The app is crashing.")

    mock_log_fallback.assert_not_called()

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
    (tmp_path / "prompts").mkdir()
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

    fallback_log = tmp_path / "fallback_log.jsonl"
    entry = json.loads(fallback_log.read_text().splitlines()[-1])
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
    (tmp_path / "prompts").mkdir()
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

    fallback_log = tmp_path / "fallback_log.jsonl"
    entry = json.loads(fallback_log.read_text().splitlines()[-1])
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
    (tmp_path / "prompts").mkdir()
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

    fallback_log = tmp_path / "fallback_log.jsonl"
    entry = json.loads(fallback_log.read_text().splitlines()[-1])
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
    (tmp_path / "prompts").mkdir()
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

    fallback_log = tmp_path / "fallback_log.jsonl"
    fallback_entry = json.loads(fallback_log.read_text().splitlines()[-1])
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
    (tmp_path / "prompts").mkdir()
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
    result = {"ticket_type": "unknown", "confidence_score": 0.0, "contains_pii": True}
    with patch('bot_engine.router.get_project_root', return_value=tmp_path):
        log_fallback(ticket_text, result)

    entry = json.loads((tmp_path / "fallback_log.jsonl").read_text().splitlines()[-1])
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
    (tmp_path / "prompts").mkdir()
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

    fallback_log = tmp_path / "fallback_log.jsonl"
    entry = json.loads(fallback_log.read_text().splitlines()[-1])
    assert "555-123-4567" not in json.dumps(entry)
    assert "[REDACTED_PHONE]" in entry["ticket"]
    assert entry["pii_types"] == ["phone"]
