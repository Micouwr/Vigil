# tests/test_gui_status.py
import pytest
from main_gui import build_status_text

def _make_result(preprocessing=None, api_failed=False, model_skip=None,
                 route="human_agent", record_status="recorded", routing_decision="default"):
    """Builds one hand-built result dict for the status matrix.

    preprocessing, api_failed, and model_skip are the exact values to
    set, or None to omit the key. routing_decision "default" builds a
    dict with the given route; "missing" omits the key; any other value
    is used as-is, so a string exercises the non-dictionary case.
    record_status None omits the key.
    """
    result = {
        "ticket_type": "access_request",
        "confidence_score": 0.99,
        "contains_pii": False,
        "model": "gemini-3.8-flash",
        "api_failed": api_failed,
        "decision_id": "11111111-1111-4111-8111-111111111111",
    }
    if preprocessing is not None:
        result["preprocessing_failed"] = preprocessing
    if model_skip is not None:
        result["model_skip"] = model_skip
    if routing_decision == "default":
        result["routing_decision"] = {"route": route, "reason": "Test reason."}
    elif routing_decision == "missing":
        pass
    else:
        result["routing_decision"] = routing_decision
    if record_status is not None:
        result["record_status"] = record_status
    return result

CASES = [
    # Successful classification across the three persistence states.
    ("success recorded",
     _make_result(route="automated_response", record_status="recorded"),
     "Classification complete. Recorded."),
    ("success record_failed",
     _make_result(route="automated_response", record_status="record_failed"),
     "Classification complete. Persistence not confirmed."),
    ("success missing status",
     _make_result(route="automated_response", record_status=None),
     "Classification complete. Persistence unknown."),
    ("success unrecognized status",
     _make_result(route="automated_response", record_status="weird"),
     "Classification complete. Persistence unknown."),
    # PII preprocessing failure, confirmed and unconfirmed routing.
    ("preprocessing confirmed recorded",
     _make_result(preprocessing=True, record_status="recorded"),
     "PII preprocessing failed; routed to human review. Recorded."),
    ("preprocessing confirmed record_failed",
     _make_result(preprocessing=True, record_status="record_failed"),
     "PII preprocessing failed; routed to human review. Persistence not confirmed."),
    ("preprocessing confirmed missing status",
     _make_result(preprocessing=True, record_status=None),
     "PII preprocessing failed; routed to human review. Persistence unknown."),
    ("preprocessing unconfirmed recorded",
     _make_result(preprocessing=True, route="automated_response", record_status="recorded"),
     "PII preprocessing failed. Recorded."),
    ("preprocessing unconfirmed record_failed",
     _make_result(preprocessing=True, route="automated_response", record_status="record_failed"),
     "PII preprocessing failed. Persistence not confirmed."),
    ("preprocessing unconfirmed missing status",
     _make_result(preprocessing=True, route="automated_response", record_status=None),
     "PII preprocessing failed. Persistence unknown."),
    ("preprocessing malformed routing",
     _make_result(preprocessing=True, record_status="recorded", routing_decision="human_agent"),
     "PII preprocessing failed. Recorded."),
    # API failure, confirmed and unconfirmed routing.
    ("api confirmed recorded",
     _make_result(api_failed=True, record_status="recorded"),
     "API call failed; ticket routed to human review. Recorded."),
    ("api confirmed record_failed",
     _make_result(api_failed=True, record_status="record_failed"),
     "API call failed; ticket routed to human review. Persistence not confirmed."),
    ("api confirmed missing status",
     _make_result(api_failed=True, record_status=None),
     "API call failed; ticket routed to human review. Persistence unknown."),
    ("api unconfirmed recorded",
     _make_result(api_failed=True, route="automated_response", record_status="recorded"),
     "API call failed. Recorded."),
    ("api unconfirmed record_failed",
     _make_result(api_failed=True, route="automated_response", record_status="record_failed"),
     "API call failed. Persistence not confirmed."),
    ("api unconfirmed missing status",
     _make_result(api_failed=True, route="automated_response", record_status=None),
     "API call failed. Persistence unknown."),
    ("api malformed routing",
     _make_result(api_failed=True, record_status="recorded", routing_decision="human_agent"),
     "API call failed. Recorded."),
    # Meaningful-text skip, confirmed and unconfirmed routing.
    ("skip confirmed recorded",
     _make_result(model_skip=True, record_status="recorded"),
     "No meaningful text to classify; routed to human review. Recorded."),
    ("skip confirmed record_failed",
     _make_result(model_skip=True, record_status="record_failed"),
     "No meaningful text to classify; routed to human review. Persistence not confirmed."),
    ("skip confirmed missing status",
     _make_result(model_skip=True, record_status=None),
     "No meaningful text to classify; routed to human review. Persistence unknown."),
    ("skip unconfirmed recorded",
     _make_result(model_skip=True, route="automated_response", record_status="recorded"),
     "No meaningful text to classify. Recorded."),
    ("skip unconfirmed record_failed",
     _make_result(model_skip=True, route="automated_response", record_status="record_failed"),
     "No meaningful text to classify. Persistence not confirmed."),
    ("skip unconfirmed missing status",
     _make_result(model_skip=True, route="automated_response", record_status=None),
     "No meaningful text to classify. Persistence unknown."),
    ("skip malformed routing",
     _make_result(model_skip=True, record_status="recorded", routing_decision="human_agent"),
     "No meaningful text to classify. Recorded."),
    # Overlapping flags verify the priority order.
    ("preprocessing plus api",
     _make_result(preprocessing=True, api_failed=True, record_status="recorded"),
     "PII preprocessing failed; routed to human review. Recorded."),
    ("api plus skip",
     _make_result(api_failed=True, model_skip=True, record_status="recorded"),
     "API call failed; ticket routed to human review. Recorded."),
    ("all three flags",
     _make_result(preprocessing=True, api_failed=True, model_skip=True, record_status="recorded"),
     "PII preprocessing failed; routed to human review. Recorded."),
    # Malformed processing-flag values do not activate their message.
    ("preprocessing string False",
     _make_result(preprocessing="False", record_status="recorded"),
     "Classification complete. Recorded."),
    ("api string false",
     _make_result(api_failed="false", record_status="recorded"),
     "Classification complete. Recorded."),
    ("skip integer 1",
     _make_result(model_skip=1, record_status="recorded"),
     "Classification complete. Recorded."),
]

@pytest.mark.parametrize("label,result,expected", CASES, ids=[c[0] for c in CASES])
def test_status_matrix(label, result, expected):
    """The full status matrix: every processing failure type with confirmed and unconfirmed routing, across each persistence state, plus overlaps and malformed values."""
    assert build_status_text(result) == expected

def test_status_never_claims_not_recorded():
    """No composed status uses the wording Not recorded, because a partial write may have occurred."""
    for label, result, expected in CASES:
        assert "Not recorded" not in build_status_text(result)
