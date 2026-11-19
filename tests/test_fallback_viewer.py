# tests/test_fallback_viewer.py
import csv
import json
import sys
import pytest
from tools.fallback_viewer import FallbackLogViewer, main

TS = "2026-11-19T10:00:00+00:00"

def _make_entry(decision_id, timestamp, **overrides):
    """Builds one decision-log entry with defaults, overridden by keyword arguments.

    Result overrides go under the 'result' key as a dict; entry-level
    overrides are the remaining keywords.
    """
    result = {
        "decision_id": decision_id,
        "ticket_type": "access_request",
        "confidence_score": 0.99,
        "contains_pii": False,
        "model": "gemini-3.8-flash",
        "api_failed": False,
        "routing_decision": {"route": "automated_response", "reason": "Confident classification."},
    }
    result_overrides = overrides.pop("result", {})
    result.update(result_overrides)
    entry = {
        "decision_id": decision_id,
        "ticket": "Please reset my password.",
        "result": result,
        "timestamp": timestamp,
    }
    entry.update(overrides)
    return entry

def _write_entries(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")

def _write_fixture(tmp_path, entries, day="2026-11-19.jsonl", legacy=None):
    """Writes day entries under tmp_path/logs/ and an optional legacy file at the root."""
    _write_entries(tmp_path / "logs" / day, entries)
    if legacy is not None:
        _write_entries(tmp_path / "fallback_log.jsonl", legacy)
    return tmp_path

def _stats_fixture():
    """Eight entries: five completed decisions, one API failure, one preprocessing failure, one model skip."""
    return [
        _make_entry("e1", TS),
        _make_entry("e2", TS, result={"ticket_type": "billing_question", "confidence_score": 0.80,
                                       "routing_decision": {"route": "human_agent", "reason": "Low confidence."}}),
        _make_entry("e3", TS, result={"ticket_type": "software_issue", "confidence_score": 0.95,
                                       "contains_pii": True,
                                       "routing_decision": {"route": "human_agent", "reason": "PII detected."}}),
        _make_entry("e4", TS, result={"ticket_type": "unknown", "confidence_score": 0.0,
                                       "api_failed": True,
                                       "routing_decision": {"route": "human_agent", "reason": "API failure."}}),
        _make_entry("e5", TS, result={"ticket_type": "unknown", "confidence_score": 0.0,
                                       "contains_pii": True, "preprocessing_failed": True,
                                       "routing_decision": {"route": "human_agent", "reason": "PII preprocessing failed."}},
                    redaction_failed=True, pii_types=["email"]),
        _make_entry("e6", TS, result={"ticket_type": "unknown", "confidence_score": 0.0,
                                       "model_skip": True,
                                       "routing_decision": {"route": "human_agent", "reason": "Meaningful text absent."}}),
        _make_entry("e7", TS, result={"ticket_type": "hardware_issue", "confidence_score": 0.60,
                                       "routing_decision": {"route": "human_agent", "reason": "Low confidence."}}),
        _make_entry("e8", TS, result={"ticket_type": "password_reset", "confidence_score": 0.70}),
    ]

def test_filter_by_routing(tmp_path):
    """filter_by_routing returns only human_agent entries, then only automated_response entries."""
    _write_fixture(tmp_path, _stats_fixture())
    viewer = FallbackLogViewer(tmp_path)
    assert len(viewer.entries) == 8
    viewer.filter_by_routing("human_agent")
    assert len(viewer.entries) == 6
    assert all(e["result"]["routing_decision"]["route"] == "human_agent" for e in viewer.entries)
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_routing("automated_response")
    assert len(viewer.entries) == 2
    assert all(e["result"]["routing_decision"]["route"] == "automated_response" for e in viewer.entries)

def test_search_filters(tmp_path):
    """Each search filter returns exact matches, including case-insensitive substrings and missing-field exclusion."""
    entries = [
        _make_entry("id-aaa", TS, subject="Password reset help", sender="Jane Smith <jane@example.com>",
                    sender_domain="example.com", source_ticket_id="INC-100"),
        _make_entry("id-bbb", TS, subject="Billing question", sender="Bob Jones <bob@corp.com>",
                    sender_domain="corp.com", source_ticket_id="INC-200"),
        _make_entry("id-ccc", TS, ticket="The printer is jammed."),
    ]
    _write_fixture(tmp_path, entries)
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_decision_id("id-aaa")
    assert [e["decision_id"] for e in viewer.entries] == ["id-aaa"]

    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_source_ticket_id("INC-200")
    assert [e["decision_id"] for e in viewer.entries] == ["id-bbb"]

    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_subject("PASSWORD")
    assert [e["decision_id"] for e in viewer.entries] == ["id-aaa"]

    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_sender("bob jones")
    assert [e["decision_id"] for e in viewer.entries] == ["id-bbb"]

    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_sender_domain("EXAMPLE.COM")
    assert [e["decision_id"] for e in viewer.entries] == ["id-aaa"]

    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_text("PRINTER")
    assert [e["decision_id"] for e in viewer.entries] == ["id-ccc"]

    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_text("password reset")
    assert [e["decision_id"] for e in viewer.entries] == ["id-aaa"]

    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_subject("nonexistent")
    assert viewer.entries == []

def test_malformed_exact_match_filters(tmp_path):
    """Non-string stored values never match an exact filter and never raise, including the sender_domain lowercasing path."""
    entries = [
        _make_entry("good-id", TS, subject="A subject", sender="Jane", sender_domain="example.com",
                    source_ticket_id="INC-1"),
        _make_entry(123, TS, subject="Bad id", sender="Bob", sender_domain=456, source_ticket_id=["INC-9"]),
    ]
    _write_fixture(tmp_path, entries)
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_decision_id("good-id")
    assert [e["decision_id"] for e in viewer.entries] == ["good-id"]
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_decision_id("123")
    assert viewer.entries == []
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_source_ticket_id("INC-1")
    assert [e["decision_id"] for e in viewer.entries] == ["good-id"]
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_source_ticket_id("INC-9")
    assert viewer.entries == []
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_sender_domain("example.com")
    assert [e["decision_id"] for e in viewer.entries] == ["good-id"]
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_sender_domain("456")
    assert viewer.entries == []

def test_multi_file_load(tmp_path):
    """A logs directory with two day files plus a legacy file loads as one sorted population with the correct total."""
    day1 = [_make_entry("d1-a", "2026-11-19T09:00:00+00:00"), _make_entry("d1-b", "2026-11-19T11:00:00+00:00")]
    day2 = [_make_entry("d2-a", "2026-11-20T09:00:00+00:00")]
    legacy = [_make_entry("legacy-a", "2026-11-18T09:00:00+00:00")]
    _write_entries(tmp_path / "logs" / "2026-11-19.jsonl", day1)
    _write_entries(tmp_path / "logs" / "2026-11-20.jsonl", day2)
    _write_entries(tmp_path / "fallback_log.jsonl", legacy)
    viewer = FallbackLogViewer(tmp_path)
    assert len(viewer.entries) == 4
    assert [e["decision_id"] for e in viewer.entries] == ["legacy-a", "d1-a", "d1-b", "d2-a"]

def test_single_file_load(tmp_path):
    """An explicit --log-path pointing at a file loads only that file."""
    day_file = tmp_path / "logs" / "2026-11-19.jsonl"
    _write_entries(day_file, [_make_entry("only-1", TS), _make_entry("only-2", TS)])
    _write_entries(tmp_path / "fallback_log.jsonl", [_make_entry("legacy-1", TS)])
    viewer = FallbackLogViewer(day_file)
    assert [e["decision_id"] for e in viewer.entries] == ["only-1", "only-2"]

def test_stats_known_fixture(tmp_path):
    """Stats on a known fixture return the hand-computed Human Routing Rate and averages."""
    _write_fixture(tmp_path, _stats_fixture())
    viewer = FallbackLogViewer(tmp_path)
    stats = viewer.generate_summary_stats()
    assert stats["total_parseable"] == 8
    assert stats["completed_decisions"] == 5
    assert stats["human"] == 3
    assert stats["automated"] == 2
    assert stats["human_routing_rate"] == pytest.approx(3 / 5)
    assert stats["average_confidence"] == pytest.approx(4.04 / 5)
    assert stats["average_confidence_by_route"]["human_agent"] == pytest.approx(2.35 / 3)
    assert stats["average_confidence_by_route"]["automated_response"] == pytest.approx(1.69 / 2)
    assert stats["pii_count"] == 2
    assert stats["api_failed"] == 1
    assert stats["preprocessing_failed"] == 1
    assert stats["model_skip"] == 1
    assert stats["api_failed_human_routed"] == 1
    assert stats["preprocessing_failed_human_routed"] == 1
    assert stats["incomplete_routing"] == 0
    assert stats["incomplete_confidence"] == 0
    assert stats["incomplete_classification"] == 0
    assert stats["malformed_status"] == 0

def test_stats_denominator_zero(tmp_path, capsys):
    """With no completed decisions, the rate and averages are None and display as Unavailable, not 0 percent."""
    entries = [_make_entry("f1", TS, result={"ticket_type": "unknown", "confidence_score": 0.0,
                                              "api_failed": True,
                                              "routing_decision": {"route": "human_agent", "reason": "API failure."}})]
    _write_fixture(tmp_path, entries)
    viewer = FallbackLogViewer(tmp_path)
    stats = viewer.generate_summary_stats()
    assert stats["total_parseable"] == 1
    assert stats["completed_decisions"] == 0
    assert stats["human_routing_rate"] is None
    assert stats["average_confidence"] is None
    assert stats["average_confidence_by_route"]["human_agent"] is None
    assert stats["average_confidence_by_route"]["automated_response"] is None
    assert stats["api_failed"] == 1
    assert stats["api_failed_human_routed"] == 1
    output = capsys.readouterr().out
    assert "Human Routing Rate: Unavailable" in output
    assert "Human Routing Rate: 0.0%" not in output

def test_stats_malformed_data(tmp_path):
    """Missing and unrecognized routes are excluded from the denominator and counted in incomplete_routing; missing confidence is excluded from the average and counted in incomplete_confidence; bad ticket types are excluded from population 2 and counted in incomplete_classification."""
    m1 = _make_entry("m1", TS)
    del m1["result"]["routing_decision"]
    m2 = _make_entry("m2", TS, result={"routing_decision": {"route": "pending", "reason": "Waiting."}})
    m3 = _make_entry("m3", TS, result={"routing_decision": {"route": "human_agent", "reason": "Low confidence."}})
    del m3["result"]["confidence_score"]
    m4 = _make_entry("m4", TS)
    del m4["result"]["ticket_type"]
    m5 = _make_entry("m5", TS, result={"ticket_type": "teleportation"})
    m6 = _make_entry("m6", TS, result={"confidence_score": 0.90})
    _write_fixture(tmp_path, [m1, m2, m3, m4, m5, m6])
    viewer = FallbackLogViewer(tmp_path)
    stats = viewer.generate_summary_stats()
    assert stats["total_parseable"] == 6
    assert stats["completed_decisions"] == 2
    assert stats["human"] == 1
    assert stats["automated"] == 1
    assert stats["human_routing_rate"] == pytest.approx(1 / 2)
    assert stats["incomplete_routing"] == 2
    assert stats["incomplete_confidence"] == 1
    assert stats["average_confidence"] == pytest.approx(0.90)
    assert stats["incomplete_classification"] == 2
    assert stats["malformed_status"] == 0
    # Second reconciliation identity: completed decisions plus incomplete routing equals population 2.
    assert stats["completed_decisions"] + stats["incomplete_routing"] == 4

def test_load_order_fixture(tmp_path):
    """Same-timestamp entries across the legacy file and two day files load in deterministic order: legacy first, then day files lexicographically."""
    legacy = [_make_entry("legacy-1", TS)]
    day1 = [_make_entry("day1-1", TS)]
    day2 = [_make_entry("day2-1", TS)]
    _write_entries(tmp_path / "fallback_log.jsonl", legacy)
    _write_entries(tmp_path / "logs" / "2026-11-19.jsonl", day1)
    _write_entries(tmp_path / "logs" / "2026-11-20.jsonl", day2)
    viewer = FallbackLogViewer(tmp_path)
    assert [e["decision_id"] for e in viewer.entries] == ["legacy-1", "day1-1", "day2-1"]

def test_timestamp_fixture(tmp_path, capsys):
    """A naive timestamp sorts as UTC, a non-string and an unparseable timestamp load with a warning and sort last, and the load order is deterministic."""
    t3 = _make_entry("t3", TS)
    t3["timestamp"] = 12345
    t1 = _make_entry("t1", "2026-11-19T10:00:00")
    t5 = _make_entry("t5", TS)
    del t5["timestamp"]
    t2 = _make_entry("t2", "2026-11-19T12:00:00+02:00")
    t6 = _make_entry("t6", "2026-11-19T09:00:00+00:00")
    t4 = _make_entry("t4", "not-a-date")
    _write_fixture(tmp_path, [t3, t1, t5, t2, t6, t4])
    viewer = FallbackLogViewer(tmp_path)
    assert [e["decision_id"] for e in viewer.entries] == ["t6", "t1", "t2", "t3", "t5", "t4"]
    output = capsys.readouterr().out
    assert output.count("Warning: Skipping unparseable timestamp") == 2

def test_malformed_field_fixture(tmp_path):
    """A numeric subject and a list ticket do not crash filtering, stats, display, or export, and never match a substring filter."""
    f1 = _make_entry("f1", TS, subject=12345)
    f2 = _make_entry("f2", TS, subject="A subject", ticket=["a", "list"])
    _write_fixture(tmp_path, [f1, f2])
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_subject("subject")
    assert [e["decision_id"] for e in viewer.entries] == ["f2"]
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_text("reset")
    assert [e["decision_id"] for e in viewer.entries] == ["f1"]
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_text("subject")
    assert [e["decision_id"] for e in viewer.entries] == ["f2"]
    viewer = FallbackLogViewer(tmp_path)
    stats = viewer.generate_summary_stats()
    assert stats["total_parseable"] == 2
    viewer.display_entries()
    out = tmp_path / "out.csv"
    viewer.export_to_csv(out)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    assert rows[1]["ticket_text"] == "['a', 'list']"

def test_nested_structure_fixture(tmp_path):
    """Malformed nested structures do not crash, do not count as human or automated, and do not qualify as classification evidence, while a valid ticket string still matches a text search."""
    n1 = {"decision_id": "n1", "ticket": "Please reset my password.", "result": "not a dict", "timestamp": TS}
    n2 = {"decision_id": "n2", "ticket": "Please reset my password.", "result": None, "timestamp": TS}
    n3 = _make_entry("n3", TS, result={"routing_decision": "human_agent"})
    n4 = _make_entry("n4", TS, result={"ticket_type": ["access_request"]})
    n5 = _make_entry("n5", TS, subject=123, ticket="Please reset my password for the widget.")
    _write_fixture(tmp_path, [n1, n2, n3, n4, n5])
    viewer = FallbackLogViewer(tmp_path)
    stats = viewer.generate_summary_stats()
    assert stats["total_parseable"] == 5
    assert stats["incomplete_classification"] == 3
    assert stats["incomplete_routing"] == 1
    assert stats["completed_decisions"] == 1
    assert stats["human"] == 0
    assert stats["automated"] == 1
    assert stats["human_routing_rate"] == pytest.approx(0.0)
    assert stats["average_confidence"] == pytest.approx(0.99)
    viewer.display_entries()
    out = tmp_path / "out.csv"
    viewer.export_to_csv(out)
    assert len(list(csv.DictReader(out.open()))) == 5
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_text("widget")
    assert [e["decision_id"] for e in viewer.entries] == ["n5"]
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_routing("automated_response")
    assert [e["decision_id"] for e in viewer.entries] == ["n4", "n5"]
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_routing("human_agent")
    assert viewer.entries == []

def test_malformed_flag_fixture(tmp_path):
    """Malformed flag values are excluded from population 2 and counted in malformed_status, never silently treated as False."""
    g1 = _make_entry("g1", TS, result={"api_failed": "false"})
    g2 = _make_entry("g2", TS, result={"preprocessing_failed": None})
    g3 = _make_entry("g3", TS, result={"model_skip": 0})
    g4 = _make_entry("g4", TS, result={"confidence_score": 0.90})
    _write_fixture(tmp_path, [g1, g2, g3, g4])
    viewer = FallbackLogViewer(tmp_path)
    stats = viewer.generate_summary_stats()
    assert stats["total_parseable"] == 4
    assert stats["malformed_status"] == 3
    assert stats["api_failed"] == 0
    assert stats["preprocessing_failed"] == 0
    assert stats["model_skip"] == 0
    assert stats["completed_decisions"] == 1
    assert stats["automated"] == 1
    assert stats["human_routing_rate"] == pytest.approx(0.0)
    assert stats["incomplete_classification"] == 0

def test_counter_overlap_fixture(tmp_path):
    """An entry with invalid evidence and api_failed True counts in both counters, and an entry with two malformed flags increments malformed_status only once."""
    o1 = _make_entry("o1", TS, result={"api_failed": True,
                                        "routing_decision": {"route": "human_agent", "reason": "API failure."}})
    del o1["result"]["ticket_type"]
    o2 = _make_entry("o2", TS, result={"api_failed": "false", "preprocessing_failed": None})
    o3 = _make_entry("o3", TS, result={"confidence_score": 0.90})
    _write_fixture(tmp_path, [o1, o2, o3])
    viewer = FallbackLogViewer(tmp_path)
    stats = viewer.generate_summary_stats()
    assert stats["total_parseable"] == 3
    assert stats["incomplete_classification"] == 1
    assert stats["api_failed"] == 1
    assert stats["preprocessing_failed"] == 0
    assert stats["malformed_status"] == 1
    assert stats["completed_decisions"] == 1
    assert stats["automated"] == 1

def test_top_level_values_skipped(tmp_path):
    """Null, list, string, and number lines are skipped safely with a warning and counted in nothing."""
    content = "\n".join([
        "null",
        "[1, 2]",
        '"a string"',
        "42",
        json.dumps(_make_entry("ok-1", TS)),
        "",
    ]) + "\n"
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "2026-11-19.jsonl").write_text(content)
    viewer = FallbackLogViewer(tmp_path)
    assert len(viewer.entries) == 1
    stats = viewer.generate_summary_stats()
    assert stats["total_parseable"] == 1

def test_finite_confidence_fixture(tmp_path):
    """NaN and infinite confidence values are excluded from average confidence and counted in incomplete_confidence."""
    c1 = _make_entry("c1", TS, result={"confidence_score": float("nan"),
                                       "routing_decision": {"route": "human_agent", "reason": "Low confidence."}})
    c2 = _make_entry("c2", TS, result={"confidence_score": float("inf"),
                                       "routing_decision": {"route": "human_agent", "reason": "Low confidence."}})
    c3 = _make_entry("c3", TS, result={"confidence_score": float("-inf")})
    c4 = _make_entry("c4", TS, result={"confidence_score": 0.80})
    _write_fixture(tmp_path, [c1, c2, c3, c4])
    viewer = FallbackLogViewer(tmp_path)
    stats = viewer.generate_summary_stats()
    assert stats["total_parseable"] == 4
    assert stats["completed_decisions"] == 4
    assert stats["human"] == 2
    assert stats["automated"] == 2
    assert stats["human_routing_rate"] == pytest.approx(0.5)
    assert stats["incomplete_confidence"] == 3
    assert stats["average_confidence"] == pytest.approx(0.80)
    assert stats["average_confidence_by_route"]["human_agent"] is None
    assert stats["average_confidence_by_route"]["automated_response"] == pytest.approx(0.80)

def test_identity_fixture(tmp_path):
    """Every routing_decision shape is accounted for, and malformed-flag and evidence-failing entries are excluded from population 2 before the identity is evaluated."""
    i1 = _make_entry("i1", TS)
    del i1["result"]["routing_decision"]
    i2 = _make_entry("i2", TS, result={"routing_decision": "human_agent"})
    i3 = _make_entry("i3", TS, result={"routing_decision": {}})
    i4 = _make_entry("i4", TS, result={"routing_decision": {"route": 123}})
    i5 = _make_entry("i5", TS, result={"routing_decision": {"route": "pending"}})
    i6 = _make_entry("i6", TS, result={"routing_decision": {"route": "human_agent", "reason": "Low confidence."}})
    i7 = _make_entry("i7", TS)
    i8 = _make_entry("i8", TS, result={"api_failed": "false",
                                        "routing_decision": {"route": "human_agent", "reason": "Low confidence."}})
    i9 = _make_entry("i9", TS, result={"routing_decision": {"route": "human_agent", "reason": "Low confidence."}})
    del i9["result"]["ticket_type"]
    _write_fixture(tmp_path, [i1, i2, i3, i4, i5, i6, i7, i8, i9])
    viewer = FallbackLogViewer(tmp_path)
    stats = viewer.generate_summary_stats()
    assert stats["total_parseable"] == 9
    assert stats["incomplete_routing"] == 5
    assert stats["completed_decisions"] == 2
    assert stats["human"] == 1
    assert stats["automated"] == 1
    assert stats["malformed_status"] == 1
    assert stats["incomplete_classification"] == 1
    assert stats["completed_decisions"] + stats["incomplete_routing"] == 7

def test_export_columns_and_order(tmp_path):
    """Export writes one row per filtered entry with the new columns present, in the stated order."""
    entry = _make_entry("exp-1", TS, subject="Password help", sender="Jane <jane@example.com>",
                        sender_domain="example.com", source_ticket_id="INC-7")
    _write_fixture(tmp_path, [entry])
    viewer = FallbackLogViewer(tmp_path)
    out = tmp_path / "report.csv"
    viewer.export_to_csv(out)
    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "timestamp", "ticket_type", "confidence_score", "contains_pii", "ticket_text",
            "decision_id", "subject", "sender", "sender_domain", "source_ticket_id",
            "route", "reason",
        ]
        rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    assert row["decision_id"] == "exp-1"
    assert row["subject"] == "Password help"
    assert row["sender"] == "Jane <jane@example.com>"
    assert row["sender_domain"] == "example.com"
    assert row["source_ticket_id"] == "INC-7"
    assert row["route"] == "automated_response"
    assert row["reason"] == "Confident classification."
    assert row["timestamp"] == TS
    assert row["ticket_type"] == "access_request"

def test_export_formula_protection(tmp_path):
    """Every exported cell starting with =, +, -, or @, including after leading space or tab, carries the single-quote prefix."""
    entry = _make_entry("-abc", TS, ticket="=1+1", subject=" =2+2", sender="\t=3+3",
                        sender_domain="@evil.com", source_ticket_id="+123",
                        result={"confidence_score": -0.5,
                                "routing_decision": {"route": "automated_response", "reason": "=reason"}})
    _write_fixture(tmp_path, [entry])
    viewer = FallbackLogViewer(tmp_path)
    out = tmp_path / "report.csv"
    viewer.export_to_csv(out)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["ticket_text"] == "'=1+1"
    assert row["subject"] == "' =2+2"
    assert row["sender"] == "'\t=3+3"
    assert row["sender_domain"] == "'@evil.com"
    assert row["source_ticket_id"] == "'+123"
    assert row["decision_id"] == "'-abc"
    assert row["reason"] == "'=reason"
    assert row["confidence_score"] == "'-0.5"
    assert row["ticket_type"] == "access_request"
    assert row["route"] == "automated_response"

def test_export_missing_values_are_empty(tmp_path):
    """Missing values export as empty strings."""
    entry = _make_entry("exp-2", TS)
    _write_fixture(tmp_path, [entry])
    viewer = FallbackLogViewer(tmp_path)
    out = tmp_path / "report.csv"
    viewer.export_to_csv(out)
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["subject"] == ""
    assert rows[0]["sender"] == ""
    assert rows[0]["sender_domain"] == ""
    assert rows[0]["source_ticket_id"] == ""

def test_old_shape_entry_no_crash(tmp_path):
    """An entry missing routing_decision and all new keys does not crash filtering, stats, display, or export."""
    old = {
        "ticket": "Please reset my password.",
        "result": {
            "ticket_type": "access_request",
            "confidence_score": 0.99,
            "contains_pii": False,
            "model": "gemini-3.8-flash",
            "api_failed": False,
            "routing_decision": {"route": "automated_response", "reason": "Confident classification."},
        },
        "timestamp": TS,
    }
    _write_fixture(tmp_path, [old])
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_routing("automated_response")
    assert len(viewer.entries) == 1
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_text("reset")
    assert len(viewer.entries) == 1
    viewer = FallbackLogViewer(tmp_path)
    stats = viewer.generate_summary_stats()
    assert stats["completed_decisions"] == 1
    viewer.display_entries()
    out = tmp_path / "report.csv"
    viewer.export_to_csv(out)
    assert len(list(csv.DictReader(out.open()))) == 1

def test_failure_entries_load_and_display(tmp_path):
    """Detection-failure and redaction-failure entries load, filter, and display without error."""
    det = _make_entry("det-1", TS, ticket="[Ticket withheld: PII detection failed]",
                      result={"ticket_type": "unknown", "confidence_score": 0.0, "contains_pii": True,
                              "preprocessing_failed": True,
                              "routing_decision": {"route": "human_agent", "reason": "PII preprocessing failed."}},
                      detection_failed=True)
    red = _make_entry("red-1", TS, ticket="[Ticket withheld: PII redaction failed]",
                      result={"ticket_type": "unknown", "confidence_score": 0.0, "contains_pii": True,
                              "preprocessing_failed": True,
                              "routing_decision": {"route": "human_agent", "reason": "PII preprocessing failed."}},
                      redaction_failed=True, pii_types=["email"])
    _write_fixture(tmp_path, [det, red])
    viewer = FallbackLogViewer(tmp_path)
    stats = viewer.generate_summary_stats()
    assert stats["total_parseable"] == 2
    assert stats["preprocessing_failed"] == 2
    assert stats["pii_count"] == 2
    assert stats["preprocessing_failed_human_routed"] == 2
    viewer.display_entries()
    viewer.filter_by_routing("human_agent")
    assert len(viewer.entries) == 2
    out = tmp_path / "report.csv"
    viewer.export_to_csv(out)
    assert len(list(csv.DictReader(out.open()))) == 2

def test_pii_display(tmp_path, capsys):
    """Display shows Yes for Boolean True, No for Boolean False, and Unknown for missing or malformed contains_pii, while stats and filtering keep strict Boolean identity."""
    yes = _make_entry("p-yes", TS, result={"contains_pii": True})
    no = _make_entry("p-no", TS, result={"contains_pii": False})
    missing = _make_entry("p-missing", TS)
    del missing["result"]["contains_pii"]
    malformed = _make_entry("p-malformed", TS, result={"contains_pii": "yes"})
    _write_fixture(tmp_path, [yes, no, missing, malformed])
    viewer = FallbackLogViewer(tmp_path)
    viewer.display_entries()
    output = capsys.readouterr().out
    assert "PII:         Yes" in output
    assert "PII:         No" in output
    assert output.count("PII:         Unknown") == 2
    stats = viewer.generate_summary_stats()
    assert stats["pii_count"] == 1
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_pii(contains_pii=True)
    assert [e["decision_id"] for e in viewer.entries] == ["p-yes"]
    viewer = FallbackLogViewer(tmp_path)
    viewer.filter_by_pii(contains_pii=False)
    assert [e["decision_id"] for e in viewer.entries] == ["p-no"]

def test_malformed_lines_skipped(tmp_path, capsys):
    """Malformed lines and partial lines are skipped with a warning and do not crash load, stats, or export."""
    content = "\n".join([
        "{not json",
        '{"decision_id": "partial-1", "ticket": "trunc',
        json.dumps(_make_entry("good-1", TS)),
        "",
    ]) + "\n"
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "2026-11-19.jsonl").write_text(content)
    viewer = FallbackLogViewer(tmp_path)
    assert len(viewer.entries) == 1
    assert viewer.entries[0]["decision_id"] == "good-1"
    output = capsys.readouterr().out
    assert output.count("Warning: Skipping malformed log entry") == 2
    stats = viewer.generate_summary_stats()
    assert stats["total_parseable"] == 1
    viewer.display_entries()
    out = tmp_path / "report.csv"
    viewer.export_to_csv(out)
    assert len(list(csv.DictReader(out.open()))) == 1

def test_complete_looking_line_loaded(tmp_path):
    """A complete-looking line from a failed write is loaded as a parseable entry, because the viewer cannot distinguish it from an ordinary successful write."""
    complete = _make_entry("complete-1", TS)
    _write_fixture(tmp_path, [complete])
    viewer = FallbackLogViewer(tmp_path)
    assert len(viewer.entries) == 1
    stats = viewer.generate_summary_stats()
    assert stats["total_parseable"] == 1
    assert stats["completed_decisions"] == 1

def test_missing_path_warning(tmp_path, capsys):
    """A missing log path keeps the warning and an empty result."""
    viewer = FallbackLogViewer(tmp_path / "does-not-exist")
    assert viewer.entries == []
    output = capsys.readouterr().out
    assert "Warning: Log file not found" in output
    stats = viewer.generate_summary_stats()
    assert stats["total_parseable"] == 0
    assert stats["completed_decisions"] == 0
    assert stats["human_routing_rate"] is None
    assert stats["average_confidence"] is None
    assert stats["human"] == 0
    assert stats["automated"] == 0

def test_cli_route_rejects_bad_value(monkeypatch):
    """--route accepts only human_agent or automated_response and fails with a usage error otherwise."""
    monkeypatch.setattr(sys, "argv", ["fallback_viewer.py", "--route", "bogus"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2

def test_stats_empty_population_note(tmp_path, capsys):
    """Stats on a known fixture state that they describe the parseable records, not a guaranteed complete population."""
    _write_fixture(tmp_path, _stats_fixture())
    viewer = FallbackLogViewer(tmp_path)
    viewer.generate_summary_stats()
    output = capsys.readouterr().out
    assert "parseable records" in output
    assert "Fallback Rate" not in output
