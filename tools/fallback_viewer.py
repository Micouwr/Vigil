#!/usr/bin/env python3
"""
Decision Log Viewer - Vigil
ISO/IEC 42001 Compliance Tool

Purpose: Human-readable interface for reviewing every controlled decision
Vigil records, in per-day log files plus the legacy fallback log.
Usage: python tools/fallback_viewer.py [options]

The viewer reads the decision log: all *.jsonl files in the logs/ directory
plus the legacy fallback_log.jsonl at the project root, as one logical log.
The default view is all entries. Use --route human_agent for the human
review queue. Routing to human review means the ticket was queued for
human review, not that a person reviewed, approved, assigned, or acted
on it.

The viewer population is dictionary-shaped JSON entries successfully
loaded by the viewer, not a guaranteed list of confirmed persistence
operations. A complete-looking line from a failed write cannot be
distinguished from an ordinary successful write, so the statistics
describe the available recorded population, not a guaranteed complete
population of all attempts.

Exported values are formula-protected: a cell whose final string starts
with =, +, -, or @, after stripping leading whitespace, is prefixed with
a single quote. This is a practical mitigation for common spreadsheet
formula interpretation, not a universal guarantee across every
spreadsheet application and CSV import method.
"""

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ticket categories in scope.yaml. The viewer does not invent categories.
TICKET_TYPES = (
    "access_request",
    "password_reset",
    "software_issue",
    "hardware_issue",
    "billing_question",
    "compliance_flag",
    "unknown",
)

ROUTES = ("human_agent", "automated_response")

CSV_COLUMNS = [
    "timestamp", "ticket_type", "confidence_score", "contains_pii",
    "ticket_text", "decision_id", "subject", "sender", "sender_domain",
    "source_ticket_id", "route", "reason",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _result(entry):
    """Returns the entry result dict, or an empty dict when missing or not a dict."""
    result = entry.get("result")
    return result if isinstance(result, dict) else {}


def _route(entry):
    """Returns the entry routing decision route, or None when missing or malformed."""
    routing = _result(entry).get("routing_decision")
    if isinstance(routing, dict):
        route = routing.get("route")
        return route if isinstance(route, str) else None
    return None


def _reason(entry):
    """Returns the entry routing decision reason, or None when missing or malformed."""
    routing = _result(entry).get("routing_decision")
    if isinstance(routing, dict):
        reason = routing.get("reason")
        return reason if isinstance(reason, str) else None
    return None


def _valid_confidence(entry):
    """Returns the entry confidence score when it is a finite JSON number, else None.

    A boolean is not a valid confidence score, and neither are NaN or
    positive or negative infinity, because the JSON parser accepts those
    non-standard numeric values and they must not enter the average.
    """
    value = _result(entry).get("confidence_score")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def _flag_status(entry, key):
    """Returns 'true', 'false', or 'malformed' for a result flag.

    Absent means False, for entries written before the key existed.
    An actual Boolean is that value. Any other present value is
    malformed and is never treated as False.
    """
    result = _result(entry)
    if key not in result:
        return "false"
    value = result[key]
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "malformed"


def _has_classification_evidence(entry):
    """True when the result carries a present, non-empty ticket_type in the seven in-scope categories."""
    ticket_type = _result(entry).get("ticket_type")
    return isinstance(ticket_type, str) and ticket_type != "" and ticket_type in TICKET_TYPES


def _exact_match(stored, value):
    """Exact match that never matches a non-string stored value and never raises."""
    return isinstance(stored, str) and stored == value


def _substring_match(stored, needle):
    """Case-insensitive substring match on string values only; a non-string never matches."""
    return isinstance(stored, str) and needle in stored.lower()


def _pii_display(entry):
    """Yes for Boolean True, No for Boolean False, Unknown for missing or malformed."""
    value = _result(entry).get("contains_pii")
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unknown"


def _protect_cell(value):
    """Formula-injection mitigation for one exported cell.

    Applied to every final string cell value immediately before
    serialization, after type conversion, regardless of the original
    JSON value type. Leading spaces, tabs, carriage returns, and line
    feeds are stripped from the final string for the check; when the
    first remaining character is =, +, -, or @, the final string is
    prefixed with a single quote character. This is a practical
    mitigation for common spreadsheet formula interpretation, not a
    universal guarantee across every spreadsheet application and CSV
    import method. The stored log entries are not changed.
    """
    text = "" if value is None else str(value)
    stripped = text.lstrip(" \t\r\n")
    if stripped and stripped[0] in ("=", "+", "-", "@"):
        return "'" + text
    return text


def _parse_timestamp(entry):
    """Parses the entry timestamp into an aware UTC datetime, or None.

    A timestamp that is absent is treated as missing with no warning.
    A non-string timestamp, or a timestamp that is present but not
    parseable as ISO 8601, is treated as missing with a warning. A
    timestamp without timezone information is interpreted as UTC.
    """
    raw = entry.get("timestamp")
    if raw is None:
        return None
    if not isinstance(raw, str):
        print(f"  Warning: Skipping unparseable timestamp: {raw!r}")
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        print(f"  Warning: Skipping unparseable timestamp: {raw!r}")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class FallbackLogViewer:
    """Interactive viewer for decision log entries."""

    def __init__(self, log_path: Path = None):
        """Initialize the viewer with a log file, a log directory, or the default log root.

        A path pointing at a file loads only that file. A path pointing
        at a directory loads that directory's log files: the legacy
        fallback_log.jsonl first when present, then the logs/ day files
        in lexicographic order, then any other *.jsonl files directly in
        the directory. No path loads the project root the same way.
        """
        self.log_path = log_path if log_path is not None else _PROJECT_ROOT
        self.entries = []
        self.load_entries()

    def _collect_log_files(self):
        """Returns the log files to load, in load order, for the configured path."""
        path = self.log_path
        if path.is_file():
            return [path]
        if not path.is_dir():
            return []
        files = []
        legacy = path / "fallback_log.jsonl"
        if legacy.is_file():
            files.append(legacy)
        logs_dir = path / "logs"
        if logs_dir.is_dir():
            files.extend(sorted(logs_dir.glob("*.jsonl")))
        for other in sorted(path.glob("*.jsonl")):
            if other != legacy:
                files.append(other)
        return files

    def load_entries(self):
        """Load all entries from the configured log files as one logical log."""
        files = self._collect_log_files()
        if not files:
            if not self.log_path.exists():
                print(f"  Warning: Log file not found: {self.log_path}")
                print("    No decision log entries to display.")
            return

        for log_file in files:
            with log_file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        print("  Warning: Skipping malformed log entry")
                        continue
                    if not isinstance(entry, dict):
                        print("  Warning: Skipping malformed log entry")
                        continue
                    parsed = _parse_timestamp(entry)
                    if parsed is not None:
                        entry["timestamp_parsed"] = parsed
                    self.entries.append(entry)

        # Stable sort by timestamp ascending, oldest first. Entries with
        # a missing timestamp sort after timestamped entries, preserving
        # global load order across the legacy file and day files.
        def sort_key(entry):
            parsed = entry.get("timestamp_parsed")
            if parsed is None:
                return (1, datetime.min.replace(tzinfo=timezone.utc))
            return (0, parsed)

        self.entries.sort(key=sort_key)
        print(f"Loaded {len(self.entries)} decision log entries")

    def filter_by_date_range(self, days: int):
        """Filter entries within the last N days."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        self.entries = [
            e for e in self.entries
            if "timestamp_parsed" in e and e["timestamp_parsed"] >= cutoff_date
        ]
        print(f"Filtered to last {days} days: {len(self.entries)} entries")

    def filter_by_confidence(self, min_confidence: float, max_confidence: float = 1.0):
        """Filter entries by confidence score range."""
        self.entries = [
            e for e in self.entries
            if _valid_confidence(e) is not None
            and min_confidence <= _valid_confidence(e) <= max_confidence
        ]
        print(f"Filtered by confidence [{min_confidence}-{max_confidence}]: {len(self.entries)} entries")

    def filter_by_category(self, category: str):
        """Filter entries by ticket classification category."""
        self.entries = [
            e for e in self.entries
            if _result(e).get("ticket_type") == category
        ]
        print(f"Filtered by category '{category}': {len(self.entries)} entries")

    def filter_by_pii(self, contains_pii: bool = True):
        """Filter entries that contain (or don't contain) PII, by strict Boolean identity."""
        self.entries = [
            e for e in self.entries
            if _result(e).get("contains_pii") == contains_pii
        ]
        pii_status = "with" if contains_pii else "without"
        print(f"Filtered {pii_status} PII: {len(self.entries)} entries")

    def filter_by_routing(self, route: str):
        """Filter entries by routing decision route. Missing routing_decision excludes the entry."""
        self.entries = [e for e in self.entries if _route(e) == route]
        print(f"Filtered by route '{route}': {len(self.entries)} entries")

    def filter_by_decision_id(self, value: str):
        """Filter entries by exact decision ID. A non-string stored value never matches."""
        self.entries = [e for e in self.entries if _exact_match(e.get("decision_id"), value)]
        print(f"Filtered by decision ID '{value}': {len(self.entries)} entries")

    def filter_by_source_ticket_id(self, value: str):
        """Filter entries by exact source ticket ID. A non-string stored value never matches."""
        self.entries = [e for e in self.entries if _exact_match(e.get("source_ticket_id"), value)]
        print(f"Filtered by source ticket ID '{value}': {len(self.entries)} entries")

    def filter_by_subject(self, value: str):
        """Filter entries by case-insensitive substring on the subject. A non-string never matches."""
        needle = value.lower()
        self.entries = [e for e in self.entries if _substring_match(e.get("subject"), needle)]
        print(f"Filtered by subject '{value}': {len(self.entries)} entries")

    def filter_by_sender(self, value: str):
        """Filter entries by case-insensitive substring on the sender. A non-string never matches."""
        needle = value.lower()
        self.entries = [e for e in self.entries if _substring_match(e.get("sender"), needle)]
        print(f"Filtered by sender '{value}': {len(self.entries)} entries")

    def filter_by_sender_domain(self, value: str):
        """Filter entries by exact sender domain, lowercased on both sides. A non-string never matches."""
        needle = value.lower() if isinstance(value, str) else None
        self.entries = [
            e for e in self.entries
            if isinstance(e.get("sender_domain"), str)
            and needle is not None
            and e["sender_domain"].lower() == needle
        ]
        print(f"Filtered by sender domain '{value}': {len(self.entries)} entries")

    def filter_by_text(self, value: str):
        """Filter entries by case-insensitive substring across subject plus ticket text."""
        needle = value.lower()

        def matches(entry):
            for field in (entry.get("subject"), entry.get("ticket")):
                if isinstance(field, str) and needle in field.lower():
                    return True
            return False

        self.entries = [e for e in self.entries if matches(e)]
        print(f"Filtered by text '{value}': {len(self.entries)} entries")

    def generate_summary_stats(self):
        """Generate and display summary statistics, and return them as a dict.

        The statistics describe the parseable records available in the
        log files, not a guaranteed complete population of all attempts.
        """
        total_parseable = len(self.entries)
        pii_count = 0
        api_failed = 0
        preprocessing_failed = 0
        model_skip = 0
        api_failed_human_routed = 0
        preprocessing_failed_human_routed = 0
        incomplete_classification = 0
        malformed_status = 0
        population2 = 0
        completed_decisions = 0
        human = 0
        automated = 0
        incomplete_routing = 0
        incomplete_confidence = 0
        confidence_values = []
        confidence_by_route = {"human_agent": [], "automated_response": []}

        for entry in self.entries:
            result = _result(entry)
            if result.get("contains_pii") is True:
                pii_count += 1
            api_status = _flag_status(entry, "api_failed")
            prep_status = _flag_status(entry, "preprocessing_failed")
            skip_status = _flag_status(entry, "model_skip")
            if api_status == "true":
                api_failed += 1
            if prep_status == "true":
                preprocessing_failed += 1
            if skip_status == "true":
                model_skip += 1
            if "malformed" in (api_status, prep_status, skip_status):
                malformed_status += 1
            route = _route(entry)
            if api_status == "true" and route == "human_agent":
                api_failed_human_routed += 1
            if prep_status == "true" and route == "human_agent":
                preprocessing_failed_human_routed += 1
            if not _has_classification_evidence(entry):
                incomplete_classification += 1
                continue
            if api_status == "true" or prep_status == "true" or skip_status == "true":
                continue
            if "malformed" in (api_status, prep_status, skip_status):
                continue
            population2 += 1
            confidence = _valid_confidence(entry)
            if confidence is None:
                incomplete_confidence += 1
            if route == "human_agent":
                completed_decisions += 1
                human += 1
                if confidence is not None:
                    confidence_values.append(confidence)
                    confidence_by_route["human_agent"].append(confidence)
            elif route == "automated_response":
                completed_decisions += 1
                automated += 1
                if confidence is not None:
                    confidence_values.append(confidence)
                    confidence_by_route["automated_response"].append(confidence)
            else:
                incomplete_routing += 1

        human_routing_rate = None
        if completed_decisions > 0:
            human_routing_rate = human / completed_decisions
        average_confidence = None
        if confidence_values:
            average_confidence = sum(confidence_values) / len(confidence_values)
        average_confidence_by_route = {}
        for route_name in ("human_agent", "automated_response"):
            values = confidence_by_route[route_name]
            average_confidence_by_route[route_name] = (
                sum(values) / len(values) if values else None
            )

        stats = {
            "total_parseable": total_parseable,
            "completed_decisions": completed_decisions,
            "human": human,
            "automated": automated,
            "human_routing_rate": human_routing_rate,
            "average_confidence": average_confidence,
            "average_confidence_by_route": average_confidence_by_route,
            "pii_count": pii_count,
            "api_failed": api_failed,
            "preprocessing_failed": preprocessing_failed,
            "model_skip": model_skip,
            "api_failed_human_routed": api_failed_human_routed,
            "preprocessing_failed_human_routed": preprocessing_failed_human_routed,
            "incomplete_routing": incomplete_routing,
            "incomplete_confidence": incomplete_confidence,
            "incomplete_classification": incomplete_classification,
            "malformed_status": malformed_status,
        }

        print("\n" + "=" * 70)
        print("DECISION LOG SUMMARY STATISTICS")
        print("=" * 70)

        print(f"\nTotal Parseable Entries: {total_parseable}")

        categories = Counter(
            _result(e).get("ticket_type")
            if isinstance(_result(e).get("ticket_type"), str)
            else "unknown"
            for e in self.entries
        )
        print("\nClassification Distribution:")
        for category, count in categories.most_common():
            percentage = (count / total_parseable) * 100 if total_parseable else 0
            print(f"   {category:20s}: {count:4d} ({percentage:5.1f}%)")

        print("\nRouting:")
        print(f"   Human-routed:    {human}")
        print(f"   Automated:       {automated}")
        print(f"   Completed decisions: {completed_decisions}")
        if human_routing_rate is None:
            print("   Human Routing Rate: Unavailable")
        else:
            print(f"   Human Routing Rate: {human_routing_rate:.1%} ({human}/{completed_decisions})")

        print("\nConfidence Scores:")
        if average_confidence is None:
            print("   Average: Unavailable")
        else:
            print(f"   Average: {average_confidence:.2f}")
        for route_name in ("human_agent", "automated_response"):
            route_average = average_confidence_by_route[route_name]
            if route_average is None:
                print(f"   Average ({route_name}): Unavailable")
            else:
                print(f"   Average ({route_name}): {route_average:.2f}")

        pii_percentage = (pii_count / total_parseable) * 100 if total_parseable else 0
        print("\nPII Detection:")
        print(f"   Tickets with PII: {pii_count} ({pii_percentage:.1f}%)")

        print("\nFailures and Skips (fail-closed human routes reported separately):")
        print(f"   API failures: {api_failed} ({api_failed_human_routed} human-routed)")
        print(f"   Preprocessing failures: {preprocessing_failed} ({preprocessing_failed_human_routed} human-routed)")
        print(f"   Model skips: {model_skip}")

        print("\nIncomplete Records:")
        print(f"   Incomplete routing: {incomplete_routing}")
        print(f"   Incomplete confidence: {incomplete_confidence}")
        print(f"   Incomplete classification: {incomplete_classification}")
        print(f"   Malformed status flags: {malformed_status}")

        if any("timestamp_parsed" in e for e in self.entries):
            timestamps = [e["timestamp_parsed"] for e in self.entries if "timestamp_parsed" in e]
            oldest = min(timestamps)
            newest = max(timestamps)
            print("\nDate Range:")
            print(f"   Oldest: {oldest.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Newest: {newest.strftime('%Y-%m-%d %H:%M:%S')}")

        print("\nNote: statistics describe the parseable records available")
        print("in the log files, not a guaranteed complete population of")
        print("all attempts. A lower Human Routing Rate is not automatically")
        print("an improvement: it could mean less oversight.")
        print("=" * 70 + "\n")
        return stats

    def display_entries(self, limit: int = None):
        """Display entries in a human-readable format."""
        if not self.entries:
            print("No entries to display")
            return

        entries_to_show = self.entries[:limit] if limit else self.entries

        print("\n" + "=" * 70)
        print("DECISION LOG ENTRIES")
        print("=" * 70 + "\n")

        for i, entry in enumerate(entries_to_show, 1):
            result = _result(entry)
            ticket_text = entry.get("ticket")
            timestamp = entry.get("timestamp", "N/A")
            subject = entry.get("subject")
            route = _route(entry)
            reason = _reason(entry)

            print(f"Entry #{i}")
            print(f"{'─' * 70}")
            print(f"Timestamp:   {timestamp}")
            print(f"Decision ID: {entry.get('decision_id', 'N/A')}")
            print(f"Category:    {result.get('ticket_type', 'N/A')}")
            print(f"Confidence:  {result.get('confidence_score', 'N/A')}")
            print(f"PII:         {_pii_display(entry)}")
            print(f"Route:       {route if route is not None else 'N/A'}")
            print(f"Reason:      {reason if reason is not None else 'N/A'}")
            if isinstance(subject, str) and subject:
                print(f"Subject:     {subject}")
            else:
                preview = ticket_text if isinstance(ticket_text, str) else ""
                print(f"Subject:     {preview[:200]}{'...' if len(preview) > 200 else ''}")
            print("Ticket Text:")
            text = ticket_text if isinstance(ticket_text, str) else "N/A"
            print(f"   {text[:200]}{'...' if len(text) > 200 else ''}")
            print()

        if limit and len(self.entries) > limit:
            print(f"... and {len(self.entries) - limit} more entries")

        print("=" * 70 + "\n")

    def export_to_csv(self, output_path: Path):
        """Export filtered entries to a CSV file.

        Exported values are formula-protected: a cell whose final string
        starts with =, +, -, or @, after stripping leading whitespace,
        is prefixed with a single quote. This is a practical mitigation
        for common spreadsheet formula interpretation, not a universal
        guarantee across every spreadsheet application and CSV import
        method. The stored log entries are not changed.
        """
        if not self.entries:
            print("  No entries to export")
            return

        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)

            writer.writeheader()
            for entry in self.entries:
                result = _result(entry)
                route = _route(entry)
                reason = _reason(entry)
                writer.writerow({
                    "timestamp": _protect_cell(entry.get("timestamp", "")),
                    "ticket_type": _protect_cell(result.get("ticket_type", "")),
                    "confidence_score": _protect_cell(result.get("confidence_score", "")),
                    "contains_pii": _protect_cell(result.get("contains_pii", "")),
                    "ticket_text": _protect_cell(entry.get("ticket", "")),
                    "decision_id": _protect_cell(entry.get("decision_id", "")),
                    "subject": _protect_cell(entry.get("subject", "")),
                    "sender": _protect_cell(entry.get("sender", "")),
                    "sender_domain": _protect_cell(entry.get("sender_domain", "")),
                    "source_ticket_id": _protect_cell(entry.get("source_ticket_id", "")),
                    "route": _protect_cell(route if route is not None else ""),
                    "reason": _protect_cell(reason if reason is not None else ""),
                })

        print(f"Exported {len(self.entries)} entries to {output_path}")


def main():
    """Main entry point for the decision log viewer."""
    parser = argparse.ArgumentParser(
        description="Vigil Decision Log Viewer - ISO/IEC 42001 Compliance Tool. "
            "Reads the decision log, day files plus the legacy fallback log. "
            "The default view is all entries; use --route human_agent for the "
            "human review queue. Exported values are formula-protected: a cell "
            "whose final string starts with =, +, -, or @ is prefixed with a "
            "single quote, as a practical mitigation for common spreadsheet "
            "formula interpretation, not a universal guarantee.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View all decision entries
  python tools/fallback_viewer.py

  # View the human review queue
  python tools/fallback_viewer.py --route human_agent

  # View entries from last 7 days
  python tools/fallback_viewer.py --date-range 7

  # View low-confidence entries
  python tools/fallback_viewer.py --max-confidence 0.3

  # View only PII-flagged tickets
  python tools/fallback_viewer.py --contains-pii

  # Search by decision ID, subject, or text
  python tools/fallback_viewer.py --decision-id 11111111-1111-4111-8111-111111111111
  python tools/fallback_viewer.py --subject password
  python tools/fallback_viewer.py --search "reset my password"

  # Export filtered results to CSV
  python tools/fallback_viewer.py --date-range 30 --export decision_report.csv
        """
    )

    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Path to a log file or a log directory (default: project logs/ plus legacy fallback_log.jsonl)"
    )

    parser.add_argument(
        "--date-range",
        type=int,
        metavar="DAYS",
        help="Show entries from last N days"
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        metavar="X",
        help="Filter by minimum confidence score (default: 0.0)"
    )

    parser.add_argument(
        "--max-confidence",
        type=float,
        default=1.0,
        metavar="X",
        help="Filter by maximum confidence score (default: 1.0)"
    )

    parser.add_argument(
        "--category",
        type=str,
        metavar="CAT",
        help="Filter by ticket category"
    )

    parser.add_argument(
        "--contains-pii",
        action="store_true",
        help="Show only PII-flagged tickets"
    )

    parser.add_argument(
        "--no-pii",
        action="store_true",
        help="Show only tickets WITHOUT PII"
    )

    parser.add_argument(
        "--route",
        type=str,
        choices=["human_agent", "automated_response"],
        metavar="ROUTE",
        help="Filter by routing decision: human_agent or automated_response"
    )

    parser.add_argument(
        "--decision-id",
        type=str,
        metavar="ID",
        help="Filter by exact decision ID"
    )

    parser.add_argument(
        "--source-ticket-id",
        type=str,
        metavar="ID",
        help="Filter by exact source ticket ID"
    )

    parser.add_argument(
        "--subject",
        type=str,
        metavar="TEXT",
        help="Filter by case-insensitive substring on the subject"
    )

    parser.add_argument(
        "--sender",
        type=str,
        metavar="TEXT",
        help="Filter by case-insensitive substring on the sender"
    )

    parser.add_argument(
        "--sender-domain",
        type=str,
        metavar="DOMAIN",
        help="Filter by exact sender domain, lowercased on both sides"
    )

    parser.add_argument(
        "--search",
        type=str,
        metavar="TEXT",
        help="Filter by case-insensitive substring across subject plus ticket text"
    )

    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Limit number of entries displayed"
    )

    parser.add_argument(
        "--export",
        type=Path,
        metavar="FILE",
        help="Export filtered results to CSV file"
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Show only summary statistics (no individual entries)"
    )

    args = parser.parse_args()

    # Initialize viewer
    viewer = FallbackLogViewer(args.log_path)

    # Apply filters
    if args.date_range:
        viewer.filter_by_date_range(args.date_range)

    if args.min_confidence > 0.0 or args.max_confidence < 1.0:
        viewer.filter_by_confidence(args.min_confidence, args.max_confidence)

    if args.category:
        viewer.filter_by_category(args.category)

    if args.contains_pii:
        viewer.filter_by_pii(contains_pii=True)

    if args.no_pii:
        viewer.filter_by_pii(contains_pii=False)

    if args.route:
        viewer.filter_by_routing(args.route)

    if args.decision_id:
        viewer.filter_by_decision_id(args.decision_id)

    if args.source_ticket_id:
        viewer.filter_by_source_ticket_id(args.source_ticket_id)

    if args.subject:
        viewer.filter_by_subject(args.subject)

    if args.sender:
        viewer.filter_by_sender(args.sender)

    if args.sender_domain:
        viewer.filter_by_sender_domain(args.sender_domain)

    if args.search:
        viewer.filter_by_text(args.search)

    # Display results
    viewer.generate_summary_stats()

    if not args.stats_only:
        viewer.display_entries(limit=args.limit)

    # Export if requested
    if args.export:
        viewer.export_to_csv(args.export)


if __name__ == "__main__":
    main()
