from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


EVENT_TYPES = {"assert", "transition", "correct", "uncertain"}
QUERY_TYPES = {"current", "at_time", "original"}
REQUIRED_EVENT_FIELDS = {
    "event_id",
    "turn",
    "speaker",
    "text",
    "fact_key",
    "value",
    "event_type",
    "event_time",
    "recorded_time",
    "source_excerpt",
}
REQUIRED_QUERY_FIELDS = {
    "query_id",
    "question",
    "query_type",
    "fact_key",
    "answer",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    errors = []
    case_ids = Counter()
    event_ids = Counter()
    query_ids = Counter()
    types = Counter()

    for case_index, case in enumerate(rows, start=1):
        case_id = case.get("case_id", f"<row-{case_index}>")
        case_ids[case_id] += 1
        types[case.get("case_type", "missing")] += 1
        events = case.get("events", [])
        queries = case.get("queries", [])
        fact_keys = {event.get("fact_key") for event in events}

        if len(events) < 2:
            errors.append(f"{case_id}: fewer than two events")
        if not queries:
            errors.append(f"{case_id}: no queries")

        for event in events:
            event_id = event.get("event_id", "<missing-event-id>")
            event_ids[event_id] += 1
            missing = REQUIRED_EVENT_FIELDS - set(event)
            if missing:
                errors.append(f"{case_id}/{event_id}: missing {sorted(missing)}")
            if event.get("event_type") not in EVENT_TYPES:
                errors.append(f"{case_id}/{event_id}: invalid event_type")
            if not event.get("source_excerpt"):
                errors.append(f"{case_id}/{event_id}: empty source_excerpt")

        for query in queries:
            query_id = query.get("query_id", "<missing-query-id>")
            query_ids[query_id] += 1
            missing = REQUIRED_QUERY_FIELDS - set(query)
            if missing:
                errors.append(f"{case_id}/{query_id}: missing {sorted(missing)}")
            if query.get("query_type") not in QUERY_TYPES:
                errors.append(f"{case_id}/{query_id}: invalid query_type")
            if query.get("fact_key") not in fact_keys:
                errors.append(f"{case_id}/{query_id}: fact_key has no event")
            if query.get("query_type") == "at_time" and not query.get("as_of"):
                errors.append(f"{case_id}/{query_id}: at_time query missing as_of")
            if query.get("query_type") in {"current", "original"} and query.get("as_of"):
                errors.append(f"{case_id}/{query_id}: as_of only applies to at_time")
            if query.get("answer") in {None, ""}:
                errors.append(f"{case_id}/{query_id}: empty answer")

    for name, values in {
        "case_id": case_ids,
        "event_id": event_ids,
        "query_id": query_ids,
    }.items():
        for value, count in values.items():
            if count > 1:
                errors.append(f"duplicate {name}: {value} ({count})")

    print(f"cases: {len(rows)}")
    print(f"case_types: {dict(sorted(types.items()))}")
    print(f"events: {sum(len(case.get('events', [])) for case in rows)}")
    print(f"queries: {sum(len(case.get('queries', [])) for case in rows)}")
    if errors:
        print("\nVALIDATION ERRORS")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("validation: OK")


if __name__ == "__main__":
    main()
