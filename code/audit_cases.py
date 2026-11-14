from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from versioned_memory import VersionedMemory


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


def parse_iso_date(value: str, location: str, errors: list[str]) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        errors.append(f"{location}: invalid ISO date {value!r}")
        return None


def normalize(value: object) -> str:
    return str(value).strip()


def audit(cases: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    errors: list[str] = []
    warnings: list[str] = []
    query_results: list[dict] = []
    case_ids = Counter()
    event_ids = Counter()
    query_ids = Counter()
    case_types = Counter()

    for case in cases:
        case_id = case.get("case_id", "<missing-case-id>")
        case_ids[case_id] += 1
        case_types[case.get("case_type", "missing")] += 1
        events = case.get("events", [])
        queries = case.get("queries", [])

        if not case.get("source_type"):
            errors.append(f"{case_id}: missing source_type")
        if len(events) < 2:
            errors.append(f"{case_id}: fewer than two events")
        if not queries:
            errors.append(f"{case_id}: no queries")

        fact_keys = {event.get("fact_key") for event in events}
        events_by_key: dict[str, list[dict]] = defaultdict(list)
        for event in events:
            event_id = event.get("event_id", "<missing-event-id>")
            event_ids[event_id] += 1
            missing = REQUIRED_EVENT_FIELDS - set(event)
            if missing:
                errors.append(f"{case_id}/{event_id}: missing {sorted(missing)}")
                continue
            if event["event_type"] not in EVENT_TYPES:
                errors.append(f"{case_id}/{event_id}: invalid event_type")
            if event["source_excerpt"] not in event["text"]:
                warnings.append(
                    f"{case_id}/{event_id}: source_excerpt is not an exact substring of text"
                )
            time_precision = event.get("time_precision", "exact")
            if event.get("event_time") is None:
                if time_precision != "unknown":
                    errors.append(
                        f"{case_id}/{event_id}: null event_time requires time_precision=unknown"
                    )
                event_date = None
            else:
                event_date = parse_iso_date(
                    event["event_time"], f"{case_id}/{event_id}/event_time", errors
                )
            recorded_date = parse_iso_date(
                event["recorded_time"], f"{case_id}/{event_id}/recorded_time", errors
            )
            if (
                event_date
                and recorded_date
                and event_date > recorded_date
                and time_precision != "scheduled"
            ):
                warnings.append(
                    f"{case_id}/{event_id}: event_time is later than recorded_time"
                )
            events_by_key[event["fact_key"]].append(event)

        for fact_key, key_events in events_by_key.items():
            if key_events[0]["event_type"] == "correct":
                errors.append(f"{case_id}/{fact_key}: correction has no prior event")
            if (
                key_events[0]["event_type"] == "transition"
                and len(key_events) == 1
            ):
                warnings.append(
                    f"{case_id}/{fact_key}: transition has no prior value for the same fact_key"
                )
            parsed_times = []
            for event in key_events:
                if event.get("event_time") is None:
                    continue
                try:
                    parsed_times.append(date.fromisoformat(event["event_time"]))
                except (TypeError, ValueError):
                    pass
            if parsed_times != sorted(parsed_times):
                warnings.append(f"{case_id}/{fact_key}: events are not chronological")

        memory = VersionedMemory(case_id)
        for event in events:
            memory.add_event(event)

        for query in queries:
            query_id = query.get("query_id", "<missing-query-id>")
            query_ids[query_id] += 1
            missing = REQUIRED_QUERY_FIELDS - set(query)
            if missing:
                errors.append(f"{case_id}/{query_id}: missing {sorted(missing)}")
                continue
            if query["query_type"] not in QUERY_TYPES:
                errors.append(f"{case_id}/{query_id}: invalid query_type")
            if query["fact_key"] not in fact_keys:
                errors.append(f"{case_id}/{query_id}: fact_key has no event")
            if query["query_type"] == "at_time":
                if not query.get("as_of"):
                    errors.append(f"{case_id}/{query_id}: at_time query missing as_of")
                else:
                    parse_iso_date(query["as_of"], f"{case_id}/{query_id}/as_of", errors)
            elif query.get("as_of"):
                errors.append(f"{case_id}/{query_id}: as_of only applies to at_time")
            if query["answer"] in {None, ""}:
                errors.append(f"{case_id}/{query_id}: empty answer")
                continue
            predicted = memory.query(query)
            correct = normalize(predicted) == normalize(query["answer"])
            query_results.append(
                {
                    "case_id": case_id,
                    "query_id": query_id,
                    "query_type": query["query_type"],
                    "answer": query["answer"],
                    "predicted": predicted,
                    "correct": correct,
                }
            )
            if not correct:
                errors.append(
                    f"{case_id}/{query_id}: standard answer {query['answer']!r} "
                    f"does not match versioned result {predicted!r}"
                )

    for name, values in {
        "case_id": case_ids,
        "event_id": event_ids,
        "query_id": query_ids,
    }.items():
        for value, count in values.items():
            if count > 1:
                errors.append(f"duplicate {name}: {value} ({count})")

    expected_types = {
        "supplement",
        "correction",
        "state_transition",
        "uncertain",
        "distractor",
    }
    if len(cases) != 20:
        errors.append(f"expected 20 cases, found {len(cases)}")
    for case_type in expected_types:
        if case_types[case_type] != 4:
            errors.append(
                f"expected 4 cases of type {case_type}, found {case_types[case_type]}"
            )

    return errors, warnings, query_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mark-machine-validated", action="store_true")
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    errors, warnings, query_results = audit(cases)
    report = {
        "validator": "audit_cases.py",
        "date": "2026-11-10",
        "cases": len(cases),
        "events": sum(len(case.get("events", [])) for case in cases),
        "queries": sum(len(case.get("queries", [])) for case in cases),
        "errors": errors,
        "warnings": warnings,
        "query_results": query_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"cases: {report['cases']}")
    print(f"events: {report['events']}")
    print(f"queries: {report['queries']}")
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")
    for warning in warnings:
        print(f"WARNING: {warning}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    if args.mark_machine_validated:
        backup = args.cases.with_suffix(".pre_machine_validation.bak.jsonl")
        shutil.copyfile(args.cases, backup)
        for case in cases:
            case["review_status"] = "machine_validated"
            case["human_review_required"] = True
            case["automated_validation"] = {
                "date": "2026-11-10",
                "result": "pass",
                "warnings": [
                    warning
                    for warning in warnings
                    if warning.startswith(case["case_id"] + "/")
                ],
            }
        args.cases.write_text(
            "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
            encoding="utf-8",
        )
        print(f"backup: {backup}")
        print("status: machine_validated")


if __name__ == "__main__":
    main()
