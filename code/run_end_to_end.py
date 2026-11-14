from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from extraction_utils import build_extracted_event, normalize_value
from versioned_memory import VersionedMemory


FIELDS = [
    "fact_key",
    "value",
    "event_type",
    "event_time",
    "time_precision",
    "epistemic_status",
]


def normalize_optional(value: object) -> str | None:
    if value is None or str(value).strip() in {"", "None", "null"}:
        return None
    return str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--extractions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-limit", type=int, default=0)
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if args.case_limit:
        cases = cases[: args.case_limit]
    extraction_rows = [
        json.loads(line)
        for line in args.extractions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    extraction_by_id = {
        row["event_id"]: row["extraction"] for row in extraction_rows
    }

    field_correct = defaultdict(int)
    field_total = defaultdict(int)
    complete_correct = 0
    complete_total = 0
    extraction_errors = []
    answer_details = []
    answer_type_correct = defaultdict(int)
    answer_type_total = defaultdict(int)

    for case in cases:
        gold_by_id = {event["event_id"]: event for event in case["events"]}
        output_events = []
        for event in case["events"]:
            extraction = extraction_by_id.get(event["event_id"])
            if extraction is None:
                extraction_errors.append(
                    f"{event['event_id']}: missing extraction"
                )
                continue

            event_complete = True
            for field in FIELDS:
                expected = event.get(field)
                actual = extraction.get(field)
                if field == "value":
                    expected = normalize_value(event["fact_key"], expected)
                    actual = normalize_value(event["fact_key"], actual)
                if field in {"event_time", "epistemic_status"}:
                    expected = normalize_optional(expected)
                    actual = normalize_optional(actual)
                    if field == "epistemic_status" and expected is None:
                        expected = "asserted"
                    if field == "epistemic_status" and actual is None:
                        actual = "asserted"
                field_total[field] += 1
                if str(expected) == str(actual):
                    field_correct[field] += 1
                else:
                    event_complete = False
            complete_total += 1
            complete_correct += int(event_complete)

            try:
                output_events.append(build_extracted_event(event, extraction))
            except Exception as exc:
                extraction_errors.append(f"{event['event_id']}: {exc}")

        memory = VersionedMemory(case["case_id"])
        for event in output_events:
            memory.add_event(event)

        for query in case["queries"]:
            try:
                predicted = memory.query(query)
            except Exception as exc:
                predicted = None
                extraction_errors.append(f"{query['query_id']}: query error {exc}")
            correct = str(predicted).strip() == str(query["answer"]).strip()
            answer_type_total[query["query_type"]] += 1
            answer_type_correct[query["query_type"]] += int(correct)
            answer_details.append(
                {
                    "case_id": case["case_id"],
                    "query_id": query["query_id"],
                    "query_type": query["query_type"],
                    "answer": query["answer"],
                    "predicted": predicted,
                    "correct": correct,
                }
            )

    result = {
        "extraction": {
            "field_accuracy": {
                field: {
                    "correct": field_correct[field],
                    "total": field_total[field],
                    "accuracy": (
                        field_correct[field] / field_total[field]
                        if field_total[field]
                        else 0.0
                    ),
                }
                for field in FIELDS
            },
            "complete_event_accuracy": (
                complete_correct / complete_total if complete_total else 0.0
            ),
            "errors": extraction_errors,
        },
        "end_to_end_answers": {
            "overall_accuracy": (
                sum(row["correct"] for row in answer_details)
                / len(answer_details)
                if answer_details
                else 0.0
            ),
            "by_query_type": {
                query_type: {
                    "correct": answer_type_correct[query_type],
                    "total": answer_type_total[query_type],
                    "accuracy": (
                        answer_type_correct[query_type]
                        / answer_type_total[query_type]
                    ),
                }
                for query_type in sorted(answer_type_total)
            },
            "details": answer_details,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result["extraction"]["field_accuracy"], ensure_ascii=False, indent=2))
    print(
        "complete_event_accuracy:",
        result["extraction"]["complete_event_accuracy"],
    )
    print(
        "end_to_end_overall:",
        result["end_to_end_answers"]["overall_accuracy"],
    )
    print("errors:", len(extraction_errors))


if __name__ == "__main__":
    main()

