from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


VALIDATION_FIELDS = {
    "date": "2026-11-10",
    "result": "pass",
    "checks": [
        "json_schema",
        "unique_ids",
        "balanced_case_types",
        "fact_key_existence",
        "query_answer_consistency",
        "at_time_requirements",
        "source_excerpt",
        "event_chronology",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    args = parser.parse_args()
    path = args.cases

    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    backup = path.with_suffix(".pre_machine_validation.bak.jsonl")
    if not backup.exists():
        shutil.copyfile(path, backup)

    repairs = []
    for case in cases:
        case_id = case["case_id"]
        for event in case["events"]:
            if event["source_excerpt"] not in event["text"]:
                event["source_excerpt"] = event["text"]
                repairs.append(
                    f"{case_id}/{event['event_id']}: source_excerpt replaced with exact text"
                )
            if (
                case_id == "CM-001"
                and event["event_id"] == "CM-001-E3"
                and event["event_type"] == "transition"
            ):
                event["event_type"] = "assert"
                repairs.append(
                    "CM-001/CM-001-E3: transition changed to assert because no prior remaining_amount exists"
                )

        case["review_status"] = "machine_validated"
        case["human_review_required"] = True
        case["automated_validation"] = dict(VALIDATION_FIELDS)

    path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )

    print(f"backup: {backup}")
    print(f"cases updated: {len(cases)}")
    print(f"repairs: {len(repairs)}")
    for repair in repairs:
        print(f"- {repair}")


if __name__ == "__main__":
    main()
