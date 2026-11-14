from __future__ import annotations

import argparse
import json
from pathlib import Path

from extraction_utils import STATUS_EQUIVALENTS, normalize_value


RUNS = (
    {
        "name": "mem0_deepseek",
        "original": "runs/mem0_deepseek_full/result.json",
        "rescored": "runs/mem0_deepseek_full/result_rescored.json",
    },
    {
        "name": "mem0_oracle",
        "original": "runs/mem0_oracle_full/result.json",
        "rescored": "runs/mem0_oracle_full/result_rescored.json",
    },
    {
        "name": "versioned_deepseek",
        "original": "runs/extraction_deepseek_full/end_to_end_result.json",
        "rescored": "runs/extraction_deepseek_full/end_to_end_result_rescored.json",
    },
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_scored_result(result: dict) -> dict:
    return result.get("end_to_end_answers", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    query_by_id = {
        query["query_id"]: query
        for case in cases
        for query in case["queries"]
    }
    experiment_root = args.cases.parent.parent

    summary = {
        "cases": len(cases),
        "normalization_rules": STATUS_EQUIVALENTS,
        "runs": {},
    }
    for run in RUNS:
        original_result = get_scored_result(
            load_json(experiment_root / run["original"])
        )
        rescored_result = get_scored_result(
            load_json(experiment_root / run["rescored"])
        )

        by_type = {}
        residual_errors = []
        for detail in rescored_result["details"]:
            query = query_by_id[detail["query_id"]]
            expected = normalize_value(query["fact_key"], query["answer"])
            raw_actual = detail.get(
                "actual_normalized",
                detail.get("actual", detail.get("predicted")),
            )
            normalized_actual = normalize_value(
                query["fact_key"],
                raw_actual,
            )
            correct = str(expected).strip() == str(normalized_actual).strip()

            bucket = by_type.setdefault(
                detail["query_type"],
                {"correct": 0, "total": 0},
            )
            bucket["total"] += 1
            bucket["correct"] += int(correct)
            if not correct:
                residual_errors.append(
                    {
                        "query_id": detail["query_id"],
                        "query_type": detail["query_type"],
                        "answer": query["answer"],
                        "actual": raw_actual,
                        "normalized_actual": normalized_actual,
                    }
                )

        for bucket in by_type.values():
            bucket["accuracy"] = bucket["correct"] / bucket["total"]
        correct = sum(bucket["correct"] for bucket in by_type.values())
        total = sum(bucket["total"] for bucket in by_type.values())

        summary["runs"][run["name"]] = {
            "strict_accuracy": original_result["overall_accuracy"],
            "strict_by_query_type": original_result["by_query_type"],
            "normalized_accuracy": correct / total,
            "normalized_by_query_type": by_type,
            "residual_errors": residual_errors,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
