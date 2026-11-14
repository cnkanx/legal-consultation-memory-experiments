from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from extraction_utils import normalize_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    result = json.loads(args.result.read_text(encoding="utf-8"))
    scored_result = result.get("end_to_end_answers", result)
    details = scored_result.get("details")
    if not isinstance(details, list):
        raise ValueError("Result does not contain a details list")

    query_by_id = {
        query["query_id"]: query
        for case in cases
        for query in case["queries"]
    }

    for detail in details:
        query = query_by_id[detail["query_id"]]
        expected = normalize_value(query["fact_key"], query["answer"])
        actual = normalize_value(
            query["fact_key"],
            detail.get(
                "actual_normalized",
                detail.get("actual", detail.get("predicted")),
            ),
        )
        detail["actual_normalized"] = actual
        detail["correct"] = str(expected).strip() == str(actual).strip()

    by_type_correct = defaultdict(int)
    by_type_total = defaultdict(int)
    for detail in details:
        query_type = detail["query_type"]
        by_type_total[query_type] += 1
        by_type_correct[query_type] += int(detail["correct"])

    scored_result["overall_accuracy"] = (
        sum(detail["correct"] for detail in details) / len(details)
        if details
        else 0.0
    )
    scored_result["by_query_type"] = {
        query_type: {
            "correct": by_type_correct[query_type],
            "total": by_type_total[query_type],
            "accuracy": by_type_correct[query_type] / by_type_total[query_type],
        }
        for query_type in sorted(by_type_total)
    }
    scored_result["rescored"] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "overall_accuracy": scored_result["overall_accuracy"],
                "by_query_type": scored_result["by_query_type"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
