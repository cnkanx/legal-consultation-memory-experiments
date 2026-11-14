from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from versioned_memory import METHODS


def load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def normalize(value: object) -> str:
    return str(value).strip()


def evaluate(cases: list[dict]) -> dict:
    result = {}
    for method_name, memory_class in METHODS.items():
        per_type_correct = defaultdict(int)
        per_type_total = defaultdict(int)
        details = []

        for case in cases:
            memory = memory_class(case["case_id"])
            for event in case["events"]:
                memory.add_event(event)

            for query in case["queries"]:
                predicted = memory.query(query)
                correct = normalize(predicted) == normalize(query["answer"])
                query_type = query["query_type"]
                per_type_total[query_type] += 1
                per_type_correct[query_type] += int(correct)
                details.append(
                    {
                        "case_id": case["case_id"],
                        "query_id": query["query_id"],
                        "query_type": query_type,
                        "question": query["question"],
                        "answer": query["answer"],
                        "predicted": predicted,
                        "correct": correct,
                    }
                )

        total = len(details)
        correct_total = sum(item["correct"] for item in details)
        result[method_name] = {
            "overall_accuracy": correct_total / total if total else 0.0,
            "by_query_type": {
                query_type: {
                    "correct": per_type_correct[query_type],
                    "total": per_type_total[query_type],
                    "accuracy": (
                        per_type_correct[query_type] / per_type_total[query_type]
                        if per_type_total[query_type]
                        else 0.0
                    ),
                }
                for query_type in sorted(per_type_total)
            },
            "details": details,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    result = evaluate(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("method\t\t\toverall")
    for method_name, data in result.items():
        print(f"{method_name:20s}\t{data['overall_accuracy']:.3f}")
        for query_type, scores in data["by_query_type"].items():
            print(
                f"  {query_type:10s}\t"
                f"{scores['correct']}/{scores['total']} = {scores['accuracy']:.3f}"
            )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

