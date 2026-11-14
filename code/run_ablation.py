from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ablation_memory import AblationMemory


CONFIGS = {
    "versioned_full": {
        "use_event_types": True,
        "use_version_chain": True,
        "use_query_routing": True,
    },
    "no_event_types": {
        "use_event_types": False,
        "use_version_chain": True,
        "use_query_routing": True,
    },
    "no_version_chain": {
        "use_event_types": True,
        "use_version_chain": False,
        "use_query_routing": True,
    },
    "no_query_routing": {
        "use_event_types": True,
        "use_version_chain": True,
        "use_query_routing": False,
    },
}


def load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    results: dict[str, dict] = {}

    for name, config in CONFIGS.items():
        type_correct = defaultdict(int)
        type_total = defaultdict(int)
        details = []
        for case in cases:
            memory = AblationMemory(**config)
            for event in case["events"]:
                memory.add_event(event)
            for query in case["queries"]:
                actual = memory.query(query)
                correct = str(actual).strip() == str(query["answer"]).strip()
                type_total[query["query_type"]] += 1
                type_correct[query["query_type"]] += int(correct)
                details.append(
                    {
                        "case_id": case["case_id"],
                        "query_id": query["query_id"],
                        "query_type": query["query_type"],
                        "answer": query["answer"],
                        "actual": actual,
                        "correct": correct,
                    }
                )
        results[name] = {
            **config,
            "overall_accuracy": (
                sum(item["correct"] for item in details) / len(details)
            ),
            "by_query_type": {
                query_type: {
                    "correct": type_correct[query_type],
                    "total": type_total[query_type],
                    "accuracy": type_correct[query_type] / type_total[query_type],
                }
                for query_type in sorted(type_total)
            },
            "details": details,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for name, result in results.items():
        print(name, result["overall_accuracy"])


if __name__ == "__main__":
    main()
