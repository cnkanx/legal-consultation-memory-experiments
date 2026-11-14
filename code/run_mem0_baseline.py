from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from extraction_utils import build_extracted_event, normalize_value
from mem0_adapter import Mem0Store


ANSWER_PROMPT = """你是基于记忆回答问题的助手。

只能使用给定Mem0检索结果，不得使用外部知识，不得猜测。
返回一个简短答案，不要解释，不要加引号，不要Markdown。

如果当前问题需要当前状态：
- 优先使用最新有效信息。

如果问题询问某一历史时点：
- 根据事件时间和记录时间选择当时可用的信息。

如果问题询问最初陈述：
- 返回最早的陈述。

如果检索结果没有足够信息或存在无法消解的冲突：
- 返回UNKNOWN。
"""


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def answer_from_memories(
    client: OpenAI,
    model: str,
    query: dict,
    memories: list[dict],
) -> str:
    evidence = [
        {
            "memory": item.get("memory"),
            "score": item.get("score"),
            "metadata": item.get("metadata", {}),
        }
        for item in memories
    ]
    prompt = (
        f"问题：{query['question']}\n"
        f"查询类型：{query['query_type']}\n"
        f"历史时点：{query.get('as_of')}\n"
        f"目标事实类别：{query['fact_key']}\n"
        f"Mem0检索结果：\n{json.dumps(evidence, ensure_ascii=False, indent=2)}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=128,
    )
    return (response.choices[0].message.content or "").strip()


def answer_from_metadata(query: dict, memories: list[dict]) -> str:
    candidates = []
    for item in memories:
        metadata = item.get("metadata") or {}
        if metadata.get("fact_key") != query["fact_key"]:
            continue
        if metadata.get("event_type") == "uncertain":
            continue
        candidates.append(
            {
                "value": metadata.get("value"),
                "event_time": metadata.get("event_time"),
                "recorded_time": metadata.get("recorded_time"),
            }
        )
    if not candidates:
        return "UNKNOWN"

    if query["query_type"] == "current":
        selected = max(
            candidates,
            key=lambda item: item["recorded_time"] or item["event_time"] or "",
        )
    elif query["query_type"] == "original":
        selected = min(
            candidates,
            key=lambda item: item["recorded_time"] or item["event_time"] or "",
        )
    elif query["query_type"] == "at_time":
        as_of = query["as_of"]
        valid = [
            item
            for item in candidates
            if item["event_time"] and item["event_time"] <= as_of
        ]
        if not valid:
            return "UNKNOWN"
        selected = max(valid, key=lambda item: item["event_time"])
    else:
        return "UNKNOWN"
    return str(selected["value"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument(
        "--extractions",
        type=Path,
        help="Optional DeepSeek extraction JSONL. Omit to use oracle gold events.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--infer", action="store_true")
    parser.add_argument(
        "--answer-mode",
        choices=["llm", "metadata"],
        default="llm",
    )
    args = parser.parse_args()

    api_key = os.environ.get("J1_EVAL_API_KEY")
    api_base = os.environ.get("J1_EVAL_API_BASE")
    model = os.environ.get("J1_EVAL_MODEL_NAME", "deepseek-chat")
    if args.answer_mode == "llm" or args.infer:
        if not api_key or not api_base:
            raise SystemExit("Set J1_EVAL_API_KEY and J1_EVAL_API_BASE")
    elif not api_key:
        api_key = "dummy-key-for-offline-smoke"
    if not api_base:
        api_base = "https://api.deepseek.com"

    cases = load_jsonl(args.cases)
    if args.case_limit:
        cases = cases[: args.case_limit]

    extraction_by_id = {}
    if args.extractions:
        extraction_rows = load_jsonl(args.extractions)
        extraction_by_id = {
            row["event_id"]: row["extraction"] for row in extraction_rows
        }

    run_id = uuid.uuid4().hex[:8]
    collection_name = f"mem0_baseline_{run_id}"
    store = Mem0Store(
        collection_name,
        api_key=api_key,
        api_base=api_base,
        model=model,
    )
    answer_client = (
        OpenAI(api_key=api_key, base_url=api_base)
        if args.answer_mode == "llm"
        else None
    )

    details = []
    add_log = []
    type_correct = defaultdict(int)
    type_total = defaultdict(int)

    for case in cases:
        case_id = case["case_id"]
        for gold_event in case["events"]:
            if extraction_by_id:
                extraction = extraction_by_id.get(gold_event["event_id"])
                if extraction is None:
                    add_log.append(
                        {
                            "case_id": case_id,
                            "event_id": gold_event["event_id"],
                            "error": "missing extraction",
                        }
                    )
                    continue
                try:
                    event = build_extracted_event(gold_event, extraction)
                    event["case_id"] = case_id
                except Exception as exc:
                    add_log.append(
                        {
                            "case_id": case_id,
                            "event_id": gold_event["event_id"],
                            "error": f"event construction failed: {exc}",
                        }
                    )
                    continue
            else:
                event = {
                    **gold_event,
                    "case_id": case_id,
                }
            try:
                add_result = store.add_event(event, infer=args.infer)
                add_log.append(
                    {
                        "case_id": case_id,
                        "event_id": gold_event["event_id"],
                        "result": add_result,
                    }
                )
            except Exception as exc:
                add_log.append(
                    {
                        "case_id": case_id,
                        "event_id": gold_event["event_id"],
                        "error": str(exc),
                    }
                )

        for query in case["queries"]:
            memories = store.search(
                query["question"],
                case_id,
                top_k=args.top_k,
            )
            try:
                if args.answer_mode == "metadata":
                    actual = answer_from_metadata(query, memories)
                else:
                    actual = answer_from_memories(
                        answer_client,
                        model,
                        query,
                        memories,
                    )
            except Exception as exc:
                actual = f"ERROR: {exc}"
            actual_normalized = normalize_value(query["fact_key"], actual)
            expected_normalized = normalize_value(
                query["fact_key"], query["answer"]
            )
            correct = (
                str(actual_normalized).strip()
                == str(expected_normalized).strip()
            )
            type_total[query["query_type"]] += 1
            type_correct[query["query_type"]] += int(correct)
            details.append(
                {
                    "case_id": case_id,
                    "query_id": query["query_id"],
                    "query_type": query["query_type"],
                    "question": query["question"],
                    "answer": query["answer"],
                    "actual": actual,
                    "actual_normalized": actual_normalized,
                    "correct": correct,
                    "retrieved": memories,
                }
            )

    result = {
        "run_id": run_id,
        "collection_name": collection_name,
        "infer": args.infer,
        "top_k": args.top_k,
        "answer_mode": args.answer_mode,
        "case_count": len(cases),
        "overall_accuracy": (
            sum(item["correct"] for item in details) / len(details)
            if details
            else 0.0
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
        "add_log": add_log,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "overall_accuracy": result["overall_accuracy"],
                "by_query_type": result["by_query_type"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
