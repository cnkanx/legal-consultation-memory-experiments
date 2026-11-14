from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from openai import OpenAI

from extraction_utils import normalize_value, parse_json_response


COMPARED_FIELDS = [
    "fact_key",
    "value",
    "event_type",
    "event_time",
    "time_precision",
    "epistemic_status",
]

ANALYSIS_PROMPT = """你是法律记忆智能体实验的错误分析员。

只能分析所给 evidence。不得虚构案例、数字、原因或论文结论。
需要区分：
- model_extraction_error：模型确实抽错
- value_canonicalization：同义值、金额单位、状态表达不一致
- time_semantics：event_time、recorded_time、valid_from含义混用
- schema_validation：输出结构或字段组合不合法
- epistemic_ambiguity：大约、我记得、听说等语气判断存在争议
- dataset_label_ambiguity：人工标签可能不合理或定义不清
- memory_logic_error：结构化记忆版本选择本身错误
- unknown：证据不足

返回一个JSON对象：
{
  "summary": {
    "total_errors": 0,
    "complete_event_accuracy": 0,
    "end_to_end_accuracy": 0,
    "main_conclusion": "只基于证据的一句总结"
  },
  "taxonomy_counts": {
    "model_extraction_error": 0,
    "value_canonicalization": 0,
    "time_semantics": 0,
    "schema_validation": 0,
    "epistemic_ambiguity": 0,
    "dataset_label_ambiguity": 0,
    "memory_logic_error": 0,
    "unknown": 0
  },
  "errors": [
    {
      "evidence_id": "原样返回",
      "category": "上述类别之一",
      "is_model_error": true,
      "is_dataset_issue": false,
      "is_expected_behavior": false,
      "explanation": "为什么",
      "affected_queries": [],
      "suggested_action": "可执行修改",
      "confidence": 0.0
    }
  ],
  "limitations": ["证据支持的局限"]
}

不要输出Markdown，不要输出JSON之外的文字。
"""


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def normalize_field(field: str, fact_key: str, value: object) -> object:
    if field == "value":
        return normalize_value(fact_key, value)
    if field in {"event_time", "epistemic_status"}:
        if value is None or str(value).strip() in {"", "None", "null"}:
            return "asserted" if field == "epistemic_status" else None
    return value


def build_evidence(
    cases: list[dict],
    extraction_rows: list[dict],
    result: dict,
) -> dict:
    extraction_by_id = {
        row["event_id"]: row["extraction"] for row in extraction_rows
    }
    evidence = []

    for case in cases:
        for event in case["events"]:
            extraction = extraction_by_id.get(event["event_id"])
            if extraction is None:
                evidence.append(
                    {
                        "evidence_id": f"MISSING-{event['event_id']}",
                        "case_id": case["case_id"],
                        "event_id": event["event_id"],
                        "kind": "missing_extraction",
                        "source_text": event["text"],
                    }
                )
                continue

            mismatches = {}
            for field in COMPARED_FIELDS:
                expected = normalize_field(
                    field, event["fact_key"], event.get(field)
                )
                actual = normalize_field(
                    field, event["fact_key"], extraction.get(field)
                )
                if str(expected) != str(actual):
                    mismatches[field] = {
                        "expected": expected,
                        "actual": actual,
                    }
            if mismatches:
                evidence.append(
                    {
                        "evidence_id": f"EVENT-{event['event_id']}",
                        "case_id": case["case_id"],
                        "event_id": event["event_id"],
                        "kind": "event_field_mismatch",
                        "source_text": event["text"],
                        "gold_event": {
                            field: normalize_field(
                                field, event["fact_key"], event.get(field)
                            )
                            for field in COMPARED_FIELDS
                        },
                        "extracted_event": {
                            field: normalize_field(
                                field, event["fact_key"], extraction.get(field)
                            )
                            for field in COMPARED_FIELDS
                        },
                        "mismatches": mismatches,
                    }
                )

    for message in result["extraction"]["errors"]:
        evidence.append(
            {
                "evidence_id": "STRUCT-" + message.split(":")[0],
                "kind": "structure_or_validation_error",
                "message": message,
            }
        )

    for answer in result["end_to_end_answers"]["details"]:
        if not answer["correct"]:
            evidence.append(
                {
                    "evidence_id": f"ANSWER-{answer['query_id']}",
                    "case_id": answer["case_id"],
                    "query_id": answer["query_id"],
                    "kind": "wrong_end_to_end_answer",
                    "query_type": answer["query_type"],
                    "expected": answer["answer"],
                    "actual": answer["predicted"],
                }
            )

    return {
        "run_summary": {
            "case_count": len(cases),
            "event_count": sum(len(case["events"]) for case in cases),
            "query_count": sum(len(case["queries"]) for case in cases),
            "extraction_metrics": result["extraction"]["field_accuracy"],
            "complete_event_accuracy": result["extraction"][
                "complete_event_accuracy"
            ],
            "end_to_end_accuracy": result["end_to_end_answers"][
                "overall_accuracy"
            ],
        },
        "evidence": evidence,
        "evidence_kind_counts": dict(
            Counter(item["kind"] for item in evidence)
        ),
    }


def analyze_with_deepseek(evidence_pack: dict) -> dict:
    api_key = os.environ.get("J1_EVAL_API_KEY")
    api_base = os.environ.get("J1_EVAL_API_BASE")
    model = os.environ.get("J1_EVAL_JUDGE_MODEL", "deepseek-chat")
    if not api_key or not api_base:
        raise SystemExit(
            "Set J1_EVAL_API_KEY and J1_EVAL_API_BASE, or use --evidence-only"
        )
    client = OpenAI(api_key=api_key, base_url=api_base)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANALYSIS_PROMPT},
            {
                "role": "user",
                "content": json.dumps(evidence_pack, ensure_ascii=False, indent=2),
            },
        ],
        temperature=0,
        max_tokens=6000,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    return parse_json_response(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--extractions", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--evidence-output", required=True, type=Path)
    parser.add_argument("--analysis-output", type=Path)
    parser.add_argument("--evidence-only", action="store_true")
    args = parser.parse_args()

    cases = read_jsonl(args.cases)
    extraction_rows = read_jsonl(args.extractions)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    evidence_pack = build_evidence(cases, extraction_rows, result)

    args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_output.write_text(
        json.dumps(evidence_pack, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"evidence: {args.evidence_output}")
    print(
        "evidence items:",
        len(evidence_pack["evidence"]),
        evidence_pack["evidence_kind_counts"],
    )

    if args.evidence_only:
        return
    if args.analysis_output is None:
        raise SystemExit("--analysis-output is required unless --evidence-only")
    analysis = analyze_with_deepseek(evidence_pack)
    args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"analysis: {args.analysis_output}")


if __name__ == "__main__":
    main()
