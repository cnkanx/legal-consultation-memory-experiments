from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI

from extraction_utils import parse_json_response


SYSTEM_PROMPT = """你是法律咨询对话中的记忆事件抽取器。

只根据提供的当前用户陈述、说话者、记录时间和之前已出现的陈述抽取一个事件。
不得读取或推断评测答案，不得补充原对话没有明确提供的事实。

返回且只返回一个 JSON 对象，字段如下：
{
  "fact_key": "允许字段之一",
  "value": "标准化后的值",
  "event_type": "assert|transition|correct|uncertain",
  "event_time": "YYYY-MM-DD或null",
  "time_precision": "exact|scheduled|unknown",
  "epistemic_status": "asserted|claimed|uncertain|not_applicable",
  "source_excerpt": "当前陈述中的精确子串"
}

规则：
1. event_type=assert：首次提出事实或增加独立事实。
2. event_type=transition：同一事实随时间真实变化。
3. event_type=correct：明确表示此前说错并纠正。
4. event_type=uncertain：听说、可能、但缺少依据。
5. event_time表示事实实际发生时间；未来已确定日期可用scheduled。
6. 无法确定实际发生时间时，event_time=null且time_precision=unknown。
7. “现在”“目前”“截至本次陈述”等相对时间，应以recorded_time作为事件的当前观察日期。
8. 对没有单独发生时间、但作为当前情况陈述的事实，event_time使用recorded_time，time_precision=exact。
9. recorded_time只用于理解相对日期，不要无条件重复成event_time。
10. “我认为”属于claimed，“普通陈述”属于asserted。
11. 对claimed类型中涉及过去事件、但未说明实际发生时间的陈述，event_time=null且time_precision=unknown。
12. 金额使用阿拉伯数字并去掉“元”“万元”等单位，例如8万元输出80000。
13. 数量、金额和状态使用规范值：未履行、部分履行、已履行。
14. source_excerpt必须逐字来自当前陈述。
"""


def build_user_prompt(
    case: dict,
    event_index: int,
    allowed_fact_keys: list[str],
) -> str:
    event = case["events"][event_index]
    context_events = case["events"][:event_index]
    context = [
        {
            "turn": previous["turn"],
            "speaker": previous["speaker"],
            "text": previous["text"],
            "recorded_time": previous["recorded_time"],
        }
        for previous in context_events
    ]
    payload = {
        "case_id": case["case_id"],
        "allowed_fact_keys": allowed_fact_keys,
        "previous_statements": context,
        "current_statement": {
            "turn": event["turn"],
            "speaker": event["speaker"],
            "text": event["text"],
            "recorded_time": event["recorded_time"],
        },
    }
    return (
        "请抽取下面当前陈述对应的记忆事件。\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def request_structured(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int = 3,
) -> tuple[dict, str]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return parse_json_response(content), content
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"DeepSeek extraction failed after retries: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("J1_EVAL_API_KEY")
    api_base = os.environ.get("J1_EVAL_API_BASE")
    model = os.environ.get("J1_EVAL_MODEL_NAME", "deepseek-chat")
    if not api_key:
        raise SystemExit("Set J1_EVAL_API_KEY before running extraction")
    if not api_base:
        raise SystemExit("Set J1_EVAL_API_BASE before running extraction")

    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if args.case_limit:
        cases = cases[: args.case_limit]
    client = OpenAI(api_key=api_key, base_url=api_base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()

    processed = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                processed.add(json.loads(line)["event_id"])

    completed = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for case in cases:
            allowed_fact_keys = sorted(
                {event["fact_key"] for event in case["events"]}
            )
            for event_index, event in enumerate(case["events"]):
                if event["event_id"] in processed:
                    continue
                prompt = build_user_prompt(case, event_index, allowed_fact_keys)
                extraction, raw_response = request_structured(
                    client, model, prompt
                )
                record = {
                    "case_id": case["case_id"],
                    "event_id": event["event_id"],
                    "turn": event["turn"],
                    "speaker": event["speaker"],
                    "text": event["text"],
                    "recorded_time": event["recorded_time"],
                    "extraction": extraction,
                    "raw_response": raw_response,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                print(
                    f"[{completed}] {case['case_id']} {event['event_id']}",
                    flush=True,
                )
                if args.limit and completed >= args.limit:
                    return


if __name__ == "__main__":
    main()

