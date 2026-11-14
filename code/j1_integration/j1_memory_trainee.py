from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from agents.trainee_LC import GPTTrainee_consult
from utils.register import register_class


EXTRACTION_PROMPT = """你是法律咨询会话的“案件事实记忆写入器”。

你的任务：只从客户最新一条发言中提取可独立保存的案件事实，供律师后续查阅。

规则：
1. 只能记录客户明确陈述的事实，不得根据法律知识补充事实。
2. 不记录客户的疑问、寒暄、情绪表达或对律师的要求。
3. fact_key 使用稳定的英文 snake_case。例如：loan_amount、contract_signed_date、
   repayment_status、employment_status、lease_status、rent_amount。
4. 如果新事实是在更新已有事实，优先复用“已有事实键”中的键。
5. event_type 只能是：
   - assert：首次或一般性陈述；
   - correct：客户明确纠正此前说法；
   - transition：同一事实随时间发生了有效变化；
   - uncertain：听说、可能、记不清、没有证据等不确定信息。
6. 不要编造 event_time。客户未说明时间时填 null，time_precision 填 unknown。
7. uncertain 信息必须保留，但不得当作已确认事实。
8. 没有可保存事实时返回空 events。

已有事实键：
{existing_keys}

客户最新发言：
{utterance}

只输出一个 JSON 对象，不要输出 Markdown：
{{
  "events": [
    {{
      "fact_key": "英文事实键",
      "value": "事实值",
      "event_type": "assert|correct|transition|uncertain",
      "event_time": null,
      "time_precision": "exact|scheduled|unknown",
      "epistemic_status": "asserted|claimed|uncertain",
      "source_excerpt": "支持该事实的原文短句"
    }}
  ]
}}
"""


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("memory extractor did not return a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("memory extractor response is not an object")
    return value


@register_class(alias="Agent.Trainee.LC_VersionedMemory")
class VersionedMemoryTrainee(GPTTrainee_consult):
    _shared_audit: dict[str, list[dict[str, Any]]] = {}
    _audit_lock = threading.Lock()

    @staticmethod
    def add_parser_args(parser):
        GPTTrainee_consult.add_parser_args(parser)
        parser.add_argument(
            "--memory_window_turns",
            type=int,
            default=2,
            help="number of recent turns kept verbatim",
        )
        parser.add_argument(
            "--memory_log_path",
            type=str,
            default="",
            help="JSON audit log for memory writes and retrieved context",
        )

    def __init__(self, args=None, trainee_info=None, name="A"):
        super().__init__(args=args, trainee_info=trainee_info, name=name)
        self.memory_window_turns = max(1, int(args.memory_window_turns))
        self.memory_log_path = (
            Path(args.memory_log_path).resolve()
            if args.memory_log_path
            else None
        )
        self.memory_by_case: dict[str, list[dict[str, Any]]] = {}
        self.audit_by_case: dict[str, list[dict[str, Any]]] = {}

    def _records(self, case_id: str) -> list[dict[str, Any]]:
        return self.memory_by_case.setdefault(case_id, [])

    def _audit(self, case_id: str) -> list[dict[str, Any]]:
        return self.audit_by_case.setdefault(case_id, [])

    def _active_keys(self, case_id: str) -> list[str]:
        return sorted(
            {
                record["fact_key"]
                for record in self._records(case_id)
                if record["status"] == "active"
            }
        )

    def _extract_events(
        self,
        case_id: str,
        utterance: str,
        turn: int,
    ) -> list[dict[str, Any]]:
        prompt = EXTRACTION_PROMPT.format(
            existing_keys=", ".join(self._active_keys(case_id)) or "无",
            utterance=utterance,
        )
        response = self.engine.get_response(
            [
                {
                    "role": "system",
                    "content": "你负责维护结构化案件事实，只输出合法 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
            flag=0,
        )
        parsed = parse_json_object(response)
        raw_events = parsed.get("events", [])
        if not isinstance(raw_events, list):
            raise ValueError("events must be a list")

        events = []
        for raw in raw_events:
            if not isinstance(raw, dict):
                continue
            fact_key = str(raw.get("fact_key", "")).strip()
            value = str(raw.get("value", "")).strip()
            event_type = str(raw.get("event_type", "")).strip()
            if not fact_key or not value:
                continue
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", fact_key):
                continue
            if event_type not in {"assert", "correct", "transition", "uncertain"}:
                continue
            events.append(
                {
                    "event_id": f"{case_id}-T{turn:03d}-E{len(events) + 1}",
                    "fact_key": fact_key,
                    "value": value,
                    "event_type": event_type,
                    "statement_event_time": raw.get("event_time"),
                    "time_precision": str(
                        raw.get("time_precision", "unknown")
                    ).strip(),
                    "epistemic_status": str(
                        raw.get(
                            "epistemic_status",
                            "uncertain"
                            if event_type == "uncertain"
                            else "asserted",
                        )
                    ).strip(),
                    "source_excerpt": str(
                        raw.get("source_excerpt", utterance)
                    ).strip(),
                }
            )
        return events

    def _write_records(
        self,
        case_id: str,
        events: list[dict[str, Any]],
        turn: int,
    ) -> None:
        records = self._records(case_id)
        for event in events:
            previous = [
                record
                for record in records
                if record["fact_key"] == event["fact_key"]
                and record["status"] == "active"
            ]
            status = "pending" if event["event_type"] == "uncertain" else "active"
            supersedes = None
            if previous and event["event_type"] in {"correct", "transition"}:
                latest_previous = max(previous, key=lambda item: item["sequence"])
                latest_previous["status"] = "superseded"
                latest_previous["superseded_at_turn"] = turn
                supersedes = latest_previous["event_id"]

            records.append(
                {
                    **event,
                    "case_id": case_id,
                    "turn": turn,
                    "sequence": len(records) + 1,
                    "status": status,
                    "supersedes": supersedes,
                }
            )

    def _memory_context(self, case_id: str) -> str:
        records = self._records(case_id)
        active: dict[str, dict[str, Any]] = {}
        latest_active_sequence: dict[str, int] = {}
        for record in records:
            if record["status"] == "active":
                active[record["fact_key"]] = record
                latest_active_sequence[record["fact_key"]] = record["sequence"]

        pending = [
            record
            for record in records
            if record["status"] == "pending"
            and record["sequence"] > latest_active_sequence.get(record["fact_key"], 0)
        ]

        lines = [
            "【版本化案件记忆】",
            "以下信息来自客户此前发言。律师应区分已确认事实与待核实信息。",
        ]
        if active:
            lines.append("当前有效事实：")
            for fact_key, record in sorted(active.items()):
                lines.append(
                    f"- {fact_key} = {record['value']} "
                    f"（第{record['turn']}轮，{record['event_type']}；"
                    f"原文：{record['source_excerpt'][:120]}）"
                )
        else:
            lines.append("当前有效事实：暂无。")

        if pending:
            lines.append("待核实信息（不得当作已确认事实）：")
            for record in pending:
                lines.append(
                    f"- {record['fact_key']} 的可能值：{record['value']} "
                    f"（第{record['turn']}轮；原文："
                    f"{record['source_excerpt'][:120]}）"
                )
        return "\n".join(lines)

    def _persist_audit(self) -> None:
        if self.memory_log_path is None:
            return
        with self._audit_lock:
            self._shared_audit.update(self.audit_by_case)
            self.memory_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.memory_log_path.write_text(
                json.dumps(
                    self._shared_audit,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def speak(self, content, case_id, save_to_memory=True):
        memories = self.memories[case_id]
        turn = sum(1 for role, _ in memories if role == "user") + 1

        extraction_error = None
        events: list[dict[str, Any]] = []
        try:
            events = self._extract_events(case_id, content, turn)
            self._write_records(case_id, events, turn)
        except Exception as exc:
            extraction_error = f"{type(exc).__name__}: {exc}"

        memory_context = self._memory_context(case_id)
        recent = memories[1:][-(self.memory_window_turns * 2) :]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": memory_context},
        ]
        messages.extend(
            {"role": role, "content": value}
            for role, value in recent
        )
        messages.append({"role": "user", "content": content})

        response = self.get_response(messages)
        if save_to_memory:
            self.memorize(("user", content), case_id)
            self.memorize(("assistant", response), case_id)

        self._audit(case_id).append(
            {
                "turn": turn,
                "client_utterance": content,
                "extracted_events": events,
                "extraction_error": extraction_error,
                "memory_context": memory_context,
                "verbatim_turns": self.memory_window_turns,
            }
        )
        self._persist_audit()
        return response
