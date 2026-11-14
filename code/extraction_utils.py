from __future__ import annotations

import json
import re
from typing import Any


EVENT_TYPES = {"assert", "transition", "correct", "uncertain"}
TIME_PRECISIONS = {"exact", "scheduled", "unknown"}
EPISTEMIC_STATUSES = {"asserted", "claimed", "uncertain", "not_applicable"}

STATUS_EQUIVALENTS = {
    "lease_status": {
        "生效": "生效",
        "已生效": "生效",
        "解除": "解除",
        "已解除": "解除",
    },
    "employment_status": {
        "在职": "在职",
        "已入职": "在职",
        "离职": "离职",
        "已离职": "离职",
    },
    "payment_status": {
        "全部付款": "全部付款",
        "已全部付款": "全部付款",
        "全部支付": "全部付款",
        "已履行": "全部付款",
        "部分付款": "部分付款",
        "部分支付": "部分付款",
        "部分履行": "部分付款",
        "未付款": "未付款",
        "未支付": "未付款",
        "未履行": "未付款",
    },
    "repayment_status": {
        "已履行": "已履行",
        "全部履行": "已履行",
        "已还清": "已履行",
        "全部还清": "已履行",
        "已经全部还清": "已履行",
        "部分履行": "部分履行",
        "部分还款": "部分履行",
        "还了一部分": "部分履行",
        "未履行": "未履行",
        "未还款": "未履行",
        "没有履行": "未履行",
        "没有还款": "未履行",
    },
    "supplemental_agreement": {
        "未确认存在": "未确认存在",
        "未见过": "未确认存在",
        "没有见过": "未确认存在",
        "可能存在": "未确认存在",
        "可能签过": "未确认存在",
    },
}


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Extractor response is not a JSON object")
    return value


def normalize_nullable(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "unknown", "未知"}:
        return None
    return text


def normalize_value(fact_key: str, value: Any) -> str:
    text = str(value).strip().strip("。；;")
    if not text:
        return text

    chinese_date = re.fullmatch(
        r"(\d{4})年(\d{1,2})月(\d{1,2})日",
        text,
    )
    if chinese_date:
        year, month, day = chinese_date.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    iso_date = re.fullmatch(
        r"(\d{4})-(\d{1,2})-(\d{1,2})",
        text,
    )
    if iso_date:
        year, month, day = iso_date.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    uncertainty_sensitive_status = fact_key in {
        "payment_status",
        "repayment_status",
    }
    if not (
        uncertainty_sensitive_status
        and ("可能" in text or "不确定" in text)
    ):
        status_value = STATUS_EQUIVALENTS.get(fact_key, {}).get(text)
        if status_value is not None:
            return status_value

    money_text = text.replace(",", "").replace("，", "")
    money_text = re.sub(r"^(?:大约|大概|约)", "", money_text)
    money_text = re.sub(r"左右$", "", money_text)
    money_match = re.fullmatch(
        r"([+-]?\d+(?:\.\d+)?)\s*(万元|万|元)?",
        money_text,
    )
    if money_match and (
        "amount" in fact_key
        or "fee" in fact_key
        or "deposit" in fact_key
        or "rent" in fact_key
        or "salary" in fact_key
    ):
        number = float(money_match.group(1))
        if money_match.group(2) in {"万元", "万"}:
            number *= 10_000
        if number.is_integer():
            return str(int(number))
        return f"{number:g}"

    quantity_match = re.fullmatch(
        r"([+-]?\d+(?:\.\d+)?)\s*(?:件|个|套|台|笔)",
        text,
    )
    if quantity_match and (
        "quantity" in fact_key or "count" in fact_key
    ):
        number = float(quantity_match.group(1))
        if number.is_integer():
            return str(int(number))
        return f"{number:g}"

    return text


def build_extracted_event(
    metadata: dict[str, Any],
    extraction: dict[str, Any],
) -> dict[str, Any]:
    fact_key = str(extraction.get("fact_key", "")).strip()
    value = normalize_value(fact_key, extraction.get("value"))
    event_type = str(extraction.get("event_type", "")).strip()
    event_time = normalize_nullable(extraction.get("event_time"))
    time_precision = str(extraction.get("time_precision", "")).strip()
    epistemic_status = str(
        extraction.get("epistemic_status", "not_applicable")
    ).strip()
    source_excerpt = str(extraction.get("source_excerpt", "")).strip()

    if not fact_key:
        raise ValueError("missing fact_key")
    if value is None or str(value).strip() == "":
        raise ValueError("missing value")
    if event_type not in EVENT_TYPES:
        raise ValueError(f"invalid event_type: {event_type}")
    if time_precision not in TIME_PRECISIONS:
        raise ValueError(f"invalid time_precision: {time_precision}")
    if epistemic_status not in EPISTEMIC_STATUSES:
        raise ValueError(f"invalid epistemic_status: {epistemic_status}")
    if event_time is None and time_precision != "unknown":
        raise ValueError("null event_time requires time_precision=unknown")
    if event_time is not None and time_precision == "unknown":
        raise ValueError("non-null event_time cannot use unknown precision")

    return {
        "event_id": metadata["event_id"],
        "turn": metadata["turn"],
        "speaker": metadata["speaker"],
        "text": metadata["text"],
        "fact_key": fact_key,
        "value": str(value).strip(),
        "event_type": event_type,
        "event_time": event_time,
        "recorded_time": metadata["recorded_time"],
        "source_excerpt": source_excerpt,
        "time_precision": time_precision,
        "epistemic_status": epistemic_status,
    }
