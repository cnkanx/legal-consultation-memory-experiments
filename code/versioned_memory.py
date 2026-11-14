from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MemoryVersion:
    record_id: str
    case_id: str
    event_id: str
    fact_key: str
    value: str
    event_type: str
    event_time: str
    recorded_time: str
    source_excerpt: str
    sequence: int
    time_precision: str = "exact"
    epistemic_status: str | None = None
    status: str = "active"
    valid_to: str | None = None
    supersedes: str | None = None


class LatestOverwriteMemory:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.values: dict[str, str] = {}

    def add_event(self, event: dict[str, Any]) -> None:
        if event["event_type"] != "uncertain":
            self.values[event["fact_key"]] = event["value"]

    def query(self, query: dict[str, Any]) -> str | None:
        return self.values.get(query["fact_key"])


class AppendOnlyMemory:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.values: dict[str, list[str]] = {}

    def add_event(self, event: dict[str, Any]) -> None:
        if event["event_type"] != "uncertain":
            self.values.setdefault(event["fact_key"], []).append(event["value"])

    def query(self, query: dict[str, Any]) -> str | None:
        values = self.values.get(query["fact_key"], [])
        if not values:
            return None
        if query["query_type"] == "original":
            return values[0]
        if len(values) == 1:
            return values[0]
        return "AMBIGUOUS[" + "|".join(values) + "]"


class VersionedMemory:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.versions: list[MemoryVersion] = []
        self.sequence = 0

    def _active_versions(self, fact_key: str) -> list[MemoryVersion]:
        return [
            version
            for version in self.versions
            if version.fact_key == fact_key and version.status == "active"
        ]

    def add_event(self, event: dict[str, Any]) -> None:
        self.sequence += 1
        event_type = event["event_type"]
        active = self._active_versions(event["fact_key"])
        previous = max(active, key=lambda item: (item.event_time, item.sequence), default=None)

        status = "pending" if event_type == "uncertain" else "active"
        supersedes = previous.record_id if previous and event_type in {"correct", "transition"} else None

        if previous and event_type in {"correct", "transition"}:
            previous.status = "superseded"
            previous.valid_to = event["event_time"]

        record = MemoryVersion(
            record_id=f"{event['event_id']}-V{self.sequence}",
            case_id=self.case_id,
            event_id=event["event_id"],
            fact_key=event["fact_key"],
            value=event["value"],
            event_type=event_type,
            event_time=event["event_time"],
            recorded_time=event["recorded_time"],
            source_excerpt=event["source_excerpt"],
            sequence=self.sequence,
            time_precision=event.get("time_precision", "exact"),
            epistemic_status=event.get("epistemic_status"),
            status=status,
            supersedes=supersedes,
        )
        self.versions.append(record)

    def query(self, query: dict[str, Any]) -> str | None:
        fact_key = query["fact_key"]
        candidates = [
            version
            for version in self.versions
            if version.fact_key == fact_key and version.status != "pending"
        ]
        if not candidates:
            return None

        if query["query_type"] == "current":
            selected = [item for item in candidates if item.status == "active"]
            if not selected:
                return None
            return max(
                selected,
                key=lambda item: (item.event_time or "9999-12-31", item.sequence),
            ).value

        if query["query_type"] == "original":
            return min(candidates, key=lambda item: item.sequence).value

        if query["query_type"] == "at_time":
            as_of = query["as_of"]
            valid = [
                item
                for item in candidates
                if item.event_time is not None
                and item.event_time <= as_of
                and (item.valid_to is None or as_of < item.valid_to)
            ]
            if not valid:
                return None
            return max(valid, key=lambda item: (item.event_time, item.sequence)).value

        raise ValueError(f"Unsupported query_type: {query['query_type']}")


METHODS = {
    "latest_overwrite": LatestOverwriteMemory,
    "append_only": AppendOnlyMemory,
    "versioned": VersionedMemory,
}

