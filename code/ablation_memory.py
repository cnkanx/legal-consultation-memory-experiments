from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AblationVersion:
    record_id: str
    fact_key: str
    value: str
    event_type: str
    event_time: str | None
    recorded_time: str
    sequence: int
    status: str = "active"
    valid_to: str | None = None
    supersedes: str | None = None


class AblationMemory:
    def __init__(
        self,
        *,
        use_event_types: bool = True,
        use_version_chain: bool = True,
        use_query_routing: bool = True,
    ) -> None:
        self.use_event_types = use_event_types
        self.use_version_chain = use_version_chain
        self.use_query_routing = use_query_routing
        self.versions: list[AblationVersion] = []
        self.sequence = 0

    def add_event(self, event: dict[str, Any]) -> None:
        self.sequence += 1
        event_type = event["event_type"] if self.use_event_types else "assert"
        active = [
            item
            for item in self.versions
            if item.fact_key == event["fact_key"] and item.status == "active"
        ]
        previous = max(
            active,
            key=lambda item: (
                item.event_time or "9999-12-31",
                item.sequence,
            ),
            default=None,
        )
        is_update = event_type in {"correct", "transition"} or (
            not self.use_event_types and previous is not None
        )
        status = "pending" if event_type == "uncertain" else "active"
        supersedes = None
        if self.use_version_chain and previous and is_update:
            previous.status = "superseded"
            previous.valid_to = event["event_time"]
            supersedes = previous.record_id

        self.versions.append(
            AblationVersion(
                record_id=f"{event['event_id']}-V{self.sequence}",
                fact_key=event["fact_key"],
                value=event["value"],
                event_type=event_type,
                event_time=event.get("event_time"),
                recorded_time=event["recorded_time"],
                sequence=self.sequence,
                status=status,
                supersedes=supersedes,
            )
        )

    def query(self, query: dict[str, Any]) -> str | None:
        candidates = [
            item
            for item in self.versions
            if item.fact_key == query["fact_key"] and item.status != "pending"
        ]
        if not candidates:
            return None

        if query["query_type"] == "original":
            return min(candidates, key=lambda item: item.sequence).value

        active = [item for item in candidates if item.status == "active"]
        if not active:
            return None
        current = max(
            active,
            key=lambda item: (
                item.event_time or "9999-12-31",
                item.sequence,
            ),
        )

        if query["query_type"] == "current" or not self.use_query_routing:
            return current.value

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
            return max(valid, key=lambda item: item.event_time).value

        raise ValueError(f"Unsupported query_type: {query['query_type']}")

