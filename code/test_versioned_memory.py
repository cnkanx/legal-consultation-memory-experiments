from __future__ import annotations

import unittest

from versioned_memory import VersionedMemory


def event(
    event_id: str,
    fact_key: str,
    value: str,
    event_type: str,
    event_time: str | None,
    sequence_time: str | None = None,
) -> dict:
    return {
        "event_id": event_id,
        "turn": 1,
        "speaker": "client",
        "text": value,
        "fact_key": fact_key,
        "value": value,
        "event_type": event_type,
        "event_time": event_time,
        "recorded_time": sequence_time or event_time or "2026-01-01",
        "source_excerpt": value,
        "time_precision": "exact" if event_time else "unknown",
    }


class VersionedMemoryTests(unittest.TestCase):
    def test_assert_current(self) -> None:
        memory = VersionedMemory("case")
        memory.add_event(event("e1", "amount", "100", "assert", "2025-01-01"))
        result = memory.query(
            {"query_type": "current", "fact_key": "amount"}
        )
        self.assertEqual(result, "100")

    def test_transition_preserves_history(self) -> None:
        memory = VersionedMemory("case")
        memory.add_event(event("e1", "status", "pending", "assert", "2025-01-01"))
        memory.add_event(event("e2", "status", "done", "transition", "2025-02-01"))
        historical = memory.query(
            {
                "query_type": "at_time",
                "fact_key": "status",
                "as_of": "2025-01-15",
            }
        )
        current = memory.query(
            {"query_type": "current", "fact_key": "status"}
        )
        self.assertEqual(historical, "pending")
        self.assertEqual(current, "done")

    def test_correction_preserves_original(self) -> None:
        memory = VersionedMemory("case")
        memory.add_event(event("e1", "date", "2025-01-01", "assert", "2025-01-01"))
        memory.add_event(event("e2", "date", "2025-01-02", "correct", "2025-01-02"))
        current = memory.query({"query_type": "current", "fact_key": "date"})
        original = memory.query({"query_type": "original", "fact_key": "date"})
        self.assertEqual(current, "2025-01-02")
        self.assertEqual(original, "2025-01-01")
        self.assertEqual(memory.versions[1].supersedes, memory.versions[0].record_id)

    def test_uncertain_does_not_overwrite(self) -> None:
        memory = VersionedMemory("case")
        memory.add_event(event("e1", "amount", "100", "assert", "2025-01-01"))
        memory.add_event(event("e2", "amount", "80", "uncertain", None))
        current = memory.query({"query_type": "current", "fact_key": "amount"})
        self.assertEqual(current, "100")
        self.assertEqual(memory.versions[1].status, "pending")

    def test_unknown_time_is_not_used_for_at_time(self) -> None:
        memory = VersionedMemory("case")
        memory.add_event(event("e1", "amount", "100", "assert", None))
        result = memory.query(
            {
                "query_type": "at_time",
                "fact_key": "amount",
                "as_of": "2025-01-01",
            }
        )
        self.assertIsNone(result)

    def test_unrelated_fact_keys_do_not_interfere(self) -> None:
        memory = VersionedMemory("case")
        memory.add_event(event("e1", "amount", "100", "assert", "2025-01-01"))
        memory.add_event(event("e2", "address", "A", "assert", "2025-01-01"))
        amount = memory.query({"query_type": "current", "fact_key": "amount"})
        address = memory.query({"query_type": "current", "fact_key": "address"})
        self.assertEqual(amount, "100")
        self.assertEqual(address, "A")

    def test_missing_fact_returns_none(self) -> None:
        memory = VersionedMemory("case")
        self.assertIsNone(
            memory.query({"query_type": "current", "fact_key": "missing"})
        )


if __name__ == "__main__":
    unittest.main()
