from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path


PROJECT_DIR = Path(
    r"D:\Projects\J1Bench\experiments\03_controlled_memory\mem0_runtime"
)
MODEL_CACHE = Path(
    r"D:\Projects\J1Bench\experiments\03_controlled_memory\model_cache\fastembed"
)

os.environ.setdefault("MEM0_DIR", str(PROJECT_DIR / "home"))
os.environ.setdefault("MEM0_TELEMETRY", "false")
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(MODEL_CACHE))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from mem0 import Memory


class Mem0Store:
    def __init__(
        self,
        collection_name: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str = "deepseek-chat",
    ) -> None:
        runtime_dir = PROJECT_DIR / collection_name
        runtime_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "llm": {
                "provider": "deepseek",
                "config": {
                    "model": model,
                    "api_key": api_key or "dummy-key-for-infer-false",
                    "deepseek_base_url": api_base
                    or "https://api.deepseek.com",
                    "temperature": 0,
                    "max_tokens": 2048,
                },
            },
            "embedder": {
                "provider": "fastembed",
                "config": {
                    "model": "BAAI/bge-small-zh-v1.5",
                    "embedding_dims": 512,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": collection_name,
                    "embedding_model_dims": 512,
                    "path": str(runtime_dir / "qdrant"),
                    "on_disk": True,
                },
            },
            "history_db_path": str(runtime_dir / "history.db"),
        }
        self.memory = Memory.from_config(config)

    def add_event(
        self,
        event: dict,
        *,
        infer: bool,
    ) -> dict:
        content = (
            f"事实类别：{event['fact_key']}\n"
            f"值：{event['value']}\n"
            f"事件类型：{event['event_type']}\n"
            f"事件时间：{event.get('event_time')}\n"
            f"记录时间：{event['recorded_time']}\n"
            f"原文：{event['text']}"
        )
        metadata = {
            "case_id": event["case_id"],
            "event_id": event["event_id"],
            "fact_key": event["fact_key"],
            "value": event["value"],
            "event_type": event["event_type"],
            "event_time": event.get("event_time"),
            "recorded_time": event["recorded_time"],
            "source_excerpt": event.get("source_excerpt"),
        }
        return self.memory.add(
            [{"role": "user", "content": content}],
            user_id=event["case_id"],
            metadata=metadata,
            infer=infer,
        )

    def search(
        self,
        query: str,
        case_id: str,
        *,
        top_k: int = 10,
    ) -> list[dict]:
        response = self.memory.search(
            query=query,
            filters={"user_id": case_id},
            top_k=top_k,
        )
        if isinstance(response, dict):
            return response.get("results", [])
        return response or []


def smoke_test() -> None:
    store = Mem0Store(f"smoke_{uuid.uuid4().hex[:8]}")
    event = {
        "case_id": "smoke-case",
        "event_id": "smoke-event-1",
        "fact_key": "remaining_amount",
        "value": "50000",
        "event_type": "assert",
        "event_time": "2025-03-01",
        "recorded_time": "2025-03-01",
        "text": "当前剩余欠款是5万元。",
        "source_excerpt": "当前剩余欠款是5万元",
    }
    added = store.add_event(event, infer=False)
    results = store.search(
        "当前剩余欠款是多少？",
        "smoke-case",
        top_k=5,
    )
    print("ADD RESULT")
    print(json.dumps(added, ensure_ascii=False, indent=2))
    print("SEARCH RESULT")
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
