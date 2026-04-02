from __future__ import annotations

import hashlib
from typing import Any

from .base import LLMClient


class MockLLMClient(LLMClient):
    def __init__(self, seed: int = 2026) -> None:
        self.seed = seed
        self.call_count = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        digest = hashlib.md5(f"{self.seed}:{prompt}".encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) / 0xFFFFFFFF
        return f"{value:.4f}"

    def get_runtime_stats(self) -> dict[str, Any]:
        return {
            "provider": "mock",
            "total_calls": self.call_count,
            "total_retries": 0,
            "total_failures": 0,
            "adaptive_concurrency_limit": None,
            "max_concurrency": None,
        }
