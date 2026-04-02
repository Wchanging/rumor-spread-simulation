from __future__ import annotations

from .base import LLMClient


class DeepSeekClient(LLMClient):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("DeepSeek client is not wired to a concrete SDK yet. Implement API calls in this method.")
