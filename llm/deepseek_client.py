from __future__ import annotations

from .base import LLMClient


class DeepSeekClient(LLMClient):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("DeepSeek 客户端尚未接入具体 SDK，请在此方法中完成 API 调用。")
