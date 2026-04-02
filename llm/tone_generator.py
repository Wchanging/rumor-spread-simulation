from __future__ import annotations

from .base import LLMClient


class DebunkToneGenerator:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def generate(self, style: str, event_description: str, evidence: str) -> str:
        if self.llm_client is None:
            return f"[{style}] 关于事件：{event_description}。证据：{evidence}。请理性核验信息来源。"

        prompt = (
            f"请使用{style}语气写一段简短辟谣文案。"
            f"\n事件：{event_description}"
            f"\n证据：{evidence}"
            "\n要求：不攻击用户，强调可核验事实。"
        )
        return self.llm_client.generate(prompt)
