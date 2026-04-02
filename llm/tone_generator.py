from __future__ import annotations

from .base import LLMClient


class DebunkToneGenerator:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def generate(self, style: str, event_description: str, evidence: str) -> str:
        if self.llm_client is None:
            return f"[{style}] About the event: {event_description}. Evidence: {evidence}. Please verify information sources rationally."

        prompt = (
            f"Write a short debunking post in a {style} tone."
            f"\nEvent: {event_description}"
            f"\nEvidence: {evidence}"
            "\nRequirements: avoid attacking users, and emphasize verifiable facts."
        )
        return self.llm_client.generate(prompt)
