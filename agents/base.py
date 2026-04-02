from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from domain.content import ContentItem


@dataclass
class Action:
    action_type: str
    actor_id: str
    event_id: str
    content: ContentItem | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    @abstractmethod
    def perceive(self, messages: list[ContentItem], global_state) -> None:
        raise NotImplementedError

    @abstractmethod
    def decide_actions(self) -> list[Action]:
        raise NotImplementedError

    @abstractmethod
    def update_state(self, feedback) -> None:
        raise NotImplementedError
