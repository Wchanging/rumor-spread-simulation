from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContentItem:
    content_id: str
    event_id: str
    author_id: str
    text: str
    images: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    timestamp: int = 0
    is_rumor: bool = False
    popularity: float = 0.0
    parent_content_id: str | None = None


@dataclass
class RumorPost(ContentItem):
    is_rumor: bool = True


@dataclass
class NormalPost(ContentItem):
    is_rumor: bool = False


@dataclass
class DebunkPost(ContentItem):
    tone_style: str = "cautious"
    evidence: str = ""
    is_rumor: bool = False
