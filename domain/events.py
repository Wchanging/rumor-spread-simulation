from __future__ import annotations

from dataclasses import dataclass, field

from .content import ContentItem


@dataclass
class Event:
    event_id: str
    description: str
    is_fake: bool
    evidence: str = ""
    evidence_posts: list[str] = field(default_factory=list)
    content_items: list[ContentItem] = field(default_factory=list)
    evidence_source_posts: list[ContentItem] = field(default_factory=list)

    def add_content(self, content: ContentItem) -> None:
        self.content_items.append(content)

    def add_evidence_source_post(self, content: ContentItem) -> None:
        self.evidence_source_posts.append(content)

    def rumor_ratio(self) -> float:
        if not self.content_items:
            return 0.0
        rumor_count = sum(1 for item in self.content_items if item.is_rumor)
        return rumor_count / len(self.content_items)
