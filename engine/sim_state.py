from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from domain.content import ContentItem
from domain.events import Event
from domain.users import UserState
from network.builder import Network


@dataclass
class SimulationState:
    network: Network
    users: dict[str, UserState]
    events: dict[str, Event]
    timestep: int = 0
    content_pool: dict[str, ContentItem] = field(default_factory=dict)
    content_added_timestep: dict[str, int] = field(default_factory=dict)
    timeline: dict[int, list[str]] = field(default_factory=lambda: defaultdict(list))
    user_share_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    user_like_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    action_log: list[dict] = field(default_factory=list)
    intervention_count: int = 0
    intervention_cost: float = 0.0

    def add_content(
        self,
        content: ContentItem,
        timestep: int | None = None,
        is_intervention: bool = False,
        cost: float = 0.0,
    ) -> None:
        ts = self.timestep if timestep is None else timestep
        self.content_pool[content.content_id] = content
        self.content_added_timestep[content.content_id] = int(ts)
        self.timeline[ts].append(content.content_id)
        if content.event_id in self.events:
            self.events[content.event_id].add_content(content)
        if is_intervention:
            self.intervention_count += 1
            self.intervention_cost += float(cost)

    def get_recent_contents(self, current_timestep: int, window: int = 2) -> list[ContentItem]:
        content_items: list[ContentItem] = []
        start = max(0, current_timestep - window)
        for ts in range(start, current_timestep + 1):
            for content_id in self.timeline.get(ts, []):
                item = self.content_pool.get(content_id)
                if item is not None:
                    content_items.append(item)
        return content_items
