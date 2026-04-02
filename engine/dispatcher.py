from __future__ import annotations

from collections import defaultdict

from domain.content import ContentItem


class Dispatcher:
    def __init__(self) -> None:
        self._inboxes: dict[str, list[ContentItem]] = defaultdict(list)

    def dispatch(self, contents: list[ContentItem], sim_state) -> None:
        for content in contents:
            author = content.author_id
            neighbors = sim_state.network.neighbors(author)
            for neighbor_id in neighbors:
                self._inboxes[neighbor_id].append(content)

    def dispatch_to_targets(self, contents: list[ContentItem], targets: list[str]) -> None:
        for target_id in targets:
            for content in contents:
                self._inboxes[target_id].append(content)

    def get_messages_for(self, user_id: str, clear: bool = True) -> list[ContentItem]:
        messages = list(self._inboxes.get(user_id, []))
        if clear:
            self._inboxes[user_id] = []
        return messages
