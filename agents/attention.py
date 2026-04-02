from __future__ import annotations

import math

from domain.content import ContentItem, DebunkPost
from domain.users import UserState


class AttentionModule:
    def __init__(
        self,
        ensure_rumor_in_selection: bool = False,
        min_rumor_items_in_selection: int = 1,
        ensure_non_fake_in_selection: bool = False,
        min_non_fake_items_in_selection: int = 1,
        max_fake_items_in_selection: int = -1,
        rumor_priority_boost: float = 0.0,
        intervention_priority_boost: float = 0.0,
        event_repeat_penalty: float = 0.0,
    ) -> None:
        self.ensure_rumor_in_selection = bool(ensure_rumor_in_selection)
        self.min_rumor_items_in_selection = max(0, int(min_rumor_items_in_selection))
        self.ensure_non_fake_in_selection = bool(ensure_non_fake_in_selection)
        self.min_non_fake_items_in_selection = max(0, int(min_non_fake_items_in_selection))
        self.max_fake_items_in_selection = int(max_fake_items_in_selection)
        self.rumor_priority_boost = float(rumor_priority_boost)
        self.intervention_priority_boost = float(intervention_priority_boost)
        self.event_repeat_penalty = max(0.0, float(event_repeat_penalty))

    def _priority_score(self, item: ContentItem, user_state: UserState) -> float:
        unseen_bonus = 1.0 if not user_state.has_seen(item.event_id, item.content_id) else 0.0

        priority_bonus = 0.0
        if bool(item.is_rumor):
            priority_bonus += self.rumor_priority_boost
        if isinstance(item, DebunkPost):
            priority_bonus += self.intervention_priority_boost

        seen_count = int(user_state.get_belief(item.event_id).seen_count)
        repeat_penalty = self.event_repeat_penalty * math.log1p(max(0, seen_count))
        return unseen_bonus + priority_bonus - repeat_penalty

    @staticmethod
    def _is_fake_event_item(item: ContentItem, fake_event_ids: set[str] | None) -> bool:
        if fake_event_ids is not None:
            return str(item.event_id) in fake_event_ids
        return bool(item.is_rumor) or isinstance(item, DebunkPost)

    def select(
        self,
        feed: list[ContentItem],
        user_state: UserState,
        fake_event_ids: set[str] | None = None,
    ) -> list[ContentItem]:
        if not feed:
            return []

        budget = user_state.attention_budget
        ranked = sorted(
            feed,
            key=lambda item: (
                self._priority_score(item, user_state),
                item.popularity,
                item.timestamp,
            ),
            reverse=True,
        )

        if budget is None:
            return ranked

        final_budget = max(0, int(budget))
        selected = list(ranked[:final_budget])
        selected = self._enforce_rumor_quota(selected=selected, ranked=ranked, budget=final_budget)
        selected = self._enforce_non_fake_quota(
            selected=selected,
            ranked=ranked,
            budget=final_budget,
            fake_event_ids=fake_event_ids,
        )
        selected = self._enforce_fake_max(
            selected=selected,
            ranked=ranked,
            budget=final_budget,
            fake_event_ids=fake_event_ids,
        )
        return selected

    def _enforce_rumor_quota(
        self,
        selected: list[ContentItem],
        ranked: list[ContentItem],
        budget: int,
    ) -> list[ContentItem]:
        if not self.ensure_rumor_in_selection:
            return selected
        if self.min_rumor_items_in_selection <= 0 or budget <= 0:
            return selected

        rumor_candidates = [item for item in ranked if bool(item.is_rumor)]
        if not rumor_candidates:
            return selected

        target_rumor_count = min(self.min_rumor_items_in_selection, budget, len(rumor_candidates))
        selected_ids = {item.content_id for item in selected}
        rumor_selected_count = sum(1 for item in selected if bool(item.is_rumor))
        if rumor_selected_count >= target_rumor_count:
            return selected

        pending_rumors = [item for item in rumor_candidates if item.content_id not in selected_ids]

        while rumor_selected_count < target_rumor_count and pending_rumors:
            rumor_item = pending_rumors.pop(0)

            if len(selected) < budget:
                selected.append(rumor_item)
                selected_ids.add(rumor_item.content_id)
                rumor_selected_count += 1
                continue

            replace_index = None
            for idx in range(len(selected) - 1, -1, -1):
                if not bool(selected[idx].is_rumor):
                    replace_index = idx
                    break

            if replace_index is None:
                break

            selected_ids.discard(selected[replace_index].content_id)
            selected[replace_index] = rumor_item
            selected_ids.add(rumor_item.content_id)
            rumor_selected_count += 1

        return selected

    def _enforce_non_fake_quota(
        self,
        selected: list[ContentItem],
        ranked: list[ContentItem],
        budget: int,
        fake_event_ids: set[str] | None,
    ) -> list[ContentItem]:
        if not self.ensure_non_fake_in_selection:
            return selected
        if self.min_non_fake_items_in_selection <= 0 or budget <= 0:
            return selected

        non_fake_candidates = [item for item in ranked if not self._is_fake_event_item(item, fake_event_ids)]
        if not non_fake_candidates:
            return selected

        target_non_fake_count = min(self.min_non_fake_items_in_selection, budget, len(non_fake_candidates))
        selected_ids = {item.content_id for item in selected}
        non_fake_selected_count = sum(
            1 for item in selected if not self._is_fake_event_item(item, fake_event_ids)
        )
        if non_fake_selected_count >= target_non_fake_count:
            return selected

        pending_non_fake = [item for item in non_fake_candidates if item.content_id not in selected_ids]

        while non_fake_selected_count < target_non_fake_count and pending_non_fake:
            non_fake_item = pending_non_fake.pop(0)

            if len(selected) < budget:
                selected.append(non_fake_item)
                selected_ids.add(non_fake_item.content_id)
                non_fake_selected_count += 1
                continue

            replace_index = None
            for idx in range(len(selected) - 1, -1, -1):
                if self._is_fake_event_item(selected[idx], fake_event_ids):
                    replace_index = idx
                    break

            if replace_index is None:
                break

            selected_ids.discard(selected[replace_index].content_id)
            selected[replace_index] = non_fake_item
            selected_ids.add(non_fake_item.content_id)
            non_fake_selected_count += 1

        return selected

    def _enforce_fake_max(
        self,
        selected: list[ContentItem],
        ranked: list[ContentItem],
        budget: int,
        fake_event_ids: set[str] | None,
    ) -> list[ContentItem]:
        if self.max_fake_items_in_selection < 0:
            return selected
        if budget <= 0:
            return selected

        fake_limit = min(max(0, self.max_fake_items_in_selection), budget)
        if self.ensure_rumor_in_selection and self.min_rumor_items_in_selection > fake_limit:
            fake_limit = min(budget, self.min_rumor_items_in_selection)

        selected_ids = {item.content_id for item in selected}
        fake_selected_count = sum(1 for item in selected if self._is_fake_event_item(item, fake_event_ids))
        if fake_selected_count <= fake_limit:
            return selected

        pending_non_fake = [
            item
            for item in ranked
            if (not self._is_fake_event_item(item, fake_event_ids)) and item.content_id not in selected_ids
        ]

        while fake_selected_count > fake_limit and pending_non_fake:
            non_fake_item = pending_non_fake.pop(0)
            replace_index = None
            for idx in range(len(selected) - 1, -1, -1):
                if self._is_fake_event_item(selected[idx], fake_event_ids):
                    replace_index = idx
                    break

            if replace_index is None:
                break

            selected_ids.discard(selected[replace_index].content_id)
            selected[replace_index] = non_fake_item
            selected_ids.add(non_fake_item.content_id)
            fake_selected_count -= 1

        return selected
