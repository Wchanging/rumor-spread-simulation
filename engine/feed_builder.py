from __future__ import annotations

import math
import random

from domain.content import ContentItem


class FeedBuilder:
    def __init__(
        self,
        window: int = 2,
        seen_decay_tau: float = 5.0,
        seen_min_weight: float = 0.05,
        seen_max_weight: float = 0.85,
        unseen_weight: float = 1.0,
        enable_empty_feed_fallback: bool = True,
        empty_feed_fallback_allow_global: bool = True,
        enable_rumor_candidate_fallback: bool = False,
        rumor_candidate_fallback_count: int = 2,
        enable_event_allocator: bool = False,
        event_allocator_temperature: float = 0.9,
        event_allocator_pool_multiplier: float = 2.0,
        event_allocator_max_events: int = 6,
        event_allocator_social_weight: float = 0.7,
        event_allocator_popularity_weight: float = 1.0,
        event_allocator_novelty_weight: float = 0.5,
        event_allocator_fatigue_weight: float = 0.3,
        event_allocator_early_diversity_rounds: int = 0,
        event_allocator_early_event_cap: int = 1,
    ) -> None:
        self.window = window
        self.seen_decay_tau = max(1e-6, float(seen_decay_tau))
        self.seen_min_weight = max(0.0, float(seen_min_weight))
        self.seen_max_weight = max(self.seen_min_weight, float(seen_max_weight))
        self.unseen_weight = max(self.seen_max_weight, float(unseen_weight))
        self.enable_empty_feed_fallback = bool(enable_empty_feed_fallback)
        self.empty_feed_fallback_allow_global = bool(empty_feed_fallback_allow_global)
        self.enable_rumor_candidate_fallback = bool(enable_rumor_candidate_fallback)
        self.rumor_candidate_fallback_count = max(0, int(rumor_candidate_fallback_count))
        self.enable_event_allocator = bool(enable_event_allocator)
        self.event_allocator_temperature = max(1e-3, float(event_allocator_temperature))
        self.event_allocator_pool_multiplier = max(1.0, float(event_allocator_pool_multiplier))
        self.event_allocator_max_events = max(1, int(event_allocator_max_events))
        self.event_allocator_social_weight = float(event_allocator_social_weight)
        self.event_allocator_popularity_weight = float(event_allocator_popularity_weight)
        self.event_allocator_novelty_weight = float(event_allocator_novelty_weight)
        self.event_allocator_fatigue_weight = float(event_allocator_fatigue_weight)
        self.event_allocator_early_diversity_rounds = max(0, int(event_allocator_early_diversity_rounds))
        self.event_allocator_early_event_cap = max(1, int(event_allocator_early_event_cap))

    @staticmethod
    def _has_rumor_candidate(candidates: dict[str, ContentItem], sim_state) -> bool:
        for item in candidates.values():
            event = sim_state.events.get(str(item.event_id))
            if bool(getattr(event, "is_fake", False)) or bool(getattr(item, "is_rumor", False)):
                return True
        return False

    def _add_rumor_candidates_if_missing(self, candidates: dict[str, ContentItem], sim_state) -> None:
        if not self.enable_rumor_candidate_fallback or self.rumor_candidate_fallback_count <= 0:
            return
        if self._has_rumor_candidate(candidates, sim_state):
            return

        rumor_pool = []
        for item in sim_state.content_pool.values():
            if item.content_id in candidates:
                continue
            event = sim_state.events.get(str(item.event_id))
            is_fake_event = bool(getattr(event, "is_fake", False))
            if is_fake_event or bool(getattr(item, "is_rumor", False)):
                rumor_pool.append(item)

        rumor_pool = sorted(
            rumor_pool,
            key=lambda item: (
                int(sim_state.content_added_timestep.get(item.content_id, 0)),
                self._popularity_score(item),
            ),
            reverse=True,
        )

        for item in rumor_pool[: self.rumor_candidate_fallback_count]:
            candidates[item.content_id] = item

    @staticmethod
    def _popularity_score(item: ContentItem) -> float:
        return math.log1p(max(0.0, float(item.popularity)))

    def _item_priority_score(self, user_state, item: ContentItem, sim_state) -> float:
        seen_weight = user_state.get_seen_weight(
            event_id=item.event_id,
            content_id=item.content_id,
            current_timestep=sim_state.timestep,
            seen_decay_tau=self.seen_decay_tau,
            seen_min_weight=self.seen_min_weight,
            seen_max_weight=self.seen_max_weight,
            unseen_weight=self.unseen_weight,
        )
        added_timestep = int(sim_state.content_added_timestep.get(item.content_id, 0))
        return float(seen_weight) + 0.15 * float(added_timestep) + 0.4 * self._popularity_score(item)

    def _event_utility(
        self,
        event_id: str,
        items: list[ContentItem],
        user_state,
        sim_state,
        inbox_event_counts: dict[str, int],
    ) -> float:
        if not items:
            return -1e9

        top_item = items[0]
        top_item_priority = self._item_priority_score(user_state, top_item, sim_state)
        popularity_signal = self._popularity_score(top_item)
        social_signal = float(inbox_event_counts.get(event_id, 0))
        seen_count = float(user_state.get_belief(event_id).seen_count)
        novelty_signal = 1.0 / (1.0 + seen_count)
        fatigue_signal = math.log1p(max(0.0, seen_count))

        return (
            self.event_allocator_popularity_weight * (0.6 * top_item_priority + 0.4 * popularity_signal)
            + self.event_allocator_social_weight * social_signal
            + self.event_allocator_novelty_weight * novelty_signal
            - self.event_allocator_fatigue_weight * fatigue_signal
        )

    def _sample_event_softmax(self, rng: random.Random, event_ids: list[str], utilities: dict[str, float]) -> str:
        if not event_ids:
            return ""
        if len(event_ids) == 1:
            return event_ids[0]

        max_u = max(utilities[event_id] for event_id in event_ids)
        weights: list[tuple[str, float]] = []
        total = 0.0
        for event_id in event_ids:
            scaled = (utilities[event_id] - max_u) / self.event_allocator_temperature
            weight = math.exp(max(-60.0, min(60.0, scaled)))
            weights.append((event_id, weight))
            total += weight

        if total <= 0:
            return rng.choice(event_ids)

        threshold = rng.random() * total
        cumulative = 0.0
        for event_id, weight in weights:
            cumulative += weight
            if cumulative >= threshold:
                return event_id
        return weights[-1][0]

    def _apply_event_allocator(
        self,
        user_id: str,
        candidates: dict[str, ContentItem],
        user_state,
        sim_state,
        max_feed_size: int,
        inbox_messages: list[ContentItem],
    ) -> dict[str, ContentItem]:
        if not self.enable_event_allocator or len(candidates) <= max_feed_size:
            return candidates

        grouped: dict[str, list[ContentItem]] = {}
        for item in candidates.values():
            grouped.setdefault(str(item.event_id), []).append(item)

        for event_id, items in grouped.items():
            grouped[event_id] = sorted(
                items,
                key=lambda item: self._item_priority_score(user_state, item, sim_state),
                reverse=True,
            )

        inbox_event_counts: dict[str, int] = {}
        for item in inbox_messages:
            event_id = str(item.event_id)
            inbox_event_counts[event_id] = inbox_event_counts.get(event_id, 0) + 1

        event_ids = list(grouped.keys())
        event_ids = sorted(
            event_ids,
            key=lambda eid: self._event_utility(eid, grouped[eid], user_state, sim_state, inbox_event_counts),
            reverse=True,
        )
        event_ids = event_ids[: self.event_allocator_max_events]
        grouped = {eid: grouped[eid] for eid in event_ids if grouped.get(eid)}

        target_pool_size = max(max_feed_size, int(round(max_feed_size * self.event_allocator_pool_multiplier)))
        rng = random.Random(f"{user_id}:{int(sim_state.timestep)}:event_allocator")
        selected: dict[str, ContentItem] = {}
        selected_event_counts: dict[str, int] = {}
        use_early_cap = (
            self.event_allocator_early_diversity_rounds > 0
            and int(sim_state.timestep) < self.event_allocator_early_diversity_rounds
        )
        cap_enforced = bool(use_early_cap)

        while len(selected) < target_pool_size and grouped:
            if cap_enforced:
                active_events = [
                    eid
                    for eid, items in grouped.items()
                    if items and selected_event_counts.get(eid, 0) < self.event_allocator_early_event_cap
                ]
            else:
                active_events = [eid for eid, items in grouped.items() if items]
            if not active_events:
                if cap_enforced:
                    cap_enforced = False
                    continue
                break

            utilities = {
                eid: self._event_utility(eid, grouped[eid], user_state, sim_state, inbox_event_counts)
                for eid in active_events
            }
            chosen_event = self._sample_event_softmax(rng, active_events, utilities)
            if not chosen_event:
                break

            next_item = grouped[chosen_event].pop(0)
            selected[next_item.content_id] = next_item
            selected_event_counts[chosen_event] = selected_event_counts.get(chosen_event, 0) + 1
            if not grouped[chosen_event]:
                grouped.pop(chosen_event, None)

        if not selected:
            return candidates
        return selected

    def build_feed(self, user_id: str, sim_state, dispatcher, max_feed_size: int = 20) -> list[ContentItem]:
        inbox_messages = dispatcher.get_messages_for(user_id, clear=True)
        neighbors = set(sim_state.network.neighbors(user_id))
        recent = sim_state.get_recent_contents(current_timestep=sim_state.timestep, window=self.window)

        candidates: dict[str, ContentItem] = {msg.content_id: msg for msg in inbox_messages}
        for item in recent:
            if item.author_id in neighbors or item.author_id == "platform":
                candidates[item.content_id] = item

        if not candidates and self.enable_empty_feed_fallback:
            for item in sim_state.content_pool.values():
                if item.author_id in neighbors or item.author_id == "platform":
                    candidates[item.content_id] = item

            if not candidates and self.empty_feed_fallback_allow_global:
                for item in sim_state.content_pool.values():
                    candidates[item.content_id] = item

        self._add_rumor_candidates_if_missing(candidates=candidates, sim_state=sim_state)

        user_state = sim_state.users[user_id]
        candidates = self._apply_event_allocator(
            user_id=user_id,
            candidates=candidates,
            user_state=user_state,
            sim_state=sim_state,
            max_feed_size=max_feed_size,
            inbox_messages=inbox_messages,
        )
        ranked = sorted(
            candidates.values(),
            key=lambda x: (
                user_state.get_seen_weight(
                    event_id=x.event_id,
                    content_id=x.content_id,
                    current_timestep=sim_state.timestep,
                    seen_decay_tau=self.seen_decay_tau,
                    seen_min_weight=self.seen_min_weight,
                    seen_max_weight=self.seen_max_weight,
                    unseen_weight=self.unseen_weight,
                ),
                int(sim_state.content_added_timestep.get(x.content_id, 0)),
                self._popularity_score(x),
            ),
            reverse=True,
        )
        return ranked[: max_feed_size]
