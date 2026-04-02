from __future__ import annotations

from typing import Any

from domain.content import DebunkPost
from llm.base import LLMClient
from llm.tone_generator import DebunkToneGenerator
from network.builder import indegree_map

from .base import InterventionStrategy


class OfficialAccountDebunk(InterventionStrategy):
    def __init__(
        self,
        official_account_count: int = 6,
        min_degree_quantile: float = 0.6,
        max_degree_quantile: float = 0.9,
        tone_style: str = "authoritative",
        start_timestep: int = 3,
        interval: int = 3,
        end_timestep: int | None = None,
        posts_per_round: int = 4,
        round_post_schedule: dict[int | str, int] | None = None,
        event_selection_mode: str = "round_robin",
        recent_action_window: int = 3,
        max_posts_per_event: int = 2,
        event_heat_weight: float = 1.0,
        event_misbelief_weight: float = 1.0,
        event_exposure_weight: float = 0.0,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.official_account_count = max(1, int(official_account_count))
        self.min_degree_quantile = max(0.0, min(1.0, float(min_degree_quantile)))
        self.max_degree_quantile = max(self.min_degree_quantile, min(1.0, float(max_degree_quantile)))
        self.tone_style = tone_style
        self.start_timestep = max(0, int(start_timestep))
        self.interval = max(1, int(interval))
        self.end_timestep = None if end_timestep is None else int(end_timestep)
        self.posts_per_round = max(1, int(posts_per_round))
        self.round_post_schedule = self._normalize_round_post_schedule(round_post_schedule)
        self.event_selection_mode = self._normalize_event_selection_mode(event_selection_mode)
        self.recent_action_window = max(1, int(recent_action_window))
        self.max_posts_per_event = max(1, int(max_posts_per_event))
        self.event_heat_weight = max(0.0, float(event_heat_weight))
        self.event_misbelief_weight = max(0.0, float(event_misbelief_weight))
        self.event_exposure_weight = max(0.0, float(event_exposure_weight))
        self._generator = DebunkToneGenerator(llm_client=llm_client)
        self._cached_accounts: list[str] = []
        self._event_pool_cursor: dict[str, int] = {}

    def select_targets(self, sim_state) -> list[str]:
        if self._cached_accounts:
            return list(self._cached_accounts)

        degree_scores = self._compute_degree_scores(sim_state)
        if not degree_scores:
            self._cached_accounts = []
            return []

        ranked = sorted(degree_scores.items(), key=lambda x: x[1], reverse=True)
        degree_values = sorted([score for _, score in ranked])
        low = self._percentile(degree_values, self.min_degree_quantile)
        high = self._percentile(degree_values, self.max_degree_quantile)

        candidates = [user_id for user_id, score in ranked if low <= score <= high]
        if len(candidates) < self.official_account_count:
            candidates = [user_id for user_id, _ in ranked]

        self._cached_accounts = candidates[: self.official_account_count]
        return list(self._cached_accounts)

    def generate_interventions(self, targets: list[str], sim_state) -> list[DebunkPost]:
        if not targets:
            return []
        posts_this_round = self._posts_for_timestep(int(sim_state.timestep))
        if posts_this_round <= 0:
            return []

        fake_events = [event for event in sim_state.events.values() if event.is_fake]
        if not fake_events:
            return []

        scheduled_events = self._schedule_events(
            fake_events=fake_events,
            sim_state=sim_state,
            posts_this_round=posts_this_round,
        )

        posts: list[DebunkPost] = []
        pointer = 0
        for idx, event in enumerate(scheduled_events):
            account = targets[pointer % len(targets)]
            pointer += 1

            evidence_text = event.evidence
            source_post = self._next_source_post(event)
            if source_post is not None:
                source_snippet = (source_post.text or "").strip()[:160]
                if source_snippet:
                    evidence_text = f"{event.evidence}\nSource-pool snippet summary: {source_snippet}"

            text = self._generator.generate(
                style=self.tone_style,
                event_description=event.description,
                evidence=evidence_text,
            )
            posts.append(
                DebunkPost(
                    content_id=f"official_{event.event_id}_{sim_state.timestep}_{idx}",
                    event_id=event.event_id,
                    author_id=account,
                    text=text,
                    tone_style=self.tone_style,
                    evidence=event.evidence,
                    timestamp=sim_state.timestep,
                )
            )
        return posts

    def _schedule_events(self, fake_events, sim_state, posts_this_round: int):
        if not fake_events:
            return []
        posts_this_round = max(0, int(posts_this_round))
        if posts_this_round <= 0:
            return []

        if self.event_selection_mode == "round_robin":
            return [fake_events[idx % len(fake_events)] for idx in range(posts_this_round)]

        score_by_event = self._risk_weighted_scores(
            sim_state=sim_state,
            fake_events=fake_events,
            heat_scores=self._recent_fake_share_counts(sim_state=sim_state, fake_events=fake_events),
        )
        assigned = {event.event_id: 0 for event in fake_events}
        scheduled = []

        while len(scheduled) < posts_this_round:
            candidates = [
                event
                for event in fake_events
                if assigned.get(event.event_id, 0) < self.max_posts_per_event
            ]
            if not candidates:
                break

            chosen = max(
                candidates,
                key=lambda event: (
                    float(score_by_event.get(event.event_id, 0.0)) / (1.0 + float(assigned.get(event.event_id, 0))),
                    -float(assigned.get(event.event_id, 0)),
                    str(event.event_id),
                ),
            )
            scheduled.append(chosen)
            assigned[chosen.event_id] = assigned.get(chosen.event_id, 0) + 1

        if len(scheduled) < posts_this_round:
            idx = 0
            while len(scheduled) < posts_this_round:
                scheduled.append(fake_events[idx % len(fake_events)])
                idx += 1

        return scheduled

    def _recent_fake_share_counts(self, sim_state, fake_events) -> dict[str, float]:
        fake_event_ids = {str(event.event_id) for event in fake_events}
        now = int(getattr(sim_state, "timestep", 0))
        start_t = max(0, now - self.recent_action_window + 1)
        scores = {str(event.event_id): 1.0 for event in fake_events}

        for action in getattr(sim_state, "action_log", []):
            action_type = str(action.get("action_type", ""))
            if action_type not in {"share", "rewrite_share"}:
                continue
            event_id = str(action.get("event_id", ""))
            if event_id not in fake_event_ids:
                continue
            timestep = int(action.get("timestep", -1))
            if timestep < start_t:
                continue
            scores[event_id] = float(scores.get(event_id, 1.0)) + 1.0
        return scores

    def _risk_weighted_scores(self, sim_state, fake_events, heat_scores: dict[str, float]) -> dict[str, float]:
        event_ids = [str(event.event_id) for event in fake_events]
        heat_values = [float(heat_scores.get(event_id, 1.0)) for event_id in event_ids]
        max_heat = max(heat_values) if heat_values else 1.0
        if max_heat <= 0:
            max_heat = 1.0

        event_stats: dict[str, dict[str, float]] = {
            event_id: {"exposed": 0.0, "misbelieving": 0.0} for event_id in event_ids
        }
        users = getattr(sim_state, "users", {})
        for user_state in users.values():
            for event_id in event_ids:
                belief = user_state.get_belief(event_id)
                if int(belief.seen_count) <= 0:
                    continue
                event_stats[event_id]["exposed"] += 1.0
                if float(belief.belief_score) > 0.0:
                    event_stats[event_id]["misbelieving"] += 1.0

        total_users = float(len(users)) if users else 1.0
        scores: dict[str, float] = {}
        for event_id in event_ids:
            heat_norm = float(heat_scores.get(event_id, 1.0)) / max_heat
            exposed = float(event_stats[event_id]["exposed"])
            misbelieving = float(event_stats[event_id]["misbelieving"])
            misbelief_ratio = (misbelieving / exposed) if exposed > 0 else 0.0
            exposure_ratio = exposed / total_users
            combined = (
                self.event_heat_weight * heat_norm
                + self.event_misbelief_weight * misbelief_ratio
                + self.event_exposure_weight * exposure_ratio
            )
            scores[event_id] = 1.0 + max(0.0, float(combined))
        return scores

    @staticmethod
    def _normalize_event_selection_mode(mode: str | None) -> str:
        value = str(mode or "round_robin").strip().lower()
        if value == "hotspot_weighted":
            return "risk_weighted"
        if value in {"round_robin", "risk_weighted"}:
            return value
        return "round_robin"

    def _next_source_post(self, event):
        pool = list(getattr(event, "evidence_source_posts", []) or [])
        if not pool:
            return None
        cursor = self._event_pool_cursor.get(event.event_id, 0)
        item = pool[cursor % len(pool)]
        self._event_pool_cursor[event.event_id] = cursor + 1
        return item

    def _should_post(self, timestep: int) -> bool:
        if timestep < self.start_timestep:
            return False
        if self.end_timestep is not None and timestep > self.end_timestep:
            return False
        return (timestep - self.start_timestep) % self.interval == 0

    def _posts_for_timestep(self, timestep: int) -> int:
        if self.round_post_schedule:
            return int(self.round_post_schedule.get(int(timestep), 0))
        return self.posts_per_round if self._should_post(timestep) else 0

    @staticmethod
    def _normalize_round_post_schedule(raw_schedule: dict[int | str, int] | None) -> dict[int, int]:
        if not isinstance(raw_schedule, dict):
            return {}
        normalized: dict[int, int] = {}
        for key, value in raw_schedule.items():
            try:
                t = int(key)
                posts = int(value)
            except (TypeError, ValueError):
                continue
            if t < 0 or posts <= 0:
                continue
            normalized[t] = posts
        return normalized

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        idx = q * (len(values) - 1)
        low = int(idx)
        high = min(len(values) - 1, low + 1)
        weight = idx - low
        return values[low] * (1.0 - weight) + values[high] * weight

    @staticmethod
    def _compute_degree_scores(sim_state) -> dict[str, float]:
        out_degree = {user_id: len(sim_state.network.neighbors(user_id)) for user_id in sim_state.users.keys()}
        in_degree = indegree_map(sim_state.network)
        score: dict[str, float] = {}
        for user_id in sim_state.users.keys():
            score[user_id] = float(out_degree.get(user_id, 0) + in_degree.get(user_id, 0))
        return score
