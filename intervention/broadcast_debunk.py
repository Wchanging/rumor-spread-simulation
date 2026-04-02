from __future__ import annotations

from typing import Any

from domain.content import DebunkPost
from llm.base import LLMClient
from llm.tone_generator import DebunkToneGenerator

from .base import InterventionStrategy


class GlobalBroadcastDebunk(InterventionStrategy):
    def __init__(
        self,
        tone_style: str = "cautious",
        cost_per_post: float = 1.0,
        start_timestep: int = 0,
        interval: int = 1,
        end_timestep: int | None = None,
        posts_per_round: int = 3,
        max_posts_per_event: int = 1,
        event_selection_mode: str = "risk_weighted",
        recent_action_window: int = 3,
        event_heat_weight: float = 1.0,
        event_misbelief_weight: float = 1.0,
        event_coverage_gap_weight: float = 0.6,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.tone_style = tone_style
        self.cost_per_post = cost_per_post
        self.start_timestep = max(0, int(start_timestep))
        self.interval = max(1, int(interval))
        self.end_timestep = None if end_timestep is None else int(end_timestep)
        self.posts_per_round = max(1, int(posts_per_round))
        self.max_posts_per_event = max(1, int(max_posts_per_event))
        self.event_selection_mode = str(event_selection_mode or "risk_weighted").strip().lower()
        self.recent_action_window = max(1, int(recent_action_window))
        self.event_heat_weight = max(0.0, float(event_heat_weight))
        self.event_misbelief_weight = max(0.0, float(event_misbelief_weight))
        self.event_coverage_gap_weight = max(0.0, float(event_coverage_gap_weight))
        self._generator = DebunkToneGenerator(llm_client=llm_client)

    def select_targets(self, sim_state) -> list[str]:
        return list(sim_state.users.keys())

    def generate_interventions(self, targets: list[str], sim_state) -> list[DebunkPost]:
        if not self._should_post(int(getattr(sim_state, "timestep", 0))):
            return []

        fake_events = [event for event in sim_state.events.values() if event.is_fake]
        if not fake_events:
            return []

        scheduled_events = self._schedule_events(fake_events=fake_events, sim_state=sim_state)
        posts: list[DebunkPost] = []
        for idx, event in enumerate(scheduled_events):
            text = self._generator.generate(
                style=self.tone_style,
                event_description=event.description,
                evidence=event.evidence,
            )
            post = DebunkPost(
                content_id=f"debunk_{event.event_id}_{sim_state.timestep}_{idx}",
                event_id=event.event_id,
                author_id="platform",
                text=text,
                tone_style=self.tone_style,
                evidence=event.evidence,
                timestamp=sim_state.timestep,
                popularity=0.0,
            )
            posts.append(post)
        return posts

    def _should_post(self, timestep: int) -> bool:
        if timestep < self.start_timestep:
            return False
        if self.end_timestep is not None and timestep > self.end_timestep:
            return False
        return (timestep - self.start_timestep) % self.interval == 0

    def _schedule_events(self, fake_events, sim_state):
        if self.event_selection_mode == "round_robin":
            return [fake_events[idx % len(fake_events)] for idx in range(self.posts_per_round)]

        score_by_event = self._risk_weighted_scores(
            sim_state=sim_state,
            fake_events=fake_events,
            heat_scores=self._recent_fake_share_counts(sim_state=sim_state, fake_events=fake_events),
        )
        assigned = {event.event_id: 0 for event in fake_events}
        scheduled = []

        while len(scheduled) < self.posts_per_round:
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

        if len(scheduled) < self.posts_per_round:
            idx = 0
            while len(scheduled) < self.posts_per_round:
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
            coverage_gap = max(0.0, 1.0 - exposure_ratio)
            combined = (
                self.event_heat_weight * heat_norm
                + self.event_misbelief_weight * misbelief_ratio
                + self.event_coverage_gap_weight * coverage_gap
            )
            scores[event_id] = 1.0 + max(0.0, float(combined))
        return scores
