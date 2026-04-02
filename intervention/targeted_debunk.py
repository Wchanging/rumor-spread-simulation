from __future__ import annotations

from collections import defaultdict

from domain.content import DebunkPost
from llm.base import LLMClient
from llm.tone_generator import DebunkToneGenerator
from network.builder import indegree_map

from .base import InterventionStrategy


def _normalize_event_selection_mode(mode: str | None) -> str:
    value = str(mode or "risk_weighted").strip().lower()
    if value == "hotspot_weighted":
        return "risk_weighted"
    if value in {"round_robin", "risk_weighted"}:
        return value
    return "risk_weighted"


def _recent_fake_share_counts(sim_state, fake_events, recent_action_window: int) -> dict[str, float]:
    fake_event_ids = {str(event.event_id) for event in fake_events}
    now = int(getattr(sim_state, "timestep", 0))
    start_t = max(0, now - max(1, int(recent_action_window)) + 1)
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


def _risk_weighted_scores(
    sim_state,
    fake_events,
    heat_scores: dict[str, float],
    event_heat_weight: float,
    event_misbelief_weight: float,
    event_coverage_gap_weight: float,
) -> dict[str, float]:
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
            max(0.0, float(event_heat_weight)) * heat_norm
            + max(0.0, float(event_misbelief_weight)) * misbelief_ratio
            + max(0.0, float(event_coverage_gap_weight)) * coverage_gap
        )
        scores[event_id] = 1.0 + max(0.0, float(combined))
    return scores


class TargetTopKSpreaders(InterventionStrategy):
    def __init__(
        self,
        k: int = 20,
        tone_style: str = "assertive",
        start_timestep: int = 0,
        interval: int = 1,
        end_timestep: int | None = None,
        posts_per_round: int = 3,
        max_posts_per_event: int = 1,
        recent_action_window: int = 3,
        min_share_count: int = 1,
        recent_share_weight: float = 1.0,
        total_share_weight: float = 0.4,
        degree_weight: float = 0.2,
        event_selection_mode: str = "risk_weighted",
        event_heat_weight: float = 1.0,
        event_misbelief_weight: float = 1.0,
        event_coverage_gap_weight: float = 0.6,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.k = k
        self.tone_style = tone_style
        self.start_timestep = max(0, int(start_timestep))
        self.interval = max(1, int(interval))
        self.end_timestep = None if end_timestep is None else int(end_timestep)
        self.posts_per_round = max(1, int(posts_per_round))
        self.max_posts_per_event = max(1, int(max_posts_per_event))
        self.recent_action_window = max(1, int(recent_action_window))
        self.min_share_count = max(0, int(min_share_count))
        self.recent_share_weight = max(0.0, float(recent_share_weight))
        self.total_share_weight = max(0.0, float(total_share_weight))
        self.degree_weight = max(0.0, float(degree_weight))
        self.event_selection_mode = _normalize_event_selection_mode(event_selection_mode)
        self.event_heat_weight = max(0.0, float(event_heat_weight))
        self.event_misbelief_weight = max(0.0, float(event_misbelief_weight))
        self.event_coverage_gap_weight = max(0.0, float(event_coverage_gap_weight))
        self._generator = DebunkToneGenerator(llm_client=llm_client)

    def select_targets(self, sim_state) -> list[str]:
        now = int(getattr(sim_state, "timestep", 0))
        start_t = max(0, now - self.recent_action_window + 1)
        recent_share_count = defaultdict(int)
        for action in getattr(sim_state, "action_log", []):
            if str(action.get("action_type", "")) not in {"share", "rewrite_share"}:
                continue
            timestep = int(action.get("timestep", -1))
            if timestep < start_t:
                continue
            user_id = str(action.get("user_id", ""))
            if user_id:
                recent_share_count[user_id] += 1

        in_deg = indegree_map(sim_state.network)
        ranked_users = []
        for user_id in sim_state.users.keys():
            total_count = float(sim_state.user_share_count.get(user_id, 0))
            recent_count = float(recent_share_count.get(user_id, 0))
            score = (
                self.recent_share_weight * recent_count
                + self.total_share_weight * total_count
                + self.degree_weight * float(in_deg.get(user_id, 0))
            )
            if total_count >= float(self.min_share_count):
                ranked_users.append((user_id, score))

        ranked_users.sort(key=lambda x: x[1], reverse=True)
        return [user_id for user_id, score in ranked_users if score > 0][: self.k]

    def generate_interventions(self, targets: list[str], sim_state) -> list[DebunkPost]:
        if not targets:
            return []
        if not self._should_post(int(getattr(sim_state, "timestep", 0))):
            return []

        fake_events = [event for event in sim_state.events.values() if event.is_fake]
        if not fake_events:
            return []

        scheduled_events = self._schedule_events(fake_events=fake_events, sim_state=sim_state)
        posts: list[DebunkPost] = []
        for idx, event in enumerate(scheduled_events):
            text = self._generator.generate(self.tone_style, event.description, event.evidence)
            posts.append(
                DebunkPost(
                    content_id=f"targeted_{event.event_id}_{sim_state.timestep}_{idx}",
                    event_id=event.event_id,
                    author_id="platform",
                    text=text,
                    tone_style=self.tone_style,
                    evidence=event.evidence,
                    timestamp=sim_state.timestep,
                )
            )
        return posts

    def _schedule_events(self, fake_events, sim_state):
        if not fake_events:
            return []

        if self.event_selection_mode == "round_robin":
            return [fake_events[idx % len(fake_events)] for idx in range(self.posts_per_round)]

        score_by_event = _risk_weighted_scores(
            sim_state=sim_state,
            fake_events=fake_events,
            heat_scores=_recent_fake_share_counts(
                sim_state=sim_state,
                fake_events=fake_events,
                recent_action_window=self.recent_action_window,
            ),
            event_heat_weight=self.event_heat_weight,
            event_misbelief_weight=self.event_misbelief_weight,
            event_coverage_gap_weight=self.event_coverage_gap_weight,
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
                    float(score_by_event.get(event.event_id, 0.0))
                    / (1.0 + float(assigned.get(event.event_id, 0))),
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

    def _should_post(self, timestep: int) -> bool:
        if timestep < self.start_timestep:
            return False
        if self.end_timestep is not None and timestep > self.end_timestep:
            return False
        return (timestep - self.start_timestep) % self.interval == 0


class PersonalizedDebunk(InterventionStrategy):
    def __init__(
        self,
        threshold: float = 0.2,
        max_target_belief: float = 0.7,
        tone_style: str = "empathetic",
        high_risk_tone_style: str = "authoritative",
        high_risk_threshold: float = 0.5,
        start_timestep: int = 0,
        interval: int = 1,
        end_timestep: int | None = None,
        max_targets_per_round: int = 40,
        per_user_post_cap: int = 1,
        posts_per_round: int = 40,
        max_posts_per_event: int = 2,
        event_selection_mode: str = "risk_weighted",
        recent_action_window: int = 3,
        event_heat_weight: float = 1.0,
        event_misbelief_weight: float = 1.0,
        event_coverage_gap_weight: float = 0.6,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.threshold = threshold
        self.max_target_belief = float(max_target_belief)
        self.tone_style = tone_style
        self.high_risk_tone_style = str(high_risk_tone_style)
        self.high_risk_threshold = float(high_risk_threshold)
        self.start_timestep = max(0, int(start_timestep))
        self.interval = max(1, int(interval))
        self.end_timestep = None if end_timestep is None else int(end_timestep)
        self.max_targets_per_round = max(1, int(max_targets_per_round))
        self.per_user_post_cap = max(1, int(per_user_post_cap))
        self.posts_per_round = max(1, int(posts_per_round))
        self.max_posts_per_event = max(1, int(max_posts_per_event))
        self.event_selection_mode = _normalize_event_selection_mode(event_selection_mode)
        self.recent_action_window = max(1, int(recent_action_window))
        self.event_heat_weight = max(0.0, float(event_heat_weight))
        self.event_misbelief_weight = max(0.0, float(event_misbelief_weight))
        self.event_coverage_gap_weight = max(0.0, float(event_coverage_gap_weight))
        self._generator = DebunkToneGenerator(llm_client=llm_client)
        self._latest_user_event_plan: dict[str, list[tuple[str, float]]] = {}

    def select_targets(self, sim_state) -> list[str]:
        if not self._should_post(int(getattr(sim_state, "timestep", 0))):
            self._latest_user_event_plan = {}
            return []

        fake_events = [event for event in sim_state.events.values() if event.is_fake]
        fake_event_ids = [str(event.event_id) for event in fake_events]
        if not fake_event_ids:
            self._latest_user_event_plan = {}
            return []

        if self.event_selection_mode == "round_robin":
            event_priority = list(fake_event_ids)
        else:
            score_by_event = _risk_weighted_scores(
                sim_state=sim_state,
                fake_events=fake_events,
                heat_scores=_recent_fake_share_counts(
                    sim_state=sim_state,
                    fake_events=fake_events,
                    recent_action_window=self.recent_action_window,
                ),
                event_heat_weight=self.event_heat_weight,
                event_misbelief_weight=self.event_misbelief_weight,
                event_coverage_gap_weight=self.event_coverage_gap_weight,
            )
            event_priority = sorted(
                fake_event_ids,
                key=lambda event_id: (float(score_by_event.get(event_id, 0.0)), str(event_id)),
                reverse=True,
            )

        priority_rank = {event_id: idx for idx, event_id in enumerate(event_priority)}
        event_post_budget = {event_id: self.max_posts_per_event for event_id in fake_event_ids}
        ranked_users: list[tuple[str, float, list[tuple[str, float]]]] = []
        for user_id, state in sim_state.users.items():
            candidates: list[tuple[str, float]] = []
            for event_id in fake_event_ids:
                score = float(state.get_belief(event_id).belief_score)
                if score >= self.threshold and score <= self.max_target_belief:
                    candidates.append((event_id, score))
            if not candidates:
                continue
            candidates.sort(key=lambda x: (-x[1], priority_rank.get(x[0], 10**9), str(x[0])))
            ranked_users.append((str(user_id), float(candidates[0][1]), candidates))

        ranked_users.sort(key=lambda x: x[1], reverse=True)
        user_event_plan: dict[str, list[tuple[str, float]]] = {}
        planned_targets = 0
        planned_posts = 0
        for user_id, _, candidates in ranked_users:
            if planned_targets >= self.max_targets_per_round:
                break
            selected_events: list[tuple[str, float]] = []
            for event_id, score in candidates:
                if event_post_budget.get(event_id, 0) <= 0:
                    continue
                selected_events.append((event_id, float(score)))
                event_post_budget[event_id] = max(0, int(event_post_budget[event_id]) - 1)
                planned_posts += 1
                if len(selected_events) >= self.per_user_post_cap or planned_posts >= self.posts_per_round:
                    break
            if not selected_events:
                continue
            user_event_plan[user_id] = selected_events
            planned_targets += 1
            if planned_posts >= self.posts_per_round:
                break

        self._latest_user_event_plan = user_event_plan
        return list(self._latest_user_event_plan.keys())

    def generate_interventions(self, targets: list[str], sim_state) -> list[DebunkPost]:
        if not targets:
            return []
        if not self._should_post(int(getattr(sim_state, "timestep", 0))):
            return []

        event_lookup = {str(event.event_id): event for event in sim_state.events.values() if event.is_fake}
        posts: list[DebunkPost] = []
        idx = 0
        for user_id in targets:
            planned_events = self._latest_user_event_plan.get(str(user_id), [])
            for event_id, score in planned_events:
                event = event_lookup.get(str(event_id))
                if event is None:
                    continue
                tone_style = (
                    self.high_risk_tone_style
                    if float(score) >= self.high_risk_threshold
                    else self.tone_style
                )
                text = self._generator.generate(tone_style, event.description, event.evidence)
                posts.append(
                    DebunkPost(
                        content_id=f"personal_{event.event_id}_{sim_state.timestep}_{idx}",
                        event_id=event.event_id,
                        author_id="platform",
                        text=text,
                        tone_style=tone_style,
                        evidence=event.evidence,
                        timestamp=sim_state.timestep,
                        parent_content_id=user_id,
                    )
                )
                idx += 1
                if len(posts) >= self.posts_per_round:
                    return posts
        return posts

    def _should_post(self, timestep: int) -> bool:
        if timestep < self.start_timestep:
            return False
        if self.end_timestep is not None and timestep > self.end_timestep:
            return False
        return (timestep - self.start_timestep) % self.interval == 0
