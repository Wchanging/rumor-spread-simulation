from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class BeliefState:
    belief_score: float = 0.0
    last_updated: int = -1
    seen_count: int = 0


@dataclass
class UserState:
    user_id: str
    gender: Optional[str] = None
    age: Optional[int] = None
    occupation: Optional[str] = None
    education_level: Optional[str] = None
    city_tier: Optional[str] = None
    big5_neuroticism: Optional[str] = None
    big5_extraversion: Optional[str] = None
    big5_openness: Optional[str] = None
    big5_agreeableness: Optional[str] = None
    big5_conscientiousness: Optional[str] = None
    attention_budget: Optional[int] = None
    online_probability: float = 1.0
    last_active_timestep: int = -1
    activation_count: int = 0
    platform_trust: float = 0.5
    trust_threshold: float = 0.5
    share_tendency: float = 0.3
    comment_tendency: float = 0.2
    beliefs: Dict[str, BeliefState] = field(default_factory=dict)
    seen_posts: Dict[str, Set[str]] = field(default_factory=dict)
    seen_timestamps: Dict[str, Dict[str, int]] = field(default_factory=dict)
    event_memories: Dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    long_term_event_memories: Dict[str, str] = field(default_factory=dict)
    believed: Dict[str, bool] = field(default_factory=dict)
    trust_evolution: list[float] = field(default_factory=list)

    def get_belief(self, event_id: str) -> BeliefState:
        if event_id not in self.beliefs:
            self.beliefs[event_id] = BeliefState()
        return self.beliefs[event_id]

    def update_belief(self, event_id: str, score: float, timestep: int) -> None:
        belief = self.get_belief(event_id)
        belief.belief_score = _clamp(score, -1.0, 1.0)
        belief.last_updated = timestep

    def increment_seen(self, event_id: str, content_id: str, timestep: int | None = None) -> None:
        belief = self.get_belief(event_id)
        belief.seen_count += 1
        if event_id not in self.seen_posts:
            self.seen_posts[event_id] = set()
        self.seen_posts[event_id].add(content_id)
        if timestep is not None:
            if event_id not in self.seen_timestamps:
                self.seen_timestamps[event_id] = {}
            self.seen_timestamps[event_id][content_id] = int(timestep)

    def has_seen(self, event_id: str, content_id: str) -> bool:
        return content_id in self.seen_posts.get(event_id, set())

    def get_seen_weight(
        self,
        event_id: str,
        content_id: str,
        current_timestep: int,
        seen_decay_tau: float = 5.0,
        seen_min_weight: float = 0.05,
        seen_max_weight: float = 0.85,
        unseen_weight: float = 1.0,
    ) -> float:
        if not self.has_seen(event_id, content_id):
            return float(unseen_weight)

        last_seen = self.seen_timestamps.get(event_id, {}).get(content_id)
        if last_seen is None:
            return float(seen_min_weight)

        dt = max(0, int(current_timestep) - int(last_seen))
        tau = max(1e-6, float(seen_decay_tau))
        recovery = 1.0 - math.exp(-float(dt) / tau)
        weight = float(seen_min_weight) + (float(seen_max_weight) - float(seen_min_weight)) * recovery
        return _clamp(weight, 0.0, float(unseen_weight))

    def record_belief_binary(self, event_id: str, threshold: float = 0.0) -> None:
        belief = self.get_belief(event_id)
        self.believed[event_id] = belief.belief_score > threshold

    def snapshot_trust(self) -> None:
        self.platform_trust = _clamp(self.platform_trust, 0.0, 1.0)
        self.trust_evolution.append(self.platform_trust)

    def add_event_memory(self, event_id: str, memory: dict[str, Any], max_items: int = 5) -> None:
        if event_id not in self.event_memories:
            self.event_memories[event_id] = []
        self.event_memories[event_id].append(memory)
        keep = max(1, int(max_items))
        if len(self.event_memories[event_id]) > keep:
            self.event_memories[event_id] = self.event_memories[event_id][-keep:]

    def get_long_term_event_memory(self, event_id: str) -> str:
        return str(self.long_term_event_memories.get(event_id, ""))

    def update_long_term_event_memory(
        self,
        event_id: str,
        memory_text: str,
        *,
        max_chars: int = 500,
        replace: bool = True,
    ) -> None:
        incoming = str(memory_text or "").strip()
        if not incoming:
            return

        keep = max(80, int(max_chars))
        if replace:
            merged = incoming
        else:
            previous = str(self.long_term_event_memories.get(event_id, "")).strip()
            merged = incoming if not previous else f"{previous}；{incoming}"

        self.long_term_event_memories[event_id] = merged[-keep:]

    def get_recent_event_memories(self, event_id: str, k: int = 3) -> list[dict[str, Any]]:
        memories = self.event_memories.get(event_id, [])
        keep = max(0, int(k))
        if keep == 0:
            return []
        return list(memories[-keep:])
