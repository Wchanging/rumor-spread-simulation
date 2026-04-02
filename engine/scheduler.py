from __future__ import annotations

import math
import random

from domain.users import UserState


class Scheduler:
    def __init__(
        self,
        min_active_ratio: float = 0.0,
        target_active_ratio: float | None = None,
        max_active_ratio: float = 1.0,
        inactivity_boost_per_round: float = 0.02,
        inactivity_boost_cap: float = 0.25,
        seed: int | None = None,
    ) -> None:
        self.min_active_ratio = max(0.0, min(1.0, float(min_active_ratio)))
        self.max_active_ratio = max(self.min_active_ratio, min(1.0, float(max_active_ratio)))
        self.target_active_ratio = (
            None
            if target_active_ratio is None
            else max(self.min_active_ratio, min(self.max_active_ratio, float(target_active_ratio)))
        )
        self.inactivity_boost_per_round = max(0.0, float(inactivity_boost_per_round))
        self.inactivity_boost_cap = max(0.0, float(inactivity_boost_cap))
        self._rng = random.Random(seed)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def pick_active_users(self, users: dict[str, UserState], timestep: int = 0) -> list[str]:
        if not users:
            return []

        n_users = len(users)
        min_k = max(0, math.ceil(self.min_active_ratio * n_users))
        max_k = max(min_k, math.floor(self.max_active_ratio * n_users))
        target_k = None
        if self.target_active_ratio is not None:
            target_k = max(min_k, min(max_k, round(self.target_active_ratio * n_users)))

        sampled: list[tuple[str, float]] = []
        for user_id, user_state in users.items():
            last_active = int(getattr(user_state, "last_active_timestep", -1))
            inactive_rounds = (timestep - last_active - 1) if last_active >= 0 else (timestep + 1)
            boost = min(self.inactivity_boost_cap, self.inactivity_boost_per_round * max(0, inactive_rounds))
            activation_prob = self._clamp(float(user_state.online_probability) + boost, 0.0, 1.0)
            score = activation_prob + self._rng.random() * 0.1
            if self._rng.random() <= activation_prob:
                sampled.append((user_id, score))

        sampled.sort(key=lambda x: x[1], reverse=True)
        active_ids = [user_id for user_id, _ in sampled]

        if len(active_ids) > max_k:
            active_ids = active_ids[:max_k]

        if target_k is not None and len(active_ids) > target_k:
            active_ids = active_ids[:target_k]

        if len(active_ids) < min_k:
            missing = min_k - len(active_ids)
            rest_candidates = [
                (user_id, users[user_id])
                for user_id in users.keys()
                if user_id not in set(active_ids)
            ]
            rest_candidates.sort(
                key=lambda item: (timestep - int(getattr(item[1], "last_active_timestep", -1))),
                reverse=True,
            )
            for user_id, _ in rest_candidates[:missing]:
                active_ids.append(user_id)

        active_set = set(active_ids)
        for user_id, user_state in users.items():
            if user_id in active_set:
                user_state.last_active_timestep = timestep
                user_state.activation_count += 1

        return active_ids
