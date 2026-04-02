from __future__ import annotations

import random
from typing import Any

from domain.content import ContentItem
from domain.users import UserState


class SharingPolicy:
    def __init__(
        self,
        belief_threshold: float = 0.15,
        enable_rewrite_share: bool = False,
        rewrite_share_ratio: float = 0.35,
        enable_like: bool = False,
        like_probability_base: float = 0.08,
        like_confidence_bonus: float = 0.25,
        disagree_like_probability: float = 0.02,
        blind_user_mode: bool = True,
        use_truth_label_in_policy: bool = False,
    ) -> None:
        self.belief_threshold = belief_threshold
        self.enable_rewrite_share = bool(enable_rewrite_share)
        self.rewrite_share_ratio = max(0.0, min(1.0, float(rewrite_share_ratio)))
        self.enable_like = bool(enable_like)
        self.like_probability_base = max(0.0, min(1.0, float(like_probability_base)))
        self.like_confidence_bonus = max(0.0, min(1.0, float(like_confidence_bonus)))
        self.disagree_like_probability = max(0.0, min(1.0, float(disagree_like_probability)))
        self.blind_user_mode = bool(blind_user_mode)
        self.use_truth_label_in_policy = bool(use_truth_label_in_policy)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "SharingPolicy":
        cfg = config or {}
        return cls(
            belief_threshold=float(cfg.get("belief_threshold", 0.15)),
            enable_rewrite_share=bool(cfg.get("enable_rewrite_share", False)),
            rewrite_share_ratio=float(cfg.get("rewrite_share_ratio", 0.35)),
            enable_like=bool(cfg.get("enable_like", False)),
            like_probability_base=float(cfg.get("like_probability_base", 0.08)),
            like_confidence_bonus=float(cfg.get("like_confidence_bonus", 0.25)),
            disagree_like_probability=float(cfg.get("disagree_like_probability", 0.02)),
            blind_user_mode=bool(cfg.get("blind_user_mode", True)),
            use_truth_label_in_policy=bool(cfg.get("use_truth_label_in_policy", False)),
        )

    def should_share(self, user_state: UserState, content: ContentItem, belief_score: float) -> bool:
        if self.blind_user_mode and not self.use_truth_label_in_policy:
            if belief_score < self.belief_threshold:
                return False

            confidence = abs(belief_score)
            probability = min(1.0, user_state.share_tendency * (0.5 + confidence))
            return random.random() < probability

        if content.is_rumor and belief_score < self.belief_threshold:
            return False

        if not content.is_rumor and belief_score > -self.belief_threshold:
            return False

        confidence = abs(belief_score)
        probability = min(1.0, user_state.share_tendency * (0.5 + confidence))
        return random.random() < probability

    def choose_share_action_type(self, content: ContentItem, belief_score: float) -> str:
        if not self.enable_rewrite_share:
            return "share"

        confidence = abs(float(belief_score))
        rewrite_prob = min(1.0, self.rewrite_share_ratio * (0.5 + 0.5 * confidence))
        return "rewrite_share" if random.random() < rewrite_prob else "share"

    def should_like(self, content: ContentItem, belief_score: float) -> bool:
        if not self.enable_like:
            return False

        if self.blind_user_mode and not self.use_truth_label_in_policy:
            if belief_score <= 0:
                return random.random() < self.disagree_like_probability
            confidence = abs(float(belief_score))
            probability = min(1.0, self.like_probability_base + self.like_confidence_bonus * confidence)
            return random.random() < probability

        aligned = (content.is_rumor and belief_score > 0) or ((not content.is_rumor) and belief_score < 0)
        if not aligned:
            return random.random() < self.disagree_like_probability

        confidence = abs(float(belief_score))
        probability = min(1.0, self.like_probability_base + self.like_confidence_bonus * confidence)
        return random.random() < probability
