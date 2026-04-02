from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .preprocess import parse_int


@dataclass
class UserProfile:
    user_id: str
    age: int | None = None
    gender: str | None = None
    occupation: str | None = None
    education_level: str | None = None
    city_tier: str | None = None
    big5_neuroticism: str | None = None
    big5_extraversion: str | None = None
    big5_openness: str | None = None
    big5_agreeableness: str | None = None
    big5_conscientiousness: str | None = None
    attention_budget: int | None = None
    online_probability: float = 1.0
    platform_trust: float = 0.5
    trust_threshold: float = 0.5
    share_tendency: float = 0.25


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class UserProfileFactory:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def load_from_csv(self, csv_path: str | Path, defaults: dict[str, Any] | None = None) -> list[UserProfile]:
        defaults = defaults or {}
        path = Path(csv_path)
        users: list[UserProfile] = []
        with path.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.DictReader(fp)
            for idx, row in enumerate(reader):
                user_id = str(row.get("User", "")).strip() or str(row.get("user_id", "")).strip()
                if not user_id:
                    user_id = f"user_{idx}"

                age_raw = row.get("Age", row.get("age"))
                age = parse_int(age_raw, default=-1)
                users.append(
                    UserProfile(
                        user_id=user_id,
                        age=age if age >= 0 else None,
                        gender=(str(row.get("Gender", row.get("gender", ""))).strip() or None),
                        occupation=(str(row.get("Occupation", row.get("occupation", ""))).strip() or None),
                        education_level=(
                            str(row.get("education_level", row.get("EducationLevel", ""))).strip() or None
                        ),
                        city_tier=(str(row.get("city_tier", row.get("CityTier", ""))).strip() or None),
                        big5_neuroticism=(
                            str(row.get("big5_neuroticism", row.get("Big5Neuroticism", ""))).strip() or None
                        ),
                        big5_extraversion=(
                            str(row.get("big5_extraversion", row.get("Big5Extraversion", ""))).strip() or None
                        ),
                        big5_openness=(
                            str(row.get("big5_openness", row.get("Big5Openness", ""))).strip() or None
                        ),
                        big5_agreeableness=(
                            str(row.get("big5_agreeableness", row.get("Big5Agreeableness", ""))).strip() or None
                        ),
                        big5_conscientiousness=(
                            str(row.get("big5_conscientiousness", row.get("Big5Conscientiousness", ""))).strip() or None
                        ),
                        attention_budget=self._read_optional_int(
                            row.get("attention_budget"),
                            defaults.get("attention_budget"),
                        ),
                        online_probability=self._read_float(
                            row.get("online_probability"),
                            defaults.get("online_probability", 1.0),
                        ),
                        platform_trust=self._read_float(
                            row.get("platform_trust"),
                            defaults.get("platform_trust", 0.5),
                        ),
                        trust_threshold=self._read_float(
                            row.get("trust_threshold"),
                            defaults.get("trust_threshold", 0.5),
                        ),
                        share_tendency=self._read_float(
                            row.get("share_tendency"),
                            defaults.get("share_tendency", 0.25),
                        ),
                    )
                )
        return users

    def generate(self, n_users: int, generate_config: dict[str, Any], defaults: dict[str, Any] | None = None) -> list[UserProfile]:
        defaults = defaults or {}
        users: list[UserProfile] = []
        fields_cfg = generate_config.get("fields", {})
        for idx in range(n_users):
            user_id = f"user_{idx}"
            users.append(
                UserProfile(
                    user_id=user_id,
                    age=self._sample_int_field(fields_cfg.get("age"), fallback=defaults.get("age")),
                    gender=self._sample_str_field(fields_cfg.get("gender"), fallback=defaults.get("gender")),
                    occupation=self._sample_str_field(fields_cfg.get("occupation"), fallback=defaults.get("occupation")),
                    education_level=self._sample_str_field(
                        fields_cfg.get("education_level"),
                        fallback=defaults.get("education_level"),
                    ),
                    city_tier=self._sample_str_field(
                        fields_cfg.get("city_tier"),
                        fallback=defaults.get("city_tier"),
                    ),
                    big5_neuroticism=self._sample_str_field(
                        fields_cfg.get("big5_neuroticism"),
                        fallback=defaults.get("big5_neuroticism"),
                    ),
                    big5_extraversion=self._sample_str_field(
                        fields_cfg.get("big5_extraversion"),
                        fallback=defaults.get("big5_extraversion"),
                    ),
                    big5_openness=self._sample_str_field(
                        fields_cfg.get("big5_openness"),
                        fallback=defaults.get("big5_openness"),
                    ),
                    big5_agreeableness=self._sample_str_field(
                        fields_cfg.get("big5_agreeableness"),
                        fallback=defaults.get("big5_agreeableness"),
                    ),
                    big5_conscientiousness=self._sample_str_field(
                        fields_cfg.get("big5_conscientiousness"),
                        fallback=defaults.get("big5_conscientiousness"),
                    ),
                    attention_budget=self._sample_int_field(
                        fields_cfg.get("attention_budget"),
                        fallback=defaults.get("attention_budget"),
                    ),
                    online_probability=_clamp(
                        self._sample_float_field(
                            fields_cfg.get("online_probability"),
                            fallback=float(defaults.get("online_probability", 1.0)),
                        ),
                        0.0,
                        1.0,
                    ),
                    platform_trust=_clamp(
                        self._sample_float_field(
                            fields_cfg.get("platform_trust"),
                            fallback=float(defaults.get("platform_trust", 0.5)),
                        ),
                        0.0,
                        1.0,
                    ),
                    trust_threshold=_clamp(
                        self._sample_float_field(
                            fields_cfg.get("trust_threshold"),
                            fallback=float(defaults.get("trust_threshold", 0.5)),
                        ),
                        0.0,
                        1.0,
                    ),
                    share_tendency=_clamp(
                        self._sample_float_field(
                            fields_cfg.get("share_tendency"),
                            fallback=float(defaults.get("share_tendency", 0.25)),
                        ),
                        0.0,
                        1.0,
                    ),
                )
            )
        return users

    def dump_to_csv(self, users: list[UserProfile], csv_path: str | Path) -> None:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "User",
                    "Age",
                    "Gender",
                    "Occupation",
                    "education_level",
                    "city_tier",
                    "big5_neuroticism",
                    "big5_extraversion",
                    "big5_openness",
                    "big5_agreeableness",
                    "big5_conscientiousness",
                    "attention_budget",
                    "online_probability",
                    "platform_trust",
                    "trust_threshold",
                    "share_tendency",
                ],
            )
            writer.writeheader()
            for user in users:
                writer.writerow(
                    {
                        "User": user.user_id,
                        "Age": "" if user.age is None else user.age,
                        "Gender": user.gender or "",
                        "Occupation": user.occupation or "",
                        "education_level": user.education_level or "",
                        "city_tier": user.city_tier or "",
                        "big5_neuroticism": user.big5_neuroticism or "",
                        "big5_extraversion": user.big5_extraversion or "",
                        "big5_openness": user.big5_openness or "",
                        "big5_agreeableness": user.big5_agreeableness or "",
                        "big5_conscientiousness": user.big5_conscientiousness or "",
                        "attention_budget": "" if user.attention_budget is None else user.attention_budget,
                        "online_probability": user.online_probability,
                        "platform_trust": user.platform_trust,
                        "trust_threshold": user.trust_threshold,
                        "share_tendency": user.share_tendency,
                    }
                )

    def _sample_float_field(self, spec: dict[str, Any] | None, fallback: float) -> float:
        if not spec:
            return float(fallback)
        spec_type = str(spec.get("type", "constant"))

        if spec_type == "constant":
            return float(spec.get("value", fallback))

        if spec_type == "normal":
            mean = float(spec.get("mean", fallback))
            std = float(spec.get("std", 0.1))
            value = self._rng.gauss(mean, std)
            low = float(spec.get("min", value))
            high = float(spec.get("max", value))
            return _clamp(value, low, high)

        if spec_type in {"categorical", "choice"}:
            value = self._sample_categorical(spec)
            return float(value if value is not None else fallback)

        if spec_type in {"bucket_gaussian", "mixture_gaussian"}:
            components = list(spec.get("components", []))
            if not components:
                return float(fallback)
            comp = self._pick_weighted_component(components)
            mean = float(comp.get("mean", fallback))
            std = float(comp.get("std", 0.05))
            value = self._rng.gauss(mean, std)
            low = float(comp.get("min", comp.get("low", 0.0)))
            high = float(comp.get("max", comp.get("high", 1.0)))
            return _clamp(value, low, high)

        return float(fallback)

    def _sample_int_field(self, spec: dict[str, Any] | None, fallback: int | None = None) -> int | None:
        if spec is None:
            return fallback
        value = self._sample_float_field(spec, fallback=float(fallback or 0))
        return int(round(value))

    def _sample_str_field(self, spec: dict[str, Any] | None, fallback: str | None = None) -> str | None:
        if not spec:
            return fallback
        if self._is_polarity_word_spec(spec):
            return self._sample_polarity_word_field(spec, fallback=fallback)
        value = self._sample_categorical(spec)
        if value is None:
            return fallback
        return str(value)

    @staticmethod
    def _is_polarity_word_spec(spec: dict[str, Any]) -> bool:
        spec_type = str(spec.get("type", "")).lower()
        if spec_type in {"polarity_words", "big5_polarity"}:
            return True
        return "high_words" in spec or "low_words" in spec

    def _sample_polarity_word_field(self, spec: dict[str, Any], fallback: str | None = None) -> str | None:
        high_words = [str(word).strip() for word in list(spec.get("high_words", [])) if str(word).strip()]
        low_words = [str(word).strip() for word in list(spec.get("low_words", [])) if str(word).strip()]

        if not high_words and not low_words:
            return fallback

        high_prob = float(spec.get("high_prob", 0.5))
        high_prob = _clamp(high_prob, 0.0, 1.0)
        choose_high = self._rng.random() < high_prob
        candidate_words = high_words if choose_high else low_words
        if not candidate_words:
            candidate_words = low_words if choose_high else high_words

        if not candidate_words:
            return fallback
        return self._rng.choice(candidate_words)

    def _sample_categorical(self, spec: dict[str, Any]) -> Any:
        choices = list(spec.get("choices", []))
        if not choices:
            return spec.get("value")
        pick = self._pick_weighted_component(choices)
        return pick.get("value")

    def _pick_weighted_component(self, components: list[dict[str, Any]]) -> dict[str, Any]:
        total_prob = 0.0
        weights: list[float] = []
        for comp in components:
            prob = float(comp.get("prob", comp.get("weight", 0.0)))
            prob = max(0.0, prob)
            weights.append(prob)
            total_prob += prob

        if total_prob <= 0:
            return self._rng.choice(components)

        threshold = self._rng.uniform(0.0, total_prob)
        cumulative = 0.0
        for comp, w in zip(components, weights):
            cumulative += w
            if cumulative >= threshold:
                return comp
        return components[-1]

    @staticmethod
    def _read_optional_int(value: Any, default: Any = None) -> int | None:
        if value is None or str(value).strip() == "":
            if default is None:
                return None
            try:
                return int(default)
            except (TypeError, ValueError):
                return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _read_float(value: Any, default: float) -> float:
        try:
            if value is None or str(value).strip() == "":
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)
