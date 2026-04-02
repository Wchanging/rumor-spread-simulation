from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .belief_metrics import compute_belief_metrics
from .cascade_metrics import compute_cascade_metrics
from .policy_cost_metrics import compute_policy_cost_metrics
from .polarization_metrics import compute_polarization_metrics
from .trust_metrics import compute_trust_metrics


@dataclass
class MetricsCollector:
    history: list[dict[str, Any]] = field(default_factory=list)
    keep_history: bool = True
    last_snapshot: dict[str, Any] = field(default_factory=dict)

    def collect(self, t: int, sim_state) -> dict[str, Any]:
        snapshot = {"timestep": t}
        snapshot.update(compute_cascade_metrics(sim_state))
        snapshot.update(compute_belief_metrics(sim_state))
        snapshot.update(compute_trust_metrics(sim_state))
        snapshot.update(compute_polarization_metrics(sim_state))
        snapshot.update(compute_policy_cost_metrics(sim_state))
        self.last_snapshot = snapshot
        if self.keep_history:
            self.history.append(snapshot)
        return snapshot

    def finalize(self) -> dict[str, Any]:
        if not self.history and not self.last_snapshot:
            return {"history": [], "summary": {}}

        if self.history:
            last = self.history[-1]
            history = self.history
        else:
            last = self.last_snapshot
            history = [self.last_snapshot]

        series_metrics = self._compute_series_summary(history)

        return {
            "history": history,
            "summary": {
                "final_misbelief_ratio": last.get("misbelief_ratio", 0.0),
                "final_fake_event_misbelief_ratio_among_exposed_pairs": last.get(
                    "fake_event_misbelief_ratio_among_exposed_pairs", 0.0
                ),
                "final_fake_event_misbelief_ratio_among_exposed_users": last.get(
                    "fake_event_misbelief_ratio_among_exposed_users", 0.0
                ),
                "final_platform_trust": last.get("platform_trust_mean", 0.0),
                "final_total_shares": last.get("total_shares", 0.0),
                "final_intervention_cost": last.get("intervention_cost", 0.0),
                "final_users_exposed_to_any_fake_ratio": last.get("users_exposed_to_any_fake_ratio", 0.0),
                "final_users_believing_any_fake_ratio_among_exposed": last.get(
                    "users_believing_any_fake_ratio_among_exposed", 0.0
                ),
                "final_fake_event_belief_metrics": last.get("fake_event_belief_metrics", {}),
                "final_event_belief_metrics": last.get("event_belief_metrics", {}),
                "final_fake_event_exposure_metrics": last.get("fake_event_exposure_metrics", {}),
                "final_event_exposure_metrics": last.get("event_exposure_metrics", {}),
                "final_mean_event_exposure_ratio": last.get("mean_event_exposure_ratio", 0.0),
                "final_mean_fake_event_exposure_ratio": last.get("mean_fake_event_exposure_ratio", 0.0),
                "final_mean_fake_event_misbelief_ratio_among_exposed": last.get(
                    "mean_fake_event_misbelief_ratio_among_exposed", 0.0
                ),
                "final_rumor_exposure_rate": last.get("rumor_exposure_rate", 0.0),
                "final_debunk_exposure_rate": last.get("debunk_exposure_rate", 0.0),
                "final_normal_exposure_rate": last.get("normal_exposure_rate", 0.0),
                "final_empty_feed_rate": last.get("empty_feed_rate", 0.0),
                "final_fake_event_action_metrics": last.get("fake_event_action_metrics", {}),
                "final_event_cascade_metrics": last.get("event_cascade_metrics", {}),
                "final_event_opinion_metrics": last.get("event_opinion_metrics", {}),
                "final_fake_event_opinion_metrics": last.get("fake_event_opinion_metrics", {}),
                "final_event_neighbor_disagreement": last.get("event_neighbor_disagreement", {}),
                "final_global_disagreement": last.get("global_disagreement", 0.0),
                "final_fake_event_global_disagreement": last.get("fake_event_global_disagreement", 0.0),
                **series_metrics,
            },
        }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _series_auc(self, history: list[dict[str, Any]], key: str) -> float:
        if not history:
            return 0.0
        auc = 0.0
        for idx, current in enumerate(history):
            y = self._safe_float(current.get(key, 0.0))
            if idx + 1 < len(history):
                t0 = self._safe_float(current.get("timestep", idx), float(idx))
                t1 = self._safe_float(history[idx + 1].get("timestep", idx + 1), float(idx + 1))
                dt = max(0.0, t1 - t0)
            else:
                dt = 1.0
            auc += y * dt
        return auc

    def _peak_with_timestep(self, history: list[dict[str, Any]], key: str) -> tuple[float, int]:
        if not history:
            return 0.0, 0
        best_value = float("-inf")
        best_t = 0
        for idx, row in enumerate(history):
            value = self._safe_float(row.get(key, 0.0))
            if value > best_value:
                best_value = value
                best_t = int(self._safe_float(row.get("timestep", idx), float(idx)))
        if best_value == float("-inf"):
            return 0.0, 0
        return best_value, best_t

    def _time_to_threshold(self, history: list[dict[str, Any]], key: str, threshold: float) -> int | None:
        for idx, row in enumerate(history):
            value = self._safe_float(row.get(key, 0.0))
            if value >= threshold:
                return int(self._safe_float(row.get("timestep", idx), float(idx)))
        return None

    def _compute_series_summary(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        if not history:
            return {}

        first = history[0]
        last = history[-1]

        peak_misbelief, peak_misbelief_t = self._peak_with_timestep(history, "misbelief_ratio")
        peak_fake_exposed_users, peak_fake_exposed_users_t = self._peak_with_timestep(
            history,
            "fake_event_misbelief_ratio_among_exposed_users",
        )

        initial_trust = self._safe_float(first.get("platform_trust_mean", 0.0))
        final_trust = self._safe_float(last.get("platform_trust_mean", 0.0))
        final_cost = self._safe_float(last.get("intervention_cost", 0.0))

        misbelief_auc = self._series_auc(history, "misbelief_ratio")
        fake_exposed_users_auc = self._series_auc(history, "fake_event_misbelief_ratio_among_exposed_users")

        return {
            "misbelief_auc": misbelief_auc,
            "fake_event_misbelief_auc_among_exposed_users": fake_exposed_users_auc,
            "peak_misbelief_ratio": peak_misbelief,
            "peak_misbelief_timestep": peak_misbelief_t,
            "peak_fake_event_misbelief_ratio_among_exposed_users": peak_fake_exposed_users,
            "peak_fake_event_misbelief_timestep_among_exposed_users": peak_fake_exposed_users_t,
            "initial_platform_trust": initial_trust,
            "platform_trust_delta": final_trust - initial_trust,
            "time_to_misbelief_0_20": self._time_to_threshold(history, "misbelief_ratio", 0.20),
            "time_to_misbelief_0_30": self._time_to_threshold(history, "misbelief_ratio", 0.30),
            "efficiency_misbelief_auc_per_cost": (misbelief_auc / final_cost) if final_cost > 0 else None,
            "efficiency_fake_exposed_auc_per_cost": (fake_exposed_users_auc / final_cost) if final_cost > 0 else None,
        }
