from __future__ import annotations


def compute_policy_cost_metrics(sim_state) -> dict[str, float]:
    return {
        "intervention_count": float(sim_state.intervention_count),
        "intervention_cost": float(sim_state.intervention_cost),
    }
