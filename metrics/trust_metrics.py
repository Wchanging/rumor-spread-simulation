from __future__ import annotations


def compute_trust_metrics(sim_state) -> dict[str, float]:
    if not sim_state.users:
        return {"platform_trust_mean": 0.0}
    value = sum(user.platform_trust for user in sim_state.users.values()) / len(sim_state.users)
    return {"platform_trust_mean": value}
