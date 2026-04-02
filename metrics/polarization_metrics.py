from __future__ import annotations


def compute_polarization_metrics(sim_state) -> dict:
    fake_event_ids = [event_id for event_id, event in sim_state.events.items() if event.is_fake]
    event_ids = list(sim_state.events.keys())
    if not event_ids or not sim_state.users:
        return {
            "polarization_abs_mean": 0.0,
            "polarization_variance": 0.0,
            "event_opinion_metrics": {},
            "fake_event_opinion_metrics": {},
            "event_neighbor_disagreement": {},
            "mean_event_opinion_variance": 0.0,
            "mean_fake_event_opinion_variance": 0.0,
            "global_disagreement": 0.0,
            "fake_event_global_disagreement": 0.0,
        }

    values: list[float] = []
    for user in sim_state.users.values():
        for event_id in fake_event_ids:
            values.append(user.get_belief(event_id).belief_score)

    if values:
        abs_mean = sum(abs(v) for v in values) / len(values)
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
    else:
        abs_mean = 0.0
        variance = 0.0

    undirected_neighbors = _build_undirected_neighbors(sim_state)
    event_opinion_metrics: dict[str, dict[str, float]] = {}
    fake_event_opinion_metrics: dict[str, dict[str, float]] = {}
    event_neighbor_disagreement: dict[str, dict[str, float]] = {}

    for event_id in event_ids:
        event_values = [float(user.get_belief(event_id).belief_score) for user in sim_state.users.values()]
        event_mean = (sum(event_values) / len(event_values)) if event_values else 0.0
        event_var = (sum((v - event_mean) ** 2 for v in event_values) / len(event_values)) if event_values else 0.0

        event_opinion_metrics[event_id] = {
            "mean_belief": event_mean,
            "variance": event_var,
        }
        if event_id in fake_event_ids:
            fake_event_opinion_metrics[event_id] = dict(event_opinion_metrics[event_id])

        dg = _compute_event_neighbor_disagreement(sim_state, event_id, undirected_neighbors=undirected_neighbors)
        event_neighbor_disagreement[event_id] = {
            "dg": dg,
        }

    mean_event_opinion_variance = (
        sum(item.get("variance", 0.0) for item in event_opinion_metrics.values()) / len(event_opinion_metrics)
        if event_opinion_metrics
        else 0.0
    )
    mean_fake_event_opinion_variance = (
        sum(item.get("variance", 0.0) for item in fake_event_opinion_metrics.values()) / len(fake_event_opinion_metrics)
        if fake_event_opinion_metrics
        else 0.0
    )
    global_disagreement = (
        sum(item.get("dg", 0.0) for item in event_neighbor_disagreement.values()) / len(event_neighbor_disagreement)
        if event_neighbor_disagreement
        else 0.0
    )
    fake_event_global_disagreement = (
        sum(event_neighbor_disagreement.get(event_id, {}).get("dg", 0.0) for event_id in fake_event_ids)
        / len(fake_event_ids)
        if fake_event_ids
        else 0.0
    )

    return {
        "polarization_abs_mean": abs_mean,
        "polarization_variance": variance,
        "event_opinion_metrics": event_opinion_metrics,
        "fake_event_opinion_metrics": fake_event_opinion_metrics,
        "event_neighbor_disagreement": event_neighbor_disagreement,
        "mean_event_opinion_variance": mean_event_opinion_variance,
        "mean_fake_event_opinion_variance": mean_fake_event_opinion_variance,
        "global_disagreement": global_disagreement,
        "fake_event_global_disagreement": fake_event_global_disagreement,
    }


def _build_undirected_neighbors(sim_state) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = {str(user_id): set() for user_id in sim_state.users.keys()}
    adjacency = getattr(sim_state.network, "adjacency", {})
    if not isinstance(adjacency, dict):
        return neighbors

    for src, dst_set in adjacency.items():
        src_id = str(src)
        neighbors.setdefault(src_id, set())
        for dst in dst_set:
            dst_id = str(dst)
            neighbors.setdefault(dst_id, set())
            if src_id == dst_id:
                continue
            neighbors[src_id].add(dst_id)
            neighbors[dst_id].add(src_id)
    return neighbors


def _compute_event_neighbor_disagreement(sim_state, event_id: str, undirected_neighbors: dict[str, set[str]]) -> float:
    user_ids = [str(user_id) for user_id in sim_state.users.keys()]
    n_users = len(user_ids)
    if n_users <= 0:
        return 0.0

    total = 0.0
    for user_id in user_ids:
        belief_i = float(sim_state.users[user_id].get_belief(event_id).belief_score)
        local_neighbors = list(undirected_neighbors.get(user_id, set()))
        if not local_neighbors:
            continue

        local_sq_sum = 0.0
        for neighbor_id in local_neighbors:
            if neighbor_id not in sim_state.users:
                continue
            belief_j = float(sim_state.users[neighbor_id].get_belief(event_id).belief_score)
            local_sq_sum += (belief_i - belief_j) ** 2
        total += local_sq_sum / max(1, len(local_neighbors))

    return total / (2.0 * n_users)
