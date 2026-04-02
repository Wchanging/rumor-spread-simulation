from __future__ import annotations


def compute_belief_metrics(sim_state) -> dict:
    event_ids = list(sim_state.events.keys())
    fake_event_ids = [event_id for event_id, event in sim_state.events.items() if event.is_fake]
    if not sim_state.users:
        return {
            "misbelief_ratio": 0.0,
            "mean_belief_score": 0.0,
            "fake_event_belief_metrics": {},
            "event_belief_metrics": {},
            "event_exposure_metrics": {},
            "fake_event_exposure_metrics": {},
            "mean_event_exposure_ratio": 0.0,
            "mean_fake_event_exposure_ratio": 0.0,
            "mean_fake_event_misbelief_ratio_among_exposed": 0.0,
            "users_exposed_to_any_fake_ratio": 0.0,
            "users_believing_any_fake_ratio_among_exposed": 0.0,
            "fake_event_misbelief_ratio_among_exposed_pairs": 0.0,
            "fake_event_misbelief_ratio_among_exposed_users": 0.0,
        }

    event_belief_metrics: dict[str, dict] = {}
    event_exposure_metrics: dict[str, dict] = {}
    fake_event_exposure_metrics: dict[str, dict] = {}

    users_exposed_to_any_fake = 0
    users_believing_any_fake = 0
    fake_exposed_pairs = 0
    fake_positive_pairs = 0

    for user in sim_state.users.values():
        exposed_any_fake = False
        believing_any_fake = False
        for event_id in fake_event_ids:
            seen_count = int(user.get_belief(event_id).seen_count)
            if seen_count > 0:
                exposed_any_fake = True
                fake_exposed_pairs += 1
                if user.get_belief(event_id).belief_score > 0:
                    fake_positive_pairs += 1
                    believing_any_fake = True
        if exposed_any_fake:
            users_exposed_to_any_fake += 1
            if believing_any_fake:
                users_believing_any_fake += 1

    for event_id in event_ids:
        exposed_user_count = 0
        positive_user_count = 0
        seen_count_sum = 0
        exposed_belief_sum = 0.0
        exposed_abs_belief_sum = 0.0

        all_user_count = len(sim_state.users)
        all_positive_user_count = 0
        all_belief_sum = 0.0

        for user in sim_state.users.values():
            belief_state = user.get_belief(event_id)
            belief_score = float(belief_state.belief_score)
            all_belief_sum += belief_score
            if belief_score > 0:
                all_positive_user_count += 1

            seen_count = int(belief_state.seen_count)
            if seen_count <= 0:
                continue
            exposed_user_count += 1
            seen_count_sum += seen_count
            exposed_belief_sum += belief_score
            exposed_abs_belief_sum += abs(belief_score)
            if belief_score > 0:
                positive_user_count += 1

        exposure_ratio = (exposed_user_count / len(sim_state.users)) if sim_state.users else 0.0
        positive_ratio_among_exposed = (positive_user_count / exposed_user_count) if exposed_user_count else 0.0
        mean_seen_count_among_exposed = (seen_count_sum / exposed_user_count) if exposed_user_count else 0.0
        mean_belief_score_among_exposed = (exposed_belief_sum / exposed_user_count) if exposed_user_count else 0.0
        mean_abs_belief_score_among_exposed = (
            (exposed_abs_belief_sum / exposed_user_count) if exposed_user_count else 0.0
        )

        event_belief_metrics[event_id] = {
            "mean_belief_score_all_users": (all_belief_sum / all_user_count) if all_user_count else 0.0,
            "positive_belief_ratio_all_users": (all_positive_user_count / all_user_count) if all_user_count else 0.0,
            "positive_belief_user_count": all_positive_user_count,
            "user_count": all_user_count,
        }

        event_exposure_metrics[event_id] = {
            "exposed_user_count": exposed_user_count,
            "exposure_ratio": exposure_ratio,
            "mean_seen_count_among_exposed": mean_seen_count_among_exposed,
            "positive_belief_user_count_among_exposed": positive_user_count,
            "positive_belief_ratio_among_exposed": positive_ratio_among_exposed,
            "mean_belief_score_among_exposed": mean_belief_score_among_exposed,
            "mean_abs_belief_score_among_exposed": mean_abs_belief_score_among_exposed,
        }

        if event_id in fake_event_ids:
            fake_event_exposure_metrics[event_id] = {
                "exposed_user_count": exposed_user_count,
                "exposure_ratio": exposure_ratio,
                "mean_seen_count_among_exposed": mean_seen_count_among_exposed,
                "misbelief_user_count_among_exposed": positive_user_count,
                "misbelief_ratio_among_exposed": positive_ratio_among_exposed,
                "mean_belief_score_among_exposed": mean_belief_score_among_exposed,
                "mean_abs_belief_score_among_exposed": mean_abs_belief_score_among_exposed,
            }

    mean_event_exposure_ratio = (
        sum(float(item.get("exposure_ratio", 0.0)) for item in event_exposure_metrics.values()) / len(event_exposure_metrics)
        if event_exposure_metrics
        else 0.0
    )
    mean_fake_event_exposure_ratio = (
        sum(float(item.get("exposure_ratio", 0.0)) for item in fake_event_exposure_metrics.values())
        / len(fake_event_exposure_metrics)
        if fake_event_exposure_metrics
        else 0.0
    )
    mean_fake_event_misbelief_ratio_among_exposed = (
        sum(float(item.get("misbelief_ratio_among_exposed", 0.0)) for item in fake_event_exposure_metrics.values())
        / len(fake_event_exposure_metrics)
        if fake_event_exposure_metrics
        else 0.0
    )

    if not fake_event_ids:
        return {
            "misbelief_ratio": 0.0,
            "mean_belief_score": 0.0,
            "fake_event_belief_metrics": {},
            "event_belief_metrics": event_belief_metrics,
            "event_exposure_metrics": event_exposure_metrics,
            "fake_event_exposure_metrics": fake_event_exposure_metrics,
            "mean_event_exposure_ratio": mean_event_exposure_ratio,
            "mean_fake_event_exposure_ratio": mean_fake_event_exposure_ratio,
            "mean_fake_event_misbelief_ratio_among_exposed": mean_fake_event_misbelief_ratio_among_exposed,
            "users_exposed_to_any_fake_ratio": 0.0,
            "users_believing_any_fake_ratio_among_exposed": 0.0,
            "fake_event_misbelief_ratio_among_exposed_pairs": 0.0,
            "fake_event_misbelief_ratio_among_exposed_users": 0.0,
        }

    positive = 0
    total = 0
    score_sum = 0.0
    fake_event_belief_metrics: dict[str, dict] = {}

    for user in sim_state.users.values():
        for event_id in fake_event_ids:
            belief = user.get_belief(event_id).belief_score
            score_sum += belief
            total += 1
            if belief > 0:
                positive += 1

    for event_id in fake_event_ids:
        event_positive = 0
        event_total = 0
        event_score_sum = 0.0
        for user in sim_state.users.values():
            belief = user.get_belief(event_id).belief_score
            event_score_sum += belief
            event_total += 1
            if belief > 0:
                event_positive += 1

        fake_event_belief_metrics[event_id] = {
            "misbelief_ratio": (event_positive / event_total) if event_total else 0.0,
            "mean_belief_score": (event_score_sum / event_total) if event_total else 0.0,
            "positive_user_count": event_positive,
            "user_count": event_total,
        }

    return {
        "misbelief_ratio": (positive / total) if total else 0.0,
        "mean_belief_score": (score_sum / total) if total else 0.0,
        "fake_event_belief_metrics": fake_event_belief_metrics,
        "event_belief_metrics": event_belief_metrics,
        "event_exposure_metrics": event_exposure_metrics,
        "fake_event_exposure_metrics": fake_event_exposure_metrics,
        "mean_event_exposure_ratio": mean_event_exposure_ratio,
        "mean_fake_event_exposure_ratio": mean_fake_event_exposure_ratio,
        "mean_fake_event_misbelief_ratio_among_exposed": mean_fake_event_misbelief_ratio_among_exposed,
        "users_exposed_to_any_fake_ratio": (users_exposed_to_any_fake / len(sim_state.users)) if sim_state.users else 0.0,
        "users_believing_any_fake_ratio_among_exposed": (
            (users_believing_any_fake / users_exposed_to_any_fake) if users_exposed_to_any_fake else 0.0
        ),
        "fake_event_misbelief_ratio_among_exposed_pairs": (
            (fake_positive_pairs / fake_exposed_pairs) if fake_exposed_pairs else 0.0
        ),
        "fake_event_misbelief_ratio_among_exposed_users": (
            (users_believing_any_fake / users_exposed_to_any_fake) if users_exposed_to_any_fake else 0.0
        ),
    }
