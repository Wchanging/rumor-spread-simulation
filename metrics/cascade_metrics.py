from __future__ import annotations

from domain.content import DebunkPost


def compute_cascade_metrics(sim_state) -> dict:
    total_posts = len(sim_state.content_pool)
    total_shares = sum(1 for item in sim_state.action_log if item.get("action_type") in {"share", "rewrite_share"})
    total_rewrite_shares = sum(1 for item in sim_state.action_log if item.get("action_type") == "rewrite_share")
    total_likes = sum(1 for item in sim_state.action_log if item.get("action_type") == "like")
    event_ids = list(sim_state.events.keys())
    user_count = len(sim_state.users)
    fake_event_ids = [event_id for event_id, event in sim_state.events.items() if event.is_fake]

    all_event_action_metrics: dict[str, dict] = {
        event_id: {
            "share_count": 0,
            "rewrite_share_count": 0,
            "like_count": 0,
            "total_actions": 0,
            "content_count": 0,
            "rumor_content_count": 0,
            "debunk_content_count": 0,
            "normal_content_count": 0,
            "unique_spreader_count": 0,
            "spreader_ratio": 0.0,
            "max_cascade_depth": 0,
        }
        for event_id in event_ids
    }

    fake_event_action_metrics: dict[str, dict] = {
        event_id: {
            "share_count": 0,
            "rewrite_share_count": 0,
            "like_count": 0,
            "total_actions": 0,
            "content_count": 0,
        }
        for event_id in fake_event_ids
    }

    event_spreaders: dict[str, set[str]] = {event_id: set() for event_id in event_ids}

    for action in sim_state.action_log:
        event_id = str(action.get("event_id", ""))
        if event_id not in all_event_action_metrics:
            continue
        action_type = str(action.get("action_type", ""))
        all_event_action_metrics[event_id]["total_actions"] += 1
        if action_type == "like":
            all_event_action_metrics[event_id]["like_count"] += 1
        elif action_type == "rewrite_share":
            all_event_action_metrics[event_id]["rewrite_share_count"] += 1
            all_event_action_metrics[event_id]["share_count"] += 1
            event_spreaders[event_id].add(str(action.get("actor_id", "")))
        elif action_type == "share":
            all_event_action_metrics[event_id]["share_count"] += 1
            event_spreaders[event_id].add(str(action.get("actor_id", "")))

    for content in sim_state.content_pool.values():
        event_id = str(getattr(content, "event_id", ""))
        if event_id not in all_event_action_metrics:
            continue
        all_event_action_metrics[event_id]["content_count"] += 1
        if bool(getattr(content, "is_rumor", False)):
            all_event_action_metrics[event_id]["rumor_content_count"] += 1
        elif isinstance(content, DebunkPost):
            all_event_action_metrics[event_id]["debunk_content_count"] += 1
        else:
            all_event_action_metrics[event_id]["normal_content_count"] += 1

    parent_map = {
        content_id: item.parent_content_id
        for content_id, item in sim_state.content_pool.items()
    }

    depth_cache: dict[str, int] = {}

    def depth(content_id: str) -> int:
        if content_id in depth_cache:
            return depth_cache[content_id]
        parent = parent_map.get(content_id)
        if not parent or parent not in parent_map:
            depth_cache[content_id] = 1
            return 1
        value = 1 + depth(parent)
        depth_cache[content_id] = value
        return value

    max_depth = 0
    for content_id in sim_state.content_pool:
        max_depth = max(max_depth, depth(content_id))

    for content_id, item in sim_state.content_pool.items():
        event_id = str(getattr(item, "event_id", ""))
        if event_id not in all_event_action_metrics:
            continue
        all_event_action_metrics[event_id]["max_cascade_depth"] = max(
            int(all_event_action_metrics[event_id].get("max_cascade_depth", 0)),
            int(depth(content_id)),
        )

    for event_id in event_ids:
        spreader_count = len(event_spreaders.get(event_id, set()) - {""})
        all_event_action_metrics[event_id]["unique_spreader_count"] = spreader_count
        all_event_action_metrics[event_id]["spreader_ratio"] = (spreader_count / user_count) if user_count else 0.0

    for event_id in fake_event_ids:
        if event_id not in all_event_action_metrics:
            continue
        fake_event_action_metrics[event_id] = {
            "share_count": int(all_event_action_metrics[event_id]["share_count"]),
            "rewrite_share_count": int(all_event_action_metrics[event_id]["rewrite_share_count"]),
            "like_count": int(all_event_action_metrics[event_id]["like_count"]),
            "total_actions": int(all_event_action_metrics[event_id]["total_actions"]),
            "content_count": int(all_event_action_metrics[event_id]["content_count"]),
        }

    return {
        "total_posts": float(total_posts),
        "total_shares": float(total_shares),
        "total_rewrite_shares": float(total_rewrite_shares),
        "total_likes": float(total_likes),
        "max_cascade_depth": float(max_depth),
        "event_cascade_metrics": all_event_action_metrics,
        "fake_event_action_metrics": fake_event_action_metrics,
    }
