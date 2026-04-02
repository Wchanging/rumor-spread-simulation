from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

# fmt: off
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


os.environ.setdefault("MPLBACKEND", "Agg")

from agents.attention import AttentionModule
from agents.rumor_agent import RumorAgent
from agents.sharing_policy import SharingPolicy
from data.event_dataset import EventDatasetLoader
from data.user_factory import UserProfile, UserProfileFactory
from domain.content import RumorPost
from domain.events import Event
from domain.users import UserState
from engine.experiment_recorder import ExperimentRecorder
from engine.feed_builder import FeedBuilder
from engine.scheduler import Scheduler
from engine.sim_state import SimulationState
from engine.simulation import SimulationEngine
from intervention.broadcast_debunk import GlobalBroadcastDebunk
from intervention.no_intervention import NoIntervention
from intervention.official_account_debunk import OfficialAccountDebunk
from intervention.targeted_debunk import PersonalizedDebunk, TargetTopKSpreaders
from llm.mock_client import MockLLMClient
from llm.openai_client import OpenAIClient
from metrics.collector import MetricsCollector
from network.builder import NetworkBuilder, indegree_map, load_network, relabel_network_nodes, save_network
# fmt: on


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")
    if suffix == ".json":
        return json.loads(text)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except Exception as exc:
            raise RuntimeError("Failed to read YAML. Install PyYAML or use a JSON config file.") from exc

    raise ValueError(f"Unsupported config format: {suffix}")


def _resolve_path(base_dir: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _normalize_user_ids(user_profiles: list[UserProfile]) -> list[UserProfile]:
    seen: dict[str, int] = {}
    for profile in user_profiles:
        if profile.user_id not in seen:
            seen[profile.user_id] = 1
            continue
        count = seen[profile.user_id]
        seen[profile.user_id] += 1
        profile.user_id = f"{profile.user_id}_{count}"
    return user_profiles


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _resolve_user_source_config(config: dict, base_dir: Path) -> dict[str, Any]:
    inline_user_source = config.get("user_source", {})
    if not isinstance(inline_user_source, dict):
        inline_user_source = {}

    raw_path = config.get("user_source_config_path", config.get("user_source_config"))
    if not raw_path:
        return dict(inline_user_source)

    resolved_path = _resolve_path(base_dir, str(raw_path))
    if resolved_path is None or not resolved_path.exists():
        raise FileNotFoundError(f"user_source_config_path does not exist: {raw_path}")

    loaded = load_config(resolved_path)
    if not isinstance(loaded, dict):
        raise ValueError("user_source_config_path must be a JSON/YAML object")

    file_user_source = loaded.get("user_source", loaded)
    if not isinstance(file_user_source, dict):
        raise ValueError("user_source in user_source_config_path must be an object")

    return _deep_merge_dict(dict(file_user_source), dict(inline_user_source))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _is_valid_profile_value(field_name: str, value: Any) -> bool:
    if _is_missing(value):
        return False

    if field_name in {"online_probability", "platform_trust", "trust_threshold", "share_tendency"}:
        try:
            number = float(value)
            return 0.0 <= number <= 1.0
        except (TypeError, ValueError):
            return False

    if field_name == "age":
        try:
            number = int(value)
            return 0 < number < 120
        except (TypeError, ValueError):
            return False

    if field_name == "attention_budget":
        try:
            number = int(value)
            return number >= 0
        except (TypeError, ValueError):
            return False

    return True


def _is_value_compatible_with_field_spec(value: Any, field_spec: dict[str, Any] | None) -> bool:
    if _is_missing(value):
        return False
    if not isinstance(field_spec, dict) or not field_spec:
        return True

    spec_type = str(field_spec.get("type", "constant")).lower()

    if spec_type in {"categorical", "choice"}:
        allowed_values = {
            str(choice.get("value"))
            for choice in list(field_spec.get("choices", []))
            if isinstance(choice, dict) and "value" in choice
        }
        if not allowed_values:
            return True
        return str(value) in allowed_values

    if spec_type in {"polarity_words", "big5_polarity"} or "high_words" in field_spec or "low_words" in field_spec:
        allowed_values = {
            str(word).strip()
            for word in list(field_spec.get("high_words", [])) + list(field_spec.get("low_words", []))
            if str(word).strip()
        }
        if not allowed_values:
            return True
        return str(value).strip() in allowed_values

    return True


def _build_required_user_fields(generate_cfg: dict[str, Any]) -> list[str]:
    required = {
        "user_id",
        "online_probability",
        "platform_trust",
        "trust_threshold",
        "share_tendency",
    }
    fields_cfg = generate_cfg.get("fields", {})
    required.update(str(name) for name in fields_cfg.keys())
    return sorted(required)


def _is_existing_user_file_reusable(
    profiles: list[UserProfile],
    required_fields: list[str],
    expected_n_users: int,
    fields_cfg: dict[str, Any] | None = None,
) -> bool:
    if not profiles:
        return False
    if expected_n_users > 0 and len(profiles) != expected_n_users:
        return False

    fields_cfg = fields_cfg or {}

    for profile in profiles:
        for field_name in required_fields:
            value = getattr(profile, field_name, None)
            if not _is_valid_profile_value(field_name, value):
                return False
            if not _is_value_compatible_with_field_spec(value, fields_cfg.get(field_name)):
                return False
    return True


def build_user_profiles(config: dict, base_dir: Path, run_seed: int) -> list[UserProfile]:
    user_source = _resolve_user_source_config(config, base_dir=base_dir)
    user_defaults = config.get("user_model", {})
    mode = str(user_source.get("mode", "generate"))
    factory = UserProfileFactory(seed=run_seed)

    if mode == "csv":
        csv_path = _resolve_path(base_dir, user_source.get("csv_path"))
        if csv_path is None or not csv_path.exists():
            raise FileNotFoundError("user_source.mode=csv but csv_path does not exist")
        profiles = factory.load_from_csv(csv_path=csv_path, defaults=user_defaults)
    elif mode == "generate":
        generate_cfg = user_source.get("generate", {})
        n_users = int(generate_cfg.get("n_users", config.get("network", {}).get("n_users", 100)))
        save_path = _resolve_path(base_dir, generate_cfg.get("save_csv_path"))
        prefer_existing_csv = bool(generate_cfg.get("prefer_existing_csv", True))
        required_fields = _build_required_user_fields(generate_cfg)
        fields_cfg = generate_cfg.get("fields", {}) if isinstance(generate_cfg.get("fields", {}), dict) else {}

        if save_path is not None and prefer_existing_csv and save_path.exists():
            existing_profiles = factory.load_from_csv(csv_path=save_path, defaults=user_defaults)
            if _is_existing_user_file_reusable(
                existing_profiles,
                required_fields=required_fields,
                expected_n_users=n_users,
                fields_cfg=fields_cfg,
            ):
                profiles = existing_profiles
            else:
                profiles = factory.generate(n_users=n_users, generate_config=generate_cfg, defaults=user_defaults)
                factory.dump_to_csv(profiles, save_path)
        else:
            profiles = factory.generate(n_users=n_users, generate_config=generate_cfg, defaults=user_defaults)

        if save_path is not None and (not save_path.exists() or not prefer_existing_csv):
            factory.dump_to_csv(profiles, save_path)
    else:
        raise ValueError(f"Unsupported user_source.mode: {mode}")

    # Global experiment setting should take precedence over per-user CSV fields.
    if isinstance(user_defaults, dict) and "attention_budget" in user_defaults:
        try:
            forced_budget = max(0, int(user_defaults.get("attention_budget")))
        except (TypeError, ValueError):
            forced_budget = None
        if forced_budget is not None:
            for profile in profiles:
                profile.attention_budget = forced_budget

    if not profiles:
        raise RuntimeError("User list is empty. Please check user_source configuration")
    return _normalize_user_ids(profiles)


def build_user_states(user_profiles: list[UserProfile]) -> dict[str, UserState]:
    users: dict[str, UserState] = {}
    for profile in user_profiles:
        users[profile.user_id] = UserState(
            user_id=profile.user_id,
            age=profile.age,
            gender=profile.gender,
            occupation=profile.occupation,
            education_level=profile.education_level,
            city_tier=profile.city_tier,
            big5_neuroticism=profile.big5_neuroticism,
            big5_extraversion=profile.big5_extraversion,
            big5_openness=profile.big5_openness,
            big5_agreeableness=profile.big5_agreeableness,
            big5_conscientiousness=profile.big5_conscientiousness,
            attention_budget=profile.attention_budget,
            online_probability=profile.online_probability,
            platform_trust=profile.platform_trust,
            trust_threshold=profile.trust_threshold,
            share_tendency=profile.share_tendency,
        )
    return users


def build_events_from_config(config: dict) -> tuple[dict[str, Event], list]:
    event_cfg = config.get("event", {})
    event = Event(
        event_id=str(event_cfg.get("event_id", "event_1")),
        description=str(event_cfg.get("description", "Unnamed event")),
        is_fake=bool(event_cfg.get("is_fake", True)),
        evidence=str(event_cfg.get("evidence", "No authoritative evidence available")),
        evidence_posts=list(event_cfg.get("evidence_posts", [])),
    )
    return {event.event_id: event}, []


def build_events_and_contents(config: dict, base_dir: Path, run_seed: int) -> tuple[dict[str, Event], list]:
    event_source = config.get("event_source", {})
    mode = str(event_source.get("mode", "config"))

    if mode != "csv":
        return build_events_from_config(config)

    events_file = _resolve_path(base_dir, event_source.get("events_file"))
    posts_dir = _resolve_path(base_dir, event_source.get("posts_dir"))
    if events_file is None or posts_dir is None:
        raise FileNotFoundError("event_source.mode=csv requires events_file and posts_dir")

    generated_posts_raw = event_source.get("generated_rumor_posts", False)
    generated_posts_cfg = generated_posts_raw if isinstance(generated_posts_raw, dict) else {}
    include_generated_rumor_posts = bool(
        generated_posts_raw if isinstance(generated_posts_raw, bool) else generated_posts_cfg.get("enabled", False)
    )
    generated_posts_dir = _resolve_path(base_dir, generated_posts_cfg.get("posts_dir", "data/processed/generated_rumor_posts_v2"))
    generated_posts_template = str(generated_posts_cfg.get("posts_template", "{event_id}_generated.csv"))

    randomize_selection = bool(event_source.get("randomize_selection", event_source.get("shuffle_selection", False)))
    raw_selection_seed = event_source.get("selection_seed", event_source.get("sample_seed", None))
    simulation_cfg = config.get("simulation", {}) if isinstance(config.get("simulation", {}), dict) else {}
    default_selection_seed = simulation_cfg.get("seed", run_seed)
    selection_seed = default_selection_seed if raw_selection_seed is None else int(raw_selection_seed)

    loader = EventDatasetLoader()
    dataset = loader.load(
        events_file=events_file,
        posts_dir=posts_dir,
        posts_template=str(event_source.get("posts_template", "{event_id}.csv")),
        include_generated_rumor_posts=include_generated_rumor_posts,
        generated_posts_dir=generated_posts_dir,
        generated_posts_template=generated_posts_template,
        max_events=event_source.get("max_events"),
        ensure_fake_event=bool(event_source.get("ensure_fake_event", False)),
        fake_event_count=event_source.get("fake_event_count", None),
        exclude_evidence_posts_in_fake=bool(
            event_source.get("exclude_evidence_posts_in_fake", event_source.get("pure_rumor_mode", False))
        ),
        keep_evidence_posts_for_official_only=bool(event_source.get("official_only_evidence_pool", False)),
        randomize_selection=randomize_selection,
        selection_seed=selection_seed,
    )
    if dataset.events:
        return dataset.events, dataset.initial_contents

    return build_events_from_config(config)


def create_initial_rumor_seeds(
    sim_state: SimulationState,
    seed_count: int = 3,
) -> None:
    degree = indegree_map(sim_state.network)
    ranked_users = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    fake_events = [event_id for event_id, event in sim_state.events.items() if event.is_fake]
    if not fake_events:
        return

    for event_idx, event_id in enumerate(fake_events):
        for idx in range(min(seed_count, len(ranked_users))):
            user_id = ranked_users[(event_idx + idx) % len(ranked_users)][0]
            post = RumorPost(
                content_id=f"seed_{event_id}_{idx}",
                event_id=event_id,
                author_id=user_id,
                text=f"Unverified message about {event_id} #{idx}",
                timestamp=0,
                popularity=1.0,
            )
            sim_state.add_content(post, timestep=0)


def attach_initial_contents(
    sim_state: SimulationState,
    initial_contents: list,
    run_seed: int,
    release_cfg: dict[str, Any] | None = None,
) -> None:
    if not initial_contents:
        return

    rng = random.Random(run_seed)
    user_ids = list(sim_state.users.keys())
    if not user_ids:
        return

    release_cfg = release_cfg or {}
    release_enabled = bool(release_cfg.get("enabled", False))
    use_source_timestamp = bool(release_cfg.get("use_source_timestamp", True))
    max_release_timestep = max(0, int(release_cfg.get("max_timestep", 0)))

    valid_timestamps = [
        int(getattr(content, "timestamp", 0))
        for content in initial_contents
        if int(getattr(content, "timestamp", 0)) > 0
    ]
    ts_min = min(valid_timestamps) if valid_timestamps else 0
    ts_max = max(valid_timestamps) if valid_timestamps else 0
    ts_span = max(1, ts_max - ts_min)

    for content in initial_contents:
        if content.author_id not in sim_state.users:
            content.author_id = rng.choice(user_ids)

        release_timestep = 0
        if release_enabled and max_release_timestep > 0:
            if use_source_timestamp and int(getattr(content, "timestamp", 0)) > 0 and ts_max > ts_min:
                norm = (int(content.timestamp) - ts_min) / ts_span
                release_timestep = int(round(norm * max_release_timestep))
            elif use_source_timestamp and int(getattr(content, "timestamp", 0)) > 0:
                release_timestep = max_release_timestep // 2
            else:
                release_timestep = rng.randint(0, max_release_timestep)

        content.timestamp = int(release_timestep)
        sim_state.add_content(content, timestep=release_timestep)


def build_intervention_strategy(config: dict, llm_client):
    intervention_cfg = config.get("intervention", {})
    strategy_name = str(intervention_cfg.get("strategy", "NoIntervention"))
    if strategy_name == "NoIntervention":
        return NoIntervention()
    if strategy_name == "GlobalBroadcastDebunk":
        return GlobalBroadcastDebunk(
            tone_style=str(intervention_cfg.get("tone_style", "cautious")),
            cost_per_post=float(intervention_cfg.get("cost_per_post", 1.0)),
            start_timestep=int(intervention_cfg.get("start_timestep", 0)),
            interval=int(intervention_cfg.get("interval", 1)),
            end_timestep=intervention_cfg.get("end_timestep", None),
            posts_per_round=int(intervention_cfg.get("posts_per_round", 3)),
            max_posts_per_event=int(intervention_cfg.get("max_posts_per_event", 1)),
            event_selection_mode=str(intervention_cfg.get("event_selection_mode", "risk_weighted")),
            recent_action_window=int(intervention_cfg.get("recent_action_window", 3)),
            event_heat_weight=float(intervention_cfg.get("event_heat_weight", 1.0)),
            event_misbelief_weight=float(intervention_cfg.get("event_misbelief_weight", 1.0)),
            event_coverage_gap_weight=float(intervention_cfg.get("event_coverage_gap_weight", 0.6)),
            llm_client=llm_client,
        )
    if strategy_name == "TargetTopKSpreaders":
        return TargetTopKSpreaders(
            k=int(intervention_cfg.get("k", 20)),
            tone_style=str(intervention_cfg.get("tone_style", "assertive")),
            start_timestep=int(intervention_cfg.get("start_timestep", 0)),
            interval=int(intervention_cfg.get("interval", 1)),
            end_timestep=intervention_cfg.get("end_timestep", None),
            posts_per_round=int(intervention_cfg.get("posts_per_round", 3)),
            max_posts_per_event=int(intervention_cfg.get("max_posts_per_event", 1)),
            recent_action_window=int(intervention_cfg.get("recent_action_window", 3)),
            min_share_count=int(intervention_cfg.get("min_share_count", 1)),
            recent_share_weight=float(intervention_cfg.get("recent_share_weight", 1.0)),
            total_share_weight=float(intervention_cfg.get("total_share_weight", 0.4)),
            degree_weight=float(intervention_cfg.get("degree_weight", 0.2)),
            event_selection_mode=str(intervention_cfg.get("event_selection_mode", "risk_weighted")),
            event_heat_weight=float(intervention_cfg.get("event_heat_weight", 1.0)),
            event_misbelief_weight=float(intervention_cfg.get("event_misbelief_weight", 1.0)),
            event_coverage_gap_weight=float(intervention_cfg.get("event_coverage_gap_weight", 0.6)),
            llm_client=llm_client,
        )
    if strategy_name == "PersonalizedDebunk":
        return PersonalizedDebunk(
            threshold=float(intervention_cfg.get("threshold", 0.2)),
            max_target_belief=float(intervention_cfg.get("max_target_belief", 0.7)),
            tone_style=str(intervention_cfg.get("tone_style", "empathetic")),
            high_risk_tone_style=str(intervention_cfg.get("high_risk_tone_style", "authoritative")),
            high_risk_threshold=float(intervention_cfg.get("high_risk_threshold", 0.5)),
            start_timestep=int(intervention_cfg.get("start_timestep", 0)),
            interval=int(intervention_cfg.get("interval", 1)),
            end_timestep=intervention_cfg.get("end_timestep", None),
            max_targets_per_round=int(intervention_cfg.get("max_targets_per_round", 40)),
            per_user_post_cap=int(intervention_cfg.get("per_user_post_cap", 1)),
            posts_per_round=int(intervention_cfg.get("posts_per_round", 40)),
            max_posts_per_event=int(intervention_cfg.get("max_posts_per_event", 2)),
            event_selection_mode=str(intervention_cfg.get("event_selection_mode", "risk_weighted")),
            recent_action_window=int(intervention_cfg.get("recent_action_window", 3)),
            event_heat_weight=float(intervention_cfg.get("event_heat_weight", 1.0)),
            event_misbelief_weight=float(intervention_cfg.get("event_misbelief_weight", 1.0)),
            event_coverage_gap_weight=float(intervention_cfg.get("event_coverage_gap_weight", 0.6)),
            llm_client=llm_client,
        )
    if strategy_name == "OfficialAccountDebunk":
        return OfficialAccountDebunk(
            official_account_count=int(intervention_cfg.get("official_account_count", 6)),
            min_degree_quantile=float(intervention_cfg.get("min_degree_quantile", 0.6)),
            max_degree_quantile=float(intervention_cfg.get("max_degree_quantile", 0.9)),
            tone_style=str(intervention_cfg.get("tone_style", "authoritative")),
            start_timestep=int(intervention_cfg.get("start_timestep", 3)),
            interval=int(intervention_cfg.get("interval", 3)),
            end_timestep=intervention_cfg.get("end_timestep", None),
            posts_per_round=int(intervention_cfg.get("posts_per_round", 4)),
            round_post_schedule=intervention_cfg.get("round_post_schedule", None),
            event_selection_mode=str(intervention_cfg.get("event_selection_mode", "round_robin")),
            recent_action_window=int(intervention_cfg.get("recent_action_window", 3)),
            max_posts_per_event=int(intervention_cfg.get("max_posts_per_event", 2)),
            event_heat_weight=float(intervention_cfg.get("event_heat_weight", 1.0)),
            event_misbelief_weight=float(intervention_cfg.get("event_misbelief_weight", 1.0)),
            event_exposure_weight=float(intervention_cfg.get("event_exposure_weight", 0.0)),
            llm_client=llm_client,
        )
    raise ValueError(f"Unknown intervention strategy: {strategy_name}")


def build_or_load_network(config: dict, user_ids: list[str], run_seed: int, base_dir: Path):
    network_cfg = config.get("network", {})
    persistence_cfg = network_cfg.get("persistence", {}) if isinstance(network_cfg.get("persistence", {}), dict) else {}
    persistence_enabled = bool(persistence_cfg.get("enabled", False))
    persistence_path = _resolve_path(base_dir, persistence_cfg.get("path")) if persistence_enabled else None
    reuse_if_exists = bool(persistence_cfg.get("reuse_if_exists", True))
    save_after_build = bool(persistence_cfg.get("save_after_build", True))

    if persistence_enabled and persistence_path is not None and reuse_if_exists and persistence_path.exists():
        try:
            loaded = load_network(persistence_path)
            if set(loaded.nodes()) == set(user_ids):
                return loaded
        except Exception:
            pass

    builder = NetworkBuilder()
    network = builder.build_synthetic(
        network_type=str(network_cfg.get("type", "small_world")),
        n=len(user_ids),
        params=network_cfg,
        seed=run_seed,
    )
    network = relabel_network_nodes(network, user_ids)

    if persistence_enabled and persistence_path is not None and save_after_build:
        save_network(
            network,
            persistence_path,
            metadata={
                "network_type": str(network_cfg.get("type", "small_world")),
                "seed": run_seed,
                "user_count": len(user_ids),
            },
        )
    return network


def build_llm_client(config: dict, base_dir: Path, run_seed: int):
    llm_cfg = config.get("llm", {})
    provider = str(llm_cfg.get("provider", "mock")).lower()
    if provider == "mock":
        return MockLLMClient(seed=run_seed)
    if provider == "openai":
        cfg = dict(llm_cfg)
        env_file = cfg.get("env_file")
        if env_file:
            cfg["env_file"] = str(_resolve_path(base_dir, str(env_file)))
        return OpenAIClient.from_config(cfg)
    raise ValueError(f"Unsupported llm.provider: {provider}")


def build_recorder(
    config: dict,
    run_dir: Path,
    run_id: int,
    run_seed: int,
    base_dir: Path,
) -> ExperimentRecorder | None:
    recording_cfg = config.get("recording", {})
    enabled = bool(recording_cfg.get("enabled", True))
    if not enabled:
        return None

    recorder = ExperimentRecorder(
        run_dir=run_dir,
        enabled=True,
        record_user_trace=bool(recording_cfg.get("record_user_trace", True)),
        max_content_chars=int(recording_cfg.get("max_content_chars", 180)),
        flush_every_round=bool(recording_cfg.get("flush_every_round", True)),
        save_plots=bool(recording_cfg.get("save_plots", True)),
        plot_profile=str(recording_cfg.get("plot_profile", "concise")),
        pretty_json_output=bool(recording_cfg.get("pretty_json_output", True)),
        log_pruning_config=dict(recording_cfg.get("log_pruning", {})),
    )
    recorder.save_run_metadata(
        {
            "experiment_name": config.get("experiment_name", "unnamed_experiment"),
            "run_id": run_id,
            "seed": run_seed,
            "base_dir": str(base_dir),
        }
    )
    return recorder


def create_experiment_output_dir(config: dict, base_dir: Path) -> Path:
    recording_cfg = config.get("recording", {})
    output_root = _resolve_path(base_dir, recording_cfg.get("output_root", "output"))
    if output_root is None:
        output_root = (base_dir / "output").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = str(config.get("experiment_name", "exp")).replace(" ", "_")
    exp_tag = str(recording_cfg.get("experiment_tag", f"exp_{ts}_{exp_name}"))
    exp_dir = output_root / exp_tag
    exp_dir.mkdir(parents=True, exist_ok=True)

    (exp_dir / "config_snapshot.json").write_text(
        json.dumps({k: v for k, v in config.items() if k != "_base_dir"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return exp_dir


def run_once(config: dict, run_seed: int, base_dir: Path, run_id: int, run_dir: Path) -> dict:
    random.seed(run_seed)

    user_profiles = build_user_profiles(config, base_dir=base_dir, run_seed=run_seed)
    users = build_user_states(user_profiles)
    user_ids = [profile.user_id for profile in user_profiles]

    network = build_or_load_network(config=config, user_ids=user_ids, run_seed=run_seed, base_dir=base_dir)

    events, initial_contents = build_events_and_contents(config, base_dir=base_dir, run_seed=run_seed)
    sim_state = SimulationState(network=network, users=users, events=events)
    initial_release_cfg = (
        config.get("simulation", {}).get("initial_content_release", {})
        if isinstance(config.get("simulation", {}).get("initial_content_release", {}), dict)
        else {}
    )
    attach_initial_contents(
        sim_state,
        initial_contents=initial_contents,
        run_seed=run_seed,
        release_cfg=initial_release_cfg,
    )
    event_source_cfg = config.get("event_source", {}) if isinstance(config.get("event_source", {}), dict) else {}

    if not initial_contents:
        seed_count = int(event_source_cfg.get("seed_count", config.get("event", {}).get("seed_count", 3)))
        create_initial_rumor_seeds(sim_state, seed_count=seed_count)

    llm_client = build_llm_client(config, base_dir=base_dir, run_seed=run_seed)
    user_model_cfg = dict(config.get("user_model", {}))
    llm_user_sim_cfg = dict(config.get("llm_user_simulation", {}))
    action_model_cfg = dict(config.get("action_model", {}))
    blind_user_mode = bool(llm_user_sim_cfg.get("blind_user_mode", True))
    if "blind_user_mode" not in action_model_cfg:
        action_model_cfg["blind_user_mode"] = blind_user_mode
    if "use_truth_label_in_policy" not in action_model_cfg:
        action_model_cfg["use_truth_label_in_policy"] = False
    agents = {
        user_id: RumorAgent(
            user_state=state,
            llm_client=llm_client,
            attention_module=AttentionModule(
                ensure_rumor_in_selection=bool(user_model_cfg.get("ensure_rumor_in_attention", False)),
                min_rumor_items_in_selection=int(user_model_cfg.get("min_rumor_items_in_attention", 1)),
                ensure_non_fake_in_selection=bool(user_model_cfg.get("ensure_non_fake_event_in_attention", False)),
                min_non_fake_items_in_selection=int(user_model_cfg.get("min_non_fake_items_in_attention", 1)),
                max_fake_items_in_selection=int(user_model_cfg.get("max_fake_items_in_attention", -1)),
                rumor_priority_boost=float(user_model_cfg.get("rumor_priority_boost", 0.0)),
                intervention_priority_boost=float(user_model_cfg.get("intervention_priority_boost", 0.0)),
                event_repeat_penalty=float(user_model_cfg.get("event_repeat_penalty", 0.0)),
            ),
            sharing_policy=SharingPolicy.from_config(action_model_cfg),
            llm_user_simulation=llm_user_sim_cfg,
        )
        for user_id, state in users.items()
    }
    intervention_strategy = build_intervention_strategy(config, llm_client=llm_client)
    intervention_cfg = config.get("intervention", {})
    intervention_activation_cfg = (
        intervention_cfg.get("activation", {})
        if isinstance(intervention_cfg.get("activation", {}), dict)
        else {}
    )
    simulation_cfg = config.get("simulation", {})
    recording_cfg = config.get("recording", {})
    metrics_collector = MetricsCollector(keep_history=bool(recording_cfg.get("keep_metrics_history", True)))
    recorder = build_recorder(config=config, run_dir=run_dir, run_id=run_id, run_seed=run_seed, base_dir=base_dir)

    feed_ranking_cfg = simulation_cfg.get("feed_ranking", {})
    feed_fallback_cfg = (
        simulation_cfg.get("feed_fallback", {})
        if isinstance(simulation_cfg.get("feed_fallback", {}), dict)
        else {}
    )
    feed_event_allocator_cfg = (
        simulation_cfg.get("feed_event_allocator", {})
        if isinstance(simulation_cfg.get("feed_event_allocator", {}), dict)
        else {}
    )
    feed_builder = FeedBuilder(
        window=int(simulation_cfg.get("feed_window", 2)),
        seen_decay_tau=float(feed_ranking_cfg.get("seen_decay_tau", 5.0)),
        seen_min_weight=float(feed_ranking_cfg.get("seen_min_weight", 0.05)),
        seen_max_weight=float(feed_ranking_cfg.get("seen_max_weight", 0.85)),
        unseen_weight=float(feed_ranking_cfg.get("unseen_weight", 1.0)),
        enable_empty_feed_fallback=bool(feed_fallback_cfg.get("enabled", True)),
        empty_feed_fallback_allow_global=bool(feed_fallback_cfg.get("allow_global", True)),
        enable_rumor_candidate_fallback=bool(feed_fallback_cfg.get("rumor_candidate_fallback", False)),
        rumor_candidate_fallback_count=int(feed_fallback_cfg.get("rumor_candidate_fallback_count", 2)),
        enable_event_allocator=bool(feed_event_allocator_cfg.get("enabled", False)),
        event_allocator_temperature=float(feed_event_allocator_cfg.get("temperature", 0.9)),
        event_allocator_pool_multiplier=float(feed_event_allocator_cfg.get("pool_multiplier", 2.0)),
        event_allocator_max_events=int(feed_event_allocator_cfg.get("max_events", 6)),
        event_allocator_social_weight=float(feed_event_allocator_cfg.get("social_weight", 0.7)),
        event_allocator_popularity_weight=float(feed_event_allocator_cfg.get("popularity_weight", 1.0)),
        event_allocator_novelty_weight=float(feed_event_allocator_cfg.get("novelty_weight", 0.5)),
        event_allocator_fatigue_weight=float(feed_event_allocator_cfg.get("fatigue_weight", 0.3)),
        event_allocator_early_diversity_rounds=int(feed_event_allocator_cfg.get("early_diversity_rounds", 0)),
        event_allocator_early_event_cap=int(feed_event_allocator_cfg.get("early_event_cap", 1)),
    )

    decision_workers = int(
        simulation_cfg.get(
            "decision_workers",
            config.get("llm", {}).get("max_concurrency", 1),
        )
    )
    runtime_monitor_cfg = (
        simulation_cfg.get("runtime_monitor", {})
        if isinstance(simulation_cfg.get("runtime_monitor", {}), dict)
        else {}
    )

    engine = SimulationEngine(
        sim_state=sim_state,
        agents=agents,
        intervention_strategy=intervention_strategy,
        metrics_collector=metrics_collector,
        scheduler=build_scheduler(config),
        feed_builder=feed_builder,
        max_feed_size=int(simulation_cfg.get("max_feed_size", 20)),
        intervention_cost_per_post=float(intervention_cfg_cost(config)),
        intervention_start_timestep=int(intervention_activation_cfg.get("start_timestep", 0)),
        intervention_min_fake_shares=int(intervention_activation_cfg.get("min_fake_shares", 0)),
        intervention_deactivate_fake_misbelief_below=(
            float(intervention_activation_cfg.get("deactivate_fake_misbelief_below"))
            if intervention_activation_cfg.get("deactivate_fake_misbelief_below", None) is not None
            else None
        ),
        intervention_deactivate_min_exposed_users=int(
            intervention_activation_cfg.get("deactivate_min_exposed_users", 0)
        ),
        intervention_force_targets_active_when_posting=bool(
            intervention_activation_cfg.get("force_targets_active_when_posting", False)
        ),
        decision_workers=decision_workers,
        recorder=recorder,
        show_progress_bar=bool(runtime_monitor_cfg.get("show_progress_bar", True)),
        show_round_summary=bool(runtime_monitor_cfg.get("show_round_summary", True)),
        progress_mininterval=float(runtime_monitor_cfg.get("progress_mininterval", 0.2)),
        share_relabel_neutral_band=float(action_model_cfg.get("share_relabel_neutral_band", 0.0)),
    )
    return engine.run(T=int(simulation_cfg.get("T", 30)))


def intervention_cfg_cost(config: dict) -> float:
    intervention_cfg = config.get("intervention", {})
    return float(intervention_cfg.get("cost_per_post", 1.0))


def build_scheduler(config: dict) -> Scheduler:
    simulation_cfg = config.get("simulation", {})
    activation_cfg = simulation_cfg.get("activation", {}) if isinstance(simulation_cfg.get("activation", {}), dict) else {}
    return Scheduler(
        min_active_ratio=float(activation_cfg.get("min_active_ratio", 0.0)),
        target_active_ratio=activation_cfg.get("target_active_ratio", None),
        max_active_ratio=float(activation_cfg.get("max_active_ratio", 1.0)),
        inactivity_boost_per_round=float(activation_cfg.get("inactivity_boost_per_round", 0.02)),
        inactivity_boost_cap=float(activation_cfg.get("inactivity_boost_cap", 0.25)),
        seed=int(simulation_cfg.get("seed", 2026)),
    )


def run_experiment(config: dict) -> dict:
    simulation_cfg = config.get("simulation", {})
    n_runs = int(simulation_cfg.get("n_runs", 1))
    base_seed = int(simulation_cfg.get("seed", 2026))

    base_dir = Path(config.get("_base_dir", ".")).resolve()
    exp_dir = create_experiment_output_dir(config=config, base_dir=base_dir)

    run_results = []
    for i in range(n_runs):
        run_dir = exp_dir / f"run_{i:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result = run_once(
            config,
            base_seed + i,
            base_dir=base_dir,
            run_id=i,
            run_dir=run_dir,
        )
        run_results.append(result)

    summaries = [result.get("summary", {}) for result in run_results]

    aggregate = {
        "avg_final_misbelief_ratio": mean([float(s.get("final_misbelief_ratio", 0.0)) for s in summaries]) if summaries else 0.0,
        "avg_final_platform_trust": mean([float(s.get("final_platform_trust", 0.0)) for s in summaries]) if summaries else 0.0,
        "avg_final_total_shares": mean([float(s.get("final_total_shares", 0.0)) for s in summaries]) if summaries else 0.0,
        "avg_final_intervention_cost": mean([float(s.get("final_intervention_cost", 0.0)) for s in summaries]) if summaries else 0.0,
    }

    return {
        "experiment_name": config.get("experiment_name", "unnamed_experiment"),
        "n_runs": n_runs,
        "output_dir": str(exp_dir),
        "aggregate": aggregate,
        "runs": run_results,
    }


def run_experiment_from_file(config_path: str | Path) -> dict:
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    base_dir = config_file.parent
    if base_dir.name.lower() == "configs":
        base_dir = base_dir.parent
    config["_base_dir"] = str(base_dir)
    return run_experiment(config)


def build_terminal_summary(result: dict[str, Any]) -> dict[str, Any]:
    runs = list(result.get("runs", []))
    run_summaries: list[dict[str, Any]] = []
    for idx, run_result in enumerate(runs):
        summary = dict(run_result.get("summary", {}))
        run_summaries.append(
            {
                "run_id": idx,
                "final_misbelief_ratio": summary.get("final_misbelief_ratio", 0.0),
                "misbelief_auc": summary.get("misbelief_auc", 0.0),
                "peak_misbelief_ratio": summary.get("peak_misbelief_ratio", 0.0),
                "peak_misbelief_timestep": summary.get("peak_misbelief_timestep", 0),
                "final_intervention_cost": summary.get("final_intervention_cost", 0.0),
                "efficiency_misbelief_auc_per_cost": summary.get("efficiency_misbelief_auc_per_cost", None),
                "final_rumor_exposure_rate": summary.get("final_rumor_exposure_rate", 0.0),
                "final_debunk_exposure_rate": summary.get("final_debunk_exposure_rate", 0.0),
                "final_normal_exposure_rate": summary.get("final_normal_exposure_rate", 0.0),
                "final_empty_feed_rate": summary.get("final_empty_feed_rate", 0.0),
            }
        )

    aggregate = dict(result.get("aggregate", {}))
    aggregate_compact = {
        "avg_final_misbelief_ratio": aggregate.get("avg_final_misbelief_ratio", 0.0),
        "avg_final_intervention_cost": aggregate.get("avg_final_intervention_cost", 0.0),
        "avg_final_total_shares": aggregate.get("avg_final_total_shares", 0.0),
    }

    return {
        "experiment_name": result.get("experiment_name"),
        "n_runs": result.get("n_runs"),
        "output_dir": result.get("output_dir"),
        "aggregate": aggregate_compact,
        "run_summaries": run_summaries,
    }


def print_terminal_summary(summary: dict[str, Any]) -> None:
    experiment_name = str(summary.get("experiment_name", "unnamed_experiment"))
    n_runs = int(summary.get("n_runs", 0) or 0)
    output_dir = str(summary.get("output_dir", ""))
    aggregate = dict(summary.get("aggregate", {}))
    run_summaries = list(summary.get("run_summaries", []))

    print(f"Experiment: {experiment_name}")
    print(f"Runs: {n_runs}")
    print(f"Output directory: {output_dir}")
    print(
        "Aggregate metrics: "
        f"FinalMisbelief(avg)={float(aggregate.get('avg_final_misbelief_ratio', 0.0)):.4f}, "
        f"Cost(avg)={float(aggregate.get('avg_final_intervention_cost', 0.0)):.2f}, "
        f"Shares(avg)={float(aggregate.get('avg_final_total_shares', 0.0)):.1f}"
    )

    if not run_summaries:
        return

    print("Per-run key metrics:")
    for row in run_summaries:
        run_id = int(row.get("run_id", -1))
        final_misbelief = float(row.get("final_misbelief_ratio", 0.0))
        misbelief_auc = float(row.get("misbelief_auc", 0.0))
        peak_misbelief = float(row.get("peak_misbelief_ratio", 0.0))
        peak_t = int(row.get("peak_misbelief_timestep", 0) or 0)
        final_cost = float(row.get("final_intervention_cost", 0.0))
        rumor_exposure = float(row.get("final_rumor_exposure_rate", 0.0))
        debunk_exposure = float(row.get("final_debunk_exposure_rate", 0.0))
        normal_exposure = float(row.get("final_normal_exposure_rate", 0.0))
        empty_feed = float(row.get("final_empty_feed_rate", 0.0))
        efficiency = row.get("efficiency_misbelief_auc_per_cost", None)
        efficiency_text = "NA" if efficiency is None else f"{float(efficiency):.6f}"
        print(
            f"- run={run_id} | final_misbelief={final_misbelief:.4f} | auc={misbelief_auc:.4f} "
            f"| peak={peak_misbelief:.4f}@t{peak_t} | cost={final_cost:.2f} "
            f"| exposure(rumor/debunk/normal)={rumor_exposure:.3f}/{debunk_exposure:.3f}/{normal_exposure:.3f} "
            f"| empty_feed={empty_feed:.3f} | efficiency={efficiency_text}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rumor simulation experiments.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config file")
    parser.add_argument(
        "--full-output",
        action="store_true",
        help="Print full experiment result JSON (including complete run details).",
    )
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help="Print compact summary in JSON format instead of plain text.",
    )
    args = parser.parse_args()

    result = run_experiment_from_file(args.config)
    if args.full_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    summary = build_terminal_summary(result)
    if args.summary_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    print_terminal_summary(summary)


if __name__ == "__main__":
    main()
