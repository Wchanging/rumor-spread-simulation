from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _series_auc(history: list[dict[str, Any]], key: str) -> float:
    if not history:
        return 0.0
    auc = 0.0
    for idx, current in enumerate(history):
        y = _safe_float(current.get(key, 0.0))
        if idx + 1 < len(history):
            t0 = _safe_float(current.get("timestep", idx), float(idx))
            t1 = _safe_float(history[idx + 1].get("timestep", idx + 1), float(idx + 1))
            dt = max(0.0, t1 - t0)
        else:
            dt = 1.0
        auc += y * dt
    return auc


def _peak_with_t(history: list[dict[str, Any]], key: str) -> tuple[float, int]:
    best_value = float("-inf")
    best_t = 0
    for idx, row in enumerate(history):
        value = _safe_float(row.get(key, 0.0))
        if value > best_value:
            best_value = value
            best_t = int(_safe_float(row.get("timestep", idx), float(idx)))
    if best_value == float("-inf"):
        return 0.0, 0
    return best_value, best_t


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _discover_runs(output_root: Path) -> list[Path]:
    runs: list[Path] = []
    for exp_dir in sorted(output_root.glob("exp_*")):
        if not exp_dir.is_dir():
            continue
        for run_dir in sorted(exp_dir.glob("run_*")):
            if (run_dir / "final_summary.json").exists():
                runs.append(run_dir)
    return runs


def _infer_exposure_mode(cfg: dict[str, Any], experiment_name: str) -> str:
    if "rq3_hard" in experiment_name:
        return "hard"
    if "rq3_soft" in experiment_name:
        return "soft"
    if "rq3_loose" in experiment_name:
        return "loose"

    user_model = cfg.get("user_model", {}) if isinstance(cfg, dict) else {}
    ensure_rumor = bool(user_model.get("ensure_rumor_in_attention", False))
    rumor_boost = _safe_float(user_model.get("rumor_priority_boost", 0.0))
    min_rumor = int(_safe_float(user_model.get("min_rumor_items_in_attention", 0), 0))

    if ensure_rumor:
        return "hard"
    if rumor_boost > 0.0 or min_rumor > 0:
        return "soft"
    return "loose"


def _load_config(configs_dir: Path, experiment_name: str) -> dict[str, Any]:
    path = configs_dir / f"{experiment_name}.json"
    loaded = _load_json(path)
    if isinstance(loaded, dict):
        return loaded
    return {}


def _build_run_row(run_dir: Path, configs_dir: Path) -> dict[str, Any] | None:
    final_summary_raw = _load_json(run_dir / "final_summary.json")
    if not isinstance(final_summary_raw, dict):
        return None

    run_meta_raw = _load_json(run_dir / "run_meta.json")
    run_meta = run_meta_raw if isinstance(run_meta_raw, dict) else {}

    summary = final_summary_raw.get("summary", {}) if isinstance(final_summary_raw.get("summary", {}), dict) else {}
    llm_stats = final_summary_raw.get("llm_stats", {}) if isinstance(final_summary_raw.get("llm_stats", {}), dict) else {}

    experiment_name = str(run_meta.get("experiment_name", run_dir.parent.name))
    cfg = _load_config(configs_dir, experiment_name)
    network_type = str(cfg.get("network", {}).get("type", "unknown")) if isinstance(cfg.get("network", {}), dict) else "unknown"
    strategy = (
        str(cfg.get("intervention", {}).get("strategy", "unknown"))
        if isinstance(cfg.get("intervention", {}), dict)
        else "unknown"
    )
    exposure_mode = _infer_exposure_mode(cfg, experiment_name)

    history_raw = _load_json(run_dir / "metrics" / "metrics_history.json")
    history: list[dict[str, Any]] = history_raw if isinstance(history_raw, list) else []

    misbelief_auc = _safe_float(summary.get("misbelief_auc", None), default=-1.0)
    if misbelief_auc < 0:
        misbelief_auc = _series_auc(history, "misbelief_ratio")

    fake_exposed_auc = _safe_float(summary.get("fake_event_misbelief_auc_among_exposed_users", None), default=-1.0)
    if fake_exposed_auc < 0:
        fake_exposed_auc = _series_auc(history, "fake_event_misbelief_ratio_among_exposed_users")

    rumor_exposure_auc = _series_auc(history, "rumor_exposure_rate")
    debunk_exposure_auc = _series_auc(history, "debunk_exposure_rate")
    normal_exposure_auc = _series_auc(history, "normal_exposure_rate")

    peak_misbelief = _safe_float(summary.get("peak_misbelief_ratio", None), default=-1.0)
    peak_misbelief_t = int(_safe_float(summary.get("peak_misbelief_timestep", 0), 0))
    if peak_misbelief < 0:
        peak_misbelief, peak_misbelief_t = _peak_with_t(history, "misbelief_ratio")

    peak_rumor_exposure, peak_rumor_exposure_t = _peak_with_t(history, "rumor_exposure_rate")

    peak_fake_exposed = _safe_float(
        summary.get("peak_fake_event_misbelief_ratio_among_exposed_users", None),
        default=-1.0,
    )
    peak_fake_exposed_t = int(
        _safe_float(summary.get("peak_fake_event_misbelief_timestep_among_exposed_users", 0), 0)
    )
    if peak_fake_exposed < 0:
        peak_fake_exposed, peak_fake_exposed_t = _peak_with_t(history, "fake_event_misbelief_ratio_among_exposed_users")

    initial_trust = _safe_float(summary.get("initial_platform_trust", None), default=-1.0)
    final_trust = _safe_float(summary.get("final_platform_trust", 0.0))
    if initial_trust < 0:
        initial_trust = _safe_float(history[0].get("platform_trust_mean", final_trust), final_trust) if history else final_trust
    trust_delta = _safe_float(summary.get("platform_trust_delta", final_trust - initial_trust), final_trust - initial_trust)

    final_cost = _safe_float(summary.get("final_intervention_cost", 0.0))
    eff_auc_per_cost = (misbelief_auc / final_cost) if final_cost > 0 else None

    return {
        "output_exp_dir": run_dir.parent.name,
        "run_dir": str(run_dir),
        "experiment_name": experiment_name,
        "run_id": int(_safe_float(run_meta.get("run_id", 0), 0)),
        "seed": int(_safe_float(run_meta.get("seed", 0), 0)),
        "network_type": network_type,
        "strategy": strategy,
        "exposure_mode": exposure_mode,
        "final_misbelief_ratio": _safe_float(summary.get("final_misbelief_ratio", 0.0)),
        "final_fake_event_misbelief_ratio_among_exposed_users": _safe_float(
            summary.get("final_fake_event_misbelief_ratio_among_exposed_users", 0.0)
        ),
        "final_users_exposed_to_any_fake_ratio": _safe_float(summary.get("final_users_exposed_to_any_fake_ratio", 0.0)),
        "misbelief_auc": misbelief_auc,
        "fake_event_misbelief_auc_among_exposed_users": fake_exposed_auc,
        "peak_misbelief_ratio": peak_misbelief,
        "peak_misbelief_timestep": peak_misbelief_t,
        "peak_rumor_exposure_rate": peak_rumor_exposure,
        "peak_rumor_exposure_timestep": peak_rumor_exposure_t,
        "peak_fake_event_misbelief_ratio_among_exposed_users": peak_fake_exposed,
        "peak_fake_event_misbelief_timestep_among_exposed_users": peak_fake_exposed_t,
        "initial_platform_trust": initial_trust,
        "final_platform_trust": final_trust,
        "platform_trust_delta": trust_delta,
        "final_intervention_cost": final_cost,
        "efficiency_misbelief_auc_per_cost": eff_auc_per_cost,
        "final_rumor_exposure_rate": _safe_float(summary.get("final_rumor_exposure_rate", 0.0)),
        "final_debunk_exposure_rate": _safe_float(summary.get("final_debunk_exposure_rate", 0.0)),
        "final_normal_exposure_rate": _safe_float(summary.get("final_normal_exposure_rate", 0.0)),
        "final_empty_feed_rate": _safe_float(summary.get("final_empty_feed_rate", 0.0)),
        "rumor_exposure_auc": rumor_exposure_auc,
        "debunk_exposure_auc": debunk_exposure_auc,
        "normal_exposure_auc": normal_exposure_auc,
        "llm_total_calls": int(_safe_float(llm_stats.get("total_calls", 0), 0)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def _aggregate_by_experiment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["experiment_name"]), []).append(row)

    keys_to_agg = [
        "final_misbelief_ratio",
        "misbelief_auc",
        "peak_misbelief_ratio",
        "peak_misbelief_timestep",
        "peak_rumor_exposure_rate",
        "peak_rumor_exposure_timestep",
        "final_fake_event_misbelief_ratio_among_exposed_users",
        "final_users_exposed_to_any_fake_ratio",
        "platform_trust_delta",
        "final_intervention_cost",
        "efficiency_misbelief_auc_per_cost",
        "final_rumor_exposure_rate",
        "final_debunk_exposure_rate",
        "final_normal_exposure_rate",
        "final_empty_feed_rate",
        "rumor_exposure_auc",
        "debunk_exposure_auc",
        "normal_exposure_auc",
    ]

    outputs: list[dict[str, Any]] = []
    for experiment_name, items in sorted(grouped.items()):
        first = items[0]
        out: dict[str, Any] = {
            "experiment_name": experiment_name,
            "n_runs": len(items),
            "network_type": first.get("network_type", "unknown"),
            "strategy": first.get("strategy", "unknown"),
            "exposure_mode": first.get("exposure_mode", "unknown"),
        }

        for key in keys_to_agg:
            vals = [float(v) for v in [item.get(key) for item in items] if v is not None]
            mean, std = _mean_std(vals)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std

        outputs.append(out)

    return outputs


def _build_topline_table(agg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in agg_rows
        if any(tag in str(row.get("experiment_name", "")).lower() for tag in ("rq1", "rq2", "rq3"))
    ]
    if not selected:
        selected = list(agg_rows)

    topline: list[dict[str, Any]] = []

    def _rq_group(name: str) -> str:
        lower = name.lower()
        if "rq1" in lower:
            return "RQ1"
        if "rq2" in lower:
            return "RQ2"
        if "rq3" in lower:
            return "RQ3"
        return "OTHER"

    selected = sorted(selected, key=lambda row: (_rq_group(str(row.get("experiment_name", ""))), str(row.get("experiment_name", ""))))

    for row in selected:
        exp_name = str(row.get("experiment_name", ""))
        topline.append(
            {
                "RQ": _rq_group(exp_name),
                "Experiment": row.get("experiment_name"),
                "Network": row.get("network_type"),
                "Strategy": row.get("strategy"),
                "FinalMisbelief": f"{_safe_float(row.get('final_misbelief_ratio_mean')):.4f}±{_safe_float(row.get('final_misbelief_ratio_std')):.4f}",
                "MisbeliefAUC": f"{_safe_float(row.get('misbelief_auc_mean')):.4f}±{_safe_float(row.get('misbelief_auc_std')):.4f}",
                "PeakMisbelief": f"{_safe_float(row.get('peak_misbelief_ratio_mean')):.4f}±{_safe_float(row.get('peak_misbelief_ratio_std')):.4f}",
                "PeakTime": f"{_safe_float(row.get('peak_misbelief_timestep_mean')):.2f}±{_safe_float(row.get('peak_misbelief_timestep_std')):.2f}",
                "TrustDelta": f"{_safe_float(row.get('platform_trust_delta_mean')):.4f}±{_safe_float(row.get('platform_trust_delta_std')):.4f}",
                "Cost": f"{_safe_float(row.get('final_intervention_cost_mean')):.4f}±{_safe_float(row.get('final_intervention_cost_std')):.4f}",
            }
        )
    return topline


def _build_rq1_rq2_key_metrics_table(agg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in agg_rows:
        exp_name = str(row.get("experiment_name", ""))
        lower = exp_name.lower()
        if ("rq1" not in lower) and ("rq2" not in lower):
            continue

        rq = "RQ1" if "rq1" in lower else "RQ2"
        rows.append(
            {
                "RQ": rq,
                "experiment_name": exp_name,
                "network_type": row.get("network_type", "unknown"),
                "strategy": row.get("strategy", "unknown"),
                "n_runs": int(_safe_float(row.get("n_runs", 0), 0)),
                "final_misbelief_ratio_mean": _safe_float(row.get("final_misbelief_ratio_mean", 0.0)),
                "final_misbelief_ratio_std": _safe_float(row.get("final_misbelief_ratio_std", 0.0)),
                "misbelief_auc_mean": _safe_float(row.get("misbelief_auc_mean", 0.0)),
                "misbelief_auc_std": _safe_float(row.get("misbelief_auc_std", 0.0)),
                "peak_misbelief_ratio_mean": _safe_float(row.get("peak_misbelief_ratio_mean", 0.0)),
                "peak_misbelief_ratio_std": _safe_float(row.get("peak_misbelief_ratio_std", 0.0)),
                "peak_misbelief_timestep_mean": _safe_float(row.get("peak_misbelief_timestep_mean", 0.0)),
                "peak_misbelief_timestep_std": _safe_float(row.get("peak_misbelief_timestep_std", 0.0)),
                "platform_trust_delta_mean": _safe_float(row.get("platform_trust_delta_mean", 0.0)),
                "platform_trust_delta_std": _safe_float(row.get("platform_trust_delta_std", 0.0)),
                "final_intervention_cost_mean": _safe_float(row.get("final_intervention_cost_mean", 0.0)),
                "final_intervention_cost_std": _safe_float(row.get("final_intervention_cost_std", 0.0)),
                "final_rumor_exposure_rate_mean": _safe_float(row.get("final_rumor_exposure_rate_mean", 0.0)),
                "final_rumor_exposure_rate_std": _safe_float(row.get("final_rumor_exposure_rate_std", 0.0)),
                "final_debunk_exposure_rate_mean": _safe_float(row.get("final_debunk_exposure_rate_mean", 0.0)),
                "final_debunk_exposure_rate_std": _safe_float(row.get("final_debunk_exposure_rate_std", 0.0)),
                "final_normal_exposure_rate_mean": _safe_float(row.get("final_normal_exposure_rate_mean", 0.0)),
                "final_normal_exposure_rate_std": _safe_float(row.get("final_normal_exposure_rate_std", 0.0)),
            }
        )

    return sorted(rows, key=lambda x: (str(x["RQ"]), str(x["experiment_name"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="从 output 目录聚合论文主表所需指标（AUC/峰值/信任变化等）。")
    parser.add_argument("--output-root", type=str, default="output", help="实验输出目录")
    parser.add_argument("--configs-dir", type=str, default="configs", help="配置目录")
    parser.add_argument("--dest-dir", type=str, default="output/paper_tables", help="聚合结果输出目录")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_root = (project_root / args.output_root).resolve()
    configs_dir = (project_root / args.configs_dir).resolve()
    dest_dir = (project_root / args.dest_dir).resolve()

    run_dirs = _discover_runs(output_root)
    run_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        row = _build_run_row(run_dir, configs_dir)
        if row is not None:
            run_rows.append(row)

    if not run_rows:
        raise RuntimeError(f"在 {output_root} 下未发现可用运行结果（run_*/final_summary.json）。")

    run_fieldnames = [
        "output_exp_dir",
        "run_dir",
        "experiment_name",
        "run_id",
        "seed",
        "network_type",
        "strategy",
        "exposure_mode",
        "final_misbelief_ratio",
        "misbelief_auc",
        "peak_misbelief_ratio",
        "peak_misbelief_timestep",
        "peak_rumor_exposure_rate",
        "peak_rumor_exposure_timestep",
        "final_fake_event_misbelief_ratio_among_exposed_users",
        "final_users_exposed_to_any_fake_ratio",
        "platform_trust_delta",
        "final_intervention_cost",
        "efficiency_misbelief_auc_per_cost",
        "final_rumor_exposure_rate",
        "final_debunk_exposure_rate",
        "final_normal_exposure_rate",
        "final_empty_feed_rate",
        "rumor_exposure_auc",
        "debunk_exposure_auc",
        "normal_exposure_auc",
        "llm_total_calls",
    ]
    _write_csv(dest_dir / "run_level_metrics.csv", run_rows, run_fieldnames)

    agg_rows = _aggregate_by_experiment(run_rows)
    agg_fieldnames = [
        "experiment_name",
        "n_runs",
        "network_type",
        "strategy",
        "exposure_mode",
        "final_misbelief_ratio_mean",
        "final_misbelief_ratio_std",
        "misbelief_auc_mean",
        "misbelief_auc_std",
        "peak_misbelief_ratio_mean",
        "peak_misbelief_ratio_std",
        "peak_misbelief_timestep_mean",
        "peak_misbelief_timestep_std",
        "peak_rumor_exposure_rate_mean",
        "peak_rumor_exposure_rate_std",
        "peak_rumor_exposure_timestep_mean",
        "peak_rumor_exposure_timestep_std",
        "final_fake_event_misbelief_ratio_among_exposed_users_mean",
        "final_fake_event_misbelief_ratio_among_exposed_users_std",
        "final_users_exposed_to_any_fake_ratio_mean",
        "final_users_exposed_to_any_fake_ratio_std",
        "platform_trust_delta_mean",
        "platform_trust_delta_std",
        "final_intervention_cost_mean",
        "final_intervention_cost_std",
        "efficiency_misbelief_auc_per_cost_mean",
        "efficiency_misbelief_auc_per_cost_std",
        "final_rumor_exposure_rate_mean",
        "final_rumor_exposure_rate_std",
        "final_debunk_exposure_rate_mean",
        "final_debunk_exposure_rate_std",
        "final_normal_exposure_rate_mean",
        "final_normal_exposure_rate_std",
        "final_empty_feed_rate_mean",
        "final_empty_feed_rate_std",
        "rumor_exposure_auc_mean",
        "rumor_exposure_auc_std",
        "debunk_exposure_auc_mean",
        "debunk_exposure_auc_std",
        "normal_exposure_auc_mean",
        "normal_exposure_auc_std",
    ]
    _write_csv(dest_dir / "experiment_aggregates.csv", agg_rows, agg_fieldnames)

    topline = _build_topline_table(agg_rows)
    topline_fields = [
        "RQ",
        "Experiment",
        "Network",
        "Strategy",
        "FinalMisbelief",
        "MisbeliefAUC",
        "PeakMisbelief",
        "PeakTime",
        "TrustDelta",
        "Cost",
    ]
    _write_csv(dest_dir / "table_main_topline.csv", topline, topline_fields)

    rq1_rq2_key_rows = _build_rq1_rq2_key_metrics_table(agg_rows)
    rq1_rq2_key_fields = [
        "RQ",
        "experiment_name",
        "network_type",
        "strategy",
        "n_runs",
        "final_misbelief_ratio_mean",
        "final_misbelief_ratio_std",
        "misbelief_auc_mean",
        "misbelief_auc_std",
        "peak_misbelief_ratio_mean",
        "peak_misbelief_ratio_std",
        "peak_misbelief_timestep_mean",
        "peak_misbelief_timestep_std",
        "platform_trust_delta_mean",
        "platform_trust_delta_std",
        "final_intervention_cost_mean",
        "final_intervention_cost_std",
        "final_rumor_exposure_rate_mean",
        "final_rumor_exposure_rate_std",
        "final_debunk_exposure_rate_mean",
        "final_debunk_exposure_rate_std",
        "final_normal_exposure_rate_mean",
        "final_normal_exposure_rate_std",
    ]
    _write_csv(dest_dir / "table_rq1_rq2_key_metrics.csv", rq1_rq2_key_rows, rq1_rq2_key_fields)

    print(json.dumps({
        "output_root": str(output_root),
        "dest_dir": str(dest_dir),
        "run_count": len(run_rows),
        "experiment_count": len(agg_rows),
        "files": [
            str(dest_dir / "run_level_metrics.csv"),
            str(dest_dir / "experiment_aggregates.csv"),
            str(dest_dir / "table_main_topline.csv"),
            str(dest_dir / "table_rq1_rq2_key_metrics.csv"),
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
