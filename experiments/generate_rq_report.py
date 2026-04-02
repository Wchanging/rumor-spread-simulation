from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


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


def _series_duration(history: list[dict[str, Any]]) -> float:
    if not history:
        return 0.0
    duration = 0.0
    for idx, current in enumerate(history):
        if idx + 1 < len(history):
            t0 = _safe_float(current.get("timestep", idx), float(idx))
            t1 = _safe_float(history[idx + 1].get("timestep", idx + 1), float(idx + 1))
            dt = max(0.0, t1 - t0)
        else:
            dt = 1.0
        duration += dt
    return duration


def _peak_with_t(history: list[dict[str, Any]], key: str) -> tuple[float, int]:
    best_v = float("-inf")
    best_t = 0
    for idx, row in enumerate(history):
        value = _safe_float(row.get(key, 0.0))
        if value > best_v:
            best_v = value
            best_t = int(_safe_float(row.get("timestep", idx), float(idx)))
    if best_v == float("-inf"):
        return 0.0, 0
    return best_v, best_t


def _canonical_experiment_name(output_exp_dir_name: str) -> str:
    marker = "exp_rq"
    idx = output_exp_dir_name.find(marker)
    if idx >= 0:
        return output_exp_dir_name[idx:]
    return output_exp_dir_name


def _resolve_path(project_root: Path, path_text: str) -> Path:
    p = Path(path_text)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def _discover_runs_from_source(source: Path) -> list[Path]:
    if source.is_dir() and source.name.startswith("run_"):
        return [source]

    if source.is_dir():
        run_dirs = sorted([p for p in source.glob("run_*") if p.is_dir()])
        if run_dirs:
            return run_dirs

    return []


def _infer_experiment_name(run_dir: Path) -> str:
    run_meta = _load_json(run_dir / "run_meta.json")
    if isinstance(run_meta, dict):
        exp_name = str(run_meta.get("experiment_name", "")).strip()
        if exp_name:
            return exp_name
    return _canonical_experiment_name(run_dir.parent.name)


def _load_config(configs_dir: Path, experiment_name: str) -> dict[str, Any]:
    cfg = _load_json(configs_dir / f"{experiment_name}.json")
    if isinstance(cfg, dict):
        return cfg
    return {}


def _history_from_run(run_dir: Path) -> list[dict[str, Any]]:
    hist = _load_json(run_dir / "metrics" / "metrics_history.json")
    if isinstance(hist, list):
        return hist
    return []


def _build_run_row(run_dir: Path, configs_dir: Path) -> dict[str, Any] | None:
    history = _history_from_run(run_dir)
    if not history:
        return None

    final_summary_raw = _load_json(run_dir / "final_summary.json")
    summary = {}
    llm_stats = {}
    if isinstance(final_summary_raw, dict):
        summary = final_summary_raw.get("summary", {}) if isinstance(final_summary_raw.get("summary", {}), dict) else {}
        llm_stats = final_summary_raw.get("llm_stats", {}) if isinstance(final_summary_raw.get("llm_stats", {}), dict) else {}

    last = history[-1]
    experiment_name = _infer_experiment_name(run_dir)
    cfg = _load_config(configs_dir, experiment_name)
    network_type = str(cfg.get("network", {}).get("type", "unknown")) if isinstance(cfg.get("network", {}), dict) else "unknown"
    strategy = (
        str(cfg.get("intervention", {}).get("strategy", "unknown"))
        if isinstance(cfg.get("intervention", {}), dict)
        else "unknown"
    )

    misbelief_auc = _safe_float(summary.get("misbelief_auc", -1.0), -1.0)
    if misbelief_auc < 0:
        misbelief_auc = _series_auc(history, "misbelief_ratio")

    peak_misbelief = _safe_float(summary.get("peak_misbelief_ratio", -1.0), -1.0)
    peak_misbelief_t = int(_safe_float(summary.get("peak_misbelief_timestep", 0), 0))
    if peak_misbelief < 0:
        peak_misbelief, peak_misbelief_t = _peak_with_t(history, "misbelief_ratio")

    peak_rumor_exposure, peak_rumor_exposure_t = _peak_with_t(history, "rumor_exposure_rate")

    final_trust = _safe_float(summary.get("final_platform_trust", last.get("platform_trust_mean", 0.0)))
    initial_trust = _safe_float(
        summary.get("initial_platform_trust", history[0].get("platform_trust_mean", final_trust)),
        final_trust,
    )
    trust_delta = _safe_float(summary.get("platform_trust_delta", final_trust - initial_trust), final_trust - initial_trust)

    final_cost = _safe_float(summary.get("final_intervention_cost", last.get("intervention_cost", 0.0)))
    rumor_auc = _series_auc(history, "rumor_exposure_rate")
    debunk_auc = _series_auc(history, "debunk_exposure_rate")
    normal_auc = _series_auc(history, "normal_exposure_rate")
    series_duration = _series_duration(history)
    mean_rumor_exposure_rate = rumor_auc / series_duration if series_duration > 0 else 0.0
    mean_debunk_exposure_rate = debunk_auc / series_duration if series_duration > 0 else 0.0
    mean_normal_exposure_rate = normal_auc / series_duration if series_duration > 0 else 0.0

    run_meta = _load_json(run_dir / "run_meta.json")
    run_meta = run_meta if isinstance(run_meta, dict) else {}

    return {
        "run_dir": str(run_dir),
        "experiment_name": experiment_name,
        "run_id": int(_safe_float(run_meta.get("run_id", 0), 0)),
        "seed": int(_safe_float(run_meta.get("seed", 0), 0)),
        "network_type": network_type,
        "strategy": strategy,
        "final_misbelief_ratio": _safe_float(summary.get("final_misbelief_ratio", last.get("misbelief_ratio", 0.0))),
        "misbelief_auc": misbelief_auc,
        "peak_misbelief_ratio": peak_misbelief,
        "peak_misbelief_timestep": peak_misbelief_t,
        "peak_rumor_exposure_rate": peak_rumor_exposure,
        "peak_rumor_exposure_timestep": peak_rumor_exposure_t,
        "final_fake_event_misbelief_ratio_among_exposed_users": _safe_float(
            summary.get(
                "final_fake_event_misbelief_ratio_among_exposed_users",
                last.get("fake_event_misbelief_ratio_among_exposed_users", 0.0),
            )
        ),
        "final_users_exposed_to_any_fake_ratio": _safe_float(
            summary.get("final_users_exposed_to_any_fake_ratio", last.get("users_exposed_to_any_fake_ratio", 0.0))
        ),
        "platform_trust_delta": trust_delta,
        "final_intervention_cost": final_cost,
        "efficiency_misbelief_auc_per_cost": (misbelief_auc / final_cost) if final_cost > 0 else None,
        "final_rumor_exposure_rate": _safe_float(summary.get("final_rumor_exposure_rate", last.get("rumor_exposure_rate", 0.0))),
        "final_debunk_exposure_rate": _safe_float(summary.get("final_debunk_exposure_rate", last.get("debunk_exposure_rate", 0.0))),
        "final_normal_exposure_rate": _safe_float(summary.get("final_normal_exposure_rate", last.get("normal_exposure_rate", 0.0))),
        "final_empty_feed_rate": _safe_float(summary.get("final_empty_feed_rate", last.get("empty_feed_rate", 0.0))),
        "rumor_exposure_auc": rumor_auc,
        "debunk_exposure_auc": debunk_auc,
        "normal_exposure_auc": normal_auc,
        "mean_rumor_exposure_rate": mean_rumor_exposure_rate,
        "mean_debunk_exposure_rate": mean_debunk_exposure_rate,
        "mean_normal_exposure_rate": mean_normal_exposure_rate,
        "llm_total_calls": int(_safe_float(llm_stats.get("total_calls", 0), 0)),
        "history": history,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def _positive_div(numerator: float, denominator: float) -> float | str:
    if denominator <= 0:
        return ""
    return numerator / denominator


def _aggregate_by_experiment(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["experiment_name"])].append(row)

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
        "mean_rumor_exposure_rate",
        "mean_debunk_exposure_rate",
        "mean_normal_exposure_rate",
    ]

    out_rows: list[dict[str, Any]] = []
    for exp_name, items in sorted(grouped.items()):
        first = items[0]
        out: dict[str, Any] = {
            "experiment_name": exp_name,
            "n_runs": len(items),
            "network_type": first.get("network_type", "unknown"),
            "strategy": first.get("strategy", "unknown"),
        }
        for key in keys_to_agg:
            vals = [float(v) for v in [item.get(key) for item in items] if v is not None]
            mean, std = _mean_std(vals)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
        out_rows.append(out)
    return out_rows


def _plot_rq1_figures(run_rows: list[dict[str, Any]], agg_rows: list[dict[str, Any]], dest_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    files: list[str] = []
    grouped_histories: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for row in run_rows:
        grouped_histories[str(row["experiment_name"])].append(row["history"])

    def _mean_series(histories: list[list[dict[str, Any]]], key: str) -> tuple[list[int], list[float]]:
        if not histories:
            return [], []
        max_len = max(len(h) for h in histories)
        xs = list(range(max_len))
        ys: list[float] = []
        for t in range(max_len):
            vals = [_safe_float(h[t].get(key, 0.0)) for h in histories if t < len(h)]
            ys.append(sum(vals) / max(1, len(vals)))
        return xs, ys

    def _split_rq1_experiment(exp_name: str) -> tuple[str, str] | None:
        if not exp_name.startswith("exp_rq1_"):
            return None
        if exp_name.endswith("_official"):
            return exp_name[len("exp_rq1_"): -len("_official")], "official"
        if exp_name.endswith("_no"):
            return exp_name[len("exp_rq1_"): -len("_no")], "no"
        return None

    def _level_sort_key(level: str) -> tuple[int, int, str]:
        fixed = {"isolated": 0, "moderate": 1, "high": 2}
        if level in fixed:
            return (0, fixed[level], level)
        match = re.search(r"r1n(\d+)", level)
        if match:
            return (1, int(match.group(1)), level)
        return (2, 9999, level)

    def _level_display_name(level: str) -> str:
        match = re.search(r"r1n(\d+)", level)
        if match:
            return f"1R+{int(match.group(1))}N"
        if level == "isolated":
            return "3R+0N"
        if level == "moderate":
            return "3R+3N"
        if level == "high":
            return "3R+6N"
        return level

    discovered: dict[str, dict[str, str]] = defaultdict(dict)
    for row in run_rows:
        exp_name = str(row.get("experiment_name", ""))
        split = _split_rq1_experiment(exp_name)
        if split is None:
            continue
        level, variant = split
        discovered[level][variant] = exp_name

    for row in agg_rows:
        exp_name = str(row.get("experiment_name", ""))
        split = _split_rq1_experiment(exp_name)
        if split is None:
            continue
        level, variant = split
        discovered[level][variant] = exp_name

    setting_map: dict[str, tuple[str, str]] = {}
    for level, pair in discovered.items():
        official_exp = pair.get("official")
        no_exp = pair.get("no")
        if official_exp and no_exp:
            setting_map[level] = (official_exp, no_exp)

    for metric_key, ylabel, name in [
        ("rumor_exposure_rate", "rumor_exposure_rate", "rq1_rumor_exposure_timeseries.png"),
        ("debunk_exposure_rate", "debunk_exposure_rate", "rq1_debunk_exposure_timeseries.png"),
        ("normal_exposure_rate", "background_exposure_share", "rq1_background_exposure_timeseries.png"),
    ]:
        plt.figure(figsize=(9, 4.5))
        has_line = False
        for setting in sorted(setting_map.keys(), key=_level_sort_key):
            official_exp, no_exp = setting_map[setting]
            setting_label = _level_display_name(setting)
            x1, y1 = _mean_series(grouped_histories.get(official_exp, []), metric_key)
            x2, y2 = _mean_series(grouped_histories.get(no_exp, []), metric_key)
            if y1:
                has_line = True
                plt.plot(x1, y1, label=f"{setting_label}-official", linewidth=1.8)
            if y2:
                has_line = True
                plt.plot(x2, y2, label=f"{setting_label}-no", linestyle="--", linewidth=1.3)
        plt.ylabel(ylabel)
        if has_line:
            plt.legend(ncol=2, fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = dest_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=180)
        plt.close()
        files.append(str(path))

    agg_map = {str(row["experiment_name"]): row for row in agg_rows}
    labels: list[str] = []
    auc_no: list[float] = []
    auc_off: list[float] = []
    final_no: list[float] = []
    final_off: list[float] = []
    rumor_mean_no: list[float] = []
    rumor_mean_off: list[float] = []
    debunk_mean_no: list[float] = []
    debunk_mean_off: list[float] = []

    for level in sorted(setting_map.keys(), key=_level_sort_key):
        row_no = agg_map.get(setting_map[level][1])
        row_off = agg_map.get(setting_map[level][0])
        if row_no is None or row_off is None:
            continue
        labels.append(_level_display_name(level))
        auc_no.append(_safe_float(row_no.get("misbelief_auc_mean", 0.0)))
        auc_off.append(_safe_float(row_off.get("misbelief_auc_mean", 0.0)))
        final_no.append(_safe_float(row_no.get("final_misbelief_ratio_mean", 0.0)))
        final_off.append(_safe_float(row_off.get("final_misbelief_ratio_mean", 0.0)))
        rumor_mean_no.append(_safe_float(row_no.get("mean_rumor_exposure_rate_mean", 0.0)))
        rumor_mean_off.append(_safe_float(row_off.get("mean_rumor_exposure_rate_mean", 0.0)))
        debunk_mean_no.append(_safe_float(row_no.get("mean_debunk_exposure_rate_mean", 0.0)))
        debunk_mean_off.append(_safe_float(row_off.get("mean_debunk_exposure_rate_mean", 0.0)))

    if labels:
        x = list(range(len(labels)))
        width = 0.36

        plt.figure(figsize=(8.5, 4.5))
        plt.bar([i - width / 2 for i in x], auc_no, width=width, label="NoIntervention")
        plt.bar([i + width / 2 for i in x], auc_off, width=width, label="Official")
        plt.xticks(x, labels)
        plt.ylabel("Misbelief AUC")
        plt.legend()
        plt.tight_layout()
        p1 = dest_dir / "rq1_misbelief_auc_bar.png"
        plt.savefig(p1, dpi=180)
        plt.close()
        files.append(str(p1))

        plt.figure(figsize=(8.5, 4.5))
        plt.bar([i - width / 2 for i in x], final_no, width=width, label="NoIntervention")
        plt.bar([i + width / 2 for i in x], final_off, width=width, label="Official")
        plt.xticks(x, labels)
        plt.ylabel("Final Misbelief")
        plt.legend()
        plt.tight_layout()
        p2 = dest_dir / "rq1_final_misbelief_bar.png"
        plt.savefig(p2, dpi=180)
        plt.close()
        files.append(str(p2))

        plt.figure(figsize=(8.8, 4.8))
        plt.subplot(1, 2, 1)
        plt.bar([i - width / 2 for i in x], rumor_mean_no, width=width, label="NoIntervention")
        plt.bar([i + width / 2 for i in x], rumor_mean_off, width=width, label="Official")
        plt.xticks(x, labels)
        plt.ylabel("Process Mean Rumor Exposure Rate")
        plt.title("Rumor")
        plt.grid(alpha=0.25, axis="y")

        plt.subplot(1, 2, 2)
        plt.bar([i - width / 2 for i in x], debunk_mean_no, width=width, label="NoIntervention")
        plt.bar([i + width / 2 for i in x], debunk_mean_off, width=width, label="Official")
        plt.xticks(x, labels)
        plt.ylabel("Process Mean Debunk Exposure Rate")
        plt.title("Debunk")
        plt.grid(alpha=0.25, axis="y")
        plt.legend()
        plt.tight_layout()
        p3 = dest_dir / "rq1_process_mean_exposure_rate_bar.png"
        plt.savefig(p3, dpi=180)
        plt.close()
        files.append(str(p3))

    return files


def _plot_rq2_figures(agg_rows: list[dict[str, Any]], dest_dir: Path) -> list[str]:
    import math
    import matplotlib.pyplot as plt

    files: list[str] = []
    agg_map = {str(row["experiment_name"]): row for row in agg_rows}
    networks = ["sw", "sf", "random"]
    strategies = ["no", "official", "topk", "personalized"]

    def _plot(metric_key: str, ylabel: str, filename: str) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
        axes = list(axes) if hasattr(axes, "__iter__") else [axes]
        for idx, network in enumerate(networks):
            ax = axes[idx]
            vals: list[float] = []
            for strategy in strategies:
                row = agg_map.get(f"exp_rq2_{network}_{strategy}")
                vals.append(_safe_float(row.get(metric_key, 0.0)) if row else 0.0)
            x = list(range(len(strategies)))
            ax.bar(x, vals)
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=20, ha="right")
            ax.set_title(f"network={network}")
            ax.grid(alpha=0.25, axis="y")
            if idx == 0:
                ax.set_ylabel(ylabel)
        plt.tight_layout()
        p = dest_dir / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=180)
        plt.close(fig)
        files.append(str(p))

    _plot("misbelief_auc_mean", "Misbelief AUC", "rq2_misbelief_auc_by_network_strategy.png")
    _plot("final_misbelief_ratio_mean", "Final Misbelief", "rq2_final_misbelief_by_network_strategy.png")
    _plot("final_intervention_cost_mean", "Intervention Cost", "rq2_cost_by_network_strategy.png")

    # Publication-friendly dual-axis small multiples:
    # left y-axis: final misbelief (bars), right y-axis: AUC per cost (line with hollow markers).
    display_strategies = [
        ("no", "No"),
        ("official", "Official"),
        ("personalized", "Personalized"),
        ("topk", "TopK"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    axes = list(axes) if hasattr(axes, "__iter__") else [axes]

    for idx, network in enumerate(networks):
        ax = axes[idx]
        ax2 = ax.twinx()

        bar_vals: list[float] = []
        auc_per_cost_vals: list[float] = []
        for strategy_key, _ in display_strategies:
            row = agg_map.get(f"exp_rq2_{network}_{strategy_key}")
            bar_vals.append(_safe_float(row.get("final_misbelief_ratio_mean", 0.0)) if row else 0.0)
            if not row or strategy_key == "no":
                auc_per_cost_vals.append(float("nan"))
            else:
                raw = row.get("efficiency_misbelief_auc_per_cost_mean", None)
                auc_per_cost_vals.append(_safe_float(raw, 0.0) if raw is not None else float("nan"))

        x = list(range(len(display_strategies)))
        bars = ax.bar(x, bar_vals, color="#4C78A8", alpha=0.88, width=0.62)
        ax2.plot(
            x,
            auc_per_cost_vals,
            color="#F58518",
            linewidth=2.2,
            marker="o",
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=2.0,
        )

        for i, bar in enumerate(bars):
            h = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )
            # Intentionally avoid labeling AUC-per-cost points to keep the figure clean.

        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in display_strategies], fontsize=14)
        ax.set_title(f"Network: {network}", fontsize=16)
        ax.grid(alpha=0.25, axis="y")
        ax.tick_params(axis="y", labelsize=13)
        ax2.tick_params(axis="y", labelsize=13)

        if idx == 0:
            ax.set_ylabel("Final Misbelief", fontsize=15)
            ax2.set_ylabel("AUC per Cost", fontsize=15)

    # Legend handles from proxy artists to keep a clean single legend for all subplots.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#4C78A8", alpha=0.88, label="Final Misbelief"),
        Line2D(
            [0],
            [0],
            color="#F58518",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=2.0,
            linewidth=2.2,
            label="AUC per Cost",
        ),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=14, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p_dual = dest_dir / "rq2_final_misbelief_auc_per_cost_small_multiples.png"
    p_dual.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p_dual, dpi=220)
    plt.close(fig)
    files.append(str(p_dual))

    # Paper-friendly heatmap for direct comparison of misbelief AUC mean.
    matrix: list[list[float]] = []
    for strategy in strategies:
        row_vals: list[float] = []
        for network in networks:
            row = agg_map.get(f"exp_rq2_{network}_{strategy}")
            row_vals.append(_safe_float(row.get("misbelief_auc_mean", 0.0)) if row else 0.0)
        matrix.append(row_vals)

    plt.figure(figsize=(6.8, 4.6))
    im = plt.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.xticks(range(len(networks)), networks)
    plt.yticks(range(len(strategies)), strategies)
    plt.xlabel("Network")
    plt.ylabel("Strategy")
    plt.title("RQ2 Misbelief AUC Mean Comparison")
    cbar = plt.colorbar(im)
    cbar.set_label("misbelief_auc_mean")
    for i, row_vals in enumerate(matrix):
        for j, value in enumerate(row_vals):
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    p_heat = dest_dir / "rq2_misbelief_auc_mean_heatmap.png"
    plt.savefig(p_heat, dpi=180)
    plt.close()
    files.append(str(p_heat))
    return files


def _plot_rq3_figures(agg_rows: list[dict[str, Any]], dest_dir: Path, baseline_experiment: str) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    files: list[str] = []
    agg_map = {str(row["experiment_name"]): row for row in agg_rows}
    baseline = agg_map.get(baseline_experiment)
    if baseline is None:
        return files

    baseline_final = _safe_float(baseline.get("final_misbelief_ratio_mean", 0.0))
    baseline_rumor_auc = _safe_float(baseline.get("rumor_exposure_auc_mean", 0.0))
    baseline_peak_heat = _safe_float(baseline.get("peak_rumor_exposure_rate_mean", 0.0))

    scheme_order = [
        ("burst", "early", "Burst-E"),
        ("burst", "mid", "Burst-M"),
        ("burst", "late", "Burst-L"),
        ("steady", "early", "Steady-E"),
        ("steady", "mid", "Steady-M"),
        ("steady", "late", "Steady-L"),
    ]
    data: list[dict[str, float | str]] = []
    for mode, phase, display_name in scheme_order:
        exp = f"exp_rq3_{mode}_{phase}"
        row = agg_map.get(exp)
        if row is None:
            continue
        final_misinfo = _safe_float(row.get("final_misbelief_ratio_mean", 0.0))
        rumor_auc = _safe_float(row.get("rumor_exposure_auc_mean", 0.0))
        peak_heat = _safe_float(row.get("peak_rumor_exposure_rate_mean", 0.0))
        cost = _safe_float(row.get("final_intervention_cost_mean", 0.0))

        final_red = baseline_final - final_misinfo
        rumor_red = baseline_rumor_auc - rumor_auc
        peak_red = baseline_peak_heat - peak_heat
        eff = final_red / cost if cost > 0 else 0.0
        data.append(
            {
                "label": display_name,
                "final_misinfo": final_misinfo,
                "final_reduction": final_red,
                "rumor_reduction": rumor_red,
                "peak_reduction": peak_red,
                "efficiency": eff,
            }
        )

    if not data:
        return files

    labels = [str(x["label"]) for x in data]
    final_misinfos = [float(x["final_misinfo"]) for x in data]
    final_reductions = [float(x["final_reduction"]) for x in data]
    rumor_reductions = [float(x["rumor_reduction"]) for x in data]
    peak_reductions = [float(x["peak_reduction"]) for x in data]
    efficiencies = [float(x["efficiency"]) for x in data]

    # New Figure 1: final misbelief (bar, left axis) + final reduction (line, right axis)
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    bars = ax.bar(x, final_misinfos, color="#4C78A8", alpha=0.9, width=0.65, label="Final Misbelief")
    ax.axhline(
        baseline_final,
        color="#A05195",
        linestyle="--",
        linewidth=2.0,
        label=f"Baseline = {baseline_final:.3f}",
    )
    ax2 = ax.twinx()
    ax2.plot(
        x,
        final_reductions,
        color="#F58518",
        linewidth=2.4,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2.0,
        label="Final Misbelief Reduction",
    )

    for i, b in enumerate(bars):
        h = float(b.get_height())
        ax.text(
            b.get_x() + b.get_width() / 2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
        )
        ax2.annotate(
            f"{final_reductions[i]:.3f}",
            xy=(x[i], final_reductions[i]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            color="#B25500",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel("Final Misinformation Prevalence", fontsize=17)
    ax2.set_ylabel("Final Misbelief Reduction", fontsize=17)
    ax.tick_params(axis="y", labelsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("RQ3: Final Misbelief and Improvement vs Baseline", fontsize=19)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        fontsize=14,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.20)
    p_new1 = dest_dir / "rq3_final_misbelief_vs_baseline_dual_axis.png"
    p_new1.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p_new1, dpi=260)
    p_new1_svg = p_new1.with_suffix(".svg")
    fig.savefig(p_new1_svg)
    plt.close(fig)
    files.append(str(p_new1))
    files.append(str(p_new1_svg))

    # New Figure 2: grouped bars for cumulative rumor exposure reduction and peak heat reduction.
    fig2, axg = plt.subplots(figsize=(12, 6.5))
    width = 0.36
    x_left = [i - width / 2 for i in x]
    x_right = [i + width / 2 for i in x]

    bars_rumor = axg.bar(x_left, rumor_reductions, width=width, color="#4C78A8", label="Cumulative Rumor Exposure Reduction")
    bars_peak = axg.bar(x_right, peak_reductions, width=width, color="#F58518", label="Peak Heat Reduction")
    axg.axhline(0.0, color="#555555", linestyle="--", linewidth=1.6)

    for bar in list(bars_rumor) + list(bars_peak):
        h = float(bar.get_height())
        va = "bottom" if h >= 0 else "top"
        dy = 0.01 if h >= 0 else -0.01
        axg.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            f"{h:.3f}",
            ha="center",
            va=va,
            fontsize=11,
        )

    axg.set_xticks(x)
    axg.set_xticklabels(labels, fontsize=13)
    axg.set_ylabel("Reduction vs Baseline", fontsize=14)
    axg.tick_params(axis="y", labelsize=12)
    axg.grid(alpha=0.25, axis="y")
    axg.set_title("RQ3: Cumulative Rumor Exposure and Peak Heat Improvements", fontsize=16)
    axg.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=12,
        frameon=False,
    )
    fig2.tight_layout(rect=[0, 0.02, 1, 1])
    p_new2 = dest_dir / "rq3_rumor_peak_reduction_grouped.png"
    fig2.savefig(p_new2, dpi=220)
    plt.close(fig2)
    files.append(str(p_new2))

    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, final_reductions)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Final Misinformation Reduction")
    plt.tight_layout()
    p1 = dest_dir / "rq3_final_misinfo_reduction_bar.png"
    p1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p1, dpi=180)
    plt.close()
    files.append(str(p1))

    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, rumor_reductions)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Cumulative Rumor Exposure Reduction")
    plt.tight_layout()
    p2 = dest_dir / "rq3_rumor_auc_reduction_bar.png"
    plt.savefig(p2, dpi=180)
    plt.close()
    files.append(str(p2))

    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, peak_reductions)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Peak Heat Reduction")
    plt.tight_layout()
    p3 = dest_dir / "rq3_peak_heat_reduction_bar.png"
    plt.savefig(p3, dpi=180)
    plt.close()
    files.append(str(p3))

    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, efficiencies)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Cost Efficiency (Final Reduction / Cost)")
    plt.tight_layout()
    p4 = dest_dir / "rq3_cost_efficiency_bar.png"
    plt.savefig(p4, dpi=180)
    plt.close()
    files.append(str(p4))

    return files


def _build_rq3_analysis_tables(
    agg_rows: list[dict[str, Any]],
    dest_dir: Path,
    baseline_experiment: str,
) -> list[Path]:
    agg_map = {str(row["experiment_name"]): row for row in agg_rows}
    baseline = agg_map.get(baseline_experiment)
    if baseline is None:
        return []

    baseline_final = _safe_float(baseline.get("final_misbelief_ratio_mean", 0.0))
    baseline_rumor_auc = _safe_float(baseline.get("rumor_exposure_auc_mean", 0.0))
    baseline_peak_heat = _safe_float(baseline.get("peak_rumor_exposure_rate_mean", 0.0))

    phases = ["early", "mid", "late"]
    cadence_list = ["steady", "burst"]

    core_rows: list[dict[str, Any]] = []
    for cadence in cadence_list:
        for phase in phases:
            exp_name = f"exp_rq3_{cadence}_{phase}"
            row = agg_map.get(exp_name)
            if row is None:
                continue

            final_prevalence = _safe_float(row.get("final_misbelief_ratio_mean", 0.0))
            rumor_auc = _safe_float(row.get("rumor_exposure_auc_mean", 0.0))
            peak_heat = _safe_float(row.get("peak_rumor_exposure_rate_mean", 0.0))
            cost = _safe_float(row.get("final_intervention_cost_mean", 0.0))

            final_reduction = baseline_final - final_prevalence
            rumor_reduction = baseline_rumor_auc - rumor_auc
            peak_reduction = baseline_peak_heat - peak_heat

            core_rows.append(
                {
                    "cadence": cadence,
                    "phase": phase,
                    "experiment_name": exp_name,
                    "final_misinformation_prevalence": final_prevalence,
                    "final_misbelief_reduction": final_reduction,
                    "cumulative_rumor_exposure_auc": rumor_auc,
                    "cumulative_rumor_exposure_reduction": rumor_reduction,
                    "peak_heat": peak_heat,
                    "peak_heat_reduction": peak_reduction,
                    "cost": cost,
                    "cost_efficiency_final_misbelief": _positive_div(final_reduction, cost),
                    "cost_efficiency_rumor_exposure": _positive_div(rumor_reduction, cost),
                    "cost_efficiency_peak_heat": _positive_div(peak_reduction, cost),
                }
            )

    core_rows = sorted(core_rows, key=lambda r: (str(r["cadence"]), str(r["phase"])))
    core_path = dest_dir / "rq3_core_metrics.csv"
    core_fields = [
        "cadence",
        "phase",
        "experiment_name",
        "final_misinformation_prevalence",
        "final_misbelief_reduction",
        "cumulative_rumor_exposure_auc",
        "cumulative_rumor_exposure_reduction",
        "peak_heat",
        "peak_heat_reduction",
        "cost",
        "cost_efficiency_final_misbelief",
        "cost_efficiency_rumor_exposure",
        "cost_efficiency_peak_heat",
    ]
    _write_csv(core_path, core_rows, core_fields)

    timing_rows: list[dict[str, Any]] = []
    for cadence in cadence_list:
        early = next((r for r in core_rows if r["cadence"] == cadence and r["phase"] == "early"), None)
        mid = next((r for r in core_rows if r["cadence"] == cadence and r["phase"] == "mid"), None)
        late = next((r for r in core_rows if r["cadence"] == cadence and r["phase"] == "late"), None)
        if early and late:
            timing_rows.append(
                {
                    "cadence": cadence,
                    "comparison": "early_minus_late",
                    "timing_gain_final_misbelief_reduction": _safe_float(early.get("final_misbelief_reduction")) - _safe_float(late.get("final_misbelief_reduction")),
                    "timing_gain_rumor_exposure_reduction": _safe_float(early.get("cumulative_rumor_exposure_reduction")) - _safe_float(late.get("cumulative_rumor_exposure_reduction")),
                    "timing_gain_peak_heat_reduction": _safe_float(early.get("peak_heat_reduction")) - _safe_float(late.get("peak_heat_reduction")),
                    "timing_gain_efficiency_final": _safe_float(early.get("cost_efficiency_final_misbelief")) - _safe_float(late.get("cost_efficiency_final_misbelief")),
                }
            )
        if early and mid:
            timing_rows.append(
                {
                    "cadence": cadence,
                    "comparison": "early_minus_mid",
                    "timing_gain_final_misbelief_reduction": _safe_float(early.get("final_misbelief_reduction")) - _safe_float(mid.get("final_misbelief_reduction")),
                    "timing_gain_rumor_exposure_reduction": _safe_float(early.get("cumulative_rumor_exposure_reduction")) - _safe_float(mid.get("cumulative_rumor_exposure_reduction")),
                    "timing_gain_peak_heat_reduction": _safe_float(early.get("peak_heat_reduction")) - _safe_float(mid.get("peak_heat_reduction")),
                    "timing_gain_efficiency_final": _safe_float(early.get("cost_efficiency_final_misbelief")) - _safe_float(mid.get("cost_efficiency_final_misbelief")),
                }
            )
        if mid and late:
            timing_rows.append(
                {
                    "cadence": cadence,
                    "comparison": "mid_minus_late",
                    "timing_gain_final_misbelief_reduction": _safe_float(mid.get("final_misbelief_reduction")) - _safe_float(late.get("final_misbelief_reduction")),
                    "timing_gain_rumor_exposure_reduction": _safe_float(mid.get("cumulative_rumor_exposure_reduction")) - _safe_float(late.get("cumulative_rumor_exposure_reduction")),
                    "timing_gain_peak_heat_reduction": _safe_float(mid.get("peak_heat_reduction")) - _safe_float(late.get("peak_heat_reduction")),
                    "timing_gain_efficiency_final": _safe_float(mid.get("cost_efficiency_final_misbelief")) - _safe_float(late.get("cost_efficiency_final_misbelief")),
                }
            )

    timing_path = dest_dir / "rq3_timing_gain.csv"
    timing_fields = [
        "cadence",
        "comparison",
        "timing_gain_final_misbelief_reduction",
        "timing_gain_rumor_exposure_reduction",
        "timing_gain_peak_heat_reduction",
        "timing_gain_efficiency_final",
    ]
    _write_csv(timing_path, timing_rows, timing_fields)

    marginal_rows: list[dict[str, Any]] = []
    for phase in phases:
        burst = next((r for r in core_rows if r["cadence"] == "burst" and r["phase"] == phase), None)
        steady = next((r for r in core_rows if r["cadence"] == "steady" and r["phase"] == phase), None)
        if burst is None or steady is None:
            continue
        marginal_rows.append(
            {
                "phase": phase,
                "comparison": "burst_minus_steady",
                "marginal_final_misbelief_reduction": _safe_float(burst.get("final_misbelief_reduction")) - _safe_float(steady.get("final_misbelief_reduction")),
                "marginal_rumor_exposure_reduction": _safe_float(burst.get("cumulative_rumor_exposure_reduction")) - _safe_float(steady.get("cumulative_rumor_exposure_reduction")),
                "marginal_peak_heat_reduction": _safe_float(burst.get("peak_heat_reduction")) - _safe_float(steady.get("peak_heat_reduction")),
                "marginal_efficiency_final": _safe_float(burst.get("cost_efficiency_final_misbelief")) - _safe_float(steady.get("cost_efficiency_final_misbelief")),
            }
        )

    marginal_path = dest_dir / "rq3_marginal_effects.csv"
    marginal_fields = [
        "phase",
        "comparison",
        "marginal_final_misbelief_reduction",
        "marginal_rumor_exposure_reduction",
        "marginal_peak_heat_reduction",
        "marginal_efficiency_final",
    ]
    _write_csv(marginal_path, marginal_rows, marginal_fields)

    return [core_path, timing_path, marginal_path]


def _plot_rq3_timeseries(run_rows: list[dict[str, Any]], dest_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    files: list[str] = []
    grouped_histories: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for row in run_rows:
        exp_name = str(row.get("experiment_name", ""))
        if exp_name.startswith("exp_rq3_"):
            grouped_histories[exp_name].append(row.get("history", []))

    if not grouped_histories:
        return files

    def _mean_series(histories: list[list[dict[str, Any]]], key: str) -> tuple[list[int], list[float]]:
        if not histories:
            return [], []
        max_len = max(len(h) for h in histories)
        xs = list(range(max_len))
        ys: list[float] = []
        for t in range(max_len):
            vals = [_safe_float(h[t].get(key, 0.0)) for h in histories if t < len(h)]
            ys.append(sum(vals) / max(1, len(vals)))
        return xs, ys

    order = [
        "exp_rq3_steady_early",
        "exp_rq3_steady_mid",
        "exp_rq3_steady_late",
        "exp_rq3_burst_early",
        "exp_rq3_burst_mid",
        "exp_rq3_burst_late",
    ]

    for metric_key, ylabel, filename in [
        ("misbelief_ratio", "Misbelief Ratio", "rq3_misbelief_timeseries.png"),
        ("rumor_exposure_rate", "Rumor Exposure Rate", "rq3_rumor_exposure_timeseries.png"),
    ]:
        plt.figure(figsize=(11.5, 6.6))
        has_line = False
        for exp_name in order:
            x, y = _mean_series(grouped_histories.get(exp_name, []), metric_key)
            if not y:
                continue
            has_line = True
            label = exp_name.replace("exp_rq3_", "")
            linestyle = "--" if "burst" in exp_name else "-"
            plt.plot(x, y, label=label, linewidth=2.8, linestyle=linestyle)
        plt.ylabel(ylabel, fontsize=17)
        plt.xlabel("timestep", fontsize=17)
        ax = plt.gca()
        ax.tick_params(axis="both", labelsize=14)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        plt.grid(alpha=0.3)
        if has_line:
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=13, frameon=False)
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        p = dest_dir / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=240)
        if filename == "rq3_misbelief_timeseries.png":
            plt.savefig(p.with_suffix(".svg"))
        plt.close()
        files.append(str(p))
        if filename == "rq3_misbelief_timeseries.png":
            files.append(str(p.with_suffix(".svg")))

    return files


def _build_rq_key_table(rq: str, agg_rows: list[dict[str, Any]], dest_dir: Path) -> Path:
    rq_upper = rq.upper()
    rows: list[dict[str, Any]] = []
    for row in agg_rows:
        exp_name = str(row.get("experiment_name", "")).lower()
        if rq == "rq1" and "rq1" not in exp_name:
            continue
        if rq == "rq2" and "rq2" not in exp_name:
            continue
        if rq == "rq3" and "rq3" not in exp_name:
            continue
        if rq == "rq4" and "rq4" not in exp_name:
            continue

        rows.append(
            {
                "RQ": rq_upper,
                "experiment_name": row.get("experiment_name", ""),
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
                "peak_rumor_exposure_rate_mean": _safe_float(row.get("peak_rumor_exposure_rate_mean", 0.0)),
                "peak_rumor_exposure_rate_std": _safe_float(row.get("peak_rumor_exposure_rate_std", 0.0)),
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
                "mean_rumor_exposure_rate_mean": _safe_float(row.get("mean_rumor_exposure_rate_mean", 0.0)),
                "mean_rumor_exposure_rate_std": _safe_float(row.get("mean_rumor_exposure_rate_std", 0.0)),
                "mean_debunk_exposure_rate_mean": _safe_float(row.get("mean_debunk_exposure_rate_mean", 0.0)),
                "mean_debunk_exposure_rate_std": _safe_float(row.get("mean_debunk_exposure_rate_std", 0.0)),
            }
        )

    rows = sorted(rows, key=lambda x: str(x.get("experiment_name", "")))
    out_path = dest_dir / f"table_{rq}_key_metrics.csv"
    fields = [
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
        "peak_rumor_exposure_rate_mean",
        "peak_rumor_exposure_rate_std",
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
        "mean_rumor_exposure_rate_mean",
        "mean_rumor_exposure_rate_std",
        "mean_debunk_exposure_rate_mean",
        "mean_debunk_exposure_rate_std",
    ]
    _write_csv(out_path, rows, fields)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate tables/figures for a single RQ. "
            "Process only one of rq1/rq2/rq3/rq4 per run and require explicit source paths (run_* or exp_*)."
        )
    )
    parser.add_argument("--rq", type=str, required=True, choices=["rq1", "rq2", "rq3", "rq4"], help="Generate only one RQ")
    parser.add_argument(
        "--source",
        type=str,
        action="append",
        required=True,
        help="Repeatable; supports run_* or exp_* directories with relative/absolute paths",
    )
    parser.add_argument("--configs-dir", type=str, default="configs")
    parser.add_argument("--dest-dir", type=str, default="")
    parser.add_argument("--dest-root", type=str, default="output/paper_reports")
    parser.add_argument("--report-name", type=str, default="")
    parser.add_argument("--baseline-experiment", type=str, default="exp_rq2_sw_no", help="Baseline experiment name for RQ3 gain plots")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    configs_dir = _resolve_path(project_root, args.configs_dir)

    if args.dest_dir:
        dest_dir = _resolve_path(project_root, args.dest_dir)
    else:
        report_name = args.report_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_dir = _resolve_path(project_root, args.dest_root) / args.rq / report_name

    run_dirs: list[Path] = []
    for source_text in args.source:
        source_path = _resolve_path(project_root, source_text)
        if not source_path.exists():
            print(f"[warn] source path does not exist, skipped: {source_path}")
            continue
        run_dirs.extend(_discover_runs_from_source(source_path))

    dedup = []
    seen = set()
    for rd in run_dirs:
        key = str(rd.resolve())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(rd)
    run_dirs = dedup

    if not run_dirs:
        raise RuntimeError("No usable run_* directory found. Provide run_* or exp_* paths via --source.")

    run_rows: list[dict[str, Any]] = []
    baseline_exp_lower = str(args.baseline_experiment).strip().lower()
    for run_dir in run_dirs:
        row = _build_run_row(run_dir, configs_dir)
        if row is None:
            continue
        exp = str(row.get("experiment_name", "")).lower()
        keep_for_rq = args.rq in exp
        if args.rq == "rq3" and exp == baseline_exp_lower:
            keep_for_rq = True
        if not keep_for_rq:
            continue
        run_rows.append(row)

    if not run_rows:
        raise RuntimeError(f"No run results matching {args.rq.upper()} were found under the given source paths.")

    run_fieldnames = [
        "run_dir",
        "experiment_name",
        "run_id",
        "seed",
        "network_type",
        "strategy",
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
        "mean_rumor_exposure_rate",
        "mean_debunk_exposure_rate",
        "mean_normal_exposure_rate",
        "llm_total_calls",
    ]
    _write_csv(dest_dir / "run_level_metrics.csv", run_rows, run_fieldnames)

    agg_rows = _aggregate_by_experiment(run_rows)
    agg_fields = [
        "experiment_name",
        "n_runs",
        "network_type",
        "strategy",
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
        "mean_rumor_exposure_rate_mean",
        "mean_rumor_exposure_rate_std",
        "mean_debunk_exposure_rate_mean",
        "mean_debunk_exposure_rate_std",
        "mean_normal_exposure_rate_mean",
        "mean_normal_exposure_rate_std",
    ]
    _write_csv(dest_dir / "experiment_aggregates.csv", agg_rows, agg_fields)

    table_path = _build_rq_key_table(args.rq, agg_rows, dest_dir)

    figure_files: list[str] = []
    extra_table_files: list[str] = []
    if args.rq == "rq1":
        figure_files = _plot_rq1_figures(run_rows, agg_rows, dest_dir)
    elif args.rq == "rq2":
        figure_files = _plot_rq2_figures(agg_rows, dest_dir)
    elif args.rq == "rq3":
        figure_files = _plot_rq3_figures(agg_rows, dest_dir, baseline_experiment=args.baseline_experiment)
        figure_files.extend(_plot_rq3_timeseries(run_rows, dest_dir))
        extra_table_files = [
            str(path)
            for path in _build_rq3_analysis_tables(
                agg_rows,
                dest_dir,
                baseline_experiment=args.baseline_experiment,
            )
        ]

    print(
        json.dumps(
            {
                "rq": args.rq,
                "dest_dir": str(dest_dir),
                "run_count": len(run_rows),
                "experiment_count": len(agg_rows),
                "sources": [str(x) for x in args.source],
                "files": {
                    "run_level_metrics": str(dest_dir / "run_level_metrics.csv"),
                    "experiment_aggregates": str(dest_dir / "experiment_aggregates.csv"),
                    "rq_key_table": str(table_path),
                    "rq_extra_tables": extra_table_files,
                    "figures": figure_files,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
