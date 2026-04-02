from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _discover_runs(output_root: Path) -> list[Path]:
    runs: list[Path] = []
    for exp_dir in sorted(output_root.glob("exp_*")):
        if not exp_dir.is_dir():
            continue
        for run_dir in sorted(exp_dir.glob("run_*")):
            if (run_dir / "metrics" / "metrics_history.json").exists():
                runs.append(run_dir)
    return runs


def _canonical_experiment_name(output_exp_dir_name: str) -> str:
    marker = "exp_rq"
    idx = output_exp_dir_name.find(marker)
    if idx >= 0:
        return output_exp_dir_name[idx:]
    return output_exp_dir_name


def _collect_histories(output_root: Path) -> dict[str, list[list[dict[str, Any]]]]:
    grouped: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for run_dir in _discover_runs(output_root):
        exp_name = _canonical_experiment_name(run_dir.parent.name)
        history = _load_json(run_dir / "metrics" / "metrics_history.json")
        if isinstance(history, list) and history:
            grouped[exp_name].append(history)
    return grouped


def _mean_series(histories: list[list[dict[str, Any]]], key: str) -> tuple[list[int], list[float]]:
    if not histories:
        return [], []

    max_len = max(len(h) for h in histories)
    xs = list(range(max_len))
    ys: list[float] = []

    for t in range(max_len):
        vals: list[float] = []
        for h in histories:
            if t >= len(h):
                continue
            vals.append(_safe_float(h[t].get(key, 0.0)))
        ys.append(sum(vals) / max(1, len(vals)))
    return xs, ys


def _save_plot(path: Path, title: str):
    import matplotlib.pyplot as plt

    plt.title(title)
    plt.xlabel("timestep")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def plot_rq1_exposure(output_root: Path, dest_dir: Path) -> None:
    import matplotlib.pyplot as plt

    grouped = _collect_histories(output_root)

    setting_map = {
        "isolated": ("exp_rq1_isolated_official", "exp_rq1_isolated_no"),
        "moderate": ("exp_rq1_moderate_official", "exp_rq1_moderate_no"),
        "high": ("exp_rq1_high_official", "exp_rq1_high_no"),
    }

    # rumor exposure
    plt.figure(figsize=(9, 4.5))
    for setting, (official_exp, no_exp) in setting_map.items():
        x1, y1 = _mean_series(grouped.get(official_exp, []), "rumor_exposure_rate")
        x2, y2 = _mean_series(grouped.get(no_exp, []), "rumor_exposure_rate")
        if y1:
            plt.plot(x1, y1, label=f"{setting}-official", linewidth=1.8)
        if y2:
            plt.plot(x2, y2, label=f"{setting}-no", linestyle="--", linewidth=1.3)
    plt.ylabel("rumor_exposure_rate")
    plt.legend(ncol=2, fontsize=8)
    _save_plot(dest_dir / "rq1_rumor_exposure_timeseries.png", "RQ1 Rumor Exposure over Time")

    # debunk exposure
    plt.figure(figsize=(9, 4.5))
    for setting, (official_exp, no_exp) in setting_map.items():
        x1, y1 = _mean_series(grouped.get(official_exp, []), "debunk_exposure_rate")
        x2, y2 = _mean_series(grouped.get(no_exp, []), "debunk_exposure_rate")
        if y1:
            plt.plot(x1, y1, label=f"{setting}-official", linewidth=1.8)
        if y2:
            plt.plot(x2, y2, label=f"{setting}-no", linestyle="--", linewidth=1.3)
    plt.ylabel("debunk_exposure_rate")
    plt.legend(ncol=2, fontsize=8)
    _save_plot(dest_dir / "rq1_debunk_exposure_timeseries.png", "RQ1 Debunk Exposure over Time")

    # background exposure
    plt.figure(figsize=(9, 4.5))
    for setting, (official_exp, no_exp) in setting_map.items():
        x1, y1 = _mean_series(grouped.get(official_exp, []), "normal_exposure_rate")
        x2, y2 = _mean_series(grouped.get(no_exp, []), "normal_exposure_rate")
        if y1:
            plt.plot(x1, y1, label=f"{setting}-official", linewidth=1.8)
        if y2:
            plt.plot(x2, y2, label=f"{setting}-no", linestyle="--", linewidth=1.3)
    plt.ylabel("background_exposure_share")
    plt.legend(ncol=2, fontsize=8)
    _save_plot(dest_dir / "rq1_background_exposure_timeseries.png", "RQ1 Background Exposure over Time")


def _load_aggregates_csv(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def plot_rq1_outcome_bars(aggregates_csv: Path, dest_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _load_aggregates_csv(aggregates_csv)
    if not rows:
        return

    levels = ["isolated", "moderate", "high"]
    labels: list[str] = []
    auc_no: list[float] = []
    auc_off: list[float] = []
    final_no: list[float] = []
    final_off: list[float] = []

    for level in levels:
        row_no = next((r for r in rows if r.get("experiment_name") == f"exp_rq1_{level}_no"), None)
        row_off = next((r for r in rows if r.get("experiment_name") == f"exp_rq1_{level}_official"), None)
        if row_no is None or row_off is None:
            continue
        labels.append(level)
        auc_no.append(_safe_float(row_no.get("misbelief_auc_mean", 0.0)))
        auc_off.append(_safe_float(row_off.get("misbelief_auc_mean", 0.0)))
        final_no.append(_safe_float(row_no.get("final_misbelief_ratio_mean", 0.0)))
        final_off.append(_safe_float(row_off.get("final_misbelief_ratio_mean", 0.0)))

    if not labels:
        return

    x = list(range(len(labels)))
    width = 0.36

    plt.figure(figsize=(8.5, 4.5))
    plt.bar([i - width / 2 for i in x], auc_no, width=width, label="NoIntervention")
    plt.bar([i + width / 2 for i in x], auc_off, width=width, label="Official")
    plt.xticks(x, labels)
    plt.ylabel("Misbelief AUC")
    plt.legend()
    plt.tight_layout()
    dest_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest_dir / "rq1_misbelief_auc_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 4.5))
    plt.bar([i - width / 2 for i in x], final_no, width=width, label="NoIntervention")
    plt.bar([i + width / 2 for i in x], final_off, width=width, label="Official")
    plt.xticks(x, labels)
    plt.ylabel("Final Misbelief")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dest_dir / "rq1_final_misbelief_bar.png", dpi=180)
    plt.close()


def plot_rq3_bars(aggregates_csv: Path, dest_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _load_aggregates_csv(aggregates_csv)
    baseline = next((r for r in rows if r.get("experiment_name") == "exp_rq2_sw_no"), None)
    if baseline is None:
        return

    baseline_final = _safe_float(baseline.get("final_misbelief_ratio_mean", 0.0))
    baseline_rumor_auc = _safe_float(baseline.get("rumor_exposure_auc_mean", 0.0))
    baseline_peak_heat = _safe_float(baseline.get("peak_rumor_exposure_rate_mean", 0.0))

    phases = ["early", "mid", "late"]
    cadence = ["steady", "burst"]
    data: list[tuple[str, float, float, float, float]] = []
    for mode in cadence:
        for phase in phases:
            exp = f"exp_rq3_{mode}_{phase}"
            row = next((r for r in rows if r.get("experiment_name") == exp), None)
            if row is None:
                continue
            final_misinfo = _safe_float(row.get("final_misbelief_ratio_mean", 0.0))
            rumor_auc = _safe_float(row.get("rumor_exposure_auc_mean", 0.0))
            peak_heat = _safe_float(row.get("peak_rumor_exposure_rate_mean", 0.0))
            cost = _safe_float(row.get("final_intervention_cost_mean", 0.0))

            final_red = baseline_final - final_misinfo
            rumor_auc_red = baseline_rumor_auc - rumor_auc
            peak_heat_red = baseline_peak_heat - peak_heat
            eff = (final_red / cost) if cost > 0 else 0.0
            data.append((f"{mode}-{phase}", final_red, rumor_auc_red, peak_heat_red, eff))

    if not data:
        return

    labels = [x[0] for x in data]
    final_reductions = [x[1] for x in data]
    rumor_auc_reductions = [x[2] for x in data]
    peak_heat_reductions = [x[3] for x in data]
    efficiencies = [x[4] for x in data]

    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, final_reductions)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Final Misinformation Reduction")
    plt.tight_layout()
    (dest_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(dest_dir / "rq3_final_misinfo_reduction_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, rumor_auc_reductions)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Cumulative Rumor Exposure Reduction")
    plt.tight_layout()
    plt.savefig(dest_dir / "rq3_rumor_auc_reduction_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, peak_heat_reductions)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Peak Heat Reduction")
    plt.tight_layout()
    plt.savefig(dest_dir / "rq3_peak_heat_reduction_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, efficiencies)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Cost Efficiency (Final Reduction / Cost)")
    plt.tight_layout()
    plt.savefig(dest_dir / "rq3_cost_efficiency_bar.png", dpi=180)
    plt.close()


def plot_rq2_key_bars(aggregates_csv: Path, dest_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _load_aggregates_csv(aggregates_csv)
    if not rows:
        return

    networks = ["sw", "sf", "random"]
    strategies = ["no", "official", "topk", "personalized"]

    def _get_row(network: str, strategy: str) -> dict[str, str] | None:
        exp = f"exp_rq2_{network}_{strategy}"
        return next((r for r in rows if r.get("experiment_name") == exp), None)

    def _plot_metric(metric_key: str, ylabel: str, filename: str) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
        axes = list(axes) if hasattr(axes, "__iter__") else [axes]
        for idx, network in enumerate(networks):
            ax = axes[idx]
            vals: list[float] = []
            for strategy in strategies:
                row = _get_row(network, strategy)
                vals.append(_safe_float(row.get(metric_key, 0.0)) if row is not None else 0.0)

            x = list(range(len(strategies)))
            ax.bar(x, vals)
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=20, ha="right")
            ax.set_title(f"network={network}")
            ax.grid(alpha=0.25, axis="y")
            if idx == 0:
                ax.set_ylabel(ylabel)

        plt.tight_layout()
        dest_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(dest_dir / filename, dpi=180)
        plt.close(fig)

    _plot_metric("misbelief_auc_mean", "Misbelief AUC", "rq2_misbelief_auc_by_network_strategy.png")
    _plot_metric("final_misbelief_ratio_mean", "Final Misbelief", "rq2_final_misbelief_by_network_strategy.png")
    _plot_metric("final_intervention_cost_mean", "Intervention Cost", "rq2_cost_by_network_strategy.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper figures from saved experiment outputs")
    parser.add_argument("--output-root", type=str, default="output")
    parser.add_argument("--aggregates", type=str, default="output/paper_tables/experiment_aggregates.csv")
    parser.add_argument("--dest", type=str, default="output/paper_figures")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_root = (project_root / args.output_root).resolve()
    aggregates_csv = (project_root / args.aggregates).resolve()
    dest_dir = (project_root / args.dest).resolve()

    plot_rq1_exposure(output_root, dest_dir)
    plot_rq1_outcome_bars(aggregates_csv, dest_dir)
    plot_rq2_key_bars(aggregates_csv, dest_dir)
    plot_rq3_bars(aggregates_csv, dest_dir)

    print(f"Figures generated at: {dest_dir}")


if __name__ == "__main__":
    main()
