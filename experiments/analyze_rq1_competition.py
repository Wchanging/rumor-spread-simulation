from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _pick(rows: list[dict[str, str]], exp_name: str) -> dict[str, str] | None:
    return next((r for r in rows if r.get("experiment_name") == exp_name), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 背景竞争分析（isolated/moderate/high）")
    parser.add_argument(
        "--aggregates",
        type=str,
        default="output/paper_tables/experiment_aggregates.csv",
        help="build_paper_tables.py 产出的聚合CSV",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="output/paper_tables/rq1_competition_analysis.csv",
        help="输出CSV",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    rows = _read_csv((project_root / args.aggregates).resolve())

    levels = ["isolated", "moderate", "high"]
    out_rows: list[dict[str, Any]] = []

    isolated_off = _pick(rows, "exp_rq1_isolated_official")
    isolated_debunk = (
        _safe_float(isolated_off.get("final_debunk_exposure_rate_mean")) if isolated_off else 0.0
    )

    for level in levels:
        off = _pick(rows, f"exp_rq1_{level}_official")
        no = _pick(rows, f"exp_rq1_{level}_no")
        if off is None or no is None:
            continue

        auc_off = _safe_float(off.get("misbelief_auc_mean"))
        auc_no = _safe_float(no.get("misbelief_auc_mean"))
        final_off = _safe_float(off.get("final_misbelief_ratio_mean"))
        final_no = _safe_float(no.get("final_misbelief_ratio_mean"))

        rumor_off = _safe_float(off.get("final_rumor_exposure_rate_mean"))
        rumor_no = _safe_float(no.get("final_rumor_exposure_rate_mean"))
        debunk_off = _safe_float(off.get("final_debunk_exposure_rate_mean"))
        bg_off = _safe_float(off.get("final_normal_exposure_rate_mean"))
        bg_no = _safe_float(no.get("final_normal_exposure_rate_mean"))

        out_rows.append(
            {
                "competition_level": level,
                "misbelief_auc_no": auc_no,
                "misbelief_auc_official": auc_off,
                "benefit_auc_reduction": auc_no - auc_off,
                "final_misbelief_no": final_no,
                "final_misbelief_official": final_off,
                "benefit_final_reduction": final_no - final_off,
                "rumor_exposure_no": rumor_no,
                "rumor_exposure_official": rumor_off,
                "debunk_exposure_official": debunk_off,
                "background_exposure_no": bg_no,
                "background_exposure_official": bg_off,
                "debunk_crowding_out_gap_vs_isolated": isolated_debunk - debunk_off,
            }
        )

    fields = [
        "competition_level",
        "misbelief_auc_no",
        "misbelief_auc_official",
        "benefit_auc_reduction",
        "final_misbelief_no",
        "final_misbelief_official",
        "benefit_final_reduction",
        "rumor_exposure_no",
        "rumor_exposure_official",
        "debunk_exposure_official",
        "background_exposure_no",
        "background_exposure_official",
        "debunk_crowding_out_gap_vs_isolated",
    ]
    _write_csv((project_root / args.dest).resolve(), out_rows, fields)
    print(f"RQ1 competition analysis written: {(project_root / args.dest).resolve()}")


if __name__ == "__main__":
    main()
