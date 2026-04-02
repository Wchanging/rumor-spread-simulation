from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RQ1_SOURCES = [
    "output/exp_20260323_185300_exp_rq1_high_no",
    "output/exp_20260323_191425_exp_rq1_high_official",
    "output/exp_20260323_193907_exp_rq1_isolated_no",
    "output/exp_20260323_202059_exp_rq1_isolated_official",
    "output/exp_20260323_205848_exp_rq1_moderate_no",
    "output/exp_20260323_211724_exp_rq1_moderate_official",
    "output/exp_20260324_090641_exp_rq1_high_official",
    "output/exp_20260325_120148_exp_rq1_high_no",
    "output/exp_20260325_125040_exp_rq1_high_official",
    "output/exp_20260325_141900_exp_rq1_isolated_no",
    "output/exp_20260325_145728_exp_rq1_isolated_no",
    "output/exp_20260325_151949_exp_rq1_isolated_official",
    "output/exp_20260325_163100_exp_rq1_moderate_no",
    "output/exp_20260325_171523_exp_rq1_moderate_official",
]

RQ2_SOURCES = [
    "output/exp_20260322_192405_exp_rq2_random_no",
    "output/exp_20260322_200558_exp_rq2_random_official",
    "output/exp_20260322_233316_exp_rq2_random_topk",
    "output/exp_20260323_000057_exp_rq2_random_personalized",
    "output/exp_20260323_085838_exp_rq2_sf_no",
    "output/exp_20260323_091636_exp_rq2_sf_official",
    "output/exp_20260323_093751_exp_rq2_sf_personalized",
    "output/exp_20260323_095828_exp_rq2_sf_topk",
    "output/exp_20260323_110046_exp_rq2_sw_official",
    "output/exp_20260323_104042_exp_rq2_sw_no",
    "output/exp_20260323_115330_exp_rq2_sw_personalized",
    "output/exp_20260323_124137_exp_rq2_sw_topk",
    "output/exp_20260325_183947_exp_rq2_random_no",
    "output/exp_20260325_192353_exp_rq2_random_official",
    "output/exp_20260325_201343_exp_rq2_random_personalized",
    "output/exp_20260325_213050_exp_rq2_random_topk",
    "output/exp_20260325_223037_exp_rq2_sf_no",
    "output/exp_20260326_084427_exp_rq2_sf_official",
    "output/exp_20260326_094304_exp_rq2_sf_personalized",
    "output/exp_20260326_110136_exp_rq2_sf_topk",
    "output/exp_20260326_121007_exp_rq2_sw_no",
    "output/exp_20260326_130357_exp_rq2_sw_official",
    "output/exp_20260326_140957_exp_rq2_sw_personalized",
    "output/exp_20260326_153703_exp_rq2_sw_topk",
]

RQ3_SOURCES = [
    "output/exp_20260324_095307_exp_rq3_burst_early",
    "output/exp_20260324_092450_exp_rq3_burst_early",
    "output/exp_20260324_102833_exp_rq3_burst_late",
    "output/exp_20260324_110859_exp_rq3_burst_mid",
    "output/exp_20260324_114458_exp_rq3_steady_early",
    "output/exp_20260324_122608_exp_rq3_steady_late",
    "output/exp_20260324_125546_exp_rq3_steady_mid",
    "output/exp_20260326_215702_exp_rq3_burst_early",
    "output/exp_20260327_093955_exp_rq3_burst_mid",
    "output/exp_20260327_105326_exp_rq3_steady_early",
    "output/exp_20260327_120320_exp_rq3_steady_late",
    "output/exp_20260327_142246_exp_rq3_steady_mid",
    "output/exp_20260327_084058_exp_rq3_burst_late",
]

RQ3_BASELINE_SOURCE = "output/exp_20260326_121007_exp_rq2_sw_no"
RQ3_BASELINE_EXPERIMENT = "exp_rq2_sw_no"


def _existing_sources(project_root: Path, raw_sources: list[str]) -> list[str]:
    out: list[str] = []
    for src in raw_sources:
        if (project_root / src).exists():
            out.append(src)
        else:
            print(f"[WARN] missing source, skip: {src}")
    # deduplicate while preserving order
    seen = set()
    deduped: list[str] = []
    for src in out:
        if src in seen:
            continue
        seen.add(src)
        deduped.append(src)
    return deduped


def _run(cmd: list[str], project_root: Path, dry: bool) -> None:
    print("\n[CMD]", " ".join(cmd))
    if dry:
        return
    subprocess.run(cmd, cwd=project_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch pipeline: offline relabel dry-run -> inplace relabel -> regenerate RQ1/RQ2/RQ3 reports."
        )
    )
    parser.add_argument("--neutral-band", type=float, default=0.2, help="Neutral band radius, default=0.2")
    parser.add_argument("--skip-dry-run", action="store_true", help="Skip non-inplace preview relabel")
    parser.add_argument("--skip-inplace", action="store_true", help="Skip inplace relabel")
    parser.add_argument("--skip-reports", action="store_true", help="Skip report regeneration")
    parser.add_argument("--dry", action="store_true", help="Print commands only, do not execute")
    parser.add_argument("--rq1-dest", type=str, default="output/paper_reports/rq1/runD")
    parser.add_argument("--rq2-dest", type=str, default="output/paper_reports/rq2/runC")
    parser.add_argument("--rq3-dest", type=str, default="output/paper_reports/rq3/runC")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    offline_script = project_root / "experiments" / "offline_relabel_recompute.py"
    report_script = project_root / "experiments" / "generate_rq_report.py"

    rq1_sources = _existing_sources(project_root, RQ1_SOURCES)
    rq2_sources = _existing_sources(project_root, RQ2_SOURCES)
    rq3_sources = _existing_sources(project_root, RQ3_SOURCES)
    baseline_sources = _existing_sources(project_root, [RQ3_BASELINE_SOURCE])

    all_sources = _existing_sources(project_root, rq1_sources + rq2_sources + rq3_sources + baseline_sources)
    if not all_sources:
        raise RuntimeError("No valid sources found.")

    if not args.skip_dry_run:
        cmd = [
            sys.executable,
            str(offline_script),
            "--neutral-band",
            str(max(0.0, args.neutral_band)),
        ]
        for src in all_sources:
            cmd.extend(["--source", src])
        _run(cmd, project_root, args.dry)

    if not args.skip_inplace:
        cmd = [
            sys.executable,
            str(offline_script),
            "--neutral-band",
            str(max(0.0, args.neutral_band)),
            "--inplace",
        ]
        for src in all_sources:
            cmd.extend(["--source", src])
        _run(cmd, project_root, args.dry)

    if args.skip_reports:
        return

    if rq1_sources:
        cmd = [sys.executable, str(report_script), "--rq", "rq1", "--dest-dir", args.rq1_dest]
        for src in rq1_sources:
            cmd.extend(["--source", src])
        _run(cmd, project_root, args.dry)

    if rq2_sources:
        cmd = [sys.executable, str(report_script), "--rq", "rq2", "--dest-dir", args.rq2_dest]
        for src in rq2_sources:
            cmd.extend(["--source", src])
        _run(cmd, project_root, args.dry)

    if rq3_sources and baseline_sources:
        cmd = [
            sys.executable,
            str(report_script),
            "--rq",
            "rq3",
            "--baseline-experiment",
            RQ3_BASELINE_EXPERIMENT,
            "--dest-dir",
            args.rq3_dest,
            "--source",
            baseline_sources[0],
        ]
        for src in rq3_sources:
            cmd.extend(["--source", src])
        _run(cmd, project_root, args.dry)


if __name__ == "__main__":
    main()
