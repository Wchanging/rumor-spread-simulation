from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ShareDescriptor:
    event_id: str
    action_type: str
    expected_text: str
    refutes_fake_event: bool


CONTENT_ID_PATTERN = re.compile(r"^(share|rewrite_share)_(.+)_(\d+)_([0-9a-fA-F]+)$")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _discover_run_dirs(source: Path) -> list[Path]:
    if not source.exists() or not source.is_dir():
        return []
    if (source / "metrics" / "metrics_history.json").exists() and (source / "user_memory").exists():
        return [source]
    runs = [p for p in source.glob("run_*") if p.is_dir() and (p / "metrics" / "metrics_history.json").exists()]
    runs.sort(key=lambda p: p.name)
    return runs


def _iter_user_entries(user_memory_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    files = sorted(user_memory_dir.glob("*.json"), key=lambda p: p.name)
    for user_file in files:
        payload = _json_load(user_file, default=[])
        if not isinstance(payload, list):
            continue
        for item in payload:
            if isinstance(item, dict):
                entries.append(item)
    return entries


def _is_refuting_fake_event(belief_score: float, neutral_band: float) -> bool:
    threshold = -max(0.0, float(neutral_band))
    return float(belief_score) < threshold


def _build_descriptors(
    entries: list[dict[str, Any]],
    neutral_band: float,
) -> tuple[dict[tuple[str, int, str, str], list[ShareDescriptor]], set[str]]:
    descriptors: dict[tuple[str, int, str, str], list[ShareDescriptor]] = {}
    fake_event_ids_global: set[str] = set()

    for entry in entries:
        user_id = str(entry.get("user_id", "")).strip()
        timestep = _safe_int(entry.get("timestep", -1), -1)
        if not user_id or timestep < 0:
            continue

        fake_event_ids = {str(x) for x in (entry.get("fake_event_ids") or [])}
        fake_event_ids_global.update(fake_event_ids)

        feed = entry.get("feed") or []
        feed_by_id: dict[str, dict[str, Any]] = {}
        if isinstance(feed, list):
            for item in feed:
                if isinstance(item, dict):
                    content_id = str(item.get("content_id", "")).strip()
                    if content_id:
                        feed_by_id[content_id] = item

        actions = entry.get("actions") or []
        if not isinstance(actions, list):
            continue

        for action in actions:
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("action_type", "")).strip()
            if action_type not in {"share", "rewrite_share"}:
                continue

            event_id = str(action.get("event_id", "")).strip()
            src_content_id = str(action.get("content_id", "")).strip()
            payload = action.get("payload") or {}
            if not isinstance(payload, dict):
                payload = {}

            belief_score = _safe_float(payload.get("belief_score", 0.0), 0.0)
            refutes_fake_event = bool(event_id in fake_event_ids and _is_refuting_fake_event(belief_score, neutral_band))

            expected_text = ""
            if action_type == "rewrite_share":
                expected_text = _normalize_text(payload.get("rewrite_text", ""))
            if not expected_text:
                source_item = feed_by_id.get(src_content_id, {})
                expected_text = _normalize_text(source_item.get("text", ""))

            key = (user_id, timestep, action_type, event_id)
            descriptors.setdefault(key, []).append(
                ShareDescriptor(
                    event_id=event_id,
                    action_type=action_type,
                    expected_text=expected_text,
                    refutes_fake_event=refutes_fake_event,
                )
            )

    return descriptors, fake_event_ids_global


def _classify_item(
    item: dict[str, Any],
    fake_event_ids: set[str],
    descriptors: dict[tuple[str, int, str, str], list[ShareDescriptor]],
    cache: dict[str, tuple[str, bool]],
    diagnostics: dict[str, int],
) -> str:
    content_id = str(item.get("content_id", "")).strip()
    event_id = str(item.get("event_id", "")).strip()
    is_rumor = bool(item.get("is_rumor", False))
    is_fake_event = event_id in fake_event_ids

    if not content_id:
        if is_fake_event and not is_rumor:
            return "debunk"
        if is_rumor:
            return "rumor"
        return "normal"

    cached = cache.get(content_id)
    if cached is not None:
        label, _ = cached
        return label

    parsed = CONTENT_ID_PATTERN.match(content_id)
    if parsed is None:
        if is_fake_event and not is_rumor:
            label = "debunk"
        elif is_rumor:
            label = "rumor"
        else:
            label = "normal"
        cache[content_id] = (label, False)
        return label

    action_type = parsed.group(1)
    author_id = parsed.group(2)
    created_t = _safe_int(parsed.group(3), -1)
    key = (author_id, created_t, action_type, event_id)
    options = descriptors.get(key, [])

    if not options:
        diagnostics["generated_without_descriptor"] = diagnostics.get("generated_without_descriptor", 0) + 1
        if is_fake_event and not is_rumor:
            label = "debunk"
        elif is_rumor:
            label = "rumor"
        else:
            label = "normal"
        cache[content_id] = (label, False)
        return label

    item_text = _normalize_text(item.get("text", ""))
    matched = [opt for opt in options if opt.expected_text and opt.expected_text == item_text]

    chosen_refute: bool
    ambiguous = False
    if len(matched) == 1:
        chosen_refute = matched[0].refutes_fake_event
    else:
        candidates = matched if matched else options
        refute_values = {opt.refutes_fake_event for opt in candidates}
        if len(refute_values) == 1:
            chosen_refute = next(iter(refute_values))
        else:
            chosen_refute = False
            ambiguous = True

    if is_fake_event and chosen_refute:
        label = "debunk"
    else:
        if is_fake_event and not is_rumor:
            label = "debunk"
        elif is_rumor:
            label = "rumor"
        else:
            label = "normal"

    if ambiguous:
        diagnostics["ambiguous_generated_mapping"] = diagnostics.get("ambiguous_generated_mapping", 0) + 1
    cache[content_id] = (label, ambiguous)
    return label


def _compute_series_auc(history: list[dict[str, Any]], key: str) -> float:
    if not history:
        return 0.0
    auc = 0.0
    for idx, current in enumerate(history):
        y = _safe_float(current.get(key, 0.0), 0.0)
        if idx + 1 < len(history):
            t0 = _safe_float(current.get("timestep", idx), float(idx))
            t1 = _safe_float(history[idx + 1].get("timestep", idx + 1), float(idx + 1))
            dt = max(0.0, t1 - t0)
        else:
            dt = 1.0
        auc += y * dt
    return auc


def _peak_with_timestep(history: list[dict[str, Any]], key: str) -> tuple[float, int]:
    best_value = float("-inf")
    best_t = 0
    for idx, row in enumerate(history):
        value = _safe_float(row.get(key, 0.0), 0.0)
        if value > best_value:
            best_value = value
            best_t = _safe_int(row.get("timestep", idx), idx)
    if best_value == float("-inf"):
        return 0.0, 0
    return best_value, best_t


def _write_metrics_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in headers})


def _recompute_run(run_dir: Path, inplace: bool = False, neutral_band: float = 0.0) -> dict[str, Any]:
    metrics_json_path = run_dir / "metrics" / "metrics_history.json"
    user_memory_dir = run_dir / "user_memory"
    latest_checkpoint_path = run_dir / "latest_checkpoint.json"
    final_summary_path = run_dir / "final_summary.json"

    history = _json_load(metrics_json_path, default=[])
    if not isinstance(history, list) or not history:
        return {"run_dir": str(run_dir), "status": "skipped", "reason": "missing metrics_history"}

    entries = _iter_user_entries(user_memory_dir)
    descriptors, fake_event_ids_global = _build_descriptors(entries, neutral_band=neutral_band)

    per_timestep_entries: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        timestep = _safe_int(entry.get("timestep", -1), -1)
        if timestep < 0:
            continue
        per_timestep_entries.setdefault(timestep, []).append(entry)

    corrected_history: list[dict[str, Any]] = []
    content_cache: dict[str, tuple[str, bool]] = {}
    diagnostics = {
        "ambiguous_generated_mapping": 0,
        "generated_without_descriptor": 0,
        "relabel_to_debunk_count": 0,
        "neutral_band": float(max(0.0, neutral_band)),
    }

    for row in history:
        if not isinstance(row, dict):
            continue
        t = _safe_int(row.get("timestep", len(corrected_history)), len(corrected_history))
        timestep_entries = per_timestep_entries.get(t, [])

        selected_total = 0
        rumor_selected = 0
        debunk_selected = 0
        normal_selected = 0
        empty_feed_users = 0
        event_selected_counts: dict[str, int] = {}

        for entry in timestep_entries:
            selected_ids = [str(x) for x in (entry.get("selected_content_ids") or [])]
            selected_total += len(selected_ids)
            if not selected_ids:
                empty_feed_users += 1

            feed_by_id: dict[str, dict[str, Any]] = {}
            for item in (entry.get("feed") or []):
                if isinstance(item, dict):
                    cid = str(item.get("content_id", "")).strip()
                    if cid:
                        feed_by_id[cid] = item

            fake_event_ids = {str(x) for x in (entry.get("fake_event_ids") or [])}
            if not fake_event_ids:
                fake_event_ids = set(fake_event_ids_global)

            for cid in selected_ids:
                item = feed_by_id.get(cid)
                if not item:
                    continue
                event_id = str(item.get("event_id", "")).strip()
                if event_id:
                    event_selected_counts[event_id] = event_selected_counts.get(event_id, 0) + 1

                old_is_rumor = bool(item.get("is_rumor", False))
                new_label = _classify_item(
                    item=item,
                    fake_event_ids=fake_event_ids,
                    descriptors=descriptors,
                    cache=content_cache,
                    diagnostics=diagnostics,
                )

                if old_is_rumor and new_label == "debunk" and event_id in fake_event_ids:
                    diagnostics["relabel_to_debunk_count"] += 1

                if new_label == "debunk":
                    debunk_selected += 1
                elif new_label == "rumor":
                    rumor_selected += 1
                else:
                    normal_selected += 1

        denominator = float(selected_total) if selected_total > 0 else 1.0
        decision_users = len(timestep_entries)
        event_attention_metrics = {
            event_id: {
                "selected_count": float(count),
                "selected_share": float(count) / denominator,
            }
            for event_id, count in event_selected_counts.items()
        }

        new_row = dict(row)
        new_row["attention_items_total"] = float(selected_total)
        new_row["rumor_exposure_rate"] = float(rumor_selected) / denominator
        new_row["debunk_exposure_rate"] = float(debunk_selected) / denominator
        new_row["normal_exposure_rate"] = float(normal_selected) / denominator
        new_row["empty_feed_rate"] = (float(empty_feed_users) / float(decision_users)) if decision_users > 0 else 0.0
        new_row["event_attention_exposure_metrics"] = event_attention_metrics
        new_row["offline_relabel_applied"] = True
        corrected_history.append(new_row)

    peak_rumor, peak_rumor_t = _peak_with_timestep(corrected_history, "rumor_exposure_rate")
    peak_debunk, peak_debunk_t = _peak_with_timestep(corrected_history, "debunk_exposure_rate")
    rumor_auc = _compute_series_auc(corrected_history, "rumor_exposure_rate")
    debunk_auc = _compute_series_auc(corrected_history, "debunk_exposure_rate")

    correction_summary = {
        "run_dir": str(run_dir),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "rule": (
            "for fake events, generated share/rewrite_share content is relabeled to debunk "
            f"when producer belief_score < {-max(0.0, float(neutral_band)):.3f}; "
            f"belief in [{-max(0.0, float(neutral_band)):.3f}, {max(0.0, float(neutral_band)):.3f}] keeps source label"
        ),
        "diagnostics": diagnostics,
        "final_original": {
            "rumor_exposure_rate": _safe_float(history[-1].get("rumor_exposure_rate", 0.0), 0.0),
            "debunk_exposure_rate": _safe_float(history[-1].get("debunk_exposure_rate", 0.0), 0.0),
            "normal_exposure_rate": _safe_float(history[-1].get("normal_exposure_rate", 0.0), 0.0),
            "empty_feed_rate": _safe_float(history[-1].get("empty_feed_rate", 0.0), 0.0),
        },
        "final_corrected": {
            "rumor_exposure_rate": _safe_float(corrected_history[-1].get("rumor_exposure_rate", 0.0), 0.0),
            "debunk_exposure_rate": _safe_float(corrected_history[-1].get("debunk_exposure_rate", 0.0), 0.0),
            "normal_exposure_rate": _safe_float(corrected_history[-1].get("normal_exposure_rate", 0.0), 0.0),
            "empty_feed_rate": _safe_float(corrected_history[-1].get("empty_feed_rate", 0.0), 0.0),
        },
        "series_corrected": {
            "rumor_exposure_auc": rumor_auc,
            "debunk_exposure_auc": debunk_auc,
            "peak_rumor_exposure_rate": peak_rumor,
            "peak_rumor_exposure_timestep": peak_rumor_t,
            "peak_debunk_exposure_rate": peak_debunk,
            "peak_debunk_exposure_timestep": peak_debunk_t,
        },
    }

    metrics_dir = run_dir / "metrics"
    corrected_json_path = metrics_dir / "metrics_history_corrected.json"
    corrected_csv_path = metrics_dir / "metrics_history_corrected.csv"
    _json_dump(corrected_json_path, corrected_history)
    _write_metrics_csv(corrected_history, corrected_csv_path)

    correction_path = run_dir / "offline_relabel_summary.json"
    _json_dump(correction_path, correction_summary)

    if inplace:
        backup_json = metrics_dir / "metrics_history.original.json"
        backup_csv = metrics_dir / "metrics_history.original.csv"
        if not backup_json.exists() and metrics_json_path.exists():
            backup_json.write_text(metrics_json_path.read_text(encoding="utf-8"), encoding="utf-8")
        csv_path = metrics_dir / "metrics_history.csv"
        if not backup_csv.exists() and csv_path.exists():
            backup_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

        _json_dump(metrics_json_path, corrected_history)
        _write_metrics_csv(corrected_history, csv_path)

        latest_checkpoint = _json_load(latest_checkpoint_path, default={})
        if isinstance(latest_checkpoint, dict) and isinstance(latest_checkpoint.get("latest_metrics"), dict):
            latest_metrics = dict(latest_checkpoint["latest_metrics"])
            latest_metrics["rumor_exposure_rate"] = corrected_history[-1].get("rumor_exposure_rate", 0.0)
            latest_metrics["debunk_exposure_rate"] = corrected_history[-1].get("debunk_exposure_rate", 0.0)
            latest_metrics["normal_exposure_rate"] = corrected_history[-1].get("normal_exposure_rate", 0.0)
            latest_metrics["empty_feed_rate"] = corrected_history[-1].get("empty_feed_rate", 0.0)
            latest_checkpoint["latest_metrics"] = latest_metrics
            latest_checkpoint["updated_at"] = datetime.now().isoformat(timespec="seconds")
            _json_dump(latest_checkpoint_path, latest_checkpoint)

        final_summary = _json_load(final_summary_path, default={})
        if isinstance(final_summary, dict) and isinstance(final_summary.get("summary"), dict):
            summary = dict(final_summary["summary"])
            summary["final_rumor_exposure_rate"] = corrected_history[-1].get("rumor_exposure_rate", 0.0)
            summary["final_debunk_exposure_rate"] = corrected_history[-1].get("debunk_exposure_rate", 0.0)
            summary["final_normal_exposure_rate"] = corrected_history[-1].get("normal_exposure_rate", 0.0)
            summary["final_empty_feed_rate"] = corrected_history[-1].get("empty_feed_rate", 0.0)
            summary["rumor_exposure_auc"] = rumor_auc
            summary["debunk_exposure_auc"] = debunk_auc
            summary["peak_rumor_exposure_rate"] = peak_rumor
            summary["peak_rumor_exposure_timestep"] = peak_rumor_t
            summary["peak_debunk_exposure_rate"] = peak_debunk
            summary["peak_debunk_exposure_timestep"] = peak_debunk_t
            final_summary["summary"] = summary
            final_summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
            _json_dump(final_summary_path, final_summary)

    return {
        "run_dir": str(run_dir),
        "status": "ok",
        "relabeled_to_debunk": diagnostics.get("relabel_to_debunk_count", 0),
        "final_rumor_before": correction_summary["final_original"]["rumor_exposure_rate"],
        "final_rumor_after": correction_summary["final_corrected"]["rumor_exposure_rate"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线重标 share/rewrite_share 内容并回算曝光相关指标")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="实验目录或 run 目录，可重复提供",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="原地覆盖 metrics_history 与 checkpoint/summary 中曝光相关字段（会自动保留 original 备份）",
    )
    parser.add_argument(
        "--neutral-band",
        type=float,
        default=0.0,
        help="中性区间半径。仅当 belief_score < -neutral_band 才将 fake 转发重标为 debunk；默认 0.0（旧口径）",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sources = [Path(s).resolve() for s in args.source]
    run_dirs: list[Path] = []
    for source in sources:
        run_dirs.extend(_discover_run_dirs(source))

    deduped = sorted({str(p): p for p in run_dirs}.values(), key=lambda p: str(p))
    if not deduped:
        print("[WARN] 未发现可处理的 run 目录。")
        return

    results: list[dict[str, Any]] = []
    for run_dir in deduped:
        result = _recompute_run(
            run_dir=run_dir,
            inplace=bool(args.inplace),
            neutral_band=max(0.0, float(args.neutral_band)),
        )
        results.append(result)
        status = result.get("status")
        if status == "ok":
            print(
                f"[OK] {run_dir} | relabeled={result.get('relabeled_to_debunk', 0)} | "
                f"final_rumor: {result.get('final_rumor_before', 0.0):.4f} -> {result.get('final_rumor_after', 0.0):.4f}"
            )
        else:
            print(f"[SKIP] {run_dir} | {result.get('reason', '')}")

    if results:
        print(f"[DONE] processed_runs={sum(1 for r in results if r.get('status') == 'ok')} total_runs={len(results)}")


if __name__ == "__main__":
    main()
