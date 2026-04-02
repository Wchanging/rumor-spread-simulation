import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Dict, List


EVENT_FILE_RE = re.compile(r"^event(\d+)\.csv$", re.IGNORECASE)
COMMENT_FILE_RE = re.compile(r"^event(\d+)_comments\.csv$", re.IGNORECASE)


def _parse_media_list(raw: str) -> List[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return []


def _init_stats() -> Dict[str, int]:
    return {
        "rows": 0,
        "rows_with_image": 0,
        "rows_with_video": 0,
        "rows_with_both": 0,
        "rows_with_any_media": 0,
        "rows_without_media": 0,
        "image_items": 0,
        "video_items": 0,
    }


def _accumulate_row_stats(stats: Dict[str, int], img_raw: str, video_raw: str) -> None:
    imgs = _parse_media_list(img_raw)
    vids = _parse_media_list(video_raw)
    has_img = len(imgs) > 0
    has_vid = len(vids) > 0

    stats["rows"] += 1
    stats["image_items"] += len(imgs)
    stats["video_items"] += len(vids)

    if has_img:
        stats["rows_with_image"] += 1
    if has_vid:
        stats["rows_with_video"] += 1
    if has_img and has_vid:
        stats["rows_with_both"] += 1
    if has_img or has_vid:
        stats["rows_with_any_media"] += 1
    else:
        stats["rows_without_media"] += 1


def _read_events_meta(events_csv: Path) -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}
    with events_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = (row.get("Event") or "").strip()
            if not event:
                continue
            meta[event] = {
                "is_fake": str(row.get("IsFake", "")).strip(),
                "raw_event_name": str(row.get("RawEventName", "")).strip(),
                "description": str(row.get("Description", "")).strip(),
            }
    return meta


def _scan_event_files(dataset_dir: Path) -> Dict[str, Dict[str, Path]]:
    mapping: Dict[str, Dict[str, Path]] = {}
    for path in sorted(dataset_dir.glob("*.csv")):
        name = path.name
        match_event = EVENT_FILE_RE.match(name)
        if match_event:
            event_id = f"event{match_event.group(1)}"
            mapping.setdefault(event_id, {})["posts"] = path
            continue
        match_comment = COMMENT_FILE_RE.match(name)
        if match_comment:
            event_id = f"event{match_comment.group(1)}"
            mapping.setdefault(event_id, {})["comments"] = path
    return mapping


def _compute_event_stats(posts_path: Path, comments_path: Path) -> Dict[str, Dict[str, int]]:
    post_stats = _init_stats()
    comment_stats = _init_stats()

    if posts_path and posts_path.exists():
        with posts_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                _accumulate_row_stats(post_stats, row.get("Img", ""), row.get("Video", ""))

    if comments_path and comments_path.exists():
        with comments_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                _accumulate_row_stats(comment_stats, row.get("Img", ""), row.get("Video", ""))

    return {"posts": post_stats, "comments": comment_stats}


def _sum_stats(target: Dict[str, int], source: Dict[str, int]) -> None:
    for key in target:
        target[key] += source.get(key, 0)


def compute_dataset_statistics(dataset_dir: Path) -> Dict[str, object]:
    events_csv = dataset_dir / "events.csv"
    if not events_csv.exists():
        raise FileNotFoundError(f"events.csv not found: {events_csv}")

    event_meta = _read_events_meta(events_csv)
    file_map = _scan_event_files(dataset_dir)
    event_ids = sorted(set(event_meta.keys()) | set(file_map.keys()), key=lambda x: int(x.replace("event", "")))

    total_post_stats = _init_stats()
    total_comment_stats = _init_stats()

    per_event: List[Dict[str, object]] = []

    fake_events = 0
    non_fake_events = 0

    for event_id in event_ids:
        files = file_map.get(event_id, {})
        post_file = files.get("posts")
        comment_file = files.get("comments")
        stats = _compute_event_stats(post_file, comment_file)
        post_stats = stats["posts"]
        comment_stats = stats["comments"]
        _sum_stats(total_post_stats, post_stats)
        _sum_stats(total_comment_stats, comment_stats)

        meta = event_meta.get(event_id, {})
        is_fake = str(meta.get("is_fake", "")).strip()
        if is_fake == "1":
            fake_events += 1
        elif is_fake == "0":
            non_fake_events += 1

        per_event.append(
            {
                "event": event_id,
                "is_fake": is_fake,
                "raw_event_name": meta.get("raw_event_name", ""),
                "description": meta.get("description", ""),
                "post_rows": post_stats["rows"],
                "post_rows_with_image": post_stats["rows_with_image"],
                "post_rows_with_video": post_stats["rows_with_video"],
                "post_rows_with_both": post_stats["rows_with_both"],
                "post_rows_with_any_media": post_stats["rows_with_any_media"],
                "post_rows_without_media": post_stats["rows_without_media"],
                "post_image_items": post_stats["image_items"],
                "post_video_items": post_stats["video_items"],
                "comment_rows": comment_stats["rows"],
                "comment_rows_with_image": comment_stats["rows_with_image"],
                "comment_rows_with_video": comment_stats["rows_with_video"],
                "comment_rows_with_both": comment_stats["rows_with_both"],
                "comment_rows_with_any_media": comment_stats["rows_with_any_media"],
                "comment_rows_without_media": comment_stats["rows_without_media"],
                "comment_image_items": comment_stats["image_items"],
                "comment_video_items": comment_stats["video_items"],
                "posts_file": str(post_file.name) if post_file else "",
                "comments_file": str(comment_file.name) if comment_file else "",
            }
        )

    global_stats = {
        "dataset_dir": str(dataset_dir),
        "event_count": len(event_ids),
        "fake_event_count": fake_events,
        "non_fake_event_count": non_fake_events,
        "posts": total_post_stats,
        "comments": total_comment_stats,
    }

    return {"global": global_stats, "per_event": per_event}


def save_outputs(stats: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    per_event_csv = output_dir / "per_event_stats.csv"
    global_json = output_dir / "global_stats.json"

    per_event_rows: List[Dict[str, object]] = stats["per_event"]  # type: ignore[index]
    if per_event_rows:
        fieldnames = list(per_event_rows[0].keys())
        with per_event_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_event_rows)
    else:
        with per_event_csv.open("w", encoding="utf-8", newline="") as f:
            f.write("")

    with global_json.open("w", encoding="utf-8") as f:
        json.dump(stats["global"], f, ensure_ascii=False, indent=2)

    print(f"Saved per-event stats: {per_event_csv}")
    print(f"Saved global stats:   {global_json}")

    global_stats = stats["global"]  # type: ignore[index]
    print("\nHeadline summary:")
    print(f"- events: {global_stats['event_count']} (fake={global_stats['fake_event_count']}, non_fake={global_stats['non_fake_event_count']})")
    print(f"- posts: {global_stats['posts']['rows']} | any_media={global_stats['posts']['rows_with_any_media']} | image_items={global_stats['posts']['image_items']} | video_items={global_stats['posts']['video_items']}")
    print(f"- comments: {global_stats['comments']['rows']} | any_media={global_stats['comments']['rows_with_any_media']} | image_items={global_stats['comments']['image_items']} | video_items={global_stats['comments']['video_items']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset statistics for posts/comments/media by event.")
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/events_12",
        help="Directory containing events.csv + event*.csv + event*_comments.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/events_12/stats_report",
        help="Directory to write statistics outputs",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    stats = compute_dataset_statistics(dataset_dir)
    save_outputs(stats, output_dir)


if __name__ == "__main__":
    main()
