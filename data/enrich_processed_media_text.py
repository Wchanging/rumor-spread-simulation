from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from data.clean_raw_events_dataset import CleanStats, MediaEnricher, MediaProcessConfig
    from data.preprocess import parse_media_list
except ModuleNotFoundError:
    from clean_raw_events_dataset import CleanStats, MediaEnricher, MediaProcessConfig
    from preprocess import parse_media_list

_EVENT_FILE_NAME = re.compile(r"^event\d+\.csv$", re.IGNORECASE)
_BASE_MEDIA_TEXT = re.compile(r"^图片\d+张，视频\d+个$")
_URL_PATTERN = re.compile(r"https?://[^\s\"'\]]+", re.IGNORECASE)

_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
_VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm", ".m3u8")
_IMAGE_NOISE_KEYS = (
    "default_avatar",
    "timeline_card_small_",
    "feed_icon_",
    "member_",
    "bigfan",
    "/vip_",
    "/vvip_",
    "/svip_",
    "_default.png",
)


@dataclass
class EnrichStats:
    files_total: int = 0
    files_updated: int = 0
    posts_total: int = 0
    posts_with_media: int = 0
    posts_processed: int = 0
    posts_updated: int = 0
    posts_skipped_existing: int = 0


def _collect_urls(raw_value: str | None) -> list[str]:
    urls: list[str] = []
    urls.extend(parse_media_list(raw_value))
    if raw_value:
        urls.extend(_URL_PATTERN.findall(str(raw_value)))

    normalized: list[str] = []
    seen: set[str] = set()
    for item in urls:
        url = str(item).strip().strip("\"'")
        url = url.rstrip(",;)")
        if not url or not url.lower().startswith("http"):
            continue
        if not _is_probably_valid_url(url):
            continue
        if url in seen:
            continue
        seen.add(url)
        normalized.append(url)
    return normalized


def _is_probably_valid_url(url: str) -> bool:
    if len(url) < 16:
        return False
    if " " in url:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc or "." not in parsed.netloc:
        return False
    return True


def _looks_like_image(url: str) -> bool:
    lower = url.lower()
    if any(ext in lower for ext in _IMAGE_EXT):
        return True
    return "sinaimg.cn" in lower and "video" not in lower


def _looks_like_video(url: str) -> bool:
    lower = url.lower()
    if any(ext in lower for ext in _VIDEO_EXT):
        return True
    return "video.weibocdn.com" in lower


def _is_noise_image(url: str) -> bool:
    lower = url.lower()
    return any(key in lower for key in _IMAGE_NOISE_KEYS)


def _filter_image_urls(urls: list[str]) -> list[str]:
    candidates = [url for url in urls if _looks_like_image(url)]
    if not candidates:
        return []
    cleaned = [url for url in candidates if not _is_noise_image(url)]
    return cleaned if cleaned else candidates


def _filter_video_urls(urls: list[str]) -> list[str]:
    return [url for url in urls if _looks_like_video(url)]


def _should_update_media_text(existing_text: str, force: bool) -> bool:
    if force:
        return True
    text = (existing_text or "").strip()
    if not text:
        return True
    if text.lower() in {"none", "null", "nan"}:
        return True
    return _BASE_MEDIA_TEXT.fullmatch(text) is not None


def _iter_event_files(data_dir: Path) -> list[Path]:
    files = [path for path in data_dir.glob("event*.csv") if _EVENT_FILE_NAME.fullmatch(path.name)]
    return sorted(files, key=lambda p: p.name)


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        content = line.strip()
        if not content or content.startswith("#") or "=" not in content:
            continue
        key, value = content.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _count_data_rows(file_path: Path) -> int:
    with file_path.open("r", encoding="utf-8-sig", newline="") as fp:
        return max(0, sum(1 for _ in fp) - 1)


def _enrich_single_file(
    file_path: Path,
    media_enricher: MediaEnricher,
    force: bool,
    dry_run: bool,
    run_stats: EnrichStats,
    show_progress: bool,
) -> int:
    run_stats.files_total += 1
    rows: list[dict[str, str]] = []
    updated_count = 0

    with file_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = list(reader.fieldnames or [])
        row_iter = reader
        row_bar = None
        if show_progress and tqdm is not None:
            row_total = _count_data_rows(file_path)
            row_bar = tqdm(
                reader,
                total=row_total,
                desc=f"{file_path.stem}",
                leave=False,
                dynamic_ncols=True,
                unit="post",
            )
            row_iter = row_bar

        for row in row_iter:
            run_stats.posts_total += 1
            if row_bar is not None:
                row_bar.set_postfix(
                    post_id=str(row.get("PostID", ""))[:10],
                    vlm_calls=media_enricher.stats.media_vlm_calls,
                    vlm_fail=media_enricher.stats.media_vlm_failures,
                )
                row_bar.refresh()
            text = str(row.get("Text", "") or "")
            img_urls = _filter_image_urls(_collect_urls(str(row.get("Img", "") or "")))
            video_urls = _filter_video_urls(_collect_urls(str(row.get("Video", "") or "")))
            if img_urls or video_urls:
                run_stats.posts_with_media += 1

            existing_media_text = str(row.get("MediaText", row.get("media_text", "")) or "").strip()
            if not img_urls and not video_urls:
                rows.append(row)
                continue

            if not _should_update_media_text(existing_media_text, force=force):
                run_stats.posts_skipped_existing += 1
                rows.append(row)
                continue

            run_stats.posts_processed += 1
            _, _, media_text = media_enricher.enrich(img_urls=img_urls, video_urls=video_urls, post_text=text)
            if media_text and media_text != existing_media_text:
                row["MediaText"] = media_text
                updated_count += 1
                run_stats.posts_updated += 1
            rows.append(row)

        if row_bar is not None:
            row_bar.close()

    if updated_count <= 0:
        return 0

    if "MediaText" not in fieldnames:
        fieldnames.append("MediaText")

    if not dry_run:
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        temp_path.replace(file_path)

    run_stats.files_updated += 1
    return updated_count


def main() -> None:
    parser = argparse.ArgumentParser(description="为 processed 事件帖子增量生成/回填 MediaText（复用 VLM 缓存）。")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="事件帖子目录，默认 data/processed/events_12",
    )
    parser.add_argument("--force", action="store_true", help="强制重生成所有含媒体的帖子描述")
    parser.add_argument("--dry-run", action="store_true", help="仅统计不落盘")
    parser.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条")
    parser.add_argument("--env-file", type=str, default="", help="指定 .env 文件路径，默认项目根目录 .env")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    env_path = Path(args.env_file).resolve() if args.env_file else (project_root / ".env")
    if args.env_file and not env_path.exists():
        raise RuntimeError(f"指定的 env 文件不存在：{env_path}")
    if env_path.exists():
        _load_env_file(env_path)
        os.environ.setdefault("MEDIA_ENV_FILE", str(env_path))

    os.environ.setdefault("MEDIA_VLM_REQUEST_TIMEOUT", "20")
    os.environ.setdefault("MEDIA_VLM_RETRY_MAX_ATTEMPTS", "2")
    os.environ.setdefault("MEDIA_VLM_RETRY_BASE_DELAY", "0.5")
    os.environ.setdefault("MEDIA_VLM_RETRY_MAX_DELAY", "3.0")

    data_dir = Path(args.data_dir).resolve() if args.data_dir else project_root / "data" / "processed" / "events_12"

    if not data_dir.exists() or not data_dir.is_dir():
        raise RuntimeError(f"目录不存在：{data_dir}")

    files = _iter_event_files(data_dir)
    if not files:
        raise RuntimeError(f"未找到事件帖子文件（event*.csv）：{data_dir}")

    clean_stats = CleanStats()
    media_cfg = MediaProcessConfig(
        enable=True,
        validate_urls=os.getenv("MEDIA_VALIDATE_URLS", "1") == "1",
        request_timeout=float(os.getenv("MEDIA_REQUEST_TIMEOUT", "3.0")),
        enable_vlm=os.getenv("MEDIA_VLM_ENABLE", "1") == "1",
        vlm_max_images_per_post=int(os.getenv("MEDIA_VLM_MAX_IMAGES", "2")),
        vlm_max_videos_per_post=int(os.getenv("MEDIA_VLM_MAX_VIDEOS", "1")),
        cache_file=os.getenv("MEDIA_CACHE_FILE", "media_cache.json"),
    )
    media_enricher = MediaEnricher(output_dir=data_dir, config=media_cfg, stats=clean_stats)

    if media_cfg.enable_vlm and getattr(media_enricher, "_vlm_client", None) is None:
        init_error = str(getattr(media_enricher, "_vlm_init_error", "") or "").strip()
        detail = f" 具体错误：{init_error}" if init_error else ""
        raise RuntimeError(
            "VLM 客户端初始化失败。请检查 OPENAI_API_KEY / OPENAI_BASE_URL / MEDIA_VLM_MODEL 配置后重试。"
            f"（已尝试读取 env：{env_path if env_path.exists() else '未找到 .env'}）{detail}"
        )

    run_stats = EnrichStats()
    file_updates: dict[str, int] = {}
    show_progress = (not args.no_progress) and (tqdm is not None)
    file_iter = files
    file_bar = None
    if show_progress:
        file_bar = tqdm(files, total=len(files), desc="events", leave=True, dynamic_ncols=True, unit="file")
        file_iter = file_bar

    for file_path in file_iter:
        updated = _enrich_single_file(
            file_path=file_path,
            media_enricher=media_enricher,
            force=args.force,
            dry_run=args.dry_run,
            run_stats=run_stats,
            show_progress=show_progress,
        )
        file_updates[file_path.name] = updated
        if file_bar is not None:
            file_bar.set_postfix(updated_files=run_stats.files_updated, updated_posts=run_stats.posts_updated)

    if file_bar is not None:
        file_bar.close()

    media_enricher.flush_cache()

    report = {
        "data_dir": str(data_dir),
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "media_processing": {
            "enable_vlm": media_cfg.enable_vlm,
            "vlm_max_images_per_post": media_cfg.vlm_max_images_per_post,
            "vlm_max_videos_per_post": media_cfg.vlm_max_videos_per_post,
            "cache_file": media_cfg.cache_file,
            "progress_enabled": bool(show_progress),
            "env_file": str(env_path) if env_path.exists() else "",
            "vlm_request_timeout": float(os.getenv("MEDIA_VLM_REQUEST_TIMEOUT", "20")),
            "vlm_retry_max_attempts": int(os.getenv("MEDIA_VLM_RETRY_MAX_ATTEMPTS", "2")),
        },
        "stats": {
            "files_total": run_stats.files_total,
            "files_updated": run_stats.files_updated,
            "posts_total": run_stats.posts_total,
            "posts_with_media": run_stats.posts_with_media,
            "posts_processed": run_stats.posts_processed,
            "posts_updated": run_stats.posts_updated,
            "posts_skipped_existing": run_stats.posts_skipped_existing,
            "media_vlm_calls": clean_stats.media_vlm_calls,
            "media_vlm_failures": clean_stats.media_vlm_failures,
            "media_vlm_retry_success": clean_stats.media_vlm_retry_success,
            "media_vlm_urls_skipped": clean_stats.media_vlm_urls_skipped,
        },
        "file_updates": file_updates,
    }

    if not args.dry_run:
        report_path = data_dir / "media_enrich_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
