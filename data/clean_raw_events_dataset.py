from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib import request
from urllib.error import URLError


@dataclass
class CleanStats:
    posts_total: int = 0
    posts_kept: int = 0
    posts_missing_user_filled: int = 0
    posts_non_numeric_interactions_fixed: int = 0
    comments_total: int = 0
    comments_kept: int = 0
    comments_dropped_missing_post_id: int = 0
    comments_dropped_orphan_post_id: int = 0
    comments_dropped_missing_comment_id: int = 0
    comments_missing_user_filled: int = 0
    media_urls_total: int = 0
    media_urls_reachable: int = 0
    media_vlm_calls: int = 0
    media_vlm_failures: int = 0
    media_vlm_retry_success: int = 0
    media_vlm_urls_skipped: int = 0


@dataclass
class MediaProcessConfig:
    enable: bool = True
    validate_urls: bool = False
    request_timeout: float = 3.0
    enable_vlm: bool = False
    vlm_max_images_per_post: int = 2
    vlm_max_videos_per_post: int = 1
    cache_file: str = "media_cache.json"


class MediaEnricher:
    def __init__(self, output_dir: Path, config: MediaProcessConfig, stats: CleanStats) -> None:
        self.output_dir = output_dir
        self.config = config
        self.stats = stats
        self.cache_path = output_dir / config.cache_file
        self.cache = self._load_cache()
        self._vlm_init_error: str = ""
        self._vlm_client = self._build_vlm_client()

    def _load_cache(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def flush_cache(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_vlm_client(self):
        if not self.config.enable_vlm:
            return None
        try:
            try:
                from llm.openai_client import OpenAIClient
            except ModuleNotFoundError:
                project_root = Path(__file__).resolve().parents[1]
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                from llm.openai_client import OpenAIClient

            env_file = os.getenv("MEDIA_ENV_FILE", "") or os.getenv("OPENAI_ENV_FILE", "")
            if not env_file:
                candidate = Path(__file__).resolve().parents[1] / ".env"
                if candidate.exists():
                    env_file = str(candidate)

            llm_cfg = {
                "model": os.getenv("MEDIA_VLM_MODEL", os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")),
                "vision_model": os.getenv("MEDIA_VLM_MODEL", os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")),
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "base_url": os.getenv("OPENAI_BASE_URL", ""),
                "env_file": env_file,
                "temperature": 0.0,
                "max_tokens": int(os.getenv("MEDIA_VLM_MAX_TOKENS", "300")),
                "max_concurrency": int(os.getenv("MEDIA_VLM_MAX_CONCURRENCY", "2")),
                "request_timeout": float(os.getenv("MEDIA_VLM_REQUEST_TIMEOUT", "20")),
                "retry_max_attempts": int(os.getenv("MEDIA_VLM_RETRY_MAX_ATTEMPTS", "2")),
                "retry_base_delay": float(os.getenv("MEDIA_VLM_RETRY_BASE_DELAY", "0.5")),
                "retry_max_delay": float(os.getenv("MEDIA_VLM_RETRY_MAX_DELAY", "3.0")),
                "adaptive_concurrency": True,
            }
            return OpenAIClient.from_config(llm_cfg)
        except Exception as exc:
            self._vlm_init_error = str(exc)
            return None

    def enrich(self, img_urls: list[str], video_urls: list[str], post_text: str) -> tuple[list[str], list[str], str]:
        if not self.config.enable:
            return img_urls, video_urls, ""

        valid_imgs = [url for url in img_urls if url and not self._is_vlm_bad_url(url)]
        valid_videos = [url for url in video_urls if url and not self._is_vlm_bad_url(url)]
        self.stats.media_vlm_urls_skipped += max(0, len(img_urls) - len(valid_imgs))
        self.stats.media_vlm_urls_skipped += max(0, len(video_urls) - len(valid_videos))
        if self.config.validate_urls:
            valid_imgs = [url for url in valid_imgs if self._is_url_reachable(url)]
            valid_videos = [url for url in valid_videos if self._is_url_reachable(url)]

        media_text = self._summarize_media(valid_imgs, valid_videos, post_text)
        return valid_imgs, valid_videos, media_text

    def _is_url_reachable(self, url: str) -> bool:
        cache_key = f"reachable::{url}"
        cached = self.cache.get(cache_key)
        if isinstance(cached, bool):
            self.stats.media_urls_total += 1
            self.stats.media_urls_reachable += 1 if cached else 0
            return cached

        self.stats.media_urls_total += 1
        req = request.Request(url=url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        reachable = False
        try:
            with request.urlopen(req, timeout=self.config.request_timeout) as resp:
                reachable = 200 <= int(getattr(resp, "status", 200)) < 400
        except Exception:
            try:
                req_get = request.Request(
                    url=url,
                    method="GET",
                    headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"},
                )
                with request.urlopen(req_get, timeout=self.config.request_timeout) as resp:
                    reachable = 200 <= int(getattr(resp, "status", 200)) < 500
            except (URLError, ValueError, TimeoutError):
                reachable = False

        self.cache[cache_key] = bool(reachable)
        self.stats.media_urls_reachable += 1 if reachable else 0
        return bool(reachable)

    def _is_vlm_bad_url(self, url: str) -> bool:
        return bool(self.cache.get(f"vlm_bad_url::{url}", False))

    def _mark_vlm_bad_url(self, url: str) -> None:
        if not url:
            return
        self.cache[f"vlm_bad_url::{url}"] = True

    def _summarize_media(self, img_urls: list[str], video_urls: list[str], post_text: str) -> str:
        if not img_urls and not video_urls:
            return ""

        base_summary = f"{len(img_urls)} image(s), {len(video_urls)} video(s)"
        vlm_summary = self._vlm_caption(img_urls=img_urls, video_urls=video_urls, post_text=post_text)
        if vlm_summary:
            return f"{base_summary}; media semantics: {vlm_summary}"
        return base_summary

    def _vlm_caption(self, img_urls: list[str], video_urls: list[str], post_text: str) -> str:
        if self._vlm_client is None or (not img_urls and not video_urls):
            return ""

        max_images = max(1, int(self.config.vlm_max_images_per_post))
        max_videos = max(0, int(self.config.vlm_max_videos_per_post))
        img_urls = img_urls[:max_images]
        video_urls = video_urls[:max_videos] if max_videos > 0 else []
        digest = hashlib.md5(
            ("|".join(img_urls) + "|" + "|".join(video_urls) + "|" + (post_text or "")[:200]).encode("utf-8")
        ).hexdigest()
        cache_key = f"vlm::{digest}"
        if cache_key in self.cache:
            value = self.cache.get(cache_key, "")
            return str(value) if value else ""

        video_context = "\n".join(f"- {url}" for url in video_urls) if video_urls else "None"
        prompt = (
            "Using the post text and media information, output one objective and concise media description (<=200 chars)."
            " Focus on visible scene/content and avoid speculation or value judgment."
            f"\nPost text: {(post_text or '')[:200]}"
            f"\nVideo links (context only; no need to explain each one):\n{video_context}"
        )
        candidate_sets: list[list[str]] = []
        candidate_sets.append(list(img_urls))
        if len(img_urls) > 1:
            candidate_sets.extend([[url] for url in img_urls])
        if not img_urls:
            candidate_sets.append([])

        seen_candidates: set[tuple[str, ...]] = set()
        for candidate in candidate_sets:
            key = tuple(candidate)
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            try:
                self.stats.media_vlm_calls += 1
                result = self._vlm_client.generate_multimodal(text=prompt, image_urls=candidate)
                summary = (result or "").strip().replace("\n", " ")
                summary = re.sub(r"\s+", " ", summary)
                if candidate != img_urls:
                    self.stats.media_vlm_retry_success += 1
                    for url in img_urls:
                        if url not in candidate:
                            self._mark_vlm_bad_url(url)
                self.cache[cache_key] = summary
                return summary
            except Exception:
                self.stats.media_vlm_failures += 1
                if len(candidate) == 1:
                    self._mark_vlm_bad_url(candidate[0])

        if img_urls:
            for url in img_urls:
                self._mark_vlm_bad_url(url)
        self.cache[cache_key] = ""
        return ""


def parse_list_block(raw: str) -> list[str]:
    value = (raw or "").strip()
    if not value:
        return []
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value.strip():
        return []
    parts = [part.strip().strip("'\"") for part in value.split(",")]
    return [part for part in parts if part]


def parse_int_strict(raw: str | None) -> tuple[int, bool]:
    if raw is None:
        return 0, True
    s = str(raw).strip()
    if not s:
        return 0, True
    try:
        return int(float(s)), False
    except (TypeError, ValueError):
        return 0, True


def normalize_json_list(raw: str | None) -> str:
    return json.dumps(parse_list_block(raw or ""), ensure_ascii=False)


def parse_eventset_line(line: str) -> dict | None:
    text = line.strip()
    if not text:
        return None

    list_matches = list(re.finditer(r"\[[^\]]*\]", text))
    posts_raw = "[]"
    evidence_posts_raw = "[]"

    if len(list_matches) >= 1:
        posts_raw = list_matches[0].group(0)
    if len(list_matches) >= 2:
        evidence_posts_raw = list_matches[1].group(0)

    prefix_end = list_matches[0].start() if list_matches else len(text)
    prefix = text[:prefix_end].rstrip(",")
    parts = prefix.split(",", 3)
    while len(parts) < 4:
        parts.append("")

    event_name = parts[0].strip()
    description = parts[1].strip()
    is_fake_raw = parts[2].strip()
    evidence = parts[3].strip()

    is_fake = 1
    if is_fake_raw in {"0", "1"}:
        is_fake = int(is_fake_raw)
    else:
        try:
            is_fake = 1 if float(is_fake_raw) >= 0.5 else 0
        except ValueError:
            is_fake = 0

    return {
        "raw_event_name": event_name,
        "Description": description,
        "IsFake": is_fake,
        "Evidence": evidence,
        "Posts": json.dumps(parse_list_block(posts_raw), ensure_ascii=False),
        "EvidencePosts": json.dumps(parse_list_block(evidence_posts_raw), ensure_ascii=False),
    }


def clean_events_csv(raw_events_path: Path, out_events_path: Path) -> list[dict]:
    lines = raw_events_path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise RuntimeError("Event set CSV is empty")

    cleaned_rows: list[dict] = []
    for idx, line in enumerate(lines[1:], start=1):
        parsed = parse_eventset_line(line)
        if parsed is None:
            continue
        parsed["Event"] = f"event{idx}"
        cleaned_rows.append(parsed)

    out_events_path.parent.mkdir(parents=True, exist_ok=True)
    with out_events_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["Event", "Description", "IsFake", "Evidence", "Posts", "EvidencePosts", "RawEventName"],
        )
        writer.writeheader()
        for row in cleaned_rows:
            writer.writerow(
                {
                    "Event": row["Event"],
                    "Description": row["Description"],
                    "IsFake": row["IsFake"],
                    "Evidence": row["Evidence"],
                    "Posts": row["Posts"],
                    "EvidencePosts": row["EvidencePosts"],
                    "RawEventName": row["raw_event_name"],
                }
            )
    return cleaned_rows


def clean_posts_csv(
    raw_path: Path,
    out_path: Path,
    event_idx: int,
    stats: CleanStats,
    media_enricher: MediaEnricher | None = None,
) -> set[str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept_post_ids: set[str] = set()
    with raw_path.open("r", encoding="utf-8-sig", newline="") as rfp, out_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as wfp:
        reader = csv.DictReader(rfp)
        writer = csv.DictWriter(
            wfp,
            fieldnames=[
                "PostID",
                "PostTime",
                "User",
                "Text",
                "Img",
                "Video",
                "MediaText",
                "Likes",
                "Comments",
                "Retweet",
            ],
        )
        writer.writeheader()

        for idx, row in enumerate(reader):
            stats.posts_total += 1
            post_id = str(row.get("PostID", "")).strip()
            if not post_id:
                continue

            user = str(row.get("User", "")).strip()
            if not user:
                user = f"unknown_user_event{event_idx}_{idx}"
                stats.posts_missing_user_filled += 1

            post_time, _ = parse_int_strict(row.get("PostTime"))
            likes, likes_fixed = parse_int_strict(row.get("Likes"))
            comments, comments_fixed = parse_int_strict(row.get("Comments"))
            retweet, retweet_fixed = parse_int_strict(row.get("Retweet"))
            if likes_fixed or comments_fixed or retweet_fixed:
                stats.posts_non_numeric_interactions_fixed += 1

            img_urls = parse_list_block(str(row.get("Img", "") or ""))
            video_urls = parse_list_block(str(row.get("Video", "") or ""))
            media_text = ""
            if media_enricher is not None:
                img_urls, video_urls, media_text = media_enricher.enrich(
                    img_urls=img_urls,
                    video_urls=video_urls,
                    post_text=str(row.get("Text", "") or ""),
                )

            writer.writerow(
                {
                    "PostID": post_id,
                    "PostTime": post_time,
                    "User": user,
                    "Text": str(row.get("Text", "")).strip(),
                    "Img": json.dumps(img_urls, ensure_ascii=False),
                    "Video": json.dumps(video_urls, ensure_ascii=False),
                    "MediaText": media_text,
                    "Likes": likes,
                    "Comments": comments,
                    "Retweet": retweet,
                }
            )
            kept_post_ids.add(post_id)
            stats.posts_kept += 1
    return kept_post_ids


def clean_comments_csv(
    raw_path: Path,
    out_path: Path,
    event_idx: int,
    stats: CleanStats,
    valid_post_ids: set[str] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("r", encoding="utf-8-sig", newline="") as rfp, out_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as wfp:
        reader = csv.DictReader(rfp)
        writer = csv.DictWriter(
            wfp,
            fieldnames=[
                "PostID",
                "CommentID",
                "CommentTime",
                "User",
                "Text",
                "Img",
                "Video",
                "Likes",
                "IsReplied",
                "SourceComment",
                "Replies",
            ],
        )
        writer.writeheader()

        for idx, row in enumerate(reader):
            stats.comments_total += 1
            post_id = str(row.get("PostID", "")).strip()
            comment_id = str(row.get("CommentID", "")).strip()
            if not post_id:
                stats.comments_dropped_missing_post_id += 1
                continue
            if valid_post_ids is not None and post_id not in valid_post_ids:
                stats.comments_dropped_orphan_post_id += 1
                continue
            if not comment_id:
                stats.comments_dropped_missing_comment_id += 1
                continue

            user = str(row.get("User", "")).strip()
            if not user:
                user = f"unknown_commenter_event{event_idx}_{idx}"
                stats.comments_missing_user_filled += 1

            comment_time, _ = parse_int_strict(row.get("CommentTime"))
            likes, _ = parse_int_strict(row.get("Likes"))

            writer.writerow(
                {
                    "PostID": post_id,
                    "CommentID": comment_id,
                    "CommentTime": comment_time,
                    "User": user,
                    "Text": str(row.get("Text", "")).strip(),
                    "Img": normalize_json_list(row.get("Img")),
                    "Video": normalize_json_list(row.get("Video")),
                    "Likes": likes,
                    "IsReplied": 0,
                    "SourceComment": "",
                    "Replies": 0,
                }
            )
            stats.comments_kept += 1


def clean_all(
    raw_root: Path,
    out_root: Path,
    n_events: int = 12,
    media_config: MediaProcessConfig | None = None,
) -> dict:
    stats = CleanStats()
    media_cfg = media_config or MediaProcessConfig()
    media_enricher = MediaEnricher(output_dir=out_root, config=media_cfg, stats=stats)
    events_path = raw_root / "事件集.csv"
    cleaned_events = clean_events_csv(events_path, out_root / "events.csv")

    if len(cleaned_events) < n_events:
        n_events = len(cleaned_events)

    for event_idx in range(1, n_events + 1):
        raw_post_file = raw_root / f"事件{event_idx}编码.csv"
        raw_comment_file = raw_root / f"事件{event_idx}评论编码.csv"

        out_post_file = out_root / f"event{event_idx}.csv"
        out_comment_file = out_root / f"event{event_idx}_comments.csv"

        valid_post_ids: set[str] | None = None
        if raw_post_file.exists():
            valid_post_ids = clean_posts_csv(
                raw_post_file,
                out_post_file,
                event_idx,
                stats,
                media_enricher=media_enricher,
            )
        if raw_comment_file.exists():
            clean_comments_csv(raw_comment_file, out_comment_file, event_idx, stats, valid_post_ids=valid_post_ids)

    media_enricher.flush_cache()

    report = {
        "raw_root": str(raw_root),
        "output_root": str(out_root),
        "events_cleaned": n_events,
        "stats": {
            "posts_total": stats.posts_total,
            "posts_kept": stats.posts_kept,
            "posts_missing_user_filled": stats.posts_missing_user_filled,
            "posts_non_numeric_interactions_fixed": stats.posts_non_numeric_interactions_fixed,
            "comments_total": stats.comments_total,
            "comments_kept": stats.comments_kept,
            "comments_dropped_missing_post_id": stats.comments_dropped_missing_post_id,
            "comments_dropped_orphan_post_id": stats.comments_dropped_orphan_post_id,
            "comments_dropped_missing_comment_id": stats.comments_dropped_missing_comment_id,
            "comments_missing_user_filled": stats.comments_missing_user_filled,
            "media_urls_total": stats.media_urls_total,
            "media_urls_reachable": stats.media_urls_reachable,
            "media_vlm_calls": stats.media_vlm_calls,
            "media_vlm_failures": stats.media_vlm_failures,
            "media_vlm_retry_success": stats.media_vlm_retry_success,
            "media_vlm_urls_skipped": stats.media_vlm_urls_skipped,
        },
        "media_processing": {
            "enabled": media_cfg.enable,
            "validate_urls": media_cfg.validate_urls,
            "enable_vlm": media_cfg.enable_vlm,
            "vlm_max_images_per_post": media_cfg.vlm_max_images_per_post,
            "vlm_max_videos_per_post": media_cfg.vlm_max_videos_per_post,
            "cache_file": media_cfg.cache_file,
        },
    }
    (out_root / "clean_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_root = project_root / "data" / "raw" / "raw"
    out_root = project_root / "data" / "processed" / "events_12"
    media_cfg = MediaProcessConfig(
        enable=os.getenv("MEDIA_ENRICH_ENABLE", "1") != "0",
        validate_urls=os.getenv("MEDIA_VALIDATE_URLS", "0") == "1",
        request_timeout=float(os.getenv("MEDIA_REQUEST_TIMEOUT", "3.0")),
        enable_vlm=os.getenv("MEDIA_VLM_ENABLE", "0") == "1",
        vlm_max_images_per_post=int(os.getenv("MEDIA_VLM_MAX_IMAGES", "2")),
        vlm_max_videos_per_post=int(os.getenv("MEDIA_VLM_MAX_VIDEOS", "1")),
        cache_file=os.getenv("MEDIA_CACHE_FILE", "media_cache.json"),
    )
    report = clean_all(raw_root=raw_root, out_root=out_root, n_events=12, media_config=media_cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
