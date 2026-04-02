from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loaders import EventLoader, PostLoader
from llm.mock_client import MockLLMClient
from llm.openai_client import OpenAIClient

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


DEBUNK_HINT_PATTERNS = [
    r"辟谣",
    r"澄清",
    r"不实",
    r"系谣言",
    r"造谣",
    r"勿信",
    r"别信",
    r"警方通报",
    r"官方回应",
    r"经核实",
    r"并非",
    r"不是.*(真的|事实|这样)",
]


def looks_like_debunk_text(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in DEBUNK_HINT_PATTERNS)


def filter_rumor_like_texts(items: list[str], max_items: int) -> list[str]:
    if not items:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        if looks_like_debunk_text(value):
            continue
        seen.add(value)
        cleaned.append(value[:180])
        if len(cleaned) >= max(1, int(max_items)):
            break
    return cleaned


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("Failed to read YAML. Install PyYAML or use JSON config.") from exc
        return yaml.safe_load(text)

    raise ValueError(f"Unsupported config format: {suffix}")


def resolve_path(base_dir: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def build_llm_client(
    provider: str,
    seed: int,
    env_file: str | None,
    model: str,
    max_concurrency: int,
):
    normalized = str(provider).lower()
    if normalized == "mock":
        return MockLLMClient(seed=seed)
    if normalized == "openai":
        cfg: dict[str, Any] = {
            "provider": "openai",
            "model": model,
            "max_concurrency": max_concurrency,
        }
        if env_file:
            cfg["env_file"] = env_file
        return OpenAIClient.from_config(cfg)
    raise ValueError(f"Unsupported llm provider: {provider}")


def resolve_post_file(posts_dir: Path, event_id: str, posts_template: str) -> Path | None:
    candidates = [
        posts_dir / posts_template.format(event_id=event_id),
        posts_dir / f"{event_id}_posts.csv",
        posts_dir / f"{event_id}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_generated_posts(raw: str, max_items: int) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    candidates: list[str] = []
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and isinstance(payload.get("posts"), list):
            candidates = [str(item).strip() for item in payload.get("posts", [])]
        elif isinstance(payload, list):
            candidates = [str(item).strip() for item in payload]
    except Exception:
        candidates = []

    if not candidates:
        match = re.findall(r"\[[\s\S]*\]", text)
        for block in reversed(match):
            try:
                payload = json.loads(block)
                if isinstance(payload, list):
                    candidates = [str(item).strip() for item in payload]
                    break
            except Exception:
                continue

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        value = str(item).strip().strip('"')
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value[:180])
        if len(cleaned) >= max(1, int(max_items)):
            break
    return cleaned


def collect_source_texts(raw_posts: list, source_limit: int) -> list[str]:
    ranked = sorted(raw_posts, key=lambda p: (int(getattr(p, "likes", 0)) +
                    int(getattr(p, "comments", 0)) + int(getattr(p, "retweet", 0))), reverse=True)
    texts: list[str] = []
    seen: set[str] = set()
    for post in ranked:
        text = str(getattr(post, "text", "") or "").strip()
        if not text or text in seen:
            continue
        if looks_like_debunk_text(text):
            continue
        seen.add(text)
        texts.append(text[:220])
        if len(texts) >= max(1, int(source_limit)):
            break

    if len(texts) < max(1, int(source_limit)):
        for post in ranked:
            text = str(getattr(post, "text", "") or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            texts.append(text[:220])
            if len(texts) >= max(1, int(source_limit)):
                break
    return texts


def fallback_generate_posts(source_texts: list[str], rumor_claim: str, count: int, rng: random.Random) -> list[str]:
    if not source_texts:
        source_texts = [rumor_claim] if rumor_claim else []
    if not source_texts:
        return []
    patterns = [
        "Online rumor says: {text}",
        "Someone in a group said: {text}",
        "A message claims: {text}",
        "It is said that: {text}",
        "Unverified information: {text}",
    ]
    results: list[str] = []
    for idx in range(count):
        source = source_texts[idx % len(source_texts)]
        pattern = patterns[idx % len(patterns)]
        marker = rng.choice(["", ", sounds plausible", ", many people are sharing this", ", what do you think?"])
        results.append((pattern.format(text=source[:120]) + marker)[:180])
    return results


def generate_posts_for_event(
    llm_client,
    event_id: str,
    event_desc: str,
    event_evidence: str,
    source_texts: list[str],
    total_count: int,
    batch_size: int,
    rng: random.Random,
    show_progress: bool = True,
) -> list[str]:
    if total_count <= 0:
        return []

    generated: list[str] = []
    rounds = max(1, math.ceil(total_count / max(1, batch_size)))
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(total=total_count, desc=f"Generating {event_id}", leave=False, unit="post")

    for i in range(rounds):
        before_count = len(generated)
        remain = total_count - len(generated)
        if remain <= 0:
            break
        ask_n = min(max(1, batch_size), remain)
        rumor_claim = str(event_desc or "").strip()[:220]
        evidence_fact = str(event_evidence or "").strip()[:220]
        prompt = (
            f"Event ID: {event_id}\n"
            f"Core rumor claim (can be paraphrased/expanded): {rumor_claim}\n"
            f"Debunking facts (only to prevent debunk-style writing): {evidence_fact}\n"
            f"Existing rumor-style samples: {json.dumps(source_texts, ensure_ascii=False)}\n"
            f"Already generated samples (avoid duplication, increase diversity across user styles): {json.dumps(generated[-8:], ensure_ascii=False)}\n"
            f"Generate {ask_n} new rumor-spreading posts. Requirements: colloquial wording, non-duplicative, consistent with the core rumor claim but rephrased, <= 200 characters each.\n"
            "Must satisfy:\n"
            "1) Write from rumor/forwarded/hearsay perspective; do not write debunking/clarification/official verification conclusions.\n"
            "2) Avoid phrases like 'this is a rumor', 'false', 'do not trust', 'verified', or 'officially debunked'.\n"
            "3) Do not directly restate debunking fact points.\n"
            "4) Keep diverse expression styles (e.g., marketing account, ordinary user, gossip observer)."
        )
        system_prompt = (
            "You are a social-platform rumor-post rewriting assistant."
            " Generate only rumor-propagation tone, not debunking or correction tone."
            " Output JSON only in the form {\"posts\":[\"...\",\"...\"]} with no explanations."
        )

        batch: list[str] = []
        try:
            raw = llm_client.generate(prompt, system_prompt=system_prompt)
            batch = filter_rumor_like_texts(parse_generated_posts(raw, max_items=ask_n * 2), max_items=ask_n)
        except Exception:
            batch = []

        if not batch:
            batch = fallback_generate_posts(source_texts, rumor_claim=rumor_claim, count=ask_n, rng=rng)

        for item in batch:
            if item not in generated:
                generated.append(item)
            if len(generated) >= total_count:
                break

        if progress_bar is not None:
            progress_bar.update(max(0, len(generated) - before_count))
            if hasattr(progress_bar, "set_postfix_str"):
                progress_bar.set_postfix_str(f"batch {i + 1}/{rounds}")

    if len(generated) < total_count:
        before_count = len(generated)
        supplement = fallback_generate_posts(
            source_texts,
            rumor_claim=str(event_desc or "").strip()[:220],
            count=total_count - len(generated),
            rng=rng,
        )
        for item in supplement:
            if item not in generated:
                generated.append(item)
            if len(generated) >= total_count:
                break
        if progress_bar is not None:
            progress_bar.update(max(0, len(generated) - before_count))

    if progress_bar is not None:
        progress_bar.close()

    return filter_rumor_like_texts(generated, max_items=total_count)


def write_generated_posts_csv(
    output_file: Path,
    event_id: str,
    posts: list[str],
    seed: int,
    post_times: list[int],
    base_likes: int,
    base_comments: int,
    base_retweet: int,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with output_file.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["PostID", "PostTime", "User", "Text", "Img", "Video", "Likes", "Comments", "Retweet"],
        )
        writer.writeheader()
        for idx, text in enumerate(posts):
            post_time = int(post_times[idx]) if idx < len(post_times) else int(post_times[-1] + idx + 1)
            writer.writerow(
                {
                    "PostID": f"gen_{event_id}_{idx:04d}",
                    "PostTime": post_time,
                    "User": f"gen_user_{rng.randint(1, 9999):04d}",
                    "Text": text,
                    "Img": "[]",
                    "Video": "[]",
                    "Likes": max(0, base_likes + rng.randint(0, 2)),
                    "Comments": max(0, base_comments + rng.randint(0, 1)),
                    "Retweet": max(0, base_retweet + rng.randint(0, 2)),
                }
            )


def build_post_times_from_source(source_times: list[int], count: int, seed: int) -> list[int]:
    if count <= 0:
        return []
    cleaned = sorted(int(ts) for ts in source_times if int(ts) > 0)
    if not cleaned:
        return list(range(1, count + 1))

    rng = random.Random(seed)
    if len(cleaned) == 1:
        base = cleaned[0]
        return [base + i for i in range(count)]

    gaps = [max(1, cleaned[i] - cleaned[i - 1]) for i in range(1, len(cleaned))]
    median_gap = sorted(gaps)[len(gaps) // 2]
    jitter = max(1, median_gap // 4)

    sampled = []
    for _ in range(count):
        base = rng.choice(cleaned)
        sampled.append(max(1, base + rng.randint(-jitter, jitter)))
    sampled.sort()
    return sampled


def patch_existing_generated_post_times(generated_file: Path, patched_times: list[int]) -> int:
    if not generated_file.exists() or not patched_times:
        return 0

    with generated_file.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        return 0

    if "PostTime" not in fieldnames:
        fieldnames.append("PostTime")

    updated = 0
    for idx, row in enumerate(rows):
        if idx >= len(patched_times):
            break
        row["PostTime"] = str(int(patched_times[idx]))
        updated += 1

    with generated_file.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline generation of supplemental CSV posts for each rumor event.")
    parser.add_argument("--config", default=None, help="Optional experiment config file path (JSON/YAML)")
    parser.add_argument("--events-file", default=None, help="Event master CSV path")
    parser.add_argument("--posts-dir", default=None, help="Directory of source event posts")
    parser.add_argument("--posts-template", default="{event_id}.csv", help="Source post filename template")
    parser.add_argument("--per-event", type=int, default=30, help="Number of generated posts per rumor event")
    parser.add_argument("--source-posts-per-event", type=int, default=6, help="Number of source posts per event for rewriting")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of posts generated per LLM request")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM generation (disabled by default, uses template variants)")
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "mock"], help="LLM provider for offline generation")
    parser.add_argument("--llm-env-file", default=None, help="Optional OpenAI env file path")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--llm-max-concurrency", type=int, default=4, help="OpenAI max concurrency")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing generated files")
    parser.add_argument("--output-dir", default="data/processed/generated_rumor_posts", help="Output directory for generated posts")
    parser.add_argument("--output-template", default="{event_id}_generated.csv", help="Output filename template")
    parser.add_argument("--base-likes", type=int, default=-1, help="Base likes for generated posts; <0 means estimate from source posts")
    parser.add_argument("--base-comments", type=int, default=-1, help="Base comments for generated posts; <0 means estimate from source posts")
    parser.add_argument("--base-retweet", type=int, default=-1, help="Base retweet for generated posts; <0 means estimate from source posts")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--patch-existing-timestamps", action="store_true",
                        help="Only patch PostTime in existing generated CSVs; do not generate new text")
    args = parser.parse_args()

    config: dict[str, Any] = {}
    base_dir = Path.cwd()
    if args.config:
        config_file = Path(args.config).resolve()
        config = load_config(config_file)
        base_dir = config_file.parent
        if base_dir.name.lower() == "configs":
            base_dir = base_dir.parent

    event_source = config.get("event_source", {}) if isinstance(config.get("event_source", {}), dict) else {}

    events_file_raw = args.events_file if args.events_file is not None else event_source.get("events_file")
    posts_dir_raw = args.posts_dir if args.posts_dir is not None else event_source.get("posts_dir")
    posts_template = str(args.posts_template or event_source.get("posts_template", "{event_id}.csv"))
    output_dir_raw = args.output_dir
    output_template = str(args.output_template)

    events_file = resolve_path(base_dir, events_file_raw)
    posts_dir = resolve_path(base_dir, posts_dir_raw)
    output_dir = resolve_path(base_dir, output_dir_raw)

    if events_file is None or not events_file.exists():
        raise FileNotFoundError("events_file does not exist. Pass --events-file or provide event_source.events_file in --config")
    if posts_dir is None or not posts_dir.exists():
        raise FileNotFoundError("posts_dir does not exist. Pass --posts-dir or provide event_source.posts_dir in --config")
    if output_dir is None:
        raise FileNotFoundError("Invalid output_dir")

    per_event = int(args.per_event)
    source_limit = int(args.source_posts_per_event)
    batch_size = int(args.batch_size)
    overwrite = bool(args.overwrite)
    use_llm = bool(args.use_llm)

    base_likes = int(args.base_likes)
    base_comments = int(args.base_comments)
    base_retweet = int(args.base_retweet)

    seed = int(args.seed)
    rng = random.Random(seed)

    llm_env_file = args.llm_env_file
    if llm_env_file is None and isinstance(config.get("llm", {}), dict):
        llm_env_file = config.get("llm", {}).get("env_file")
    resolved_env_file = str(resolve_path(base_dir, llm_env_file)) if llm_env_file else None

    llm_client = (
        build_llm_client(
            provider=str(args.llm_provider),
            seed=seed,
            env_file=resolved_env_file,
            model=str(args.llm_model),
            max_concurrency=int(args.llm_max_concurrency),
        )
        if use_llm
        else MockLLMClient(seed=seed)
    )
    event_loader = EventLoader()
    post_loader = PostLoader()

    raw_events = event_loader.load(events_file)
    fake_events = [event for event in raw_events if bool(event.is_fake)]

    summary: list[dict[str, Any]] = []
    event_iter = tqdm(fake_events, desc="Processing fake events", unit="event")
    for event in event_iter:
        if hasattr(event_iter, "set_postfix_str"):
            event_iter.set_postfix_str(str(event.event_id))
        src_file = resolve_post_file(Path(posts_dir), event.event_id, posts_template)
        if src_file is None:
            summary.append({"event_id": event.event_id, "status": "skip_missing_source_file"})
            continue

        out_file = Path(output_dir) / output_template.format(event_id=event.event_id)
        if out_file.exists() and args.patch_existing_timestamps:
            raw_posts_for_patch = post_loader.load(src_file, event_id=event.event_id)
            source_times_for_patch = [int(getattr(post, "post_time", 0)) for post in raw_posts_for_patch if int(getattr(post, "post_time", 0)) > 0]

            with out_file.open("r", encoding="utf-8-sig", newline="") as fp:
                reader = csv.DictReader(fp)
                existing_rows = list(reader)
            patched_times = build_post_times_from_source(
                source_times=source_times_for_patch,
                count=len(existing_rows),
                seed=seed + len(summary),
            )
            updated = patch_existing_generated_post_times(out_file, patched_times)
            summary.append({"event_id": event.event_id, "status": "patched", "updated": updated, "output": str(out_file)})
            continue

        if out_file.exists() and not overwrite:
            summary.append({"event_id": event.event_id, "status": "skip_exists", "output": str(out_file)})
            continue

        raw_posts = post_loader.load(src_file, event_id=event.event_id)
        source_texts = collect_source_texts(raw_posts, source_limit=source_limit)
        if not source_texts:
            summary.append({"event_id": event.event_id, "status": "skip_no_source", "output": str(out_file)})
            continue

        source_likes = [max(0, int(getattr(post, "likes", 0))) for post in raw_posts]
        source_comments = [max(0, int(getattr(post, "comments", 0))) for post in raw_posts]
        source_retweets = [max(0, int(getattr(post, "retweet", 0))) for post in raw_posts]

        effective_base_likes = base_likes if base_likes >= 0 else (sum(source_likes) // max(1, len(source_likes)))
        effective_base_comments = base_comments if base_comments >= 0 else (sum(source_comments) // max(1, len(source_comments)))
        effective_base_retweet = base_retweet if base_retweet >= 0 else (sum(source_retweets) // max(1, len(source_retweets)))

        generated_posts = generate_posts_for_event(
            llm_client=llm_client,
            event_id=event.event_id,
            event_desc=event.description,
            event_evidence=event.evidence,
            source_texts=source_texts,
            total_count=per_event,
            batch_size=batch_size,
            rng=rng,
            show_progress=True,
        )

        source_times = [int(getattr(post, "post_time", 0)) for post in raw_posts if int(getattr(post, "post_time", 0)) > 0]
        generated_post_times = build_post_times_from_source(
            source_times=source_times,
            count=len(generated_posts),
            seed=seed + len(summary),
        )

        write_generated_posts_csv(
            output_file=out_file,
            event_id=event.event_id,
            posts=generated_posts,
            seed=seed + len(summary),
            post_times=generated_post_times,
            base_likes=effective_base_likes,
            base_comments=effective_base_comments,
            base_retweet=effective_base_retweet,
        )
        summary.append({"event_id": event.event_id, "status": "ok", "generated": len(generated_posts), "output": str(out_file)})

    report_path = Path(output_dir) / "generation_summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"fake_events": len(fake_events), "output_dir": str(output_dir), "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
