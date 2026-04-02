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
            raise RuntimeError("读取YAML失败，请安装PyYAML或改用JSON配置。") from exc
        return yaml.safe_load(text)

    raise ValueError(f"不支持的配置格式: {suffix}")


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
    raise ValueError(f"不支持的 llm provider: {provider}")


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
        "网传消息：{text}",
        "有人在群里说：{text}",
        "看到消息称：{text}",
        "据说：{text}",
        "未证实信息：{text}",
    ]
    results: list[str] = []
    for idx in range(count):
        source = source_texts[idx % len(source_texts)]
        pattern = patterns[idx % len(patterns)]
        marker = rng.choice(["", "，听着挺真", "，很多人都在转", "，你们怎么看"])
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
        progress_bar = tqdm(total=total_count, desc=f"生成 {event_id}", leave=False, unit="post")

    for i in range(rounds):
        before_count = len(generated)
        remain = total_count - len(generated)
        if remain <= 0:
            break
        ask_n = min(max(1, batch_size), remain)
        rumor_claim = str(event_desc or "").strip()[:220]
        evidence_fact = str(event_evidence or "").strip()[:220]
        prompt = (
            f"事件ID: {event_id}\n"
            f"谣言核心说法(可改写扩写): {rumor_claim}\n"
            f"辟谣事实要点(仅用于避免写成辟谣帖): {evidence_fact}\n"
            f"已有网传说法样本: {json.dumps(source_texts, ensure_ascii=False)}\n"
            f"已生成样本(避免重复，尽量多样性一些，模拟各种类型的用户): {json.dumps(generated[-8:], ensure_ascii=False)}\n"
            f"请生成 {ask_n} 条新的谣言传播帖，要求：口语化、彼此不重复、与谣言核心说法一致但措辞不同、每条不超过200字。\n"
            "必须满足：\n"
            "1) 站在网传/听说/转述角度写，不要写辟谣、澄清、官方回应、求证结论。\n"
            "2) 不要出现‘这是谣言/不实/别信/勿信/经核实/官方辟谣/并非’等表达。\n"
            "3) 不要直接复述辟谣事实要点。 \n"
            "4) 多样性一些，模拟各种类型的用户发帖人，营销号，普通用户，吃瓜群众等不同风格的表达"
        )
        system_prompt = (
            "你是社交平台谣言帖改写助手。"
            "只生成谣言传播口吻，不生成辟谣或纠错口吻。"
            "只输出JSON，格式必须是{\"posts\":[\"...\",\"...\"]}，不要任何解释。"
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
    parser = argparse.ArgumentParser(description="为每个谣言事件离线生成补充帖CSV。")
    parser.add_argument("--config", default=None, help="可选：实验配置文件路径（JSON/YAML）")
    parser.add_argument("--events-file", default=None, help="事件总表CSV路径")
    parser.add_argument("--posts-dir", default=None, help="原始事件帖子目录")
    parser.add_argument("--posts-template", default="{event_id}.csv", help="原始帖子文件模板")
    parser.add_argument("--per-event", type=int, default=30, help="每个谣言事件生成帖子数")
    parser.add_argument("--source-posts-per-event", type=int, default=6, help="每个事件用于改写的源帖数量")
    parser.add_argument("--batch-size", type=int, default=10, help="单次LLM请求生成数量")
    parser.add_argument("--use-llm", action="store_true", help="启用LLM生成（默认关闭，使用模板变体）")
    parser.add_argument("--llm-provider", default="openai", choices=["openai", "mock"], help="离线生成使用的LLM提供方")
    parser.add_argument("--llm-env-file", default=None, help="可选：OpenAI环境变量文件路径")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI模型名")
    parser.add_argument("--llm-max-concurrency", type=int, default=4, help="OpenAI并发上限")
    parser.add_argument("--overwrite", action="store_true", help="强制覆盖已存在的生成文件")
    parser.add_argument("--output-dir", default="data/processed/generated_rumor_posts", help="生成帖输出目录")
    parser.add_argument("--output-template", default="{event_id}_generated.csv", help="生成帖文件名模板")
    parser.add_argument("--base-likes", type=int, default=-1, help="生成帖基础likes，<0表示自动按源帖估计")
    parser.add_argument("--base-comments", type=int, default=-1, help="生成帖基础comments，<0表示自动按源帖估计")
    parser.add_argument("--base-retweet", type=int, default=-1, help="生成帖基础retweet，<0表示自动按源帖估计")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument("--patch-existing-timestamps", action="store_true", help="只修复已生成CSV的PostTime，不生成新文本")
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
        raise FileNotFoundError("events_file 不存在，请传 --events-file 或在 --config 中提供 event_source.events_file")
    if posts_dir is None or not posts_dir.exists():
        raise FileNotFoundError("posts_dir 不存在，请传 --posts-dir 或在 --config 中提供 event_source.posts_dir")
    if output_dir is None:
        raise FileNotFoundError("output_dir 无效")

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
    event_iter = tqdm(fake_events, desc="处理fake事件", unit="event")
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
