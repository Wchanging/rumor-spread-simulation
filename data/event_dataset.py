from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from domain.content import ContentItem, NormalPost, RumorPost
from domain.events import Event

from .loaders import EventLoader, PostLoader


@dataclass
class LoadedEventDataset:
    events: dict[str, Event] = field(default_factory=dict)
    initial_contents: list[ContentItem] = field(default_factory=list)


class EventDatasetLoader:
    def __init__(self) -> None:
        self._event_loader = EventLoader()
        self._post_loader = PostLoader()

    def load(
        self,
        events_file: str | Path,
        posts_dir: str | Path,
        posts_template: str = "{event_id}.csv",
        include_generated_rumor_posts: bool = False,
        generated_posts_dir: str | Path | None = None,
        generated_posts_template: str = "{event_id}_generated.csv",
        max_events: int | None = None,
        ensure_fake_event: bool = False,
        fake_event_count: int | None = None,
        exclude_evidence_posts_in_fake: bool = False,
        keep_evidence_posts_for_official_only: bool = False,
        randomize_selection: bool = False,
        selection_seed: int | None = None,
    ) -> LoadedEventDataset:
        raw_events = self._event_loader.load(events_file)
        raw_events = self._select_raw_events(
            raw_events=raw_events,
            max_events=max_events,
            ensure_fake_event=ensure_fake_event,
            fake_event_count=fake_event_count,
            randomize_selection=bool(randomize_selection),
            selection_seed=selection_seed,
        )

        posts_base = Path(posts_dir)
        generated_posts_base = Path(generated_posts_dir) if generated_posts_dir is not None else posts_base
        dataset = LoadedEventDataset()

        for raw_event in raw_events:
            event = Event(
                event_id=raw_event.event_id,
                description=raw_event.description,
                is_fake=raw_event.is_fake,
                evidence=raw_event.evidence,
                evidence_posts=raw_event.evidence_posts,
            )
            dataset.events[event.event_id] = event

            post_file = self._resolve_post_file(posts_base, raw_event.event_id, posts_template)
            if post_file is None:
                continue

            raw_posts = self._post_loader.load(post_file, event_id=raw_event.event_id)
            evidence_post_ids = set(raw_event.evidence_posts)
            for raw_post in raw_posts:
                base_kwargs = {
                    "content_id": f"src_{raw_event.event_id}_{raw_post.post_id}",
                    "event_id": raw_event.event_id,
                    "author_id": raw_post.user_id,
                    "text": self._merge_text_with_media(raw_post.text, raw_post.media_text),
                    "images": raw_post.images,
                    "videos": raw_post.videos,
                    "timestamp": raw_post.post_time,
                    "popularity": float(raw_post.likes + raw_post.comments + raw_post.retweet),
                }

                if exclude_evidence_posts_in_fake and raw_event.is_fake and raw_post.post_id in evidence_post_ids:
                    if keep_evidence_posts_for_official_only:
                        event.add_evidence_source_post(NormalPost(**base_kwargs))
                    continue

                content: ContentItem
                if raw_event.is_fake:
                    content = RumorPost(**base_kwargs)
                else:
                    content = NormalPost(**base_kwargs)
                dataset.initial_contents.append(content)

            if include_generated_rumor_posts and raw_event.is_fake:
                generated_file = self._resolve_post_file(generated_posts_base, raw_event.event_id, generated_posts_template)
                if generated_file is None:
                    continue
                generated_raw_posts = self._post_loader.load(generated_file, event_id=raw_event.event_id)
                for generated_post in generated_raw_posts:
                    generated_kwargs = {
                        "content_id": f"gen_{raw_event.event_id}_{generated_post.post_id}",
                        "event_id": raw_event.event_id,
                        "author_id": generated_post.user_id,
                        "text": self._merge_text_with_media(generated_post.text, generated_post.media_text),
                        "images": generated_post.images,
                        "videos": generated_post.videos,
                        "timestamp": generated_post.post_time,
                        "popularity": float(generated_post.likes + generated_post.comments + generated_post.retweet),
                    }
                    dataset.initial_contents.append(RumorPost(**generated_kwargs))

        return dataset

    @staticmethod
    def _select_raw_events(
        raw_events: list,
        max_events: int | None,
        ensure_fake_event: bool,
        fake_event_count: int | None,
        randomize_selection: bool,
        selection_seed: int | None,
    ) -> list:
        if not raw_events:
            return []

        ordered_events = list(raw_events)
        if randomize_selection:
            rng = random.Random(selection_seed)
            rng.shuffle(ordered_events)

        limit = len(ordered_events) if max_events is None else max(0, int(max_events))
        if limit <= 0:
            return []
        limit = min(limit, len(ordered_events))

        fake_events = [event for event in ordered_events if bool(getattr(event, "is_fake", False))]
        normal_events = [event for event in ordered_events if not bool(getattr(event, "is_fake", False))]

        if ensure_fake_event and not fake_events:
            raise ValueError("event_source.ensure_fake_event=true，但数据集中没有谣言事件。")

        if fake_event_count is not None:
            target_fake = max(0, int(fake_event_count))
            if ensure_fake_event:
                target_fake = max(1, target_fake)
            target_fake = min(target_fake, limit)

            if len(fake_events) < target_fake:
                raise ValueError(
                    f"event_source.fake_event_count={target_fake}，但数据集中仅有 {len(fake_events)} 个谣言事件。"
                )

            selected_ids: list[str] = [event.event_id for event in fake_events[:target_fake]]
            normal_slots = max(0, limit - len(selected_ids))
            selected_ids.extend(event.event_id for event in normal_events[:normal_slots])

            if len(selected_ids) < limit:
                extra_needed = limit - len(selected_ids)
                extra_fakes = fake_events[target_fake: target_fake + extra_needed]
                selected_ids.extend(event.event_id for event in extra_fakes)

            selected_set = set(selected_ids)
            selected_events = [event for event in ordered_events if event.event_id in selected_set]
            return selected_events[:limit]

        selected_events = list(ordered_events[:limit])
        if ensure_fake_event and fake_events and not any(event.is_fake for event in selected_events):
            first_fake = fake_events[0]
            replaced = False
            for idx in range(len(selected_events) - 1, -1, -1):
                if not selected_events[idx].is_fake:
                    selected_events[idx] = first_fake
                    replaced = True
                    break
            if not replaced and selected_events:
                selected_events[0] = first_fake

        deduped: list = []
        seen_event_ids: set[str] = set()
        for event in selected_events:
            if event.event_id in seen_event_ids:
                continue
            seen_event_ids.add(event.event_id)
            deduped.append(event)

        if len(deduped) < limit:
            for event in ordered_events:
                if event.event_id in seen_event_ids:
                    continue
                seen_event_ids.add(event.event_id)
                deduped.append(event)
                if len(deduped) >= limit:
                    break

        return deduped[:limit]

    @staticmethod
    def _merge_text_with_media(text: str, media_text: str) -> str:
        content_text = (text or "").strip()
        media_summary = (media_text or "").strip()
        if not media_summary:
            return content_text
        if not content_text:
            return f"[媒体信息] {media_summary}"
        return f"{content_text}\n[媒体信息] {media_summary}"

    @staticmethod
    def _resolve_post_file(posts_dir: Path, event_id: str, posts_template: str) -> Path | None:
        candidates = [
            posts_dir / posts_template.format(event_id=event_id),
            posts_dir / f"{event_id}_posts.csv",
            posts_dir / f"{event_id}.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None
