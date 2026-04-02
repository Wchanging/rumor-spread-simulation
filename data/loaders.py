from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .preprocess import anonymize_id, parse_int, parse_media_list, parse_timestamp, sanitize_text


@dataclass
class RawEvent:
    event_id: str
    description: str
    is_fake: bool
    evidence: str
    posts: list[str] = field(default_factory=list)
    evidence_posts: list[str] = field(default_factory=list)


@dataclass
class RawPost:
    post_id: str
    event_id: str
    post_time: int
    user_id: str
    text: str
    images: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    media_text: str = ""
    likes: int = 0
    comments: int = 0
    retweet: int = 0


@dataclass
class RawComment:
    post_id: str
    comment_id: str
    comment_time: int
    user_id: str
    text: str
    images: list[str] = field(default_factory=list)
    videos: list[str] = field(default_factory=list)
    likes: int = 0
    is_replied: bool = False
    source_comment: str = ""
    replies: int = 0


@dataclass
class RawUser:
    user_id: str
    age: int | None = None
    gender: str | None = None
    occupation: str | None = None
    posts: str = ""
    comments: str = ""
    followees: int = 0
    followers: int = 0


class _BaseCSVLoader:
    def _read_rows(self, file_path: str | Path) -> Iterable[dict[str, Any]]:
        path = Path(file_path)
        with path.open("r", encoding="utf-8-sig", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                yield row


class EventLoader(_BaseCSVLoader):
    def load(self, file_path: str | Path) -> list[RawEvent]:
        events: list[RawEvent] = []
        for row in self._read_rows(file_path):
            event_id = str(row.get("Event", "")).strip()
            if not event_id:
                continue
            events.append(
                RawEvent(
                    event_id=event_id,
                    description=sanitize_text(row.get("Description", ""), min_len=0),
                    is_fake=parse_int(row.get("IsFake", 0)) == 1,
                    evidence=sanitize_text(row.get("Evidence", ""), min_len=0),
                    posts=parse_media_list(row.get("Posts", "[]")),
                    evidence_posts=parse_media_list(row.get("EvidencePosts", "[]")),
                )
            )
        return events


class PostLoader(_BaseCSVLoader):
    def load(self, file_path: str | Path, event_id: str) -> list[RawPost]:
        posts: list[RawPost] = []
        for row in self._read_rows(file_path):
            post_id = str(row.get("PostID", "")).strip()
            if not post_id:
                continue
            posts.append(
                RawPost(
                    post_id=post_id,
                    event_id=event_id,
                    post_time=parse_timestamp(row.get("PostTime", 0)),
                    user_id=str(row.get("User", "")).strip(),
                    text=sanitize_text(row.get("Text", ""), min_len=0),
                    images=parse_media_list(row.get("Img", "[]")),
                    videos=parse_media_list(row.get("Video", "[]")),
                    media_text=sanitize_text(row.get("MediaText", row.get("media_text", "")), min_len=0),
                    likes=parse_int(row.get("Likes", 0)),
                    comments=parse_int(row.get("Comments", 0)),
                    retweet=parse_int(row.get("Retweet", 0)),
                )
            )
        return posts


class CommentLoader(_BaseCSVLoader):
    def load(self, file_path: str | Path) -> list[RawComment]:
        comments: list[RawComment] = []
        for row in self._read_rows(file_path):
            comment_id = str(row.get("CommentID", "")).strip()
            if not comment_id:
                continue
            comments.append(
                RawComment(
                    post_id=str(row.get("PostID", "")).strip(),
                    comment_id=comment_id,
                    comment_time=parse_timestamp(row.get("CommentTime", 0)),
                    user_id=str(row.get("User", "")).strip(),
                    text=sanitize_text(row.get("Text", ""), min_len=0),
                    images=parse_media_list(row.get("Img", "[]")),
                    videos=parse_media_list(row.get("Video", "[]")),
                    likes=parse_int(row.get("Likes", 0)),
                    is_replied=parse_int(row.get("IsReplied", 0)) == 1,
                    source_comment=str(row.get("SourceComment", "")).strip(),
                    replies=parse_int(row.get("Replies", 0)),
                )
            )
        return comments


class UserLoader(_BaseCSVLoader):
    def __init__(self, anonymize: bool = True) -> None:
        self.anonymize = anonymize
        self._user_mapping: dict[str, str] = {}

    @property
    def user_mapping(self) -> dict[str, str]:
        return self._user_mapping

    def load(self, file_path: str | Path) -> list[RawUser]:
        users: list[RawUser] = []
        for row in self._read_rows(file_path):
            raw_user_id = str(row.get("User", "")).strip()
            if not raw_user_id:
                continue
            user_id = raw_user_id
            if self.anonymize:
                user_id = anonymize_id(raw_user_id, self._user_mapping)

            age_value = row.get("Age")
            age = parse_int(age_value, default=-1)
            users.append(
                RawUser(
                    user_id=user_id,
                    age=age if age >= 0 else None,
                    gender=str(row.get("Gender", "")).strip() or None,
                    occupation=str(row.get("Occupation", "")).strip() or None,
                    posts=sanitize_text(row.get("Posts", ""), min_len=0),
                    comments=sanitize_text(row.get("Comments", ""), min_len=0),
                    followees=parse_int(row.get("Followees", 0)),
                    followers=parse_int(row.get("Followers", 0)),
                )
            )
        return users
