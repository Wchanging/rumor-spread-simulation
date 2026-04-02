from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any


def sanitize_text(text: Any, min_len: int = 2) -> str:
    if text is None:
        return ""
    value = str(text)
    value = value.replace("\u3000", " ").strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\S\r\n]+", " ", value)
    if len(value) < min_len:
        return ""
    return value


def parse_media_list(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]

    value = str(raw_value).strip()
    if not value or value in {"[]", "nan", "None"}:
        return []

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass

    value = value.strip("[]")
    if not value:
        return []
    parts = [part.strip().strip("'\"") for part in value.split(",")]
    return [part for part in parts if part]


def parse_timestamp(raw_value: Any) -> int:
    if raw_value is None:
        return 0

    if isinstance(raw_value, (int, float)):
        return int(raw_value)

    value = str(raw_value).strip()
    if not value:
        return 0

    if value.isdigit():
        return int(value)

    time_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ]
    for fmt in time_formats:
        try:
            dt = datetime.strptime(value, fmt)
            return int(dt.timestamp())
        except ValueError:
            continue

    return 0


def anonymize_id(raw_id: Any, mapping: dict[str, str]) -> str:
    source = str(raw_id)
    if source not in mapping:
        mapping[source] = str(uuid.uuid5(uuid.NAMESPACE_DNS, source))
    return mapping[source]


def parse_int(raw_value: Any, default: int = 0) -> int:
    try:
        if raw_value is None:
            return default
        return int(float(raw_value))
    except (TypeError, ValueError):
        return default
