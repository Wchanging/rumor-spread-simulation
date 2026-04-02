from __future__ import annotations

import os
import random
import threading
import time
from pathlib import Path
from typing import Any

from .base import LLMClient


def _load_env_file(env_path: str | Path) -> None:
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        content = line.strip()
        if not content or content.startswith("#") or "=" not in content:
            continue
        key, value = content.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        vision_model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_concurrency: int = 8,
        min_concurrency: int = 1,
        retry_max_attempts: int = 5,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 16.0,
        request_timeout: float = 60.0,
        adaptive_concurrency: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("The openai package is not installed. Please run: pip install openai") from exc

        self.model = model
        self.vision_model = vision_model or model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.max_concurrency = max(1, int(max_concurrency))
        self.min_concurrency = max(1, min(int(min_concurrency), self.max_concurrency))
        self.retry_max_attempts = max(1, int(retry_max_attempts))
        self.retry_base_delay = max(0.1, float(retry_base_delay))
        self.retry_max_delay = max(self.retry_base_delay, float(retry_max_delay))
        self.request_timeout = max(1.0, float(request_timeout))
        self.adaptive_concurrency = bool(adaptive_concurrency)

        self._adaptive_limit = self.max_concurrency
        self._active_requests = 0
        self._peak_active_requests = 0
        self._overload_streak = 0
        self._success_streak = 0
        self._condition = threading.Condition()
        self._stats_lock = threading.Lock()
        self._total_calls = 0
        self._total_retries = 0
        self._total_failures = 0
        self._total_overload_errors = 0

    @classmethod
    def from_config(cls, llm_config: dict[str, Any]) -> "OpenAIClient":
        env_file = llm_config.get("env_file")
        if env_file:
            _load_env_file(env_file)

        api_key = str(llm_config.get("api_key") or os.getenv(str(llm_config.get("api_key_env", "OPENAI_API_KEY")), ""))
        if not api_key:
            raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY in .env or environment variables.")

        base_url = llm_config.get("base_url") or os.getenv(str(llm_config.get("base_url_env", "OPENAI_BASE_URL")), None)

        return cls(
            model=str(llm_config.get("model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))),
            api_key=api_key,
            base_url=str(base_url) if base_url else None,
            vision_model=str(llm_config.get("vision_model", os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"))),
            temperature=float(llm_config.get("temperature", 0.2)),
            max_tokens=int(llm_config.get("max_tokens", 1024)),
            max_concurrency=int(llm_config.get("max_concurrency", 8)),
            min_concurrency=int(llm_config.get("min_concurrency", 1)),
            retry_max_attempts=int(llm_config.get("retry_max_attempts", 5)),
            retry_base_delay=float(llm_config.get("retry_base_delay", 1.0)),
            retry_max_delay=float(llm_config.get("retry_max_delay", 16.0)),
            request_timeout=float(llm_config.get("request_timeout", 60.0)),
            adaptive_concurrency=bool(llm_config.get("adaptive_concurrency", True)),
        )

    def generate(self, prompt: str, **kwargs) -> str:
        model = str(kwargs.get("model", self.model))
        system_prompt = str(kwargs.get("system_prompt", "You are a helpful assistant."))
        temperature = float(kwargs.get("temperature", self.temperature))
        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))

        def _request() -> str:
            response = self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=float(kwargs.get("timeout", self.request_timeout)),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""

        return self._with_retry_and_adaptive_concurrency(_request)

    def generate_multimodal(
        self,
        text: str,
        image_urls: list[str],
        **kwargs,
    ) -> str:
        model = str(kwargs.get("model", self.vision_model))
        system_prompt = str(kwargs.get("system_prompt", "You are a helpful multimodal assistant."))
        temperature = float(kwargs.get("temperature", self.temperature))
        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))

        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for url in image_urls:
            if not url:
                continue
            content.append({"type": "image_url", "image_url": {"url": url}})

        def _request() -> str:
            response = self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=float(kwargs.get("timeout", self.request_timeout)),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
            )
            return response.choices[0].message.content or ""

        return self._with_retry_and_adaptive_concurrency(_request)

    def _with_retry_and_adaptive_concurrency(self, fn):
        last_error: Exception | None = None
        for attempt in range(1, self.retry_max_attempts + 1):
            retry_sleep = 0.0
            self._acquire_slot()
            with self._stats_lock:
                self._total_calls += 1
            try:
                result = fn()
                self._on_success()
                return result
            except Exception as exc:
                last_error = exc
                is_transient = self._is_transient_error(exc)
                is_overload = self._is_overload_error(exc)
                if is_overload:
                    self._on_overload()
                else:
                    self._on_failure_no_overload()

                if (not is_transient) or attempt >= self.retry_max_attempts:
                    with self._stats_lock:
                        self._total_failures += 1
                    raise

                with self._stats_lock:
                    self._total_retries += 1
                sleep_time = min(self.retry_max_delay, self.retry_base_delay * (2 ** (attempt - 1)))
                retry_sleep = max(0.05, sleep_time * (1.0 + random.uniform(-0.2, 0.2)))
            finally:
                self._release_slot()

            if retry_sleep > 0:
                time.sleep(retry_sleep)

        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenAI request failed with an unknown error")

    def _acquire_slot(self) -> None:
        with self._condition:
            while self._active_requests >= self._adaptive_limit:
                self._condition.wait(timeout=0.1)
            self._active_requests += 1
            if self._active_requests > self._peak_active_requests:
                self._peak_active_requests = self._active_requests

    def _release_slot(self) -> None:
        with self._condition:
            self._active_requests = max(0, self._active_requests - 1)
            self._condition.notify_all()

    def _on_overload(self) -> None:
        with self._condition:
            self._overload_streak += 1
            self._success_streak = 0
            if self.adaptive_concurrency:
                new_limit = max(self.min_concurrency, self._adaptive_limit - 1)
                self._adaptive_limit = new_limit
            with self._stats_lock:
                self._total_overload_errors += 1
            self._condition.notify_all()

    def _on_failure_no_overload(self) -> None:
        with self._condition:
            self._success_streak = 0

    def _on_success(self) -> None:
        with self._condition:
            self._overload_streak = 0
            self._success_streak += 1
            if self.adaptive_concurrency and self._success_streak >= 20 and self._adaptive_limit < self.max_concurrency:
                self._adaptive_limit += 1
                self._success_streak = 0
            self._condition.notify_all()

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        message = str(exc).lower()
        transient_tokens = [
            "rate limit",
            "429",
            "503",
            "502",
            "504",
            "timeout",
            "temporarily unavailable",
            "connection",
            "overload",
            "too many requests",
        ]
        return any(token in message for token in transient_tokens)

    @staticmethod
    def _is_overload_error(exc: Exception) -> bool:
        message = str(exc).lower()
        overload_tokens = ["rate limit", "429", "overload", "too many requests", "quota"]
        return any(token in message for token in overload_tokens)

    def get_runtime_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = {
                "provider": "openai",
                "total_calls": self._total_calls,
                "total_retries": self._total_retries,
                "total_failures": self._total_failures,
                "total_overload_errors": self._total_overload_errors,
            }
        with self._condition:
            stats.update(
                {
                    "adaptive_concurrency_limit": self._adaptive_limit,
                    "max_concurrency": self.max_concurrency,
                    "min_concurrency": self.min_concurrency,
                    "active_requests": self._active_requests,
                    "peak_active_requests": self._peak_active_requests,
                }
            )
        return stats
