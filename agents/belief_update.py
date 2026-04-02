from __future__ import annotations

import json
import re
from typing import Any

from domain.content import ContentItem
from domain.users import UserState
from llm.base import LLMClient


class BeliefUpdateModule:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        step_size: float = 0.25,
        llm_user_simulation: dict[str, Any] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.step_size = step_size
        llm_cfg = llm_user_simulation or {}
        self.llm_user_mode_enabled = bool(llm_cfg.get("enabled", False))
        self.llm_memory_items = max(0, int(llm_cfg.get("memory_items", 3)))
        self.llm_delta_clip = max(0.01, float(llm_cfg.get("belief_delta_clip", 0.3)))
        self.enable_llm_trust_update = bool(llm_cfg.get("enable_llm_trust_update", True))
        self.llm_trust_delta_clip = max(0.0, float(llm_cfg.get("trust_delta_clip", 0.08)))
        self.enable_long_term_memory = bool(llm_cfg.get("enable_long_term_memory", True))
        self.long_term_memory_max_chars = max(80, int(llm_cfg.get("long_term_memory_max_chars", 500)))
        self.llm_thought_chars = max(20, int(llm_cfg.get("max_thought_chars", 120)))
        self.blind_user_mode = bool(llm_cfg.get("blind_user_mode", True))
        self.use_truth_label_in_prompt = bool(llm_cfg.get("use_truth_label_in_prompt", False))
        self.use_truth_label_in_base_update = bool(llm_cfg.get("use_truth_label_in_base_update", False))

    def update(
        self,
        user_state: UserState,
        content: ContentItem,
        timestep: int,
        recent_memories: list[dict[str, Any]] | None = None,
        long_term_memory: str = "",
        use_llm: bool = True,
    ) -> dict:
        current = user_state.get_belief(content.event_id).belief_score
        if self.blind_user_mode and not self.use_truth_label_in_base_update:
            signal = 0.0
        else:
            signal = 1.0 if content.is_rumor else -1.0
        trust_weight = 0.5 + 0.5 * user_state.platform_trust
        base_delta = self.step_size * signal * trust_weight * (1.0 - abs(current))

        llm_delta = 0.0
        llm_raw = ""
        llm_error = ""
        llm_thought = ""
        llm_suggested_like = None
        llm_suggested_share = None
        llm_suggested_rewrite = None
        llm_rewrite_text = ""
        llm_confidence = None
        trust_before = float(user_state.platform_trust)
        llm_trust_delta = 0.0
        llm_long_term_memory_update = ""
        if self.llm_client is not None and use_llm:
            try:
                if self.llm_user_mode_enabled:
                    llm_result = self._persona_decide(
                        user_state=user_state,
                        content=content,
                        current_belief=current,
                        recent_memories=recent_memories or [],
                        long_term_memory=long_term_memory,
                    )
                    llm_raw = str(llm_result.get("llm_raw", ""))
                    llm_delta = max(-self.llm_delta_clip, min(self.llm_delta_clip, float(llm_result.get("belief_delta", 0.0))))
                    llm_thought = str(llm_result.get("thought", ""))[: self.llm_thought_chars]
                    llm_suggested_like = llm_result.get("suggested_like")
                    llm_suggested_share = llm_result.get("suggested_share")
                    llm_suggested_rewrite = llm_result.get("suggested_rewrite")
                    llm_rewrite_text = str(llm_result.get("rewrite_text", ""))[:260]
                    llm_confidence = llm_result.get("confidence")
                    llm_long_term_memory_update = str(llm_result.get("long_term_memory_update", "")).strip()[
                        : self.long_term_memory_max_chars
                    ]
                    if self.enable_llm_trust_update:
                        llm_trust_delta = max(
                            -self.llm_trust_delta_clip,
                            min(self.llm_trust_delta_clip, float(llm_result.get("platform_trust_delta", 0.0))),
                        )
                else:
                    truth_text = (
                        f"content_is_rumor={int(content.is_rumor)}; "
                        if (not self.blind_user_mode or self.use_truth_label_in_prompt)
                        else ""
                    )
                    prompt = (
                        f"Current belief on event truthfulness={current:.3f}; platform_trust={user_state.platform_trust:.3f}; "
                        f"{truth_text}content={content.text}."
                        "\nOutput one float in [-1,1] as the suggested delta for belief in the original event claim."
                        "Positive means more belief in the original claim; negative means less belief in the original claim."
                        "If this content is credible debunking, the value should usually be negative."
                        "If this content supports the original claim and is credible, the value can be positive."
                        "Output only the numeric value with no extra text."
                    )
                    llm_resp = self.llm_client.generate(prompt)
                    llm_raw = llm_resp
                    llm_delta = max(-self.llm_delta_clip, min(self.llm_delta_clip, float(llm_resp)))
            except Exception as exc:
                llm_delta = 0.0
                llm_error = str(exc)
                llm_trust_delta = 0.0

        trust_after = max(0.0, min(1.0, trust_before + llm_trust_delta))
        user_state.platform_trust = trust_after

        new_score = max(-1.0, min(1.0, current + base_delta + llm_delta))
        user_state.update_belief(content.event_id, new_score, timestep)
        user_state.record_belief_binary(content.event_id)
        return {
            "belief_before": current,
            "base_delta": base_delta,
            "llm_delta": llm_delta,
            "belief_after": new_score,
            "llm_raw": llm_raw,
            "llm_error": llm_error,
            "llm_thought": llm_thought,
            "llm_suggested_like": llm_suggested_like,
            "llm_suggested_share": llm_suggested_share,
            "llm_suggested_rewrite": llm_suggested_rewrite,
            "llm_rewrite_text": llm_rewrite_text,
            "llm_confidence": llm_confidence,
            "llm_trust_delta": llm_trust_delta,
            "platform_trust_before": trust_before,
            "platform_trust_after": trust_after,
            "llm_long_term_memory_update": llm_long_term_memory_update,
        }

    def _persona_decide(
        self,
        user_state: UserState,
        content: ContentItem,
        current_belief: float,
        recent_memories: list[dict[str, Any]],
        long_term_memory: str,
    ) -> dict[str, Any]:
        memories = recent_memories[-self.llm_memory_items:] if self.llm_memory_items > 0 else []
        memory_lines = []
        for idx, item in enumerate(memories, start=1):
            memory_lines.append(
                f"{idx}) t={item.get('timestep')}, thought={item.get('thought', '')}, belief_after={item.get('belief_after')}"
            )
        memory_block = "\n".join(memory_lines) if memory_lines else "None"
        long_memory_block = str(long_term_memory or "").strip()
        if not long_memory_block:
            long_memory_block = "None"

        profile = {
            "user_id": user_state.user_id,
            "gender": user_state.gender,
            "age": user_state.age,
            "occupation": user_state.occupation,
            "education_level": user_state.education_level,
            "city_tier": user_state.city_tier,
            "big5_neuroticism": user_state.big5_neuroticism,
            "big5_extraversion": user_state.big5_extraversion,
            "big5_openness": user_state.big5_openness,
            "big5_agreeableness": user_state.big5_agreeableness,
            "big5_conscientiousness": user_state.big5_conscientiousness,
            "platform_trust": round(float(user_state.platform_trust), 4),
            "share_tendency": round(float(user_state.share_tendency), 4),
            "trust_threshold": round(float(user_state.trust_threshold), 4),
        }

        event_line = f"\nCurrent event: {content.event_id}"
        if (not self.blind_user_mode) or self.use_truth_label_in_prompt:
            event_line += f", is_rumor={int(content.is_rumor)}"

        prompt = (
            "Complete the user behavior decision for this round."
            f"\nUser profile: {json.dumps(profile, ensure_ascii=False)}"
            f"{event_line}"
            f"\nCurrent belief={current_belief:.3f}"
            f"\nCurrent content: {(content.text or '')[:800]}"
            f"\nRecent event memories:\n{memory_block}"
            f"\nLong-term event memory (cumulative summary):\n{long_memory_block}"
        )
        system_prompt = (
            "You are a social-platform user behavior simulator."
            " Act as the given user and make cognition/action decisions in first person."
            " Output exactly one JSON object and nothing else."
            " Required fields: belief_delta(float,-1..1), confidence(float,0..1), thought(string,<=120 chars),"
            " suggested_like(bool), suggested_share(bool), suggested_rewrite(bool), rewrite_text(string,<=220 chars),"
            " platform_trust_delta(float,-1..1), long_term_memory_update(string,<=300 chars)."
            " belief_delta is the change in belief about the event's original claim (not wording sentiment)."
            " Positive means stronger belief in the original claim; negative means weaker belief."
            " For credible debunking content, belief_delta should usually be negative."
            " platform_trust_delta is trust change after this exposure: positive=more trust, negative=less trust."
            " long_term_memory_update should be concise and accumulative."
            " If the stance shifts from believing to doubting, you may moderately increase share/rewrite probability"
            " to encourage fact-checking, but keep behavior natural."
            " confidence is confidence in the delta (0..1), and thought is a short inner monologue."
            " Ensure valid JSON and field constraints."
        )
        llm_resp = self.llm_client.generate(prompt, system_prompt=system_prompt)
        payload = self._parse_json_payload(llm_resp)
        payload["llm_raw"] = llm_resp
        return payload

    @staticmethod
    def _parse_json_payload(raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            return {}

        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()

        parsed: dict[str, Any] | None = None
        try:
            candidate = json.loads(text)
            if isinstance(candidate, dict):
                parsed = candidate
        except Exception:
            parsed = None

        if parsed is None:
            matches = re.findall(r"\{[\s\S]*?\}", text)
            for snippet in reversed(matches):
                try:
                    candidate = json.loads(snippet)
                    if isinstance(candidate, dict):
                        parsed = candidate
                        break
                except Exception:
                    continue

        if parsed is None:
            return BeliefUpdateModule._parse_partial_payload(text)

        result: dict[str, Any] = {}
        result["belief_delta"] = BeliefUpdateModule._safe_float(parsed.get("belief_delta"), 0.0)
        confidence = BeliefUpdateModule._safe_float(parsed.get("confidence"), 0.5)
        result["confidence"] = max(0.0, min(1.0, confidence))
        result["thought"] = str(parsed.get("thought", "")).strip()
        result["suggested_like"] = BeliefUpdateModule._safe_bool(parsed.get("suggested_like"))
        result["suggested_share"] = BeliefUpdateModule._safe_bool(parsed.get("suggested_share"))
        result["suggested_rewrite"] = BeliefUpdateModule._safe_bool(parsed.get("suggested_rewrite"))
        result["rewrite_text"] = str(parsed.get("rewrite_text", "")).strip()
        result["platform_trust_delta"] = BeliefUpdateModule._safe_float(parsed.get("platform_trust_delta"), 0.0)
        result["long_term_memory_update"] = str(parsed.get("long_term_memory_update", "")).strip()
        return result

    @staticmethod
    def _parse_partial_payload(text: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        result["belief_delta"] = BeliefUpdateModule._extract_float_field(text, "belief_delta", 0.0)
        confidence = BeliefUpdateModule._extract_float_field(text, "confidence", 0.5)
        result["confidence"] = max(0.0, min(1.0, confidence))
        result["thought"] = BeliefUpdateModule._extract_string_field(text, "thought")
        result["suggested_like"] = BeliefUpdateModule._extract_bool_field(text, "suggested_like")
        result["suggested_share"] = BeliefUpdateModule._extract_bool_field(text, "suggested_share")
        result["suggested_rewrite"] = BeliefUpdateModule._extract_bool_field(text, "suggested_rewrite")
        result["rewrite_text"] = BeliefUpdateModule._extract_string_field(text, "rewrite_text")
        result["platform_trust_delta"] = BeliefUpdateModule._extract_float_field(text, "platform_trust_delta", 0.0)
        result["long_term_memory_update"] = BeliefUpdateModule._extract_string_field(text, "long_term_memory_update")
        return result

    @staticmethod
    def _extract_float_field(text: str, field: str, default: float) -> float:
        pattern = rf'"{re.escape(field)}"\s*:\s*(-?\d+(?:\.\d+)?)'
        match = re.search(pattern, text)
        if not match:
            return float(default)
        return BeliefUpdateModule._safe_float(match.group(1), default)

    @staticmethod
    def _extract_bool_field(text: str, field: str) -> bool | None:
        pattern = rf'"{re.escape(field)}"\s*:\s*(true|false)'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None
        token = str(match.group(1)).lower()
        if token == "true":
            return True
        if token == "false":
            return False
        return None

    @staticmethod
    def _extract_string_field(text: str, field: str) -> str:
        closed_pattern = rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
        closed = re.search(closed_pattern, text)
        if closed:
            try:
                return str(json.loads(f'"{closed.group(1)}"')).strip()
            except Exception:
                return str(closed.group(1)).strip()

        open_pattern = rf'"{re.escape(field)}"\s*:\s*"([^\n\r]*)'
        opened = re.search(open_pattern, text)
        if opened:
            return str(opened.group(1)).strip()
        return ""

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _safe_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "1", "yes", "y"}:
                return True
            if v in {"false", "0", "no", "n"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return None
