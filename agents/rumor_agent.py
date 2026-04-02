from __future__ import annotations

import json
import re
from typing import Any

from agents.attention import AttentionModule
from agents.base import Action, BaseAgent
from agents.belief_update import BeliefUpdateModule
from agents.sharing_policy import SharingPolicy
from domain.content import ContentItem
from domain.users import UserState
from llm.base import LLMClient


class RumorAgent(BaseAgent):
    def __init__(
        self,
        user_state: UserState,
        llm_client: LLMClient | None = None,
        attention_module: AttentionModule | None = None,
        belief_module: BeliefUpdateModule | None = None,
        sharing_policy: SharingPolicy | None = None,
        llm_user_simulation: dict[str, Any] | None = None,
    ) -> None:
        sim_cfg = llm_user_simulation or {}
        self.user_state = user_state
        self.llm_client = llm_client
        self.attention_module = attention_module or AttentionModule()
        self.belief_module = belief_module or BeliefUpdateModule(llm_client=llm_client, llm_user_simulation=sim_cfg)
        self.sharing_policy = sharing_policy or SharingPolicy()
        self.llm_user_mode_enabled = bool(sim_cfg.get("enabled", False))
        self.use_llm_action_decision = bool(sim_cfg.get("use_llm_action_decision", False))
        self.use_llm_attention_decision = bool(sim_cfg.get("use_llm_attention_decision", False))
        self.blind_user_mode = bool(sim_cfg.get("blind_user_mode", True))
        self.use_truth_label_in_attention_prompt = bool(sim_cfg.get("use_truth_label_in_attention_prompt", False))
        self.allow_llm_rewrite_text = bool(sim_cfg.get("allow_llm_rewrite_text", True))
        self.memory_items = max(1, int(sim_cfg.get("memory_items", 3)))
        self.enable_long_term_memory = bool(sim_cfg.get("enable_long_term_memory", True))
        self.long_term_memory_max_chars = max(80, int(sim_cfg.get("long_term_memory_max_chars", 500)))
        raw_llm_item_cap = sim_cfg.get("max_llm_items_per_round", None)
        self.max_llm_items_per_round = None if raw_llm_item_cap is None else max(0, int(raw_llm_item_cap))
        self.attention_min_browse = max(0, int(sim_cfg.get("attention_min_browse", 1)))
        self.attention_max_browse = max(1, int(sim_cfg.get("attention_max_browse", 8)))
        self.attention_candidate_limit = max(1, int(sim_cfg.get("attention_candidate_limit", 12)))
        self._last_attention_trace: dict[str, Any] = {}
        self._inbox: list[ContentItem] = []
        self._timestep: int = 0
        self._fake_event_ids: set[str] = set()
        self._last_selected_content_ids: list[str] = []
        self._last_decision_trace: list[dict] = []

    def perceive(self, messages: list[ContentItem], global_state) -> None:
        self._inbox = messages
        self._timestep = getattr(global_state, "timestep", 0)
        events = getattr(global_state, "events", {})
        if isinstance(events, dict):
            self._fake_event_ids = {
                str(event_id)
                for event_id, event in events.items()
                if bool(getattr(event, "is_fake", False))
            }
        else:
            self._fake_event_ids = set()

    def decide_actions(self) -> list[Action]:
        actions: list[Action] = []
        selected = self.attention_module.select(
            self._inbox,
            self.user_state,
            fake_event_ids=self._fake_event_ids,
        )
        self._last_attention_trace = {"mode": "rule", "candidate_count": len(selected)}
        if self.llm_user_mode_enabled and self.use_llm_attention_decision and self.llm_client is not None:
            selected = self._select_with_llm_attention(selected)
        self._last_selected_content_ids = [item.content_id for item in selected]
        self._last_decision_trace = []

        for idx, item in enumerate(selected):
            self.user_state.increment_seen(item.event_id, item.content_id, timestep=self._timestep)
            recent_memories = self.user_state.get_recent_event_memories(item.event_id, k=self.memory_items)
            long_term_memory = self.user_state.get_long_term_event_memory(item.event_id) if self.enable_long_term_memory else ""
            use_llm_for_item = True
            if self.max_llm_items_per_round is not None and idx >= self.max_llm_items_per_round:
                use_llm_for_item = False
            update_result = self.belief_module.update(
                self.user_state,
                item,
                self._timestep,
                recent_memories=recent_memories,
                long_term_memory=long_term_memory,
                use_llm=use_llm_for_item,
            )
            score = float(update_result["belief_after"])

            liked = self.sharing_policy.should_like(item, score)
            llm_like = update_result.get("llm_suggested_like")
            if self.llm_user_mode_enabled and self.use_llm_action_decision and llm_like is not None:
                liked = bool(llm_like)
            if liked:
                actions.append(
                    Action(
                        action_type="like",
                        actor_id=self.user_state.user_id,
                        event_id=item.event_id,
                        content=item,
                        payload={"belief_score": score, "like_boost": 1.0},
                    )
                )

            shared = self.sharing_policy.should_share(self.user_state, item, score)
            llm_share = update_result.get("llm_suggested_share")
            if self.llm_user_mode_enabled and self.use_llm_action_decision and llm_share is not None:
                shared = bool(llm_share)

            share_action_type = "none"
            if shared:
                share_action_type = self.sharing_policy.choose_share_action_type(item, score)
                llm_rewrite = update_result.get("llm_suggested_rewrite")
                if (
                    self.llm_user_mode_enabled
                    and self.use_llm_action_decision
                    and llm_rewrite is not None
                    and bool(llm_rewrite)
                ):
                    share_action_type = "rewrite_share"

                payload = {"belief_score": score}
                if share_action_type == "rewrite_share":
                    rewrite_text = ""
                    if self.allow_llm_rewrite_text:
                        rewrite_text = str(update_result.get("llm_rewrite_text", "")).strip()
                    payload["rewrite_text"] = rewrite_text or self._rewrite_text(item, score)
                actions.append(
                    Action(
                        action_type=share_action_type,
                        actor_id=self.user_state.user_id,
                        event_id=item.event_id,
                        content=item,
                        payload=payload,
                    )
                )

            memory_thought = str(update_result.get("llm_thought", "")).strip()
            if not memory_thought:
                memory_thought = f"看到内容后态度变化至 {score:.3f}"
            self.user_state.add_event_memory(
                item.event_id,
                {
                    "timestep": self._timestep,
                    "content_id": item.content_id,
                    "belief_after": round(score, 4),
                    "thought": memory_thought,
                    "shared": bool(shared),
                    "share_action_type": share_action_type,
                    "liked": bool(liked),
                },
                max_items=self.memory_items,
            )

            if self.enable_long_term_memory:
                llm_long_update = str(update_result.get("llm_long_term_memory_update", "")).strip()
                if llm_long_update:
                    self.user_state.update_long_term_event_memory(
                        item.event_id,
                        llm_long_update,
                        max_chars=self.long_term_memory_max_chars,
                        replace=True,
                    )
                else:
                    fallback_summary = (
                        f"t={self._timestep}: 最近态度{score:.3f}，"
                        f"对内容{item.content_id}的判断为{'倾向相信' if score >= 0 else '倾向怀疑'}。"
                    )
                    self.user_state.update_long_term_event_memory(
                        item.event_id,
                        fallback_summary,
                        max_chars=self.long_term_memory_max_chars,
                        replace=False,
                    )

            self._last_decision_trace.append(
                {
                    "content_id": item.content_id,
                    "event_id": item.event_id,
                    "is_rumor": item.is_rumor,
                    "belief_before": update_result["belief_before"],
                    "base_delta": update_result["base_delta"],
                    "llm_delta": update_result["llm_delta"],
                    "belief_after": update_result["belief_after"],
                    "llm_raw": update_result["llm_raw"],
                    "llm_error": update_result["llm_error"],
                    "llm_thought": update_result.get("llm_thought", ""),
                    "llm_suggested_like": update_result.get("llm_suggested_like"),
                    "llm_suggested_share": update_result.get("llm_suggested_share"),
                    "llm_suggested_rewrite": update_result.get("llm_suggested_rewrite"),
                    "llm_trust_delta": update_result.get("llm_trust_delta", 0.0),
                    "llm_long_term_memory_update": update_result.get("llm_long_term_memory_update", ""),
                    "platform_trust_before": update_result.get("platform_trust_before"),
                    "platform_trust_after": update_result.get("platform_trust_after"),
                    "liked": liked,
                    "shared": shared,
                    "share_action_type": share_action_type,
                }
            )
        return actions

    def update_state(self, feedback) -> None:
        trust_delta = float(feedback.get("platform_trust_delta", 0.0)) if isinstance(feedback, dict) else 0.0
        self.user_state.platform_trust = max(0.0, min(1.0, self.user_state.platform_trust + trust_delta))

    def collect_step_trace(self) -> dict:
        return {
            "selected_content_ids": list(self._last_selected_content_ids),
            "decision_trace": list(self._last_decision_trace),
            "attention_trace": dict(self._last_attention_trace),
        }

    def _rewrite_text(self, content: ContentItem, belief_score: float) -> str:
        source = (content.text or "").strip()
        if not source:
            return "转述：该信息值得关注，请自行核验来源。"

        stance = "我倾向认同这一说法" if belief_score >= 0 else "我倾向认为这一说法不够可靠"
        rewritten = f"{stance}。我的理解是：{source[:180]}"
        return rewritten[:260]

    def _select_with_llm_attention(self, candidates: list[ContentItem]) -> list[ContentItem]:
        if not candidates:
            return []

        pool = list(candidates[: self.attention_candidate_limit])
        pool_by_id = {item.content_id: item for item in pool}

        upper = min(len(pool), self.attention_max_browse)
        if self.user_state.attention_budget is not None:
            upper = min(upper, max(0, int(self.user_state.attention_budget)))
        if upper <= 0:
            self._last_attention_trace = {
                "mode": "llm",
                "fallback": "upper_bound_zero",
                "candidate_count": len(pool),
            }
            return []
        lower = min(self.attention_min_browse, upper)

        expose_truth_label = (not self.blind_user_mode) or self.use_truth_label_in_attention_prompt
        candidate_payload = []
        for item in pool:
            payload = {
                "content_id": item.content_id,
                "event_id": item.event_id,
                "timestamp": item.timestamp,
                "popularity": item.popularity,
                "text": (item.text or "")[:120],
            }
            if expose_truth_label:
                payload["is_rumor"] = item.is_rumor
            candidate_payload.append(payload)

        prompt = (
            f"用户画像: {json.dumps(self._profile_for_prompt(), ensure_ascii=False)}"
            f"\n候选内容列表: {json.dumps(candidate_payload, ensure_ascii=False)}"
            f"\n请在本轮从候选内容中选择浏览顺序。"
            f"\n输出要求: browse_count 在 [{lower}, {upper}] 之间；"
            "ordered_content_ids 必须来自候选content_id。"
        )
        system_prompt = (
            "你在模拟用户浏览行为。"
            "请只输出一个JSON对象，不要输出额外文字。"
            "字段固定：browse_count(int), ordered_content_ids(list[str]), thought(string,<=80字)。"
        )

        try:
            raw = self.llm_client.generate(prompt, system_prompt=system_prompt)
            payload = self._parse_attention_json(raw)
            browse_count = int(payload.get("browse_count", upper))
            browse_count = max(lower, min(upper, browse_count))

            ordered_ids = payload.get("ordered_content_ids", [])
            if not isinstance(ordered_ids, list):
                ordered_ids = []

            final_ids: list[str] = []
            seen: set[str] = set()
            for content_id in ordered_ids:
                cid = str(content_id)
                if cid in pool_by_id and cid not in seen:
                    final_ids.append(cid)
                    seen.add(cid)
                if len(final_ids) >= browse_count:
                    break

            if len(final_ids) < browse_count:
                for item in pool:
                    if item.content_id not in seen:
                        final_ids.append(item.content_id)
                        seen.add(item.content_id)
                    if len(final_ids) >= browse_count:
                        break

            self._last_attention_trace = {
                "mode": "llm",
                "candidate_count": len(pool),
                "selected_count": len(final_ids),
                "thought": str(payload.get("thought", ""))[:80],
            }
            return [pool_by_id[cid] for cid in final_ids if cid in pool_by_id]
        except Exception as exc:
            fallback = pool[:upper]
            self._last_attention_trace = {
                "mode": "llm",
                "fallback": "rule",
                "candidate_count": len(pool),
                "error": str(exc),
            }
            return fallback

    def _profile_for_prompt(self) -> dict[str, Any]:
        return {
            "user_id": self.user_state.user_id,
            "gender": self.user_state.gender,
            "age": self.user_state.age,
            "occupation": self.user_state.occupation,
            "education_level": self.user_state.education_level,
            "city_tier": self.user_state.city_tier,
            "big5_neuroticism": self.user_state.big5_neuroticism,
            "big5_extraversion": self.user_state.big5_extraversion,
            "big5_openness": self.user_state.big5_openness,
            "big5_agreeableness": self.user_state.big5_agreeableness,
            "big5_conscientiousness": self.user_state.big5_conscientiousness,
            "platform_trust": round(float(self.user_state.platform_trust), 4),
            "share_tendency": round(float(self.user_state.share_tendency), 4),
            "attention_budget": self.user_state.attention_budget,
        }

    @staticmethod
    def _parse_attention_json(raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            return {}
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

        matches = re.findall(r"\{[\s\S]*?\}", text)
        for snippet in reversed(matches):
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return {}
