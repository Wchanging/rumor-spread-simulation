from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from agents.base import Action
from domain.content import ContentItem
from engine.dispatcher import Dispatcher
from engine.experiment_recorder import ExperimentRecorder
from engine.feed_builder import FeedBuilder
from engine.scheduler import Scheduler
from engine.sim_state import SimulationState
from intervention.base import InterventionStrategy
from metrics.belief_metrics import compute_belief_metrics
from metrics.collector import MetricsCollector
from domain.content import DebunkPost


class SimulationEngine:
    def __init__(
        self,
        sim_state: SimulationState,
        agents: dict,
        intervention_strategy: InterventionStrategy,
        metrics_collector: MetricsCollector,
        scheduler: Scheduler | None = None,
        dispatcher: Dispatcher | None = None,
        feed_builder: FeedBuilder | None = None,
        max_feed_size: int = 20,
        intervention_cost_per_post: float = 1.0,
        intervention_start_timestep: int = 0,
        intervention_min_fake_shares: int = 0,
        intervention_deactivate_fake_misbelief_below: float | None = None,
        intervention_deactivate_min_exposed_users: int = 0,
        intervention_force_targets_active_when_posting: bool = False,
        decision_workers: int = 1,
        recorder: ExperimentRecorder | None = None,
        show_progress_bar: bool = True,
        show_round_summary: bool = True,
        progress_mininterval: float = 0.2,
        share_relabel_neutral_band: float = 0.0,
    ) -> None:
        self.sim_state = sim_state
        self.agents = agents
        self.intervention_strategy = intervention_strategy
        self.metrics_collector = metrics_collector
        self.scheduler = scheduler or Scheduler()
        self.dispatcher = dispatcher or Dispatcher()
        self.feed_builder = feed_builder or FeedBuilder()
        self.max_feed_size = max_feed_size
        self.intervention_cost_per_post = intervention_cost_per_post
        self.intervention_start_timestep = max(0, int(intervention_start_timestep))
        self.intervention_min_fake_shares = max(0, int(intervention_min_fake_shares))
        self.intervention_deactivate_fake_misbelief_below = (
            None
            if intervention_deactivate_fake_misbelief_below is None
            else max(0.0, min(1.0, float(intervention_deactivate_fake_misbelief_below)))
        )
        self.intervention_deactivate_min_exposed_users = max(0, int(intervention_deactivate_min_exposed_users))
        self.intervention_force_targets_active_when_posting = bool(intervention_force_targets_active_when_posting)
        self._intervention_permanently_deactivated = False
        self.decision_workers = max(1, int(decision_workers))
        self.recorder = recorder
        self.show_progress_bar = bool(show_progress_bar)
        self.show_round_summary = bool(show_round_summary)
        self.progress_mininterval = max(0.0, float(progress_mininterval))
        self.share_relabel_neutral_band = max(0.0, float(share_relabel_neutral_band))
        self._warned_tqdm_unavailable = False

    def run(self, T: int) -> dict:
        for t in range(T):
            self.sim_state.timestep = t
            forced_active_users = self._apply_intervention()

            active_users = self.scheduler.pick_active_users(self.sim_state.users, timestep=t)
            if self.intervention_force_targets_active_when_posting and forced_active_users:
                active_set = set(active_users)
                for user_id in forced_active_users:
                    if user_id not in self.sim_state.users:
                        continue
                    if user_id in active_set:
                        continue
                    active_users.append(user_id)
                    active_set.add(user_id)
                    user_state = self.sim_state.users[user_id]
                    user_state.last_active_timestep = t
                    user_state.activation_count += 1
            user_context: dict[str, dict] = {}

            for user_id in active_users:
                if user_id not in self.agents:
                    continue
                feed = self.feed_builder.build_feed(
                    user_id=user_id,
                    sim_state=self.sim_state,
                    dispatcher=self.dispatcher,
                    max_feed_size=self.max_feed_size,
                )
                agent = self.agents[user_id]
                agent.perceive(feed, self.sim_state)
                user_context[user_id] = {"feed": feed, "agent": agent}

            per_user_actions = self._decide_actions_in_parallel(active_users, user_context, timestep=t)

            actions_count = 0
            user_round_logs: list[dict] = []
            selected_content_ids_by_user: dict[str, list[str]] = {}
            for user_id in active_users:
                if user_id not in per_user_actions:
                    continue
                actions = per_user_actions[user_id]
                self._apply_actions(actions)
                actions_count += len(actions)

                agent = user_context[user_id]["agent"]
                step_trace = agent.collect_step_trace() if hasattr(agent, "collect_step_trace") else {}
                selected_content_ids_by_user[user_id] = list(step_trace.get("selected_content_ids", []) or [])
                if self.recorder is not None:
                    user_round_logs.append(
                        self.recorder.build_user_round_log(
                            timestep=t,
                            user_id=user_id,
                            user_state=self.sim_state.users[user_id],
                            feed=user_context[user_id]["feed"],
                            step_trace=step_trace,
                            actions=actions,
                            fake_event_ids=[eid for eid, event in self.sim_state.events.items() if event.is_fake],
                        )
                    )
                self.sim_state.users[user_id].snapshot_trust()

            snapshot = self.metrics_collector.collect(t, self.sim_state)
            snapshot.update(
                self._compute_attention_exposure_metrics(
                    selected_content_ids_by_user=selected_content_ids_by_user,
                    decision_user_count=len(user_context),
                )
            )
            self._print_round_summary(
                timestep=t,
                active_user_count=len(active_users),
                decision_user_count=len(user_context),
                completed_user_count=len(per_user_actions),
                actions_count=actions_count,
                snapshot=snapshot,
            )
            if self.recorder is not None:
                llm_stats = self._collect_llm_stats()
                self.recorder.record_round(
                    timestep=t,
                    active_users=active_users,
                    actions_count=actions_count,
                    metrics_snapshot=snapshot,
                    user_round_logs=user_round_logs,
                    llm_stats=llm_stats,
                )

        final_result = self.metrics_collector.finalize()
        if self.recorder is not None:
            self.recorder.finalize(final_result, self.sim_state, llm_stats=self._collect_llm_stats())
        completed_rounds = len(self.metrics_collector.history)
        last_timestep = completed_rounds - 1 if completed_rounds > 0 else -1
        print(
            f"[run completed] completed_rounds={completed_rounds} last_timestep={last_timestep}",
            flush=True,
        )
        return final_result

    def _decide_actions_in_parallel(
        self,
        active_users: list[str],
        user_context: dict[str, dict],
        timestep: int,
    ) -> dict[str, list[Action]]:
        if self.decision_workers <= 1:
            results: dict[str, list[Action]] = {}
            user_ids = [user_id for user_id in active_users if user_id in user_context]
            progress = self._create_progress_bar(total=len(user_ids), desc=f"Round {timestep} decisions")
            for user_id in user_ids:
                try:
                    results[user_id] = user_context[user_id]["agent"].decide_actions()
                except Exception:
                    results[user_id] = []
                if progress is not None:
                    progress.update(1)
            if progress is not None:
                progress.close()
            return results

        results: dict[str, list[Action]] = {}
        with ThreadPoolExecutor(max_workers=self.decision_workers, thread_name_prefix="sim-agent") as executor:
            future_map = {
                executor.submit(user_context[user_id]["agent"].decide_actions): user_id
                for user_id in active_users
                if user_id in user_context
            }
            progress = self._create_progress_bar(total=len(future_map), desc=f"Round {timestep} decisions")

            for future in as_completed(future_map):
                user_id = future_map[future]
                try:
                    results[user_id] = future.result()
                except Exception:
                    results[user_id] = []
                if progress is not None:
                    progress.update(1)
            if progress is not None:
                progress.close()
        return results

    def _create_progress_bar(self, total: int, desc: str):
        if total <= 0 or not self.show_progress_bar:
            return None
        if tqdm is None:
            if not self._warned_tqdm_unavailable:
                print("[runtime] tqdm 未安装，已自动跳过进度条显示。")
                self._warned_tqdm_unavailable = True
            return None
        return tqdm(total=total, desc=desc, leave=False, mininterval=self.progress_mininterval)

    def _print_round_summary(
        self,
        timestep: int,
        active_user_count: int,
        decision_user_count: int,
        completed_user_count: int,
        actions_count: int,
        snapshot: dict[str, Any],
    ) -> None:
        if not self.show_round_summary:
            return
        misbelief = float(snapshot.get("misbelief_ratio", 0.0))
        rumor_exposure = float(snapshot.get("rumor_exposure_rate", 0.0))
        debunk_exposure = float(snapshot.get("debunk_exposure_rate", 0.0))
        normal_exposure = float(snapshot.get("normal_exposure_rate", 0.0))
        empty_feed_rate = float(snapshot.get("empty_feed_rate", 0.0))
        intervention_cost = float(snapshot.get("intervention_cost", 0.0))
        message = (
            "[round {t}] active={active} decision_users={decide} completed={done} "
            "actions={actions} misbelief={mis:.3f} rumor_exp={rumor_exp:.3f} "
            "debunk_exp={debunk_exp:.3f} normal_exp={normal_exp:.3f} "
            "empty_feed={empty_feed:.3f} cost={cost:.1f}".format(
                t=timestep,
                active=active_user_count,
                decide=decision_user_count,
                done=completed_user_count,
                actions=actions_count,
                mis=misbelief,
                rumor_exp=rumor_exposure,
                debunk_exp=debunk_exposure,
                normal_exp=normal_exposure,
                empty_feed=empty_feed_rate,
                cost=intervention_cost,
            )
        )
        if self.show_progress_bar and tqdm is not None:
            try:
                tqdm.write(message)
                return
            except Exception:
                pass
        print(message, flush=True)

    def _compute_attention_exposure_metrics(
        self,
        selected_content_ids_by_user: dict[str, list[str]],
        decision_user_count: int,
    ) -> dict[str, Any]:
        total_selected = 0
        rumor_selected = 0
        debunk_selected = 0
        normal_selected = 0
        event_selected_counts: dict[str, int] = {}

        for selected_ids in selected_content_ids_by_user.values():
            total_selected += len(selected_ids)
            for content_id in selected_ids:
                item = self.sim_state.content_pool.get(str(content_id))
                if item is None:
                    continue
                event_id = str(getattr(item, "event_id", ""))
                if event_id:
                    event_selected_counts[event_id] = event_selected_counts.get(event_id, 0) + 1
                event = self.sim_state.events.get(str(getattr(item, "event_id", "")))
                is_fake_event = bool(getattr(event, "is_fake", False))
                if isinstance(item, DebunkPost) or (is_fake_event and not bool(getattr(item, "is_rumor", False))):
                    debunk_selected += 1
                elif bool(getattr(item, "is_rumor", False)):
                    rumor_selected += 1
                else:
                    normal_selected += 1

        denominator = float(total_selected) if total_selected > 0 else 1.0
        empty_feed_users = sum(1 for ids in selected_content_ids_by_user.values() if not ids)
        empty_feed_rate = (float(empty_feed_users) / float(decision_user_count)) if decision_user_count > 0 else 0.0
        event_attention_exposure_metrics: dict[str, dict[str, float]] = {}
        for event_id, count in event_selected_counts.items():
            event_attention_exposure_metrics[event_id] = {
                "selected_count": float(count),
                "selected_share": float(count) / denominator,
            }

        return {
            "attention_items_total": float(total_selected),
            "rumor_exposure_rate": float(rumor_selected) / denominator,
            "debunk_exposure_rate": float(debunk_selected) / denominator,
            "normal_exposure_rate": float(normal_selected) / denominator,
            "empty_feed_rate": float(empty_feed_rate),
            "event_attention_exposure_metrics": event_attention_exposure_metrics,
        }

    def _collect_llm_stats(self) -> dict:
        for agent in self.agents.values():
            llm_client = getattr(agent, "llm_client", None)
            if llm_client is not None and hasattr(llm_client, "get_runtime_stats"):
                try:
                    return llm_client.get_runtime_stats()
                except Exception:
                    return {}
        return {}

    def _apply_intervention(self) -> list[str]:
        if not self._is_intervention_activated():
            return []

        targets = self.intervention_strategy.select_targets(self.sim_state)
        interventions = self.intervention_strategy.generate_interventions(targets, self.sim_state)
        if not interventions:
            return []

        for post in interventions:
            self.sim_state.add_content(
                post,
                timestep=self.sim_state.timestep,
                is_intervention=True,
                cost=self.intervention_cost_per_post,
            )

        personalized_posts = [post for post in interventions if post.parent_content_id in self.sim_state.users]
        user_authored_posts = [
            post
            for post in interventions
            if post.parent_content_id not in self.sim_state.users and post.author_id in self.sim_state.users
        ]
        broadcast_posts = [
            post
            for post in interventions
            if post.parent_content_id not in self.sim_state.users and post.author_id not in self.sim_state.users
        ]

        for post in personalized_posts:
            self.dispatcher.dispatch_to_targets([post], [str(post.parent_content_id)])

        for post in user_authored_posts:
            self.dispatcher.dispatch([post], self.sim_state)

        if broadcast_posts:
            if targets:
                self.dispatcher.dispatch_to_targets(broadcast_posts, targets)
            else:
                self.dispatcher.dispatch_to_targets(broadcast_posts, list(self.sim_state.users.keys()))

        return [str(target) for target in targets]

    def _is_intervention_activated(self) -> bool:
        if self._intervention_permanently_deactivated:
            return False
        if int(self.sim_state.timestep) < self.intervention_start_timestep:
            return False
        if self._should_deactivate_intervention_now():
            self._intervention_permanently_deactivated = True
            print(
                "[intervention] auto-deactivated at t={t}: fake misbelief among exposed users dropped below threshold.".format(
                    t=int(self.sim_state.timestep)
                )
            )
            return False
        if self.intervention_min_fake_shares <= 0:
            return True
        return self._count_fake_event_shares() >= self.intervention_min_fake_shares

    def _should_deactivate_intervention_now(self) -> bool:
        if self.intervention_deactivate_fake_misbelief_below is None:
            return False
        if not self.sim_state.users:
            return False

        belief_metrics = compute_belief_metrics(self.sim_state)
        exposed_ratio = float(belief_metrics.get("users_exposed_to_any_fake_ratio", 0.0))
        exposed_users = int(round(exposed_ratio * len(self.sim_state.users)))
        if exposed_users < self.intervention_deactivate_min_exposed_users:
            return False

        misbelief_among_exposed_users = float(
            belief_metrics.get("fake_event_misbelief_ratio_among_exposed_users", 0.0)
        )
        return misbelief_among_exposed_users <= self.intervention_deactivate_fake_misbelief_below

    def _count_fake_event_shares(self) -> int:
        fake_event_ids = {event_id for event_id, event in self.sim_state.events.items() if bool(event.is_fake)}
        if not fake_event_ids:
            return 0
        return sum(
            1
            for action in self.sim_state.action_log
            if str(action.get("event_id", "")) in fake_event_ids
            and str(action.get("action_type", "")) in {"share", "rewrite_share"}
        )

    def _apply_actions(self, actions: list[Action]) -> None:
        for action in actions:
            if action.content is None:
                continue
            source = action.content

            if action.action_type == "like":
                like_boost = float(action.payload.get("like_boost", 1.0))
                source.popularity = float(source.popularity) + max(0.0, like_boost)
                self.sim_state.user_like_count[action.actor_id] += 1
                self.sim_state.action_log.append(
                    {
                        "timestep": self.sim_state.timestep,
                        "actor_id": action.actor_id,
                        "action_type": "like",
                        "event_id": action.event_id,
                        "content_id": source.content_id,
                        "parent_content_id": source.content_id,
                    }
                )
                continue

            if action.action_type not in {"share", "rewrite_share"}:
                continue

            text = source.text
            if action.action_type == "rewrite_share":
                rewrite_text = str(action.payload.get("rewrite_text", "")).strip()
                if rewrite_text:
                    text = rewrite_text

            event = self.sim_state.events.get(str(action.event_id))
            is_fake_event = bool(getattr(event, "is_fake", False))
            belief_score = float(action.payload.get("belief_score", 0.0))
            # Keep source label for users in the neutral band [-band, band].
            knows_fake_and_refutes = is_fake_event and belief_score < -self.share_relabel_neutral_band

            base_content_kwargs = {
                "content_id": f"{action.action_type}_{action.actor_id}_{self.sim_state.timestep}_{uuid.uuid4().hex[:8]}",
                "event_id": action.event_id,
                "author_id": action.actor_id,
                "text": text,
                "images": list(source.images),
                "videos": list(source.videos),
                "timestamp": self.sim_state.timestep,
                "popularity": source.popularity + 1,
                "parent_content_id": source.content_id,
            }
            if knows_fake_and_refutes:
                new_content = DebunkPost(**base_content_kwargs)
            else:
                new_content = ContentItem(
                    **base_content_kwargs,
                    is_rumor=source.is_rumor,
                )
            self.sim_state.add_content(new_content, timestep=self.sim_state.timestep)
            self.sim_state.user_share_count[action.actor_id] += 1
            self.sim_state.action_log.append(
                {
                    "timestep": self.sim_state.timestep,
                    "actor_id": action.actor_id,
                    "action_type": action.action_type,
                    "event_id": action.event_id,
                    "content_id": new_content.content_id,
                    "parent_content_id": source.content_id,
                }
            )
            self.dispatcher.dispatch([new_content], self.sim_state)
