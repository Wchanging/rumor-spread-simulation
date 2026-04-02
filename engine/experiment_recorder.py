from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ExperimentRecorder:
    def __init__(
        self,
        run_dir: str | Path,
        enabled: bool = True,
        record_user_trace: bool = True,
        max_content_chars: int = 180,
        flush_every_round: bool = True,
        save_plots: bool = True,
        plot_profile: str = "concise",
        pretty_json_output: bool = True,
        log_pruning_config: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self.record_user_trace = record_user_trace
        self.max_content_chars = max(50, int(max_content_chars))
        self.flush_every_round = flush_every_round
        self.save_plots = save_plots
        profile = str(plot_profile or "concise").strip().lower()
        self.plot_profile = profile if profile in {"full", "concise"} else "concise"
        self.pretty_json_output = bool(pretty_json_output)
        self.log_pruning_config = dict(log_pruning_config or {})
        self.log_mode = str(self.log_pruning_config.get("mode", "full"))
        self.include_user_profile_each_round = bool(self.log_pruning_config.get("include_user_profile_each_round", False))
        self.risk_score_threshold = float(self.log_pruning_config.get("risk_score_threshold", 0.65))
        self.risk_belief_threshold = float(self.log_pruning_config.get("belief_threshold", 0.55))
        self.risk_low_trust_threshold = float(self.log_pruning_config.get("low_trust_threshold", 0.25))
        self.periodic_full_round_interval = int(self.log_pruning_config.get("periodic_full_round_interval", 10))
        self.summary_decision_limit = int(self.log_pruning_config.get("summary_decision_limit", 4))
        self.summary_feed_limit = int(self.log_pruning_config.get("summary_feed_limit", 6))
        self._high_risk_users: set[str] = set()

        self.run_dir = Path(run_dir)
        self.user_memory_dir = self.run_dir / "user_memory"
        self.user_memory_summary_dir = self.run_dir / "user_memory_summary"
        self.user_profile_dir = self.run_dir / "user_profile"
        self.metrics_dir = self.run_dir / "metrics"
        self.logs_dir = self.run_dir / "logs"
        self.plots_dir = self.run_dir / "plots"

        self._metrics_headers: list[str] = []
        self._latest_metrics: dict[str, Any] = {}
        self._metrics_history: list[dict[str, Any]] = []
        self._profile_written_users: set[str] = set()

        if self.enabled:
            self.user_memory_dir.mkdir(parents=True, exist_ok=True)
            self.user_memory_summary_dir.mkdir(parents=True, exist_ok=True)
            self.user_profile_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.plots_dir.mkdir(parents=True, exist_ok=True)

    def save_run_metadata(self, metadata: dict[str, Any]) -> None:
        if not self.enabled:
            return
        metadata_file = self.run_dir / "run_meta.json"
        metadata = dict(metadata)
        metadata["recorded_at"] = datetime.now().isoformat(timespec="seconds")
        metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _records_file_path(self, base_dir: Path, name: str) -> Path:
        suffix = "json" if self.pretty_json_output else "jsonl"
        return base_dir / f"{name}.{suffix}"

    def _append_record(self, path: Path, payload: dict[str, Any]) -> None:
        if self.pretty_json_output:
            records: list[dict[str, Any]] = []
            if path.exists():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        records = loaded
                except Exception:
                    records = []
            records.append(payload)
            path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            return

        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def record_round(
        self,
        timestep: int,
        active_users: list[str],
        actions_count: int,
        metrics_snapshot: dict[str, Any],
        user_round_logs: list[dict[str, Any]],
        llm_stats: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        self._record_metrics_snapshot(metrics_snapshot)
        round_risk_stats = {"high_risk_user_count": 0, "full_log_count": 0, "summary_log_count": 0}
        if self.record_user_trace:
            for log in user_round_logs:
                self._write_user_profile_once(log)
                risk = self._assess_user_risk(log)
                log["risk_assessment"] = risk

                user_id = str(log.get("user_id", "unknown"))
                if risk["is_high_risk"]:
                    self._high_risk_users.add(user_id)
                if user_id in self._high_risk_users:
                    round_risk_stats["high_risk_user_count"] += 1

                if self._should_write_full_log(log, risk):
                    self._record_user_log(log)
                    round_risk_stats["full_log_count"] += 1

                if self.log_mode != "full":
                    self._record_user_summary_log(log, risk)
                    round_risk_stats["summary_log_count"] += 1

        self._record_round_summary(
            timestep,
            active_users,
            actions_count,
            llm_stats,
            round_risk_stats=round_risk_stats,
        )

        if self.flush_every_round:
            self._flush_latest_checkpoint()

    def finalize(self, final_result: dict[str, Any], sim_state, llm_stats: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return

        final_summary = {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "summary": final_result.get("summary", {}),
            "event_count": len(sim_state.events),
            "user_count": len(sim_state.users),
            "total_contents": len(sim_state.content_pool),
            "llm_stats": llm_stats or {},
        }
        (self.run_dir / "final_summary.json").write_text(
            json.dumps(final_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._write_user_final_states(sim_state)
        if self.save_plots:
            self._plot_metrics()

    def _record_user_log(self, log: dict[str, Any]) -> None:
        user_id = str(log.get("user_id", "unknown"))
        payload = dict(log)
        if not self.include_user_profile_each_round:
            payload.pop("user_profile", None)
        path = self._records_file_path(self.user_memory_dir, user_id)
        self._append_record(path, payload)

    def _write_user_profile_once(self, log: dict[str, Any]) -> None:
        user_id = str(log.get("user_id", "unknown"))
        if user_id in self._profile_written_users:
            return

        profile_payload = {
            "user_id": user_id,
            "user_profile": log.get("user_profile", {}),
            "first_seen_timestep": log.get("timestep", 0),
        }
        path = self.user_profile_dir / f"{user_id}.json"
        path.write_text(json.dumps(profile_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._profile_written_users.add(user_id)

    def _record_round_summary(
        self,
        timestep: int,
        active_users: list[str],
        actions_count: int,
        llm_stats: dict[str, Any] | None,
        round_risk_stats: dict[str, int] | None = None,
    ) -> None:
        path = self._records_file_path(self.logs_dir, "round_summary")
        payload = {
            "timestep": timestep,
            "active_user_count": len(active_users),
            "active_users": active_users,
            "actions_count": actions_count,
            "llm_stats": llm_stats or {},
            "risk_stats": round_risk_stats or {},
        }
        self._append_record(path, payload)

    def _assess_user_risk(self, log: dict[str, Any]) -> dict[str, Any]:
        state = log.get("state_snapshot", {})
        beliefs = state.get("beliefs", {})
        fake_event_ids = set(log.get("fake_event_ids", []))
        decision_trace = log.get("decision_trace", [])
        actions = log.get("actions", [])

        max_belief = 0.0
        for event_id, belief in beliefs.items():
            if fake_event_ids and event_id not in fake_event_ids:
                continue
            score = float(belief.get("belief_score", 0.0))
            max_belief = max(max_belief, score)

        rumor_selected_count = sum(1 for item in decision_trace if bool(item.get("is_rumor", False)))
        rumor_shared_count = sum(1 for item in decision_trace if bool(item.get("is_rumor", False)) and bool(item.get("shared", False)))
        total_selected = len(decision_trace)
        rumor_share_ratio = (rumor_shared_count / max(1, rumor_selected_count)) if rumor_selected_count > 0 else 0.0

        platform_trust = float(state.get("platform_trust", 0.5))
        low_trust_strength = 0.0
        if platform_trust < self.risk_low_trust_threshold:
            low_trust_strength = (self.risk_low_trust_threshold - platform_trust) / max(1e-6, self.risk_low_trust_threshold)

        belief_component = min(1.0, max_belief)
        share_component = min(1.0, rumor_share_ratio)
        trust_component = min(1.0, low_trust_strength)
        score = 0.5 * belief_component + 0.35 * share_component + 0.15 * trust_component

        reasons: list[str] = []
        if max_belief >= self.risk_belief_threshold:
            reasons.append(f"High misbelief tendency (max_belief={max_belief:.3f})")
        if rumor_shared_count > 0:
            reasons.append(f"Shared rumor this round (rumor_shared={rumor_shared_count})")
        if platform_trust < self.risk_low_trust_threshold:
            reasons.append(f"Low platform trust (platform_trust={platform_trust:.3f})")

        is_high_risk = score >= self.risk_score_threshold or rumor_shared_count >= 2
        return {
            "score": round(score, 4),
            "is_high_risk": bool(is_high_risk),
            "max_belief": round(max_belief, 4),
            "rumor_selected_count": rumor_selected_count,
            "rumor_shared_count": rumor_shared_count,
            "rumor_share_ratio": round(rumor_share_ratio, 4),
            "platform_trust": round(platform_trust, 4),
            "reasons": reasons,
            "total_selected": total_selected,
            "total_actions": len(actions),
        }

    def _should_write_full_log(self, log: dict[str, Any], risk: dict[str, Any]) -> bool:
        if self.log_mode == "full":
            return True

        user_id = str(log.get("user_id", "unknown"))
        timestep = int(log.get("timestep", 0))

        if risk.get("is_high_risk", False):
            return True
        if user_id in self._high_risk_users:
            return True

        if self.log_mode == "adaptive":
            interval = max(1, self.periodic_full_round_interval)
            return timestep % interval == 0

        if self.log_mode == "high_risk_only":
            return False

        return True

    def _record_user_summary_log(self, log: dict[str, Any], risk: dict[str, Any]) -> None:
        user_id = str(log.get("user_id", "unknown"))
        path = self._records_file_path(self.user_memory_summary_dir, user_id)

        decision_trace = list(log.get("decision_trace", []))
        compact_decisions = decision_trace[: max(1, self.summary_decision_limit)]

        feed = list(log.get("feed", []))
        compact_feed = feed[: max(1, self.summary_feed_limit)]

        state = log.get("state_snapshot", {})
        beliefs = state.get("beliefs", {})
        max_belief = 0.0
        max_belief_event = None
        for event_id, belief in beliefs.items():
            score = float(belief.get("belief_score", 0.0))
            if score > max_belief:
                max_belief = score
                max_belief_event = event_id

        payload = {
            "timestep": log.get("timestep"),
            "user_id": user_id,
            "risk_assessment": risk,
            "state_summary": {
                "platform_trust": state.get("platform_trust"),
                "top_belief_event": max_belief_event,
                "top_belief_score": max_belief,
                "belief_count": len(beliefs),
            },
            "feed_count": len(feed),
            "feed": compact_feed,
            "selected_content_ids": log.get("selected_content_ids", []),
            "decision_trace": compact_decisions,
            "actions_count": len(log.get("actions", [])),
        }
        self._append_record(path, payload)

    def _record_metrics_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._latest_metrics = snapshot
        self._metrics_history.append(snapshot)

        history_path = self._records_file_path(self.metrics_dir, "metrics_history")
        if self.pretty_json_output:
            history_path.write_text(json.dumps(self._metrics_history, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            with history_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

        csv_path = self.metrics_dir / "metrics_history.csv"
        if not self._metrics_headers:
            self._metrics_headers = list(snapshot.keys())
            with csv_path.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=self._metrics_headers)
                writer.writeheader()
                writer.writerow(snapshot)
        else:
            with csv_path.open("a", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=self._metrics_headers)
                writer.writerow({key: snapshot.get(key) for key in self._metrics_headers})

    def _flush_latest_checkpoint(self) -> None:
        if not self._latest_metrics:
            return
        checkpoint = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "latest_metrics": self._latest_metrics,
        }
        (self.run_dir / "latest_checkpoint.json").write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_user_final_states(self, sim_state) -> None:
        path = self._records_file_path(self.run_dir, "user_final_states")
        payloads: list[dict[str, Any]] = []
        for user_id, user_state in sim_state.users.items():
            payloads.append(
                {
                    "user_id": user_id,
                    "gender": user_state.gender,
                    "age": user_state.age,
                    "occupation": user_state.occupation,
                    "platform_trust": user_state.platform_trust,
                    "share_tendency": user_state.share_tendency,
                    "beliefs": {
                        event_id: {
                            "belief_score": belief.belief_score,
                            "seen_count": belief.seen_count,
                            "last_updated": belief.last_updated,
                        }
                        for event_id, belief in user_state.beliefs.items()
                    },
                }
            )

        if self.pretty_json_output:
            path.write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")
            return

        with path.open("w", encoding="utf-8") as fp:
            for payload in payloads:
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _plot_metrics(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        if not self._metrics_history:
            return

        x = [int(item.get("timestep", idx)) for idx, item in enumerate(self._metrics_history)]
        if self.plot_profile == "full":
            candidate_keys = [
                "misbelief_ratio",
                "platform_trust_mean",
                "total_shares",
                "intervention_cost",
                "polarization_abs_mean",
                "polarization_variance",
                "mean_event_opinion_variance",
                "mean_fake_event_opinion_variance",
                "global_disagreement",
                "fake_event_global_disagreement",
            ]
        else:
            candidate_keys = [
                "misbelief_ratio",
                "platform_trust_mean",
                "total_shares",
                "intervention_cost",
                "global_disagreement",
            ]

        for key in candidate_keys:
            y = [item.get(key) for item in self._metrics_history]
            if not any(value is not None for value in y):
                continue
            plt.figure(figsize=(8, 4))
            plt.plot(x, y, marker="o", linewidth=1.5)
            plt.title(key)
            plt.xlabel("timestep")
            plt.ylabel(key)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{key}.png", dpi=160)
            plt.close()

        self._plot_increment_metrics(x)
        self._plot_event_lines(x, concise=(self.plot_profile != "full"))
        self._plot_fake_event_lines(x, concise=(self.plot_profile != "full"))

    @staticmethod
    def _build_increment_series(values: list[Any]) -> list[float | None]:
        increments: list[float | None] = []
        prev: float | None = None
        for value in values:
            if value is None:
                increments.append(None)
                continue
            try:
                current = float(value)
            except Exception:
                increments.append(None)
                continue

            if prev is None:
                increments.append(max(0.0, current))
            else:
                increments.append(max(0.0, current - prev))
            prev = current
        return increments

    def _plot_increment_metrics(self, x: list[int]) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        if self.plot_profile == "full":
            candidate_keys = [
                ("total_shares", "new_shares_per_round"),
                ("total_rewrite_shares", "new_rewrite_shares_per_round"),
                ("total_likes", "new_likes_per_round"),
                ("total_posts", "new_posts_per_round"),
                ("intervention_cost", "new_intervention_cost_per_round"),
            ]
        else:
            candidate_keys = [
                ("total_shares", "new_shares_per_round"),
                ("total_posts", "new_posts_per_round"),
                ("intervention_cost", "new_intervention_cost_per_round"),
            ]

        for source_key, output_key in candidate_keys:
            raw_values = [item.get(source_key) for item in self._metrics_history]
            y = self._build_increment_series(raw_values)
            if not any(v is not None for v in y):
                continue

            plt.figure(figsize=(8, 4))
            plt.plot(x, y, marker="o", linewidth=1.5)
            plt.title(output_key)
            plt.xlabel("timestep")
            plt.ylabel(output_key)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_dir / f"{output_key}.png", dpi=160)
            plt.close()

    def _plot_nested_metric_lines(
        self,
        x: list[int],
        nested_key: str,
        metric_key: str,
        output_file: str,
        title: str,
        ylabel: str,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        event_ids: set[str] = set()
        for snapshot in self._metrics_history:
            nested = snapshot.get(nested_key, {})
            if isinstance(nested, dict):
                event_ids.update(str(event_id) for event_id in nested.keys())

        if not event_ids:
            return

        plt.figure(figsize=(9, 4.5))
        has_curve = False
        for event_id in sorted(event_ids):
            y = []
            for snapshot in self._metrics_history:
                nested = snapshot.get(nested_key, {})
                value = None
                if isinstance(nested, dict):
                    event_metrics = nested.get(event_id, {})
                    if isinstance(event_metrics, dict):
                        value = event_metrics.get(metric_key)
                y.append(value)
            if any(v is not None for v in y):
                has_curve = True
                plt.plot(x, y, marker="o", linewidth=1.5, label=event_id)

        if has_curve:
            plt.title(title)
            plt.xlabel("timestep")
            plt.ylabel(ylabel)
            plt.grid(alpha=0.3)
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            plt.savefig(self.plots_dir / output_file, dpi=160)
        plt.close()

    def _plot_nested_increment_lines(
        self,
        x: list[int],
        nested_key: str,
        metric_key: str,
        output_file: str,
        title: str,
        ylabel: str,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        event_ids: set[str] = set()
        for snapshot in self._metrics_history:
            nested = snapshot.get(nested_key, {})
            if isinstance(nested, dict):
                event_ids.update(str(event_id) for event_id in nested.keys())

        if not event_ids:
            return

        plt.figure(figsize=(9, 4.5))
        has_curve = False
        for event_id in sorted(event_ids):
            raw_values: list[Any] = []
            for snapshot in self._metrics_history:
                nested = snapshot.get(nested_key, {})
                value = None
                if isinstance(nested, dict):
                    event_metrics = nested.get(event_id, {})
                    if isinstance(event_metrics, dict):
                        value = event_metrics.get(metric_key)
                raw_values.append(value)

            y = self._build_increment_series(raw_values)
            if any(v is not None for v in y):
                has_curve = True
                plt.plot(x, y, marker="o", linewidth=1.5, label=event_id)

        if has_curve:
            plt.title(title)
            plt.xlabel("timestep")
            plt.ylabel(ylabel)
            plt.grid(alpha=0.3)
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            plt.savefig(self.plots_dir / output_file, dpi=160)
        plt.close()

    def _plot_event_lines(self, x: list[int], concise: bool = False) -> None:
        self._plot_event_attention_share_stacked(x)
        self._plot_nested_metric_lines(
            x=x,
            nested_key="event_attention_exposure_metrics",
            metric_key="selected_share",
            output_file="event_attention_exposure_share_by_round_lines.png",
            title="event_attention_exposure_share_by_round (line)",
            ylabel="share in selected attention items",
        )

        if concise:
            self._plot_nested_metric_lines(
                x=x,
                nested_key="event_exposure_metrics",
                metric_key="exposure_ratio",
                output_file="exposure_ratio_by_event.png",
                title="exposure_ratio_by_event",
                ylabel="exposure_ratio",
            )
            self._plot_nested_metric_lines(
                x=x,
                nested_key="fake_event_exposure_metrics",
                metric_key="misbelief_ratio_among_exposed",
                output_file="misbelief_ratio_among_exposed_by_fake_event.png",
                title="misbelief_ratio_among_exposed_by_fake_event",
                ylabel="misbelief_ratio_among_exposed",
            )
            return

        self._plot_nested_metric_lines(
            x=x,
            nested_key="event_opinion_metrics",
            metric_key="variance",
            output_file="opinion_variance_by_event.png",
            title="opinion_variance_by_event",
            ylabel="opinion_variance",
        )
        self._plot_nested_metric_lines(
            x=x,
            nested_key="event_neighbor_disagreement",
            metric_key="dg",
            output_file="neighbor_disagreement_by_event.png",
            title="neighbor_disagreement_by_event",
            ylabel="neighbor_disagreement_dg",
        )
        self._plot_nested_metric_lines(
            x=x,
            nested_key="event_exposure_metrics",
            metric_key="exposure_ratio",
            output_file="exposure_ratio_by_event.png",
            title="exposure_ratio_by_event",
            ylabel="exposure_ratio",
        )
        self._plot_nested_metric_lines(
            x=x,
            nested_key="event_exposure_metrics",
            metric_key="positive_belief_ratio_among_exposed",
            output_file="positive_belief_ratio_among_exposed_by_event.png",
            title="positive_belief_ratio_among_exposed_by_event",
            ylabel="positive_belief_ratio_among_exposed",
        )
        self._plot_nested_metric_lines(
            x=x,
            nested_key="event_exposure_metrics",
            metric_key="mean_seen_count_among_exposed",
            output_file="mean_seen_count_among_exposed_by_event.png",
            title="mean_seen_count_among_exposed_by_event",
            ylabel="mean_seen_count_among_exposed",
        )
        self._plot_nested_metric_lines(
            x=x,
            nested_key="event_cascade_metrics",
            metric_key="spreader_ratio",
            output_file="spreader_ratio_by_event.png",
            title="spreader_ratio_by_event",
            ylabel="spreader_ratio",
        )
        self._plot_nested_metric_lines(
            x=x,
            nested_key="event_cascade_metrics",
            metric_key="max_cascade_depth",
            output_file="max_cascade_depth_by_event.png",
            title="max_cascade_depth_by_event",
            ylabel="max_cascade_depth",
        )
        self._plot_nested_metric_lines(
            x=x,
            nested_key="fake_event_exposure_metrics",
            metric_key="misbelief_ratio_among_exposed",
            output_file="misbelief_ratio_among_exposed_by_fake_event.png",
            title="misbelief_ratio_among_exposed_by_fake_event",
            ylabel="misbelief_ratio_among_exposed",
        )
        self._plot_nested_increment_lines(
            x=x,
            nested_key="event_cascade_metrics",
            metric_key="share_count",
            output_file="new_share_count_by_event.png",
            title="new_share_count_by_event",
            ylabel="new_share_count",
        )
        self._plot_nested_increment_lines(
            x=x,
            nested_key="event_cascade_metrics",
            metric_key="content_count",
            output_file="new_content_count_by_event.png",
            title="new_content_count_by_event",
            ylabel="new_content_count",
        )

    def _plot_event_attention_share_stacked(self, x: list[int]) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        nested_key = "event_attention_exposure_metrics"
        event_ids: set[str] = set()
        for snapshot in self._metrics_history:
            nested = snapshot.get(nested_key, {})
            if isinstance(nested, dict):
                event_ids.update(str(event_id) for event_id in nested.keys())

        if not event_ids:
            return

        sorted_event_ids = sorted(event_ids)
        series: list[list[float]] = []
        has_signal = False
        for event_id in sorted_event_ids:
            values: list[float] = []
            for snapshot in self._metrics_history:
                nested = snapshot.get(nested_key, {})
                share = 0.0
                if isinstance(nested, dict):
                    metrics = nested.get(event_id, {})
                    if isinstance(metrics, dict):
                        try:
                            share = float(metrics.get("selected_share", 0.0))
                        except Exception:
                            share = 0.0
                values.append(max(0.0, min(1.0, share)))
            if any(v > 0 for v in values):
                has_signal = True
            series.append(values)

        if not has_signal:
            return

        plt.figure(figsize=(10, 4.8))
        plt.stackplot(x, *series, labels=sorted_event_ids, alpha=0.9)
        plt.title("event_attention_exposure_share_by_round (100% stacked)")
        plt.xlabel("timestep")
        plt.ylabel("share in selected attention items")
        plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.25)
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "event_attention_exposure_share_by_round_stacked.png", dpi=160)
        plt.close()

    def _plot_fake_event_lines(self, x: list[int], concise: bool = False) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        fake_event_ids: set[str] = set()
        for snapshot in self._metrics_history:
            per_event = snapshot.get("fake_event_belief_metrics", {})
            if isinstance(per_event, dict):
                fake_event_ids.update(str(event_id) for event_id in per_event.keys())

        if not fake_event_ids:
            return

        sorted_event_ids = sorted(fake_event_ids)

        plt.figure(figsize=(9, 4.5))
        has_curve = False
        for event_id in sorted_event_ids:
            y = []
            for snapshot in self._metrics_history:
                per_event = snapshot.get("fake_event_belief_metrics", {})
                value = None
                if isinstance(per_event, dict):
                    event_metrics = per_event.get(event_id, {})
                    if isinstance(event_metrics, dict):
                        value = event_metrics.get("misbelief_ratio")
                y.append(value)
            if any(v is not None for v in y):
                has_curve = True
                plt.plot(x, y, marker="o", linewidth=1.5, label=event_id)

        if has_curve:
            plt.title("misbelief_ratio_by_fake_event")
            plt.xlabel("timestep")
            plt.ylabel("misbelief_ratio")
            plt.grid(alpha=0.3)
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "misbelief_ratio_by_fake_event.png", dpi=160)
        plt.close()

        if concise:
            return

        plt.figure(figsize=(9, 4.5))
        has_curve = False
        for event_id in sorted_event_ids:
            y = []
            for snapshot in self._metrics_history:
                per_event = snapshot.get("fake_event_action_metrics", {})
                value = None
                if isinstance(per_event, dict):
                    event_metrics = per_event.get(event_id, {})
                    if isinstance(event_metrics, dict):
                        value = event_metrics.get("share_count")
                y.append(value)
            if any(v is not None for v in y):
                has_curve = True
                plt.plot(x, y, marker="o", linewidth=1.5, label=event_id)

        if has_curve:
            plt.title("share_count_by_fake_event")
            plt.xlabel("timestep")
            plt.ylabel("share_count")
            plt.grid(alpha=0.3)
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            plt.savefig(self.plots_dir / "share_count_by_fake_event.png", dpi=160)
        plt.close()

        self._plot_nested_increment_lines(
            x=x,
            nested_key="fake_event_action_metrics",
            metric_key="share_count",
            output_file="new_share_count_by_fake_event.png",
            title="new_share_count_by_fake_event",
            ylabel="new_share_count",
        )

    def build_user_round_log(
        self,
        timestep: int,
        user_id: str,
        user_state,
        feed,
        step_trace: dict[str, Any],
        actions: list,
        fake_event_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        feed_summary = [
            {
                "content_id": item.content_id,
                "event_id": item.event_id,
                "author_id": item.author_id,
                "is_rumor": item.is_rumor,
                "text": (item.text or "")[: self.max_content_chars],
                "timestamp": item.timestamp,
                "popularity": item.popularity,
            }
            for item in feed
        ]

        action_payload = []
        for action in actions:
            action_payload.append(
                {
                    "action_type": action.action_type,
                    "event_id": action.event_id,
                    "content_id": action.content.content_id if action.content else None,
                    "payload": action.payload,
                }
            )

        return {
            "timestep": timestep,
            "user_id": user_id,
            "user_profile": {
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
                "attention_budget": user_state.attention_budget,
                "online_probability": user_state.online_probability,
            },
            "state_snapshot": {
                "platform_trust": user_state.platform_trust,
                "trust_threshold": user_state.trust_threshold,
                "share_tendency": user_state.share_tendency,
                "long_term_event_memories": dict(getattr(user_state, "long_term_event_memories", {})),
                "beliefs": {
                    event_id: {
                        "belief_score": belief.belief_score,
                        "seen_count": belief.seen_count,
                        "last_updated": belief.last_updated,
                    }
                    for event_id, belief in user_state.beliefs.items()
                },
            },
            "feed": feed_summary,
            "selected_content_ids": step_trace.get("selected_content_ids", []),
            "decision_trace": step_trace.get("decision_trace", []),
            "actions": action_payload,
            "fake_event_ids": list(fake_event_ids or []),
        }
