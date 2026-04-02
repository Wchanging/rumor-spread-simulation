# 配置参数参考（精简版，2026-03-22）

本文档作为实验配置的主参考，优先覆盖 `exp_base_official.json` 与 RQ1~RQ3 批量配置。

---

## 1. 运行与输出

### `simulation`
- `T`：总轮数。
- `n_runs`：重复运行次数。
- `seed`：随机种子。
- `decision_workers`：并发决策线程数。
- `runtime_monitor.show_progress_bar`：是否显示决策进度条。
- `runtime_monitor.show_round_summary`：是否输出每轮摘要。

### `recording`
- `enabled`：是否保存实验输出。
- `output_root`：输出根目录。
- `keep_metrics_history`：是否持久化历史指标。
- `save_plots`：是否导出图表。

---

## 2. 事件与网络

### `event_source`
- `mode`：事件来源模式（`csv`/`config`）。
- `events_file`、`posts_dir`、`posts_template`：事件与帖子数据路径。
- `max_events`：每次实验采样事件上限。
- `ensure_fake_event`、`fake_event_count`：假事件数量控制。
- `generated_rumor_posts`：是否加载离线生成谣言补充。

### `network`
- `type`：网络类型（如 `small_world`、`scale_free`、`random`；兼容别名 `erdos_renyi`）。
- 类型参数：
  - `small_world`: `k`, `p`
  - `scale_free`: `m`
  - `random`（兼容 `erdos_renyi`）: `p`
  - 度数对齐建议：若希望 random 与 small_world 的平均出度同量级，可近似取 `p ≈ k/(n-1)`。
- `persistence.*`：网络持久化与复用。

---

## 3. 用户与行为

### `user_model`
- `attention_budget`：每轮注意力上限。
- `ensure_rumor_in_attention`、`min_rumor_items_in_attention`：谣言曝光硬约束（主要用于 RQ1 hard）。
- `ensure_non_fake_event_in_attention`、`min_non_fake_items_in_attention`：非 fake 内容最小保留。
- `intervention_priority_boost`：辟谣内容注意力优先增益。
- `event_repeat_penalty`：同事件重复惩罚。
- `online_probability`、`platform_trust`、`share_tendency`、`trust_threshold`：用户行为基础参数。

### `action_model`
- `belief_threshold`：触发分享阈值。
- `enable_rewrite_share`、`rewrite_share_ratio`：改写转发开关与概率。
- `enable_like` 及 like 概率参数。
- `blind_user_mode`：行为策略是否不使用真值标签。

### `llm_user_simulation`
- `enabled`：是否启用 LLM 用户模拟。
- `use_llm_action_decision`、`use_llm_attention_decision`：行为/注意力由 LLM 建议。
- `max_llm_items_per_round`：每轮 LLM 处理条数上限。
- `belief_delta_clip`：信念增量裁剪。
- `enable_llm_trust_update`：是否启用信任更新（当前论文主线默认关闭）。

---

## 4. Feed 机制（RQ1 关键）

### `simulation.feed_event_allocator`
- `enabled`：是否启用事件级分配。
- `temperature`：事件采样温度（越高越随机）。
- `social_weight`、`popularity_weight`、`novelty_weight`、`fatigue_weight`：事件效用权重。
- `early_diversity_rounds`、`early_event_cap`：前期多样性约束。

### `simulation.feed_fallback`
- `enabled`、`allow_global`：空 feed 回填策略。
- `rumor_candidate_fallback`：谣言候选硬回填（RQ1 hard 常用）。

---

## 5. 干预策略参数

### 通用
- `intervention.strategy`：
  - `NoIntervention`
  - `GlobalBroadcastDebunk`
  - `TargetTopKSpreaders`
  - `PersonalizedDebunk`
  - `OfficialAccountDebunk`
- `intervention.cost_per_post`：单帖成本。
- `intervention.start_timestep`：策略级发帖开始轮次（真正控制该策略何时开始发帖）。
- `intervention.activation.start_timestep`：引擎级最早激活轮次（监控/触发系统何时开始生效）。
- `intervention.activation.min_fake_shares`：激活前 fake 分享门槛。

说明：`intervention.start_timestep` 与 `intervention.activation.start_timestep` 是两层控制；前者管“策略执行排期”，后者管“引擎激活门槛”。

### `OfficialAccountDebunk`
- `official_account_count`：官方账号数。
- `start_timestep`、`interval`、`end_timestep`：发帖时间窗。
- `posts_per_round`：每轮总辟谣帖数。
- `event_selection_mode`：`round_robin` 或 `risk_weighted`。
- `recent_action_window`：近期热度回看窗口。
- `max_posts_per_event`：单轮单事件辟谣帖上限。
- `event_heat_weight`：风险评分中的传播热度权重。
- `event_misbelief_weight`：风险评分中的误信比例权重。
- `event_exposure_weight`：风险评分中的曝光覆盖权重。
- `activation.force_targets_active_when_posting`：发帖轮强制激活官方账号。

### 其他策略

#### `GlobalBroadcastDebunk`（升级后）
- 通用时序控制：`start_timestep`, `interval`, `end_timestep`
- 预算控制：`posts_per_round`, `max_posts_per_event`
- 事件选择：`event_selection_mode`（`risk_weighted` / `round_robin`）
- 评分权重：`event_heat_weight`, `event_misbelief_weight`, `event_coverage_gap_weight`
- 近期窗口：`recent_action_window`
- 语气：`tone_style`

说明：相较旧版“每轮对所有 fake 事件各发一条”，新版可按风险和覆盖缺口分配预算，更可控。

#### `TargetTopKSpreaders`（升级后）
- 目标规模：`k`
- 目标筛选增强：`recent_action_window`, `min_share_count`, `recent_share_weight`, `total_share_weight`, `degree_weight`
- 通用时序控制：`start_timestep`, `interval`, `end_timestep`
- 预算控制：`posts_per_round`, `max_posts_per_event`
- 事件选择：`event_selection_mode`（`risk_weighted` / `round_robin`）
- 评分权重：`event_heat_weight`, `event_misbelief_weight`, `event_coverage_gap_weight`
- 语气：`tone_style`

说明：相较旧版“仅按累计分享次数取 Top-K”，新版支持“近期传播 + 历史传播 + 网络结构”混合打分。

#### `PersonalizedDebunk`（升级后）
- 核心阈值：`threshold`
- 目标上界：`max_target_belief`（用于过滤极端高信用户）
- 目标与预算控制：`max_targets_per_round`, `per_user_post_cap`, `posts_per_round`
- 通用时序控制：`start_timestep`, `interval`, `end_timestep`
- 事件预算控制：`max_posts_per_event`
- 事件选择：`event_selection_mode`（`risk_weighted` / `round_robin`）
- 评分权重：`event_heat_weight`, `event_misbelief_weight`, `event_coverage_gap_weight`
- 近期窗口：`recent_action_window`
- 分层语气：`tone_style`, `high_risk_tone_style`, `high_risk_threshold`

说明：相较旧版“目标用户 × fake事件”全展开发帖，新版按用户风险排序并限制单用户/全局/单事件预算，并支持按风险加权分配跨事件预算，避免单事件独占。

---

## 6. 主指标字段（输出）

### 每轮（`metrics_history`）
- `misbelief_ratio`
- `rumor_exposure_rate`
- `debunk_exposure_rate`
- `normal_exposure_rate`
- `empty_feed_rate`
- `intervention_cost`

### 终局（summary）
- `final_misbelief_ratio`
- `misbelief_auc`
- `peak_misbelief_ratio`
- `peak_misbelief_timestep`
- `final_intervention_cost`
- `final_rumor_exposure_rate`
- `final_debunk_exposure_rate`
- `final_normal_exposure_rate`
- `final_empty_feed_rate`

注：`AUC reduction`、`Efficiency(reduction/cost)` 属于对照后处理指标。

---

## 7. RQ3 推荐口径（同预算下投放时机 × 投放节奏）

- 推荐配置集合：
  - `exp_rq3_steady_early.json`
  - `exp_rq3_steady_mid.json`
  - `exp_rq3_steady_late.json`
  - `exp_rq3_burst_early.json`
  - `exp_rq3_burst_mid.json`
  - `exp_rq3_burst_late.json`
- 策略固定：
  - 统一 `OfficialAccountDebunk`
- 总预算约束（6个配置统一）：
  - `cost_per_post = 1.0`
  - 总投放数 `=100`（即总成本=100）
  - `max_posts_per_event = 4`
- 时机组（steady，10轮平稳投放）：
  - early：`start_timestep=2`, `end_timestep=11`, `posts_per_round=10`
  - mid：`start_timestep=5`, `end_timestep=14`, `posts_per_round=10`
  - late：`start_timestep=10`, `end_timestep=19`, `posts_per_round=10`
- 节奏组（burst，3轮集中投放）：
  - 通过 `round_post_schedule` 精确分配 100 帖
  - early：`{2:34,3:33,4:33}`
  - mid：`{5:34,6:33,7:33}`
  - late：`{10:34,11:33,12:33}`
- 激活门槛建议：
  - `intervention.activation.start_timestep = 0`
  - `intervention.activation.min_fake_shares = 0`
  - 目的：聚焦时机与节奏效应，避免激活阈值引入额外混杂。

---

## 8. RQ1 推荐口径（背景内容竞争强度）

- 推荐配置集合：
  - `exp_rq1_isolated_no.json`
  - `exp_rq1_isolated_official.json`
  - `exp_rq1_moderate_no.json`
  - `exp_rq1_moderate_official.json`
  - `exp_rq1_high_no.json`
  - `exp_rq1_high_official.json`
- 竞争强度定义（通过 `event_source`）：
  - isolated：`max_events=3`, `fake_event_count=3`
  - moderate：`max_events=6`, `fake_event_count=3`
  - high：`max_events=10`, `fake_event_count=3`
- 统一口径：
  - `network.type = small_world`
  - `NoIntervention` vs `OfficialAccountDebunk`
  - official 干预时窗：`start_timestep=3`, `interval=1`, `end_timestep=14`, `posts_per_round=10`
- 建议后处理指标：
  - `Benefit (AUC reduction)`
  - `Debunk crowding-out gap`（相对 isolated 的 debunk exposure 下降）
