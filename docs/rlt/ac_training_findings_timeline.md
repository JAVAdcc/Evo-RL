# AC 模型训练发现时间线（截至 2026-04-20）

> 目的：把 `docs/rlt` 和 coder 实验日志里的 AC 训练发现收敛成一份时间线，只保留当前仍有解释力的核心结论。若前后冲突，以后者为准。
>
> 主要来源：
> - `docs/rlt/ac_overnight_experiments.md`
> - `docs/rlt/0415ac训练.md`
> - `docs/rlt/ac_hp_sweep_cp405_results_20260420.md`
> - `docs/rlt/rlt_pipeline_review_20260415_1839.md`
> - coder `outputs/ac_search_478ep_cp_sft/results.json`
> - coder `outputs/stage1_sb1_pretrain/results.json`
> - coder `outputs/stage2_sb1_mixed/results.json`
> - coder `outputs/beta_sweep_cp405_post_p0/results.json`
> - coder `outputs/datamix_cp405_post_p0/*/results.json`
> - coder `outputs/datamix_renorm_teleop_post_p0/*/results.json`

## 当前有效结论

- **pre-P0 的数值型 β 结论全部失效**。4/20 review 确认：旧代码的 BC 项按 `mean` 缩小了 120 倍，而且 offline 实际一直跑的是 UTD=1，不是配置里的 5。为了避免混淆，本文统一按 **`beta_current ~= beta_preP0 / 120`** 理解旧实验；凡是 pre-P0 的“β=3/5/10 最优”都不再直接可用。
- **当前 post-P0 的 β sweet spot 是一个平台，不是尖峰**。CP405-only sweep 里：`β=0.001` 发散，`0.003/0.01` 边缘，`β>=0.03` 全部健康；`0.1/0.3/1.0/3.0` 的 `q_gap / ref_mse / td_error` 基本持平。当前继续训练时优先看 `β=0.3~1.0`，若必须单点选型，先用 `β=0.3`。
- **现在最大的真实问题不是 critic 发散，而是 actor 严重 ref-copying**。post-P0 健康配置下 `ref_mse ~= 0.001`，但 `ref_dropped_mse ~= 0.17~0.18`，相差约 180 倍，说明 actor 仍主要靠 `ref_chunk` 决策，state-only 映射几乎没学出来。
- **数据混合是目前最有效的增益杠杆**。加 `teleop141` 会显著降低 `ref_dropped_mse`，加 `intervene156` 单独收益较小，三桶混合通常最好。当前最可信的数据配方是 **`cp405 + renorm_teleop + intervene`**。
- **teleop 的收益大部分是真的，不只是 bucket shortcut**。把 teleop 重归一化到 CP405 坐标系后，`+both` 配置的 `ref_dropped_mse` 仍持平或更好，同时 teleop bucket 上的 `expert_mse` 大致减半，说明收益主要来自更好的 state→action 泛化，而不是仅仅识别了 bucket 来源。
- **以后不要再只看 `ref_mse` 选模型**。当前最值得盯的指标顺序：`ref_dropped_mse`、`q_gap`、`td_error`、per-bucket 指标。

## 时间线

### 2026-04-11 ~ 2026-04-12：初版 stride-2 / 478ep CP+SFT 搜索（pre-P0，数值结论已过期）

- 当时的核心发现是：修复 `actual_steps/stride` 后，critic 不再是“死的”，Q 真正开始传播；因此旧时代靠低 `ref_mse` 选出来的 checkpoint 不能再当作 RL 真学到了东西。
- 当时一度得到“`β=3~4` 最优、`50K` 比 `100K` 更稳”的结论，但这些结论都建立在 **旧 β 约定 + offline UTD 实际为 1** 的前提上，已经不能直接沿用。
- 这个阶段唯一保留下来的结论是：**固定 buffer 上的 offline RL 很容易在 Q/BC 失衡时漂掉**，所以之后任何 sweep 都不能只看 `ref_mse`。

### 2026-04-15 上午：Stage1 / Stage2 SB1 日志（pre-P0，保留“趋势”，不保留绝对超参）

- `stage1_sb1_pretrain`（纯 warmup）在旧约定 `β=5` 下是健康基线：`mix_ref_mse=0.00143`、`q_gap=0.0029`、`td_error=0.0071`。
- `stage2_sb1_mixed`（`0.2/0.4/0.4` 三桶混合，warm-start 自 stage1）把 `expert_mse` 从 `0.093` 压到 `0.059`，同时 `q_gap=0.0083`、`td_error=0.0114` 仍在健康区。
- 这个阶段保留下来的核心结论是：**混合 human / rollout 数据是有效方向，能提高策略质量而不立刻把 critic 弄坏**。这个趋势后来被 post-P0 的数据混合消融再次验证。

### 2026-04-15 夜：reward_scale sweep（pre-P0，具体 winner 已废弃）

- 这一轮第一次明确说明：**只看 `ref_mse` 会选到死 critic**。例如 `rs=0` 虽然 `ref_mse` 最低，但 Q 完全塌到 0，本质上只是纯 BC。
- 同时也明确了一个现在仍然成立的原则：**actor loss 的本质是 `-Q` 和 BC 正则的量级平衡问题**。`Q` 过大或 BC 过弱，actor 就会去 chase critic 误差。
- 这轮提出的四个 critic 健康指标是后来最有价值的遗产：`mean_q_policy`、`td_error`、`q_gap`、terminal calibration。
- 但这轮里诸如“`rs=0.05, β=5` 最佳”或“`rs=0.1, β=10` 次优”的配方，今天都不应再直接使用，因为 **P0 之后 β 数值语义已经变了**。

### 2026-04-19 ~ 2026-04-20：P0 复盘 + post-P0 CP405 sweep（当前基准）

- 4/20 review 把历史结论重置了两次：
  - **β 语义重置**：旧实验里的数值型 β 全都偏弱 120 倍。
  - **训练循环语义重置**：旧 offline AC search 实际是 UTD=1，不是配置中的 UTD=5。
- 在这个新基线上，CP405-only β sweep 的结论很清楚：
  - `β=0.001` 发散。
  - `β=0.003 / 0.01` 边缘。
  - `β>=0.03` 健康。
  - `β=0.1 / 0.3 / 1.0 / 3.0` 几乎是平坦高原，没有谁明显碾压其他点。
- 这一轮最重要的新发现不是 β 本身，而是 **严重 ref-copying**：健康 β 下 `ref_mse` 已经接近 0，但 `ref_dropped_mse` 依然在 `0.17~0.18`，说明 `ref_dropout_p=0.7` 远远不够。

### 2026-04-19 夜：post-P0 数据混合消融（当前有效）

- 在 `β=0.3` 下：
  - `cp405_only`: `ref_dropped_mse=0.176`
  - `+teleop`: `0.138`（约 -21%）
  - `+intervene`: `0.161`（约 -8%）
  - `+both`: `0.137`（约 -22%）
- 在 `β=1.0` 下趋势相同：`+teleop` 和 `+both` 明显优于 `cp405_only`，`+intervene` 单独收益较小。
- 这说明 **teleop bucket 才是目前最强的 state→action 监督来源**；intervene 有帮助，但主要像辅助项，不是主增益来源。
- 同时，所有混合配置的 `q_gap / td_error` 都仍然健康，所以这不是“用更多 BC 换掉 critic”的伪提升，而是 **真正没有明显伤到 critic 的泛化改善**。

### 2026-04-20 凌晨：renormalized teleop 控制实验（当前有效）

- 这轮实验专门验证：之前 `+teleop` 的收益，会不会只是因为不同 bucket 的动作归一化范围不一致，让 actor 偷看到了“样本来自哪个 bucket”。
- 结果是：
  - `β=0.3` 下，`+teleop` 单路从 `0.138` 回退到 `0.146`，说明原始收益里确实混入了一点 shortcut。
  - 但 `+both` 从 `0.137` 进一步降到 `0.133`；`β=1.0` 下 `+teleop` / `+both` 也都持平或更好。
  - teleop bucket 上的 `expert_mse` 从约 `0.005` 下降到约 `0.0027~0.0032`，接近减半。
- 所以当前最稳妥的解释是：**bucket shortcut 确实存在，但不是主要收益来源；把 teleop 重归一化之后，真实收益反而更可信了。**

## 现在应当直接丢弃的旧结论

- 任何 pre-P0 的“`β=3/4/5/10` 最优”一类结论。
- 任何 pre-P0 的“`50K` 是绝对上限、`100K` 一定发散”一类硬规则。
- 任何把 `reward_scale` 某个点位当作当前默认 recipe 的结论。
- 任何只按 `ref_mse` 排名得到的 winner。

## 现在继续训练时的默认判断顺序

1. 先把 **`ref_dropped_mse`** 当主指标，优先解决 ref-copying。
2. 同时守住 **`q_gap`** 和 **`td_error`**，防止为了降 `ref_dropped_mse` 把 critic 训坏。
3. 数据侧优先沿着 **`cp405 + renorm_teleop + intervene`** 继续走。
4. 超参侧优先在 **`β=0.3~1.0`** 和更高 `ref_dropout_p`（至少 `0.85/0.95/1.0`）上做下一轮搜索。
