# Critic/Q Training Update 2026-04-23

## 目标

这轮工作的目标不是继续盲扫 `beta / dropout / penalty`，而是先把 critic 的训练动态和 Q 可视化链路理顺，重点回答两个问题：

1. 现有 Q 可视化是否可信。
2. 在保留 TD 训练范式的前提下，能否把成功轨迹上的 `Q(expert)` 推向更合理的 critical-phase 上升形状。

## 同步到仓库的文件

- `scripts/visualize_rlt_warmup_compare.py`
- `scripts/rlt_training/train_phase_a_ref_bootstrap.py`
- `scripts/rlt_training/train_phase_b_target_anchor.py`

说明：

- `run_visualize_with_env.py`、`diagnose_live_actor_ref_gap.py` 这类 coder 环境专用辅助脚本没有同步进仓库。
- 这次同步的是当前最有价值、可复用的主脚本，不是完整的临时调试目录镜像。

## 这次改了什么

### 1. 可视化脚本修正并扩展

`scripts/visualize_rlt_warmup_compare.py` 这次主要修了两类问题：

- 补上了 `observation.state` 的 quantile normalize，避免 live 可视化路径和训练时的 `state_vec` 口径错位。
- 新增了三条 Q 的联合可视化与导出能力：
  - `Q(actor)`
  - `Q(ref)`
  - `Q(expert)`

这一步是必要前置，因为之前 `proprio` 没有按训练口径归一化时，`Q(actor)` 与 `Q(ref)` 的差距会被夸大，不能用来判断 critic 本身是否合理。

### 2. Phase A: critic warm-start with cached next_ref bootstrap

新增脚本：`scripts/rlt_training/train_phase_a_ref_bootstrap.py`

设计：

- 冻结 actor。
- 只训练 critic。
- bootstrap 动作不再接当前 actor，而是直接用 cache 里的 `next_ref_flat`。

对应目标更接近：

- 当前 `Q(s, exec_chunk)` 的未来回报
- 但未来部分先不受“坏 actor”污染

Phase A 的作用是先把 critic 从原始 joint TD 的不稳定 bootstrap 里拉出来，得到一个更干净的 warm-start。

### 3. Phase B: target actor + reference-anchor bootstrap

新增脚本：`scripts/rlt_training/train_phase_b_target_anchor.py`

设计：

- 从 Phase A checkpoint warm-start。
- 复制一份 `target_actor`。
- bootstrap 动作用：

`a'_boot = (1 - lambda) * next_ref + lambda * mu_target`

其中：

- `lambda` 从 `bootstrap_lambda_start` 线性 ramp 到 `bootstrap_lambda_max`
- actor 使用 delayed update
- target actor 用 EMA 更新

这一步的目的是：

- 不直接退回“纯 actor bootstrap”
- 先用 `ref` 锚住 critic 的形状
- 再逐步把 actor 注入回来

## 当前实验结论

### 当前最强 fixed-seed 候选

在 fixed-seed 评估语义下，当前最强 `best_q_expert` 候选是：

- `Phase B`
- `lambda_max = 0.15`
- `seed = 1`
- `best checkpoint = step 1000`
- `mean_q_expert = 0.122867`

也就是我们当前说的 `l015_seed1 best_q`。

### 与原始 Shiki baseline 的关键差异

这套实验没有换 backbone，也没有换主数据配方。主要差异是训练动态：

1. 原始 offline AC 更接近：
   - 当前项用 `exec_chunk`
   - bootstrap 项直接用当前 actor 产出的 `a'`
2. 这轮新实验改成了两阶段：
   - Phase A：先完全去掉 actor bootstrap
   - Phase B：再用 `target_actor + ref anchor` 缓慢把 actor 加回来
3. 这轮还补了 fixed-seed eval 与 best-checkpoint 机制，避免把采样噪声误当成“最优点”

## 目前最重要的观察

### 1. Phase A 确实有帮助

它能把 `cp405` 成功轨迹上后段 Q 形状从“掉下去”推成“往上抬”，说明 actor bootstrap 确实是 critic 的污染源之一。

### 2. Phase B 比纯 ref bootstrap 更强，但不能放太猛

`lambda` 太大时会回撤。当前观察到的比较健康区间在较小的 `lambda`，例如 `0.15` 这条线优于之前更大的混合强度。

### 3. seed variance 已经足够大

在 fixed-seed 评估语义下，只换训练 seed，`best_q_expert` 就会有明显波动。这意味着继续细扫小超参之前，必须先承认 variance 已经和超参改动同量级。

### 4. best_q 和 best_td 已经分叉

这轮实验里已经观察到：

- 更低的 TD error
- 不一定对应更高、更像 progress 的 `Q(expert)`

所以后面不能只拿 `mean_anchor_td_error` 当唯一优化目标。

## 已知 caveat

这次同步的脚本已经够继续做实验，但还不是“所有语义都彻底锁死”的终版，主要 caveat 是：

1. `visualize_rlt_warmup_compare.py` 对 `cp405` 的定性诊断更可信。
2. mixed-cache 场景下，尤其 `teleop/intervene`，live 可视化仍不等价于训练 cache 的完全 faithful replay。
3. `Q(ref)` 在 live VLA 路径上仍可能受 reference sampling 影响；跨不同导出 HTML 做严格数值对比时要谨慎。
4. 因为 `best_q` 与 `best_td` 分叉，后续实验需要同时看：
   - `Q(expert)` 曲线形状
   - `mean_q_expert`
   - `mean_anchor_td_error`

## 建议的下一步

如果继续沿这条线推进，优先级建议是：

1. 继续在 fixed-seed 语义下补少量 seed，先量化 variance。
2. 对 `best_q_expert` 与 `best_anchor_td` 的 checkpoint 做成对 Q 可视化比较，而不是只看标量。
3. 如果仍想强化“整个 critical phase 都递增”的 pattern，下一步优先考虑：
   - 更保守的 actor 注入 schedule
   - progress/ranking 类辅助约束
   - 而不是回头继续大扫 `beta / penalty`
