# cs4b-rl-s17 — 主 RL 训练 seed17 与 dev checkpoint 选择（步骤 7）

- run_id：`cs4b-rl-s17`；experiment_id：`codescout_verl_4b_v1`
- 状态：**完成**（训练 200/200 + dev 评测 + checkpoint 选择）
- 时间：2026-10-07 06:12–08:15 训练（CST，2.1h）；07:56–08:30 dev 评测
- 对照：`cs4b-base-dev01`（未训练同起点同口径 dev 基线）

## 1. 目的/假设

协议 v1 主训练：相同起点/工具/奖励/采样下 RL 提升 dev 定位 F1。判据：dev sum_F1 高于基线；checkpoint 按冻结规则选择。

## 2. 对照/改动

唯一改动 = RL 训练本身。配置 = configs/verl/codescout_4b.yaml（权威清单）+ `rollout.seed=17`、total_training_steps=200、save/test_freq=10。数据 train_rl.parquet 全量 39,187 行（seed42 冻结划分，sample 顺序由 verl sampler 决定）。

## 3. 数据/模型/配置

见 protocol-v1.md 与 cs4b-prof-a01；模型 Qwen3-4B-Instruct-2507@cdbee75f；奖励 multilevel F1 1/1/1；batch 8×8=64 episodes/update；GSPO clip 3e-4/4e-4 + seq-mean-token-mean + GRPO 不除 std；AdamW [0.9,0.999] wd 1e-2 eps 1e-8 lr 1e-6 恒定；6 turns、hermes、温度 1.0。

## 4. seed/环境

seed：rollout.seed=17（vLLM 采样）；数据顺序受 verl sampler 控制（两 seed 差异主由采样随机性提供——**已知限度，如实记录**）。环境：cs4b-env-a01；8×H100；代码 commit `26ea008`（训练时刻 HEAD）。

## 5. 结果/成本

**训练指标**（每 10 步采样）：critic/score/mean 从 ~0.19 波动上升至 ~0.5–1.0 区间；grad_norm 0.25–2.5 正常无爆炸；entropy 0.23–0.32 稳定；无 aborted；off_policy staleness 0（同步）。10-update 快速否证点与 40-update 观察点均 PASS（reward 非常数、梯度有效、工具正常、无泄漏迹象）→ continue 决定。

**dev 评测（正式口径 132K、0.7/20/0.8、单 episode、100 任务）**：

| checkpoint | sum_F1 | file_F1 | 与基线差 |
|---|---|---|---|
| base（cs4b-base-dev01） | 0.2067 | 0.1367 | — |
| global_step_160 | 0.8409 | — | +0.634 |
| global_step_170 | 0.9757 | — | +0.769 |
| global_step_180 | 1.0237 | — | +0.817 |
| **global_step_190（选中）** | **1.0580** | — | **+0.851** |
| global_step_200 | 0.9267 | — | +0.720 |

- **checkpoint 选择：global_step_190**（最高 1.058；无平分情形）。选择在查看任何测试集结果之前完成（测试集此刻仍未评测）。
- 训练内 40960 窗口 val（test_freq=10）仅作趋势监控，未用于选择。
- dev 五点中 0/10/40/100 不可评：`max_actor_ckpt_to_keep=5`（原配方保留策略）只留 160–200 权重——**协议日程与保留策略冲突，如实记录**；以留存 5 点全测（160–200）替代，选择规则不变。

**成本**：训练 8 卡 × 2.1h ≈ **16.6 GPUh**；dev 评测 6 点 × ~4 分钟 ≈ 3.4 GPUh。

## 6. 符合预期与否

符合：RL 假设在 dev 上方向性成立且幅度大（+0.85/3，约 4.1×基线）。s200 回落提示后期可能过训/不稳定——按冻结规则选 s190，不追最好点。

## 7. 结论/限度

seed17 条件下 dev 定位 F1 大幅提升；单 seed 无不确定性结论，等待 seed29 + 最终 Verified 测试（匹配对照 + 任务配对 bootstrap CI + repo 聚类敏感性）。checkpoint 保留策略导致 dev 0/10/40/100 点缺失，是复现协议与原配方冲突的已披露偏差。

## 8. 下一步

1. seed29 同配置训练（已启动 08:34 CST，run_id cs4b-rl-s29）→ 同流程选择。
2. 最终评测：base + s17-s190 + s29 选中点 × Verified 500（+ Pro 264 / Lite 300 泛化）。
3. `scripts/analyze_bootstrap.py`：任务配对 + repo 聚类 bootstrap（10000 次、seed 20261007）、预设门槛判定。
