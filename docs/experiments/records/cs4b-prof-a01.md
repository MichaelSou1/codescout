# cs4b-prof-a01 — 8×H100 代表性 profile（步骤 6）

- run_id：`cs4b-prof-a01`；experiment_id：`codescout_verl_4b_v1`
- 状态：**完成**
- 时间：2026-10-07 05:37–05:47（CST，4 updates 含 2 次 checkpoint save）；GPU 0–7
- 配置：协议全量口径（configs/verl/codescout_4b.yaml 无 smoke 缩减：prompt 40960/response 32768/max_model_len 73728/gpu_mem 0.8/batch 8×8=64 episodes/update/8 workers）；数据 train_rl.parquet 前 64 行（注意：重复 epoch、克隆缓存热，见 §7 限度）

## 测量结果（每 update = 64 episodes / 8 issues）

| 指标 | step1 | step2 | step3 | step4 |
|---|---|---|---|---|
| step 总时长 (s) | 31.9（首步含编译） | 16.8（含 save 5.1） | 11.7 | 20.2（含 save 5.1） |
| gen (s) | 8.7 | 4.6 | 4.6 | 6.6 |
| old_log_prob (s) | 3.5 | 1.4 | 1.4 | 1.5 |
| update_actor (s) | 6.5 | 4.2 | 4.2 | 5.5 |
| update_weights (s) | 1.4 | 1.3 | 1.4 | 1.4 |
| actor 峰值 allocated/reserved (GB/卡) | 9.4/10.7 | 11.2/12.9 | 11.2/12.9 | 11.3/13.0 |
| num_turns mean | 5.97 | 6.16 | 5.59 | 5.28 |
| response_length mean | 630 | 632 | 611 | 639 |
| critic/score mean | 0.030 | 0 | 0.063 | 0.078 |

- 稳态（不含 save）≈ **11.7–16.8 s/update**；save 每 10 updates 一次 ≈ 5s（协议 save_freq=10 摊薄 ~0.5s/step）。
- 有效性证据：`rollout_probs_diff_valid 1.0`、`rollout_actor_probs_pearson_corr ≈0.997`、`log_ppl_abs_diff ≈0.003`——rollout 与训练 logprob 高度一致（无 retokenize 漂移的直接证据）；`off_policy staleness 0`（同步 on-policy）；`pg_clipfrac 0`（GSPO 初期 ratio≈1 符合预期）；`aborted_ratio 0`。
- 吞吐 860–2330 token/s（含/不含 save），220K tokens/update。

## 主训练预算推导（供用户冻结；口径：8 卡预留 ×墙钟）

- 稳态 12–17s/update + save 摊薄 ≈ **13–18s/update（profile 条件）**。
- 真实条件修正（未测量，保守放大 2–3×）：每 update 8 个新 issue 的首次克隆（并行 8 worker，缓存热后可忽略）、更难任务 → 轨迹更长/turns 更满 → gen 上升。估计 **30–55s/update**。
- **每 seed 200 updates ≈ 1.7–3 小时** → 2 seeds 串行 ≈ 3.5–6 小时（**28–48 GPUh**）。
- dev 评测 5 点 × 100 任务 + 最终 Verified 500×3 模型：评测吞吐未 profile，估 **6–15 GPUh**。
- 工程余量 20%（AGENTS 口径初估）：**主训练+评测合计 ≈ 45–80 GPUh**；profile 已消耗 1.4 GPUh。

## 开放审计项（profile 观察）

1. `prompt_length/clip_ratio 0.125`（恰 8/64=每 issue 1 条）——prompt mean 仅 ~2800 远小于 40960，clip 语义需查（可能是 agent-loop prompt_ids 对齐截断的计数口径），进步骤 4 token 流审计。
2. `response_length/clip_ratio 0.015625`（1/64）——同上，口径待审计。
3. cpu_memory ~124GB/进程——8 worker ×克隆与 tokenizer 缓存，正常但需在长 run 监控。
4. dev 评测吞吐未测（步骤 5/8 执行时先小样本测）。

## 决定

8 卡配置验收通过（无 OOM、无 aborted、峰值余量充足）。**主训练启动待用户冻结预算（推荐按上限 80 GPUh 预留，达到即停）**。
