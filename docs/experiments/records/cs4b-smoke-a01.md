# cs4b-smoke-a01 — verl trainer 最小 correctness smoke + 恢复验收（步骤 4/6）

- run_id：`cs4b-smoke-a01`；experiment_id：`codescout_verl_4b_v1`
- 状态：**完成（核心验收 PASS）**
- 时间：2026-10-07 04:50–05:40（CST）；执行位置 kml-1005，GPU 0/1（UUID 见部署记录 §2）
- 阶段/协议：todo.md 步骤 6 第一项（最小 correctness smoke）；配置 = configs/verl/codescout_4b.yaml + smoke overrides（见 §3）

## 1. 目的/假设

验证 verl trainer + codesearch_agent 适配全链路接线正确：真实 vLLM rollout → 多轮工具执行（terminal/localization_finish）→ 奖励计算 → optimizer 更新 → checkpoint 保存 → 恢复续训。判据：连续 ≥3 updates、奖励非平凡分布、checkpoint 可恢复且恢复后再更新。

## 2. 对照/改动

被测改动 = 本项目全套适配（src/verl_adapter + configs/verl/codescout_4b.yaml + scripts/run_verl_4b.sh）。无前序训练 run。

## 3. 数据/模型/配置

- 模型：本地 Qwen3-4B-Instruct-2507@cdbee75f（7.6GiB）。
- 数据：train_rl.parquet 首 16 行（smoke_train_rl.parquet）；validation_rl.parquet（未用，val_before_train=False）。
- smoke overrides：n_gpus_per_node=2、total_training_steps=2（恢复 run 为 3）、save_freq=1、response_length/max_response_length=8192、max_model_len=49152、gpu_memory_utilization=0.45、agent.num_workers=4、test_freq=-1。其余走权威配置（gspo/seq-mean-token-mean/clip 3e-4/4e-4/GRPO 不除 std/lr 1e-6 恒定/AdamW [0.9,0.999]/6 turns/hermes/温度 1.0）。
- 有效 batch：8 prompts × 8 rollouts = 64 episodes/update。

## 4. seed/环境

代码 commit 含 `06e6c1b` 之前全部修复；服务器 checkout 干净（smoke 期间发现的 checkout 内误写已迁移清理，见 §5 infra-3）。环境：verl v0.9.1 官方锁（cs4b-env-a01）。GPU 准入：启动前 0/1 卡 0 MiB 空闲；结束 0 MiB、无残留进程。

## 5. 结果/成本（含全部 infra 失败与修复）

| # | 失败 | 根因 | 修复 |
|---|---|---|---|
| 1 | hydra 报 actor.dtype 键不存在 | v0.9.1 无该键，默认已解析 bfloat16 | 删除 override |
| 2 | 运行期要求 log_prob_micro_batch_size(_per_gpu) | v0.9.1 强制显式 | 配置加 per_gpu=1 |
| 3 | Ray AF_UNIX socket 路径 >107 字节 | 允许根路径太长 | RAY_TMPDIR=/dev/shm/codescout-ray（RAM tmpfs） |
| 4 | vLLM 拒绝启动（KV cache 不足） | 默认 max_model_len 取模型原生 262K | 显式 rollout.max_model_len（协议口径 prompt+response） |
| 5 | checkpoint/hydra/tensorboard 写进 Git checkout | run 脚本 DATA_ROOT 回退 `$REPO_ROOT/codescout-data`（project_env 导出名是 CODESCOUT_DATA_ROOT） | run 脚本强制要求 CODESCOUT_DATA_ROOT；90GB checkpoint mv 到允许根；TENSORBOARD_DIR/hydra.run.dir 指向 run 目录；checkout 恢复干净 |

**验收结果（全部通过）**：
- 2 updates 完成：step1 critic/score mean 0.0417（max 1.667/min 0）、step2 mean 0.263（max 2.333）——奖励函数真实工作，数值在 0–3 F1 语义内合理；step1 actor loss ≈7e-10（多组全零 reward → GRPO advantage 0 + 部分序列全 mask，符合语义）、step2 -0.0078。
- 多轮真实搜索：num_turns mean 6.28/5.63（min 4 / max 12）；response_length mean ~640 tokens；prompt_length mean ~2850。
- off_policy staleness 0.0（同步 on-policy 如设计）。
- checkpoint：global_step_1/2/3 + latest 指针。
- **恢复**：resume_mode=auto 从 global_step_2 恢复（model+optimizer 分片均加载，global step 置 2）→ 完成 step 3（score mean 0.216）→ 存 global_step_3。
- 唯一 Traceback 为 torch dynamo atexit `dump_compile_times` 清理噪音，与训练无关。

成本：GPU 0+1 × 2 卡 ≈ 25 分钟 smoke + 12 分钟恢复 ≈ **0.9 GPUh**；CPU 克隆缓存复用；90GB checkpoint 占允许根（保留策略：smoke checkpoint 在 8 卡 profile 后可清理）。

## 6. 符合预期与否

符合：全部判据通过。未见预期外行为；finish 工具/失败分类的逐轨迹审计需在 profile/否证阶段从轨迹 dump 核（step 4 token 流审计仍开放）。

## 7. 结论/限度

工程链路（rollout→工具→奖励→更新→保存→恢复）**接线验收通过**。这不构成 RL 有效性证据（2+1 updates 无统计意义）。开放项：固定张量梯度对齐（D1/D3 口径）、恢复的数据游标语义细查、8 卡代表性 profile、token 流逐 token 审计。

## 8. 下一步

1. 固定张量 loss/梯度对齐测试上服务器跑（tests/test_loss_alignment.py，进行中）。
2. 8 卡代表 profile（全 batch 8×8，run_id cs4b-prof-a01）→ 产出吞吐/峰值/updates-per-hour → **训练预算表交用户冻结**（步骤 7 主训练的硬门槛）。
