# HANDOVER — CodeScout verl 复现（接手文档）

更新：2026-10-07 06:30 CST（随主训练进度滚动更新）。接手前先读 [AGENTS.md](../AGENTS.md)、[todo.md](../todo.md)、本文件与 [docs/experiments/status.md](experiments/status.md)。

## 1. 一句话现状

协议 v1 冻结后的 verl 迁移已完成工程验收全链路（环境/数据/workspace/奖励合同/损失梯度对齐/smoke/恢复/profile/未训练基线），主训练 seed17（200 updates）正在 8×H100 上运行；seed29 与最终 Verified 评测待 s17 完成后依次执行。

## 2. 已完成（全部有证据、已入 Git）

| 项 | 证据 | commit/run |
|---|---|---|
| 协议 v1 / SkyRL→verl 映射 / verl 适配设计 | docs/reproduction/{protocol-v1,skyrl-to-verl,verl-adapter-design}.md | da0b2d5/72e86d0/271bb5c |
| kml-1005 部署核验 | docs/deployment/kml-1005.md（GPU UUID/挂载/写权限实测） | da0b2d5 |
| 隔离环境（Miniforge 26.7.2-0 + Python 3.12.15 + verl v0.9.1 官方锁） | manifests/environment/{install-summary,check}.json PASS | 9d39a1e |
| 数据：模型@cdbee75f + 4 数据集固定 revision；划分 39187+100 无重叠；私有标签隔离 | manifests/data/{assets-v1.md,swe-smith-prepare-manifest.json} | 0268561 |
| workspace 构造（无 .git 快照、mutation 校验、20/20 smoke、无泄漏） | scripts/smoke_workspace.py 结果 + cs4b-oracle-a01 | 9585b9d |
| 奖励合同（15 tests）+ 适配（tools/agent_loop/reward/build_prompts）+ 服务器全套 34/35 | tests/test_reward_contract.py、tests/test_verl_adapter.py | cdf0a6c/d7290b3 |
| 固定张量损失/梯度/优势对齐（SkyRL 参考复制 vs verl v0.9.1，≤1e-8，D1 口径完全刻画） | tests/test_loss_alignment.py 服务器 10/10 PASSED | (本会话) |
| correctness smoke + 恢复（2+1 updates、真实奖励、checkpoint 恢复续训） | docs/experiments/records/cs4b-smoke-a01.md | 56c9c88 |
| 8 卡 profile（11.7–16.8s/update 稳态、11.3GB/卡 actor 峰值、预算表） | docs/experiments/records/cs4b-prof-a01.md | 10296ea |
| 未训练 dev 基线（100 任务，sum_F1 0.2067；file 0.137/module 0.050/entity 0.020） | results/experiments/codescout_verl_4b_v1/cs4b-base-dev01-summary.json | 28d9553 |

## 3. 正在运行

- **cs4b-rl-s17**：主训练 seed17（2026-10-07 06:12 CST 启动，PID 1133943，日志
  `codescout-data/logs/rl_s17.log`，checkpoint `codescout-data/runs/cs4b-rl-s17/checkpoints/`）。
  200 updates，save/test 每 10，rollout.seed=17。
- 监控要点：step1–10 的 `critic/score/mean` 分布（不得全常数）、`actor/grad_norm` 非零有限、
  `aborted_ratio` 低、`rollout_probs_diff_valid=1.0`；异常（梯度非有限/评分器崩/泄漏）立即
  `kill <PID>`（只清理本作业进程树，不 pkill 全局）。

## 4. 待执行（顺序）

1. s17 训练完成后：在 0/10/40/100/200 五点 checkpoint 上用**正式口径**（132K、0.7/20/0.8）
   重放 dev 评测（`scripts/evaluate_localization.sh` + checkpoint 路径），按 sum_F1 选
   checkpoint（平分选更早）；锁 checkpoint hash 后才可看测试集结果。
2. seed29 同配置训练（rollout.seed=29，run_id cs4b-rl-s29）。
3. 最终评测（步骤 8）：base + s17 + s29 三模型 × Verified 500（actor-only 单 episode），
   Pro 264 / Lite 300 泛化补充；任务配对 bootstrap 10000 次 95% CI + repo 聚类敏感性；
   收益门槛 sum_F1 ≥ +0.10 且两 seed 同向、CI 下界>0。
4. 交付 `docs/reproduction/final-report.md`、更新本文件与 todo.md 勾选。

## 5. 关键命令（服务器）

```bash
cd /mmu_vlm_hdd/home/rhsu/playground/codescout
source scripts/project_env.sh                      # 必须（conda/Ray/缓存路径）
# 训练（已由 scripts/run_verl_4b.sh 包装）
bash scripts/run_verl_4b.sh "$CODESCOUT_DATA_ROOT/models/Qwen3-4B-Instruct-2507" \
  "$CODESCOUT_DATA_ROOT/datasets/swe_smith_prepared/train_rl.parquet" \
  "$CODESCOUT_DATA_ROOT/datasets/swe_smith_prepared/validation_rl.parquet" \
  cs4b-rl-s29 actor_rollout_ref.rollout.seed=29
# 评测（checkpoint 路径替换 MODEL_PATH 即可评任意 checkpoint）
bash scripts/evaluate_localization.sh <CKPT_OR_MODEL_PATH> <VAL_PARQUET> <RUN_ID>
```

- 恢复：`trainer.resume_mode=auto` 从 `runs/<id>/checkpoints/global_step_N` 续训（smoke 已验收）。
- 测试：`"$CODESCOUT_PYTHON" -m pytest tests/ -q`（服务器；本机无 torch/verl，分层 skip）。

## 6. 环境与合规事实

- verl v0.9.1 @ 1876b06d（uv.lock 官方组合 + 2 处已记录修补：flash-attn wheelhouse URL、
  无 actor.dtype 键）；Python 3.12.15 conda prefix `codescout-data/envs/verl-vllm`；
  Ray session 在 `/dev/shm/codescout-ray`（AF_UNIX 107 字节限制）；全部缓存落允许根。
- 已消耗 GPUh（截至 06:30）：约 5.5（GPU 短测 0.15 + smoke 0.9 + profile 1.4 + 基线 1.6 + 
  初始化损耗）；主训练预算上限按 profile 记录 ≈ 28–48 GPUh（两 seed）+ 评测 6–15 + 20% 余量。
- 三端 git：本机 = origin/main = 服务器 checkout `9585b9d`（工作树干净；服务器 push 通道
  未核验，服务器侧只 fetch/pull）。

## 7. 已知限制/开放项（如实）

- 评测与训练内 dev 口径不同（132K 正式 vs 40960 趋势）：checkpoint 选择只用正式口径。
- `prompt_length/clip_ratio 0.125`、`response_length/clip_ratio` 的计数口径未审计（token
  流逐 token 对照表未做；`rollout_actor_probs_pearson_corr≈0.997` 已旁证无 retokenize）。
- 全 mask 序列口径 D1/D2 已数值刻画（`loss_verl = loss_skyrl × R/N`；N=R 时残差<1e-8），
  但生产 global_batch_size=64 vs 微批行数的比值需在 step 指标里复核一次。
- openhands schema 对账测试需装 openhands 的环境（1 skip）；hermes 解析等价性由
  smoke/profile 真实轨迹旁证，未做逐字节 prompt diff。
- 服务器 push 通道未核验；W&B 未启用；`use_torch_compile`/`enable_gradient_checkpointing`
  沿 verl 默认（未逐项对齐 SkyRL——只影响吞吐不影响语义，若逐项复刻需求出现需重 profile）。
