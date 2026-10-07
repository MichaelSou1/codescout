# CodeScout verl 复现最终报告（final report, protocol-v1）

- experiment_id：`codescout_verl_4b_v1`；报告日期：2026-10-07（Asia/Shanghai）
- 全程记录：[docs/experiments/index.md](../experiments/index.md)；协议：[protocol-v1.md](protocol-v1.md)
- 研究决定（按 protocol-v1 §7 预设门槛）：**有效（Verified 主指标全部门槛通过）**

## 1. 结论摘要

在相同模型起点（Qwen3-4B-Instruct-2507@cdbee75f）、相同工具（terminal + localization_finish）、
相同奖励（三级 F1 之和，0–3）与匹配推理预算下，用 **verl v0.9.1（同步 trainer）** 复现的
agentic RL 使真实仓库定位能力显著提升：

| 测试集（分母） | base | RL seed17 (step190) | RL seed29 (step200) | 平均差值 [任务配对 bootstrap 95% CI] |
|---|---|---|---|---|
| **SWE-bench Verified（500）** | 0.2616 | **0.7619** | **0.7332** | **+0.486 [0.407, 0.565]** |
| SWE-bench Lite（300） | 0.1960 | 0.6194 | 0.5161 | +0.372 [0.289, 0.459] |
| SWE-bench Pro（264） | 0.0334 | 0.0914 | 0.0774 | +0.049 [0.023, 0.080] |

指标 = 每任务 file+module+entity F1 之和的宏平均（0–3）；actor-only 单 episode；
正式口径 132K 窗口、温度 0.7/top_k 20/top_p 0.8。

门槛判定（预设，非论文阈值）：
- Verified：增益 0.486 ≥ 0.10 ✓；两 seed 方向同为正（+0.500/+0.472）✓；平均差值 CI 下界 0.407>0 ✓ → **有效**
- Lite：同门槛全过（+0.372）
- Pro：方向为正、CI 下界 >0（统计显著），但幅度 0.049 < 0.10 → **未达预设幅度门槛**（正提升、幅度小，如实分列；repo 仅 3 个，聚类 CI 宽）
- repo 聚类敏感性：Verified 12 repo 聚类 bootstrap s17 CI [0.260, 0.603]——仍为正，但确认跨仓库不确定性明显大于任务级 CI

## 2. 方法与框架差异（skyrl-to-verl.md 摘要）

- 原实现 SkyRL fully-async（允许 ≤4 updates 陈旧）→ 本复现 **verl 同步 trainer**（staleness 实测 0）。
  这是最大的方法偏差：off-policy 程度低于原实现；作为已披露差异，不宣称原框架原样复现。
- GSPO/GRPO/sequence_mean 语义经固定张量数值对齐验收（SkyRL 参考实现复制 vs verl v0.9.1
  源码，ratio/loss/梯度/优势逐项 ≤1e-8；归一化口径 `loss_verl = loss_skyrl × R/N` 在
  N=R 时仅剩 <1e-8 分母偏置，D1/D2 完全刻画；tests/test_loss_alignment.py 10/10）。
- AdamW betas [0.9,0.999]/wd 1e-2/eps 1e-8/lr 1e-6 恒定、clip 3e-4/4e-4、KL 双关、6 turns、
  hermes、batch 8×8=64 episodes/update、200 updates——逐项按 SkyRL commit `81e5a97c` 与
  原脚本解析后迁移。
- 已知口径差异（全部在映射文档 §10 分列）：末轮 `<|im_end|>` 多 1 个 mask=1 token（保留
  verl 行为并披露）、超并行工具调用 verl 丢弃而非判罚、checkpoint 保留策略（5 份）导致
  dev 0/10/40/100 点不可评测（以留存 160–200 五点替代）。

## 3. 负结果与限制（如实）

- Pro 未达 +0.10 幅度门槛：正提升但小。Pro 为 3-repo 长尾困难任务，训练分布（SWE-smith
  131 repo）覆盖不足的解释成立但未验证——不延伸结论。
- 两 seed 仅 rollout 采样种子不同；任务配对 CI 不覆盖训练随机性（协议 §8 预先声明）。
- s17 dev 在 step190 达峰后 s200 回落（1.058→0.927），提示预算末段不稳定；按冻结规则
  选 checkpoint，未追最好点。
- 数据为公开 builder 重建划分（作者 manifest 不可得），不宣称与原训练样本顺序一致；
  Pro 公开 264 vs 论文 266，分母差异已记录。
- 官方 CodeScout-4B 参考模型未评测（协议记录为可选参考；本项目未横比论文表数字）。

## 4. 成本（GPU 时，8×H100 预留口径）

| 波次 | GPUh |
|---|---|
| 环境/GPU 短测/smoke+恢复/profile | 2.5 |
| 基线 dev（cs4b-base-dev01） | 1.6 |
| 主训练 s17（200 updates） | 16.6 |
| 主训练 s29（200 updates） | 16.6 |
| dev 评测 11 点（s17/s29 五点+补跑） | ~7 |
| 最终评测 9 个（3 模型×3 集） | ~14 |
| **合计（含等待/失败/初始化）** | **≈ 58.5**（预算上限 80 内） |

失败重试入账：环境安装 7 次（不同因）、评测瞬态 vLLM 启动失败 2 次（重试成功）、
审计序列化 bug 1 次（修复重跑）。

## 5. 交付物索引

- 代码：`src/verl_adapter/`（agent_loop/tools/reward/workspace/semantics/build_prompts）、
  `configs/verl/codescout_4b.yaml`、`scripts/`（setup/run_verl/evaluate/analyze/bootstrap 等）
- 测试：`tests/test_reward_contract.py`（15）、`tests/test_verl_adapter.py`（20）、
  `tests/test_loss_alignment.py`（10，服务器全过）
- 结果：`results/experiments/codescout_verl_4b_v1/`（dev 轨迹、final-eval 9 评测 + bootstrap 3）
- 账本：`docs/experiments/records/`（stage0/env/oracle/smoke/prof/base/rl-s17/rl-s29/test 共 9 份）
- 模型：服务器 `codescout-data/runs/cs4b-rl-s17/hf/global_step_190`（dev 1.058）、
  `cs4b-rl-s29/hf/global_step_200`（dev 0.867）；完整 checkpoint/审计 JSONL 在运行时根。

## 6. 下一步（若继续）

- 逐 token mask 审计对照表与 prompt clip 口径审计（profile 遗留开放项）。
- 异步 trainer 迁移（verl async rollout）以对齐原 fully-async 协议，需重做匹配基线。
- Pro 提升不足的归因（训练分布覆盖 vs 上下文长度 vs 工具预算）。
- 官方 CodeScout-4B 在本口径下的参考评测（可比性仍受限：fork 评测协议差异未核）。
