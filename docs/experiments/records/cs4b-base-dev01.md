# cs4b-base-dev01 — 未训练起点 dev 基线（步骤 5）

- run_id：`cs4b-base-dev01`；experiment_id：`codescout_verl_4b_v1`
- 状态：**完成**
- 时间：2026-10-07 06:05–06:35（CST，含一次 audit 序列化失败重跑）；8×H100 ≈ 1.6 GPUh
- 模型：Qwen3-4B-Instruct-2507@cdbee75f（未训练起点，无任何修改）
- 被评测集：冻结 dev 100 任务（validation_rl.parquet；SWE-smith 分布）
- 评测参数：协议 §5 正式口径——温度 0.7 / top_k 20 / top_p 0.8 / 单 episode / actor-only /
  max_model_len 131072 / response 98304；实现 = verl trainer val_before_train（占位训练集 16 行，
  val 前无参数更新）
- 逐任务审计：runs/cs4b-base-dev01/reward_audit.jsonl（164 行 = 100 dev + 64 占位训练调用；
  汇总按被评测集过滤）

## 结果（dev 100 任务宏平均）

| 指标 | 值 |
|---|---|
| **sum_F1（主指标，0–3）** | **0.2067** |
| file_F1 | 0.1367 |
| module_F1 | 0.0500 |
| entity_F1 | 0.0200 |
| file P/R | 0.135 / 0.140 |
| num_turns | 5.68 |
| failure_class | no_finish 54；有效预测 46 |

- 分母：100 任务，无 infra 排除；两次确定性重算（eval_summary 重跑）一致。
- 一次 infra 失败已修复：审计 JSONL 的 numpy int64 序列化崩溃（reward.py `_audit_emit`
  加 jsonable 转换），重跑后通过。

## 解读与限度

- 未训练基线低是预期现象：54% 任务未调用 finish（多因轮数耗尽/格式漂移），三级 F1 递减
  （file > module > entity）符合任务难度结构。
- **不可与论文表直接横比**：论文 base 数字（Verified file F1 49.7%）是 Verified 分布 +
  官方评测 fork 口径；本基线是 SWE-smith dev 分布 + 本项目口径。协议对照只允许
  同口径 base vs RL 差值。
- 该基线是后续 RL dev 评测的匹配对照（同模型、同工具、同解码、同窗口）。

## 下一步

主训练 cs4b-rl-s17（seed 17，200 updates，rollout.seed=17）；训练内 test_freq=10 的
40960 窗口 val 仅作趋势监控，checkpoint 选择用本口径（132K 正式参数）在 0/10/40/100/200
五点重放。
