# 实验索引

维护日期：2026-10-07（Asia/Shanghai）。所有实验（含失败/取消/负结果）入账；单次完整记录见 [records/](records/)，状态汇总见 [status.md](status.md)。模板：[records/_TEMPLATE.md](records/_TEMPLATE.md)。

`experiment_id = codescout_verl_4b_v1`（主实验）；协议：[docs/reproduction/protocol-v1.md](../reproduction/protocol-v1.md)。

| run_id | 日期（CST） | 目的 | 状态 | 对照 | 记录 |
|---|---|---|---|---|---|
| cs4b-stage0-docs | 2026-10-07 | 步骤0：源码事实核对、kml-1005 核验、协议 v1 与映射文档建立 | 完成（文档交付物） | 无前序 | [records/cs4b-stage0-docs.md](records/cs4b-stage0-docs.md) |

## 计划中的 run_id（登记占用，未启动）

- `cs4b-env-a01`：conda/verl 环境搭建与验收（步骤 1）。
- `cs4b-oracle-a01`：数据/奖励 oracle 与 20 任务工具 smoke（步骤 2–3）。
- `cs4b-base-s17`：未训练起点 dev 基线（步骤 5）。
- `cs4b-rl-s17` / `cs4b-rl-s29`：主 RL 训练 seed17/29（步骤 6–7）。
- `cs4b-test-s17` / `cs4b-test-s29` / `cs4b-test-base`：最终评测（步骤 8）。

重试/修订沿用 `parent_run_id`；同协议续跑另记 `attempt_id`，预算累计不清零。
