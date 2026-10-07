# 实验状态汇总

更新：2026-10-07 11:45 CST（**项目主链路完成**）。

## 最终结论

- **研究决定：有效（Verified 主指标全部门槛通过）**——未训练 base 0.262 → RL 两 seed 0.762/0.733，
  平均差值 **+0.486，任务配对 bootstrap 95% CI [0.407, 0.565]**；两 seed 同向正、增益远超 +0.10 门槛。
- Lite 泛化 +0.372（门槛通过）；Pro +0.049（统计显著为正但未达 +0.10 幅度门槛，如实分列）。
- 详见 [final-report](../reproduction/final-report.md) 与 [records/cs4b-test-final.md](records/cs4b-test-final.md)。

## 已完成（全部有证据入 Git）

步骤 0–8 全流程：协议冻结、verl v0.9.1 环境验收 PASS、数据/划分/重叠审计（零污染）、
workspace 20/20、奖励合同 15 tests、适配全套、损失/梯度数值对齐 10/10、correctness smoke+恢复、
8 卡 profile、未训练基线、两 seed 200-update 主训练、dev checkpoint 选择（s17@190、s29@200）、
9 个最终评测 + bootstrap。

## 活动作业

- 无。全部 GPU 作业完成并干净退出（GPU 0 MiB、无残留进程）。

## 阻塞

- 无主链路阻塞。开放项见 final-report §3/§6（prompt turns 文本、官方 fork 参数核对、
  ACL 隔离强度、audit_trajectory 独立脚本等）。

## 预算

- 总消耗 ≈ **58.5 GPUh**（上限 80 内）：环境/短测/smoke/profile 2.5、基线 1.6、
  主训练 2×16.6、dev 评测 ~10.5、最终评测 ~14（含等待/失败/初始化）。
- infra 失败均有限重试并入账（环境 7 次、评测瞬态 2 次、审计 bug 1 次）。

## 下一步（可选）

1. 官方评测 fork 参数核对 + CodeScout-4B 参考模型评测（可比性受限）。
2. 逐 token mask 审计与 prompt clip 口径审计（profile 遗留）。
3. 异步 trainer 迁移（对齐原 fully-async 协议需重做匹配基线）。
4. Pro 提升不足的归因分析。
