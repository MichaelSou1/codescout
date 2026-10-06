# 实验状态汇总

更新：2026-10-07 04:55 CST。

## 当前结论

- 尚无 RL 结果。工程链路推进：环境（verl v0.9.1 官方锁定组合）验收 PASS；数据/划分/workspace 构造 oracle 数据面通过；verl 适配实现进行中。

## 已完成（工程验收，非方法有效）

- 环境：Python 3.12.15 prefix，torch 2.11.0+cu130 / vllm 0.24.0 / transformers 5.9.0 / flash-attn 2.8.3 / verl 0.9.1，路径合规无越界（cs4b-env-a01）。
- 数据：模型@cdbee75f + 4 数据集固定 revision 下载；划分 39,187 train + 100 dev（无重叠）；私有标签隔离（cs4b-oracle-a01）。
- Workspace：20/20 任务真实克隆+mutation 快照，无 `.git`，无泄漏。
- 奖励合同：15 测试冻结原 scorer 语义（tests/test_reward_contract.py）。

## 活动作业

- 步骤 4 适配实现（子代理，src/verl_adapter + configs + run 脚本 + 对齐测试）。
- 无 GPU 作业、无训练进程。

## 阻塞

- 步骤 5–8（基线/训练/评测）依赖：适配验收（固定张量对齐、finish 语义 A-2、轮数耗尽 A-1）、GPU 短测，以及**训练执行额度与预算冻结**（用户当次授权）。

## 预算

- 已消耗 GPUh：0（未启动任何 GPU 作业）。
- 服务器 CPU 墙钟约 1.5 小时（安装/下载/克隆）。

## 下一步

1. 适配实现完成后：服务器跑 reward 对齐（≤1e-8）、prompt 编码一致性、单任务全链路 smoke。
2. GPU 短测（模型加载+短生成+FSDP init，独立 run 记预算）。
3. 之后的训练启动需用户冻结执行额度（200 updates × 2 seeds 的 GPUh 预算）。
