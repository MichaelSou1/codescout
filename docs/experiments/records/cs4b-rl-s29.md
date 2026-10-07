# cs4b-rl-s29 — 主 RL 训练 seed29 与 dev checkpoint 选择（步骤 7）

- run_id：`cs4b-rl-s29`；experiment_id：`codescout_verl_4b_v1`
- 状态：**完成**（训练 200/200 + dev 评测 + checkpoint 选择）
- 时间：2026-10-07 08:34–10:40 训练（CST，2.1h ≈ 16.6 GPUh）；09:48–10:20 dev 评测
- 对照：`cs4b-base-dev01`；姊妹 run：`cs4b-rl-s17`（仅 rollout.seed=29 vs 17，其余配置逐项相同）

## 5. 结果

**dev 评测（正式口径，100 任务）**：

| checkpoint | sum_F1 | 与基线差 |
|---|---|---|
| global_step_160 | 0.8537 | +0.647 |
| global_step_170 | 0.7613 | +0.555 |
| global_step_180 | 0.8000 | +0.593 |
| global_step_190 | 0.8140 | +0.607 |
| **global_step_200（选中）** | **0.8673** | **+0.661** |

- **checkpoint 选择：global_step_200**（最高 0.867；选择在查看测试集结果前完成）。
- 与 s17 对比：同点位数值接近（s160 0.854 vs 0.841），s29 后期未见 s17 的 190 峰值、轨迹更平（0.76–0.87），两 seed 均大幅正提升且方向一致。
- 训练内指标健康（同 s17：reward 非常数、grad 正常、无 aborted）。

成本：训练 16.6 GPUh + dev 5 点 ≈ 3.4 GPUh。

## 7. 结论/限度

两 seed dev 提升一致为正（+0.851 / +0.661，均远超 +0.10 门槛方向），seed 间峰值点不同（190 vs 200）属训练随机性正常波动。**注意**：两 seed 仅 rollout 采样种子不同，数据顺序由 verl sampler 共享逻辑控制（s17 记录的已知限度同样适用）——任务级 bootstrap 不覆盖训练种子不确定性，报告按协议 §8 如实区分。

## 8. 下一步

最终评测（运行中）：base + s17-s190 + s29-s200 × Verified 500 / Pro 264 / Lite 300 →
`analyze_bootstrap.py`（10000 次、seed 20261007）→ final report。
