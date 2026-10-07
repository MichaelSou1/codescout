# cs4b-offref-final — 官方协议参考评测（步骤 5/8 补充口径）

- run_id：`cs4b-offref-{base,rl_s17,rl_s29,official_cso4b}-{verified,lite,pro}`（12 个评测）
- 状态：**完成**（2026-10-07 11:51–13:25 CST，8×H100 ≈ 18 GPUh）
- 动机：官方评测 fork 核对（[official-eval-fork-audit.md](../../reproduction/official-eval-fork-audit.md)）
  发现官方口径为 15 轮 + `adityasoni17/*-code-search` 真值镜像（与本项目所用 locagent 镜像
  73/500 任务 file_changes 不同）——按 protocol-v1 §5 的双口径要求补跑官方协议参考。
- 模型：base（Qwen3-4B-Instruct-2507@cdbee75f）、rl_s17@190、rl_s29@200、
  **OpenHands/CodeScout-4B（官方发布参考，仅评测不参与训练/选择）**
- 口径：15 轮（max_assistant_turns=15）、官方真值、其余同本项目评测管线
  （132K、0.7/20/0.8、单 episode、冻结 scorer——与官方 eval_infer.py 的分母/评分细节差异
  见差异表 §2，因此**不与论文表数字横比**，四模型同管线互比有效）。

## 结果（sum_F1 宏平均，0–3）

| 集（官方镜像行数） | base | rl_s17 | rl_s29 | 官方 CodeScout-4B |
|---|---|---|---|---|
| Verified 500 | 0.268 | **0.815** | 0.783 | 0.730 |
| Lite 300 | 0.219 | **0.636** | 0.552 | 0.497 |
| Pro 266 | 0.039 | **0.084** | 0.078 | 0.070 |

## 结论

1. 本项目匹配口径结论（Verified +0.486）在官方口径下**复现并放大**（s17 0.815 vs base 0.268，
   +0.547；15 轮给 RL 模型更多发挥空间，与官方 15 轮设定一致）。
2. **verl 管线复现模型全面达到并略超官方 CodeScout-4B**（Verified +0.085/+0.053、
   Lite +0.139/+0.055、Pro +0.014/+0.008）——SkyRL→verl 迁移的质量独立佐证。
3. 限定：官方 CodeScout-4B 在本项目管线下评测（非官方 eval_infer.py 完整管线），
   官方管线的温度/分母不可考（差异表 §1/§2），故此对照是"同管线互比"而非官方成绩。

## 下一步

无——交付完毕。开放项见 final-report §6。
