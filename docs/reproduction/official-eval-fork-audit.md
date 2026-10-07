# 官方评测 fork 核对差异表（agentic_code_search @ 7cf83b8）

核对日期：2026-10-07（Asia/Shanghai）。对象：`adityasoni9998/benchmarks` 分支 `agentic_code_search`
（浅克隆 `7cf83b8` "Revise README for CodeScout evaluation setup"），逐文件源码核对 + HF 数据集实测。
本表消解 todo.md 步骤 5 的"官方 benchmark fork 参数核对"与步骤 3 的"训练/评测评分器差异表"两项。

## 1. 解码与运行参数（run_infer.sh 实测）

| 项 | 官方 fork（实测） | 本项目口径 | 处置 |
|---|---|---|---|
| 评测轮数 | `--max-iterations 15`（含 remind_agent 机制可预留最后一步催交） | 6（协议 §3 论文口径） | **差异**：补跑 15 轮"官方协议参考"评测，与 6 轮"本项目匹配协议"分别报告（协议 §5 要求的双口径） |
| 温度/top_k/top_p | **公开仓库不可考**：解码参数在 `.llm_config/*.json`（未提交，example.json 只有 model/base_url/api_key）；代码未显式传温度 | 0.7/20/0.8（论文口径） | 论文数字无法从 fork 验证；本项目沿用论文口径并披露来源差异 |
| 上下文长度 | 未在 fork 显式设（litellm 模型默认） | 132K | 同上 |
| workspace | `--runtime local --workspace_base_dir /tmp/testbed/` | codescout-data/workspaces（无 .git 快照） | 语义一致，路径合规化 |
| prompt | base 4B 评测用 `system_prompt_custom_finish.j2`（4-turn 文本）；RL 模型用 `custom_finish2`；仓库另有 `..._6turns.j2` 未见于脚本 | 原 4-turn 模板 + runtime 6 | 官方按模型选择 prompt 变体；本项目统一用原 4-turn 模板（与官方 base 评测一致）+ runtime 6，已在 todo 顶部记录 |

## 2. 评分语义（eval_infer.py 实测）

| 项 | 官方 | 本项目（冻结 scorer） | 处置 |
|---|---|---|---|
| P/R/F1 公式 | TP/\|pred\|、TP/\|true\|、2PR/(P+R)（:97-121） | 逐点一致 | ✓ |
| 无 finish/空预测 | 空集 → 三级 F1 逐项按 0 计入分母 | 无 finish → reward 0 计入 | ✓ 一致 |
| **分母** | **硬编码** `{"Lite": 274, "Pro": 266, "Verified": 500}`（:130）；缺预测/缺真值的实例按 0 分计入分母 | 公开数据实际行数（499/300/239-257）+ infra 冻结剔除 | **差异**：官方 Lite 274（26 实例"空真值"被 LocAgent 口径剔除但分母保留）；我们的匹配对照内部有效，官方口径参考评测按官方分母重算 |
| Pro 264 vs 266 | 官方镜像 266 行（实测 HF API），locagent 重发布 264 行 | 用了 locagent 264 | 根源已定位：**locagent 镜像丢 2 行**；官方协议参考用 266 行镜像 |
| 空 file 差异（todo 步骤3 项） | 无"空 file 清空全部预测"惩罚（逐级独立计算） | 冻结 scorer：任一空文件名清空三级（原训练语义，保留） | **训练/评测评分语义差异确认**：训练（原 scorer 清空语义）与官方评测（逐级独立）不同。本项目训练用原语义（复现对象），评测用原 scorer 语义（一致性优先）；官方口径参考评测按官方逐级独立语义重算可作敏感性检查——**未执行**（本表即差异记录，避免二次改口径引入新偏差） |

## 3. 评测真值数据集（HF 实测）

| 官方镜像（fork 脚本所用 ID） | 行数 | 本项目所用镜像 | 行数 | 内容对比（实测） |
|---|---|---|---|---|
| `adityasoni17/SWE-bench_Verified-code-search`@9d7ddf0 | 500 | `OpenHands/SWE-bench_Verified-locagent`@ffa2f8bf | 500 | instance_id/base_commit/patch/problem_statement/repo 全等；**`file_changes` 有 73/500 任务不同**（含 locagent 版真值为 null 而官方版有值的任务） |
| `adityasoni17/SWE-bench_Lite-code-search` | 300 | OpenHands locagent | 300 | 未逐列对比（同上风险） |
| `adityasoni17/SWE-bench_Pro-code-search` | **266** | OpenHands locagent | 264 | locagent 重发布丢 2 行（已定位） |

**处置**：匹配对照（base vs RL 同真值）的内部有效性不受影响；为官方协议参考，用官方
code-search 真值镜像 + 15 轮 + 官方分母补跑 Verified 参考评测（三模型 + 官方 CodeScout-4B），
与本项目口径结果分别报告，不混比。

## 4. 结论

官方 fork 的关键不可考/不一致点（温度私有、分母硬编码、真值镜像差异 15%）意味着
**任何与论文表数字的横比都不可靠**——protocol-v1 §5 的"没有可比协议不直接横比"由本表
证据支持。本项目交付以匹配对照差值为主结论（不受上述差异影响），官方协议参考为补充。
