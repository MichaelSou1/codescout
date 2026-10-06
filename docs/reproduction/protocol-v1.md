# CodeScout verl 复现协议 v1（冻结）

冻结日期：2026-10-07（Asia/Shanghai）。本协议由源码事实与论文/模型卡交叉核对后冻结，是训练与评测的**唯一依据**；冻结后的任何改动须开新协议版本并重做匹配基线。标注【待核验】的项在对应步骤执行前补齐证据，不凭默认值补齐。

- 源码 commit：`9d05a644e02f102f91e7e03244217705246fc4a4`（本机/服务器/origin 一致，2026-10-07 03:21 核对）。
- SkyRL 参考 commit：`81e5a97c7430503c0c4e6508497cc5aa01a0c624`（原 `pyproject.toml` 固定）。
- OpenHands SDK 参考 commit：`85ecfd9333d2d2cc4404dd460fd38868d9b978e2`。
- 框架映射与逐项差异另见 [skyrl-to-verl.md](skyrl-to-verl.md)。

## 1. 任务与模型

- 任务：给定 issue 文本与真实仓库工作目录，agent 用终端工具搜索，调用 `localization_finish` 提交文件/类/函数定位，奖励为三级 F1 之和。
- RL 起点（唯一）：`Qwen/Qwen3-4B-Instruct-2507`，全参数 BF16，不做量化、不换模型、不加 SFT/PRM/judge/rerank。下载时固定 revision 并记录【待核验：实际 revision】。
- 不改 chat template；训练与评测都关闭 thinking（`enable_thinking=False`，与原脚本一致）。

## 2. 数据与划分

- 训练数据：`OpenHands/SWE-smith-py-code-search`（SWE-smith Python 定位版）。执行时固定 revision，记录行数/字段/体积【待核验】。
- 划分重建：按 `src/build_dataset.py` 语义——删除空 `problem_statement`，`random_state=42` 全量打乱，末 100 条为 validation，其余为 train。此为公开 builder 重建划分；作者实际训练 manifest 不可得，**不宣称与原训练样本顺序一致**。
- dev：冻结上述 100 条 validation，训练全程不得增改。
- 测试（仅最终评测，训练/调参期间不得看结果）：SWE-bench Verified 定位版（预期 500 任务，下载后核实）；Pro/Lite 定位版作泛化补充，revision 与分母下载后锁定【待核验】。
- SWE-smith 用 `--use_patch` 语义构建（mutation patch 引入 bug，`base_commit=None`，见 skyrl-to-verl §数据）；Verified/Pro/Lite 用 base_commit 语义。actor 可见字段只含 issue/仓库标识/版本配置；`file_changes/target/patch` 是私有标签，不进 prompt、不进工具返回、不进 actor 可读路径。
- 数据审计：按 instance_id/repo/base_commit/文本 hash 核对 train/dev/test 重叠并在训练前冻结处理规则；处理规则不看测试成绩。

## 3. Prompt、工具与轮数

- System prompt：`src/prompts/templates/system_prompt_custom_finish.j2`；user prompt：`file_module_custom_finish.j2`。两者版本化，随本协议 commit 冻结。
- **轮数冲突的冻结决定：** 论文报告 6 个 agent turns；原 system prompt 文本写 4 turns；原脚本 `generator.max_turns=10`。本协议主配置取 **6 turns**（以论文为准），system prompt 文本同步改为 6，runtime 上限同 6。实施前须验证 OpenHands SDK `max_iteration_per_run` 与实际 LLM 调用次数的对应关系【待核验：SDK iteration 计数语义】。若发布轨迹证明别的配置，冻结前统一纠正并记录依据。
- 工具：仅 `terminal`（bash）与 `localization_finish`，schema 逐字复用 `src/tools/localization_finish.py`。每轮并行 bash 调用上限按 prompt 文本执行（≤5）。
- 终止：`localization_finish` 必须恰好一次；未调用、多调用、格式错误、轮数耗尽均为**有效模型失败**（reward 0、按原语义处理 loss mask），不是 infra 重试理由。原实现的轮数耗尽整条 mask 置 0 行为按原样保留（见 §6）。
- workspace：`codescout-data/workspaces/<run_id>/<episode_id>`，替代原 `/tmp/testbed/<uuid>`；SWE-smith 应用 mutation 后导出**无 `.git`** 的代码快照，防止 `git diff` 泄漏 mutation 位置。

## 4. 奖励（冻结）

- 唯一奖励函数：`multilevel_localization_f1_reward`，权重 file/module/entity = 1/1/1，总范围 0–3。
- 空真值集合得 0（原实现行为，不"修正"）；空文件名预测使三级全空得 0；去重后按 set 计算 precision/recall/F1。
- module = `file:Class` 或 `file:func`（顶层函数），entity = `file:Class.method` 或 `file:func`；解析语义逐字复用 `src/rewards/file_localization/module_rewards.py`（`parse_structured_outputs`）。
- 最后一步 sanity check（原 `sanity_check_last_step`）：最后一次响应必须恰好一对 tool call 标记、恰好一个 `<|im_end|>`、`⿻` 后无非空白文本；失败按有效模型失败记 0。
- infra 异常（克隆失败、评分器崩溃等）与有效模型失败分列；infra 排除规则训练前冻结，不计 0 分。

## 5. 采样与上下文

- 训练：temperature 1.0，每 issue 8 条轨迹（组内同 issue 同策略采样）；vLLM `max_model_len`/`max_input_length`=40960，**每次 LLM 调用**生成上限 8192（不是整 episode 上限）；hermes tool-call parser；thinking 关闭。
- 评测（正式）：temperature 0.7、top_k 20、top_p 0.8、上下文 132K（论文口径），每任务单 episode、actor-only【待核验：官方 benchmark fork `run_infer.py` 的实际解码参数；若不同，分别报告"官方协议参考"与"本项目匹配协议"两套，不混比】。
- 原训练内 eval 温度 0.6 仅记为源码差异，本项目 dev 评测统一用正式评测参数。
- 132K 评测窗口与 40960 训练窗口分别验收；不支持时公开降级并重做匹配 base/RL。

## 6. 训练语义（verl 目标配置）

- Advantage：GRPO（组内 8 条同 issue），中心化**不除以标准差**（`grpo_norm_by_std=false`）；全同 reward 组 advantage 为 0，报告其比例。
- Policy loss：GSPO——sequence 级 importance ratio，clip `eps_low=0.0003`、`eps_high=0.0004`；loss reduction `sequence_mean`（SkyRL 名）；KL 两项关闭。
- 优化器：AdamW，恒定 lr 1e-6；`update_epochs_per_batch=1`；每 update 有效 batch = 8 issues × 8 轨迹 = 64 episodes。betas/epsilon/weight_decay 以固定 SkyRL commit 源码解析后锁定【待核验：具体值】。
- Loss mask：仅模型生成 token 计 loss；system/user/tool 结果/prefix 均为 0；覆盖**每一轮**的生成 token，不是只训最后 finish；mask 构造复刻原实现（`buffer_succeed=1`（4B-Instruct）、`buffer_precede=1`、`<|im_start|>`/`assistant` 角色切换遮蔽）。保留真实采样 token IDs，不 retokenize SDK 重写消息。
- 轮数耗尽且无 finish：整条 loss mask 置 0（原实现），reward 0；该行为按原语义保留，其比例与选择偏差入账分析。
- 异步：**主实验用同步 batch 配置**起步（verl 官方 trainer），原 fully-async（允许 ≤4 updates 陈旧）能力差异作为已知方法偏差披露；异步迁移只有在 verl 支持验收后才作为追加实验，且须重做匹配基线。
- 预算目标：论文 200 updates（每 update 8 issues×8 轨迹），checkpoint 每 10 updates、HF export 每 50 updates、保留最近 5 份；dev 评测在 0/10/40/100/200 updates，日程冻结，不增测追好。

## 7. 实验矩阵与决定规则

- seed：17 与 29，两 seed 仅随机性不同，共享其余全部冻结配置。
- 快速否证：第一 seed 先 10 updates（排除 reward 全常数/梯度失效/工具失控/标签泄漏），再至 40 updates；40 updates 无明确涨分且正确性通过 → inconclusive 决定，不自动改 reward。
- 主指标：每任务三级 F1 之和的宏平均（0–3）；file/module/entity 分别报告。
- 预设收益门槛（本项目预设，非论文阈值）：宏平均提升 ≥0.10，且两 seed 方向同为正、平均差值任务配对 bootstrap（10000 次、固定 seed）95% CI 下界 >0；同时检查分级 F1、precision、token/成本膨胀。
- 未达门槛按未见明确收益/退化/证据不足交付，负结果也是完成交付，不无限训练。

## 8. 停止与安全条件

- 立即 stop：梯度非有限、评分器异常、私有标签泄漏、动作漏 mask、GPU 通信异常按基础设施诊断。
- infra 同因重试 ≤2 次；预算耗尽即停并交付证据；GPUh 按预留卡数×时长积分记账，含等待与失败。
- 私有标签隔离：gold/mutation patch 只在私有准备器可读位置；actor workspace 不挂载标签；越权读取尝试必须失败，违规审计失败阻止训练。

## 9. 本协议未覆盖/依赖项

- verl 精确版本与依赖组合：步骤 1 锁定后回填【待核验：verl v0.9.1 支持矩阵】。
- `sequence_mean` 与 verl loss reduction 的数值对齐：步骤 4 固定张量对照后回填。
- 官方评测 fork 的 workspace/解码参数：步骤 5 核对后回填。
