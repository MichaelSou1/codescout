# SkyRL → verl 配置映射与差异账

维护日期：2026-10-07（Asia/Shanghai）。逐项列出原 SkyRL 实现（源码 commit `9d05a64` 内 `scripts/run_async_training_4b.sh` + SkyRL `81e5a97c`）到 verl 目标配置的映射。状态列：✅已核对源码事实 / 🔄计划映射（待实现验收）/ ⚠️已知差异须披露 / ❓待核验。

**定位声明：本项目是 CodeScout 方法在 verl 上的适配复现，不是 SkyRL 原样复现。** 所有数值验收见 protocol-v1.md §6 与 todo.md 步骤 4 通过条件。

## 1. 训练循环拓扑

| 项 | 原（SkyRL） | 目标（verl） | 差异/证明 | 状态 |
|---|---|---|---|---|
| 框架 | SkyRL fully-async（`FullyAsyncRayPPOTrainer`） | verl 官方 trainer | 主实验同步 batch；async 差异作为方法偏差披露；verl 若支持 async rollout（agent loop + async trainer）须单独验收后才启用 | ⚠️ |
| 更新引擎/推理引擎 | 4 training + 4 inference（8 卡，`colocate_all=false`） | verl resource pool：FSDP2 actor（TP1 rollout）+ ref policy；共置 sleep/wakeup 或分置由端到端 profile 决定 | 原拓扑不是 verl 默认，不照抄 | 🔄 |
| 生成并发 | `trainer.fully_async.num_parallel_generation_workers=8` | verl agent loop rollout 并发参数 | 单位不同，不直接照填数值 | ❓ |
| 陈旧策略 | `max_staleness_steps`（fully-async，论文 ≤4 updates） | 同步模式下为 0 | 同步消陈旧性；差异披露 | ⚠️ |
| Ray | SkyRL 自带 Ray 拓扑 | verl `ray.init` + resource pool | 节点/卡/Rank/Python 登记验收 | 🔄 |

## 2. 优化目标

| 项 | 原（SkyRL） | 目标（verl v0.9.1 实测源码） | 差异/证明 | 状态 |
|---|---|---|---|---|
| Advantage | GRPO，`grpo_norm_by_std=false`（组中心化不除 std） | verl `advantage_estimator=grpo`，`norm_adv_by_std_in_grpo=False`（v0.9.1 `core_algos.py` 支持该开关） | 用固定 reward 张量对照 advantage 逐步数值 | ✅语义/🔄验收 |
| Policy loss | GSPO（`gspo_policy_loss`，`ppo_utils.py`：mask 内 log-ratio 序列均值→合并 token 比、clamp max10、对称 clip） | verl v0.9.1 原生 `policy_loss_type="gspo"`（`core_algos.py` `compute_policy_loss_gspo`，同公式同 clamp） | **无需自写 loss adapter**；固定张量对照 ratio/loss/梯度 | ✅语义/🔄验收 |
| clip | `eps_clip_low=0.0003, eps_clip_high=0.0004` | `actor.optim.clip_ratio_low=3e-4, clip_ratio_high=4e-4`（verl 支持 low/high 分开设） | 数值对照 | ✅ |
| loss reduction | `sequence_mean` = 每序列 token-mean → batch-mean（SkyRL `reduce_loss`） | verl `loss_agg_mode="seq-mean-token-mean"`：每序列 token-mean（分母 +1e-8）→ sum/`global_batch_size`×`dp_size`，排除全 mask 序列 | 微批/累积口径不同：SkyRL `.mean()` 含全 mask 序列（贡献0），verl 分母用 global_batch_size。**以"同全局 batch 一次 optimizer step 的最终梯度"为对齐判据**，逐步张量对照仅作诊断 | ✅语义/❓梯度对齐待做 |
| KL | `use_kl_loss=False, use_kl_in_reward=False` | verl 等价关闭 `kl_loss`/`kl_in_reward` | 配置+代码路径验收 | 🔄 |
| 优化器 | AdamW：lr 1e-6（脚本覆盖）、betas [0.9,0.999]、weight_decay 1e-2、eps 未传（torch 默认 1e-8）、`max_grad_norm=1.0`、`constant_with_warmup` + `num_warmup_steps=0`（= 恒定 lr）；policy loss 无 entropy 项 | verl 同值设置（AdamW betas/wd/eps/clip、constant scheduler 0 warmup、entropy 显式 0） | 超参已从 SkyRL `81e5a97c` `ppo_base_config.yaml` + `fsdp_strategy.py` 解析；verl 侧显式配置不继承默认 | ✅解析/🔄配置 |
| batch | `train_batch_size=8` issues × `n_samples_per_prompt=8` = 64 episodes/update；`update_epochs_per_batch=1`；`policy_mini_batch_size=8`；`micro_train_batch_size_per_gpu=1` | verl 同有效 batch；核 verl mini-batch 是否含 rollout 展开维度 | 记录 optimizer.step 次数 vs 遍历次数 | 🔄❓ |
| 轨迹归一化 | 整 episode 为一条序列（generator 聚合各轮成单条 response） | verl multi-turn 数据结构 | SkyRL generator 已聚合为单序列，归一化即上表 sequence_mean；匹配非 padding 生成 token 权重 | 🔄 |

## 3. 采样与生成

| 项 | 原 | 目标 | 状态 |
|---|---|---|---|
| 引擎 | vLLM async engine，`tool_call_parser="hermes"`，`enable_auto_tool_choice=true`，TP=1 | verl rollout vLLM 同配置；Hermes parser 与 Qwen3 模板兼容性实测（多工具并行不丢调用） | 🔄❓ |
| 长度 | `max_model_len=40960`，`max_input_length=40960`，每次调用生成 ≤8192 | 同值；多轮上下文/裁剪规则冻结；总 episode token 另统计 | ✅/🔄 |
| 温度 | 训练 1.0 / 训练内 eval 0.6 | 训练 1.0；正式评测 0.7（top_k20/top_p0.8，论文口径） | ✅ |
| thinking | `enable_thinking=False`（chat_template_kwargs） | 同 | ✅ |
| 轮数 | `generator.max_turns=10`，prompt 文本 4，论文 6 | 协议冻结 6（prompt+runtime 一致）；核 SDK iteration↔LLM 调用计数 | ⚠️❓ |
| attention | `VLLM_FLASH_ATTN_VERSION=2`，`disable_cascade_attn=true`，`VLLM_FLASH_ATTN_VERSION`/`enforce_eager=false` | 按所选 verl-vLLM 组合核 kernel 支持 | ❓ |
| 调试开关 | `CUDA_LAUNCH_BLOCKING=1`、`TORCH_USE_CUDA_DSA=1`、`set -x`、`dump_data_batch=true` | 正式 profile 关闭；`set -x` 带秘密时移除 | ⚠️ |

## 4. 数据与轨迹语义（源码事实，须原样迁移）

- 原生成器把整 episode 聚合为**单条**训练序列：`response_ids = last_prompt_ids + last_response_ids - first_prompt_ids` 前缀差，loss mask 复刻 `<|im_start|>assistant` 遮蔽（4B-Instruct `buffer_succeed=1`、`buffer_precede=1`）。逐轮生成 token 全覆盖 mask=1。**verl 侧须等价复刻并审计**。
- 轮数耗尽无 finish → 整条 mask 置 0（reward 亦 0）。
- `sanity_check_last_step`：最后响应恰一对 tool 标记、恰一个 `<|im_end|>`、`⿻` 后无正文；失败得 0。
- 无 TokenEvent 的异常 rollout：占位 token `151643`、mask 0、reward 0，按 error 处理。
- 数据：SWE-smith 经 `build_dataset.py`（seed42 打乱、末100 validation、`--use_patch` 置 `base_commit=None`、应用 mutation patch）；`clone_instance` 从 GitHub 克隆 + checkout/apply patch。
- 奖励：`multilevel_localization_f1_reward`（1/1/1，0–3），解析用 `parse_structured_outputs`。

## 5. verl v0.9.1 环境组合（tag 源码实测，2026-10-07）

- `requires-python = ">=3.10,<3.13"` → **Python 3.11 主试可行**。
- 官方锁定组合（pyproject.toml + uv.lock）：`torch==2.11.0`（cu130）、`torchvision==0.26.0`、`torchaudio==2.11.0`、`vllm==0.24.0`（cu130/torch2.11 abi3）、`transformers==5.9.0`、`flash-attn==2.8.3`（cu130torch211 wheelhouse）、`liger-kernel>=0.8.2`、`trl==0.27.0`、`flash-linear-attention==0.5.2`、`cupy-cuda13x==14.0.1`、`ray[default]>=2.41.0`、TransferQueue（git）。
- 安装方式：uv 为官方支持途径（uv.lock 全局锁 + `--extra fsdp --extra vllm`）；本项目以 conda prefix 承载 Python 3.11、以 uv 从官方 uv.lock 导出锁版本安装（AGENTS 隔离约定）。
- driver 595.58.03 ≥ CUDA 13.0 最低要求（580+），cu130 wheel 兼容【待核验：安装后实测】。
- 安装后必须实测：vLLM 0.24 + Qwen3-4B 模型加载、hermes parser 与该版本兼容性（原 vLLM 0.11 用 hermes，0.24 的 parser 行为须验证）。

## 6. 未解事项（进入训练前必须消解）

1. ~~verl v0.9.1 的 GSPO/`sequence_mean` 支持矩阵~~——已消解（2026-10-07）：GSPO 原生、`seq-mean-token-mean` 对应，见 §2/§5；剩余为 multi-turn token 归一化的梯度级对齐。—— 步骤 4
2. ~~SkyRL AdamW betas/eps/weight_decay 的确切值~~——已消解（2026-10-07）：betas [0.9,0.999]、wd 1e-2、eps 1e-8（torch 默认）、max_grad_norm 1.0、constant_with_warmup(0)。—— 步骤 4 配置落实
3. verl agent loop 是否满足：原始采样 token IDs 保留、逐轮 mask、不 retokenize。—— 步骤 4
4. OpenHands SDK `max_iteration_per_run` 计数语义（turns=6 冻结的前提）。—— 步骤 3
5. Hermes parser + Qwen3-4B-Instruct-2507 + enable_thinking=False 的实际兼容性。—— 步骤 3/4
6. 官方评测 fork（benchmarks `agentic_code_search`）解码参数与 workspace 路径。—— 步骤 5

## 变更历史

- 2026-10-07：依据源码 `9d05a64`（`run_async_training_4b.sh`、`code_search_generator.py`、`file_localization.py`、`module_rewards.py`、`localization_finish.py`、`instance.py`、`build_dataset.py`、`async_trainer.py`、`pyproject.toml`）建立首版映射。
