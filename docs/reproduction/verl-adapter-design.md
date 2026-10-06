# verl v0.9.1 ToolAgentLoop 适配设计（实现者文档）

维护日期：2026-10-07（Asia/Shanghai）。研究基于 **verl v0.9.1 tag，commit `1876b06d0a3e4e71e06230be10af14492ca8a75b`**（`git clone --depth 1 --branch v0.9.1` 到 `/tmp/verl-study`，浅克隆 graft commit）。本文所有 verl 源码引用均相对 `/tmp/verl-study/`；CodeScout 侧引用相对本仓库根。上游协议：[protocol-v1.md](protocol-v1.md)，映射账：[skyrl-to-verl.md](skyrl-to-verl.md)。文中 `<tool_call>`/`</tool_call>` 即 Hermes 工具调用起、止两个特殊标记（byte-exact 定义见 `verl/experimental/agent_loop/tool_parser.py:103-104`；od -c 已核对源码字节）。

**阅读约定：**【源码事实】可从给定 path:line 复核；【设计建议】是下一步实现提案（`src/verl_adapter/` 均为拟建，尚不存在）；【阻塞风险】是 verl 机制与原语义无法逐点对齐、需显式决策或子类覆盖的点。

---

## 1. ToolAgentLoop 状态机与 response_mask 构造

### 1.1 状态机【源码事实】

注册名 `tool_agent`（`verl/experimental/agent_loop/tool_agent_loop.py:101`）。主循环 `run()`（tool_agent_loop.py:131-213）在四个状态间转移（`AgentState`，tool_agent_loop.py:48-53）：

- **PENDING**（一次）：`_handle_pending_state`（tool_agent_loop.py:215-232）调 `ct_build_initial_tokens`（传 `tools=tool_schemas`，line 222-229）编码 system+user+tools 得初始 `prompt_ids`，转 GENERATING。
- **GENERATING**（每次 LLM 调用一次）：`_handle_generating_state`（tool_agent_loop.py:234-308）：
  1. 注入 tool parser 的 stop token（line 239-241；Hermes 为空，见 §2.3）；
  2. `server_manager.generate(...)`（line 243-253），**每次调用的 prompt 是当前累积的 `agent_data.prompt_ids`**（prefix-cache 语义）；
  3. `assistant_turns += 1`（line 272）；
  4. `ct_merge_assistant_token`（line 274-280）把本轮生成 token 原样并入运行时流并扩展 mask；
  5. 终止判定（line 289-295，先于 tool 解析）：`len(response_mask) >= response_length` 或 `assistant_turns >= max_assistant_turns` 或 `user_turns >= max_user_turns` → TERMINATED；
  6. `extract_tool_calls`（line 300-302）；无 tool_calls → **TERMINATED（line 305-308）**，有 → PROCESSING_TOOLS。
- **PROCESSING_TOOLS**（每工具轮一次）：`_handle_processing_tools_state`（tool_agent_loop.py:310-411）：
  1. 截取 `tool_calls[:max_parallel_calls]` 并 `asyncio.gather` 并发执行（line 317-321）；
  2. 每个响应构造 `{"role":"tool","content":text}`（line 345-352），tool reward 收进 `agent_data.tool_rewards`（line 375-376）；
  3. `ct_merge_context_msg`（line 386-393）把 tool 段（含下一轮 generation prompt）编码并入 token 流；**若合并后 `len(response_mask) >= response_length` 直接 TERMINATED（line 394-395）**；
  4. `user_turns += 1`（line 410），转 GENERATING。

输出装配（tool_agent_loop.py:183-213）：`response_ids = prompt_ids[-len(response_mask):]`、`prompt_ids = prompt_ids[:len-len(response_mask)]`——**prompt/response 以 mask 长度切分，工具结果 token 全归 response 段**；`response_ids`/`response_mask`/`response_logprobs` 再截断到 `rollout.response_length`（line 195-202）。`num_turns = user_turns + assistant_turns + 1`（line 203）。

### 1.2 ct_merge_* 的实现位置与语义【源码事实】

基类薄封装在 `verl/experimental/agent_loop/agent_loop.py:389-435`，实现在 **Continuous Token (CT) builder**（`verl/utils/tokenizer/continuous_token.py`）：

- `ct_build_initial_tokens`（agent_loop.py:318-362）→ `build_initial_tokens`（continuous_token.py:99-111）→ `_render_tokens`（continuous_token.py:226-241）→ `apply_chat_template(tokenizer, messages, tokenize=True, add_generation_prompt=True, tools=tools, **chat_template_kwargs)`（`verl/utils/tokenizer/chat_template.py:141-168` 直接转发 HF `apply_chat_template`，含 Qwen3.5 空 user 兼容 fallback）。文本 prompt 超长左截断（agent_loop.py:437-446）。
- `ct_merge_assistant_token`（agent_loop.py:413-435）→ `merge_assistant_tokens`（continuous_token.py:197-204）：**模型生成 token ids 原样拼接**，不重新分词（`appended_token_count=len(assistant_token_ids)`），返回 `MergeResult(kind="assistant")`。
- `ct_merge_context_msg`（agent_loop.py:389-411）→ `merge_context_tokens`（continuous_token.py:180-195）→ `tokenize_context_incremental_messages`（continuous_token.py:113-166）：按 message 组（连续 tool messages 一组，continuous_token.py:311-325）做 **suffix-diff 增量编码**——用合成前缀 `[synthetic system, synthetic user, synthetic assistant(tool_calls)]` 渲染取前缀差（`render_delta_token_id`，continuous_token.py:243-261；前缀不一致直接抛错 line 258-260）。合成 assistant 仅为被 diff 掉而设（`_synthetic_assistant_for_tools`，continuous_token.py:343-363）。Qwen 边界：运行时前缀以 `<|im_end|>` 结尾而模板要求其后有换行时，`QwenContinuousTokenBuilder._merge_context_token_ids`（continuous_token.py:562-578）**插入一个换行 token 并登记为 `inserted_token_ids`**。

Qwen3 路由：HF `model_type="qwen3"` → `QwenContinuousTokenBuilder`（`_MODEL_TYPE_TO_FAMILY` qwen3→QWEN3，`verl/utils/tokenizer/continuous_token_wiring.py:126-127`；registry continuous_token_wiring.py:80；入口 `create_continuous_token_builder`，continuous_token_wiring.py:218）。`Qwen/Qwen3-4B-Instruct-2507` 的 model_type 应为 qwen3【待核验：部署时加载 config.json 确认】。

### 1.3 response_mask：哪些 token mask=1【源码事实】

mask 唯一权威实现 `ContinuousTokenBuilder.align_response_metadata`（continuous_token.py:365-419）：

| token 类别 | mask | 证据 |
|---|---|---|
| 模型生成的 assistant token（每轮全部生成 ids，含结尾 stop token） | **1** | continuous_token.py:398-411（kind=="assistant"：`aligned_mask += [1]*appended_token_count`） |
| 工具结果/上下文段（模板渲染的 user 角色标记、tool 正文、其 `<|im_end|>`、换行，及下一轮 generation prompt `<|im_start|>assistant` + 换行） | **0** | continuous_token.py:412-415（kind=="context"：`[0]*appended_token_count`） |
| CT 自建边界 token（Qwen 的 `<|im_end|>` 后补的换行） | **0** | continuous_token.py:392-396；MergeResult docstring 明确 boundary 不计 loss/logprob（line 44-48） |
| suffix-diff 删除的前缀 token | mask 同步裁剪 | continuous_token.py:386-390 |

worker 层 `_agent_loop_postprocess` 右 pad 到 `response_length` 且与 attention_mask 相乘（agent_loop.py:767-779），padding 为 0。`AgentLoopOutput` docstring 同口径 "1 for LLM generated token, 0 for tool response token"（agent_loop.py:95-96）；worker docstring 示例 `| 1..1 | 0..0 (tool) | 1..1 | 0 (padding)`（agent_loop.py:584-586）。

**推论：每轮 generation prompt `<|im_start|>assistant` + 换行属 context 段 mask=0；工具结果整段（含 user 角色标记）mask=0；模型生成 token（含其自身 stop token `<|im_end|>`，vLLM/SGLang 将 stop token 计入输出 ids）mask=1。**

### 1.4 与原 SkyRL 生成器 mask 逐点对比

原实现（`src/generator/code_search_generator.py:407-468`）：整 episode 聚合单条序列（`response = last_prompt + last_response - first_prompt` 前缀差，line 412-416），逐 token 扫描（line 426-456）：`<|im_start|>` 置 inside=True 并把前 `buffer_precede=1` 个 token 回溯置 0（line 434-437）；紧跟的 `assistant` 角色词关闭 inside 并给 `buffer_succeed` 个后续 token 置 0（line 439-450；`buffer_succeed=1` for Qwen3-4B-Instruct-2507，line 421-422）；其余 mask=1。轮数耗尽无 finish → 整条置 0（line 464-468）。

| 点位 | 原 SkyRL | verl ToolAgentLoop | 结论 |
|---|---|---|---|
| 模型生成 token | 1 | 1（continuous_token.py:400） | 相同 |
| `<|im_start|>assistant` + 换行角色标记 | 0（inside + buffer_succeed=1） | 0（context 段） | 相同 |
| `<|im_start|>` 前的 1 个 token（换行） | 0（buffer_precede=1 回溯清零，line 435-437） | 0（CT 补的边界 token，continuous_token.py:570-572） | 相同（机制不同） |
| 工具结果段（user 角色整段） | 0（inside 持续到下个 assistant 角色词） | 0（context） | 相同 |
| 每轮 assistant 结尾 `<|im_end|>` | 在序列中则 mask=1；**但最后一轮的 `<|im_end|>` 因 `include_stop_str_in_output=False`（code_search_generator.py:166-167）不在 response_ids 内** | **每轮都在 response_ids 且 mask=1**（stop token 计入生成 ids） | **不同【阻塞风险 B-1】**：verl 每 episode 多 1 个 mask=1 token（最后轮 `<|im_end|>`），gsPO 序列比率的长度分母 +1。缓解见 §10.1 |
| thinking 模型 buffer_succeed=5 默认分支 | 遮 5 个 | context 段整体遮蔽 | 语义相同（本项目 4B-Instruct + enable_thinking=False，不涉及） |
| 轮数耗尽整条置 0 | generator 内实现（line 464-468） | **无原生开关**；需自定义 agent loop 在 rollout 层做（§4） | 落地方式不同【阻塞风险 A-1】 |
| 采样 token 保留 | SDK retokenize 消息后前缀差重建 | merge_assistant_tokens 原样拼接，不 retokenize（continuous_token.py:197-204） | verl 更符合协议 §6"保留真实采样 token IDs"（skyrl-to-verl §6.3 可关闭） |

### 1.5 tokenization 一致性【源码事实】

CT 用 suffix-diff 保证"增量编码 == 全量 apply_chat_template 的后缀"（continuous_token.py:258-260 抛错校验），天然满足"训练/推理分词一致"。`multi_turn.tokenization_sanity_check_mode`（config/rollout.py:60）只被**非 agent-loop 的同步 rollout schemas** 消费（`verl/workers/rollout/schemas.py:606-655`），对 ToolAgentLoop 不生效——CT builder 的前缀校验才是这条路径的一致性保障。

---

## 2. 工具定义、ToolResponse 与终止语义

### 2.1 自定义工具的两条路【源码事实】

- **`FunctionTool`**（`verl/tools/function_tool.py:44-64`）：`@function_tool` 装饰器注册（function_tool.py:67-138），schema 从函数签名+Google docstring 经 `transformers.utils.get_json_schema` 推断（function_tool.py:187-210），或 `schema=` 直接给 `OpenAIFunctionToolSchema`（function_tool.py:107-111）。`async call(parameters)`（function_tool.py:60-64）直接调用；**stateless，`tools_kwargs`/instance 语义被有意忽略**（tool_agent_loop.py:476-482 注释明示 "use a BaseTool subclass instead"）。返回经 `normalize_function_tool_return` 归一 `(ToolResponse, reward, metrics)`（function_tool.py:213-248）。
- **`BaseTool`**（`verl/tools/base_tool.py:24-93`）：生命周期 `async create(instance_id=None, **kwargs) -> (id, ToolResponse)`（base_tool.py:46-59）、`async execute(instance_id, parameters, **kwargs) -> (ToolResponse, reward, metrics)`（base_tool.py:61-74）、`calc_reward`（76-85）、`release`（87-93）。调用点 `_call_tool`（tool_agent_loop.py:472-496）：`kwargs = tools_kwargs.get(tool_name, {})`；`instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))`（line 486）；`await tool.execute(instance_id, tool_args, agent_data=agent_data)`（line 487-489）；finally 中 release（line 493-496）。**instance_id 维度传入方式：dataset 行 `extra_info.tools_kwargs.<tool_name>.create_kwargs`**——样板见 `examples/data_preprocess/gsm8k_tool_agent_loop.py:103-109`；提取在 `verl/utils/dataset/rl_dataset.py:403`，注入 `tools_kwargs` 列（rl_dataset.py:409）→ worker 逐样本 kwargs（agent_loop.py:643-653）→ `agent_data.tools_kwargs`（tool_agent_loop.py:143,153）。`execute` 拿到完整 `agent_data`（含 `messages`/`extra_fields`；docstring 明确允许工具存 session 数据，tool_agent_loop.py:56-58,98）——这是把结构化预测写入轨迹的最佳通道。

工具声明文件：`multi_turn.tool_config_path`（yaml，`initialize_tools_from_config`，`verl/tools/tool_registry.py:55-80`；格式 `tools: [{class_name, tool_schema?, config: {type: native, ...}}]`）或 `multi_turn.function_tool_path`（py 文件，`load_function_tools_from_path`，function_tool.py:151-184），worker 启动时合并加载（agent_loop.py:526-531；重名冲突检测 tool_registry.py:91-99）。`data.tool_config_path`/`data.function_tool_path` 用 `${oc.select:...}` 引用同一值（`verl/trainer/config/data/legacy_data.yaml:46-49`），仅用于 prompt 长度过滤（rl_dataset.py:129-151）。

### 2.2 ToolResponse 结构【源码事实】

`verl/tools/schemas.py:98-127`：`text: str|None`、`image: list|None`、`video: list|None`。响应在 loop 中转 `{"role":"tool","content":text}`（tool_agent_loop.py:345-352）。截断见 §3。

### 2.3 "调用即终止"：localization_finish 语义如何落地

**【源码事实】ToolAgentLoop 无显式 terminate 接口**：终止仅四来源——①本轮生成无 tool_calls（tool_agent_loop.py:305-308）；②response_length 满（line 290,394）；③`max_assistant_turns` 满（line 292）；④`max_user_turns` 满（line 294）。BaseTool 无任何控制信号返回（base_tool.py:24-93）。

**【阻塞风险 A-2，核心】** 原语义（`src/tools/localization_finish.py:86-98` `LocalizationFinishExecutor` 置 `ConversationExecutionStatus.FINISHED`）是 **finish 执行后 episode 立即结束，不再有下一次 LLM 调用**。verl 原生 loop 执行完任意工具都回 GENERATING（tool_agent_loop.py:410-411），即 finish 后**还会多一次 LLM 调用**，其 token mask=1 进 loss——不等价：多烧推理预算、改变 6-turn 计数口径、"最后一次响应"不再含 finish 调用（sanity check 对象漂移）。

**【设计建议】** `src/verl_adapter/agent_loop.py` 定义 `CodeSearchAgentLoop(ToolAgentLoop)`，`@register("codescout_tool_agent")`，覆盖 `_handle_processing_tools_state`：父类逻辑执行完后，若本轮解析出的 tool_calls 含 `localization_finish`，直接返回 `AgentState.TERMINATED`。配套：
- finish 的返回文本仍并入 token 流（原实现 finish 也产生 Observation 文本，只是不再触发下一轮）；
- 在 `agent_data.extra_fields` 写 `structured_locations`（经 `execute(..., agent_data=...)`，tool_agent_loop.py:487-489），供 reward fn 无损读取（§5）；
- 原实现要求 finish 恰好一次（`code_search_generator.py:87-90` 计数 `cnt != 1 → None`）：适配器在 extra_fields 记 `finish_call_count`；同轮并行出现多个 finish 或之前轮已调用过，reward 侧判 0（对应原 sanity check `code_search_generator.py:270-272` 恰一对工具标记）。

**Hermes parser【源码事实】**：`multi_turn.format="hermes"`（config/rollout.py:61 默认）→ `HermesToolParser`（`verl/experimental/agent_loop/tool_parser.py:96-129`）：起止标记即 §头所述 <tool_call>/</tool_call>（line 103-104），regex `<tool_call>(.*?)</tool_call>` DOTALL（line 105）；解码整个 response 后按 regex 提取 JSON `{"name","arguments"}`（line 116-124；解析失败仅 log 并丢弃该 call），content 为去掉 tool 块的剩余文本（line 127）。stop_token_ids 为空（继承默认，tool_parser.py:54-64；Qwen3 靠自身 EOS `<|im_end|>` 停止）。**无服务端 auto tool choice**——verl 在 loop 内客户端解析，与原实现（vLLM 服务端 hermes 解析 + `enable_auto_tool_choice`）机制不同、语义应一致，须 smoke 验证（skyrl-to-verl §6.5）。

### 2.4 terminal 工具的 workspace 维度【设计建议】

原实现 bash 在 `/tmp/testbed/<uuid>` 克隆仓库执行（code_search_generator.py:129-132）。verl 侧每 episode 在 `create()` 时从 `tools_kwargs.create_kwargs` 拿离线准备好的**无 `.git` 代码快照**路径（protocol §3：`codescout-data/workspaces/<run_id>/<episode_id>`），或由 `create()` 现场从共享只读源复制；`release()` 清理本 episode 目录。**私有标签（file_changes/target/patch）只进 create_kwargs 或 reward_model 列，绝不进 prompt 列，也不出现在工具返回文本里。**

---

## 3. 工具并行 / 上限 / 截断【源码事实】

全部在 `actor_rollout_ref.rollout.multi_turn.*`（`verl/workers/config/rollout.py:47-62` `MultiTurnConfig`）：

| 键 | 默认 | 语义 | 证据 |
|---|---|---|---|
| `max_parallel_calls` | 1 | 每轮只执行解析出的前 N 个 tool_calls，其余静默丢弃（执行侧 tool_agent_loop.py:317；assistant message 序列化侧同样只记前 N 个，line 419） | config/rollout.py:56 |
| `max_tool_response_length` | 256 | 单个工具返回文本超过则截断 | config/rollout.py:57 |
| `tool_response_truncate_side` | "middle" | `left`：留尾部，前缀 "(truncated)..."；`right`：留头部，尾缀 "...(truncated)"；`middle`：首尾各 len//2，中间插 "...(truncated)..." | tool_agent_loop.py:499-506 |
| `enable` | False | 仅作为 meta_info 传给 actor update（ray_trainer.py:1329），v0.9.1 中 actor 侧无消费者；真正生效的是 `agent.default_agent_loop` + multi_turn 各键 | ray_trainer.py:1329；全库 grep 无其他消费点 |

**【设计建议】** terminal 搜索输出经常超长：`max_tool_response_length` 起步 16384（按 profile 调）、`tool_response_truncate_side=left`（保文件尾/报错），与原 OpenHands terminal 输出上限对齐【待核验：原 terminal 截断行为】。`max_parallel_calls=5`（协议 §3 每轮并行 bash ≤5）。注意 verl 是"丢弃超额调用"而非原实现的"违规判 0"——超并行违规须在 reward 侧用 extra_fields 里的每轮 tool_calls 计数另行判罚【阻塞风险 B-2】。

---

## 4. 轮数语义【源码事实】

`max_assistant_turns`（config/rollout.py:52）**= LLM 调用次数上限**：每次 `generate` 后 `assistant_turns += 1`（tool_agent_loop.py:272），随即判 `assistant_turns >= max_assistant_turns` → 终止（line 292-293）——判定先于 tool 解析，第 6 轮生成了 tool call 也不执行。`max_user_turns`（config/rollout.py:55）计工具轮（每轮 PROCESSING_TOOLS +1，tool_agent_loop.py:410）。协议冻结 6 turns（protocol §3）→ **`max_assistant_turns=6`**（配 `max_user_turns=5` 或留空；6 次调用至多 5 个工具轮）。OpenHands `max_iteration_per_run` 计数语义仍待核验（skyrl-to-verl §6.4），verl 侧口径明确即 LLM 调用数。

**per-call 生成上限**【阻塞风险 B-3】：`rollout.response_length` 是**整 episode 累计** response 预算（判定用累计 mask 长度，tool_agent_loop.py:290,394），没有"每次调用 8192"的原生 per-call cap。协议 §5 的每次调用 ≤8192 须靠 sampling_params 的 `max_tokens` 之类实现：worker 组装 sampling_params 只显式带 temperature/top_p/top_k/logprobs（agent_loop.py:590-596），其余键原样透传给 vLLM/SGLang 请求【待核验：v0.9.1 `LLMServerClient.generate` 对 `sampling_params["max_tokens"]` 的透传与多轮下的生效方式；不透传则在自定义 loop 里显式注入】。

**轮数耗尽 reward=0 + mask 全 0 的落地**：response_mask **可以**整条置 0——`agg_loss` 的 seq-mask 把全 0 序列从 sum 剔除（core_algos.py:1194-1198），但 `global_batch_size` 分母仍含它（§10），故全 0 序列只贡献 0 梯度、不改其余序列权重，与 SkyRL `sequence_mean` 中全 mask 序列贡献 0 一致。**做在哪一层**：response_mask 是 rollout 产物（`AgentLoopOutput.response_mask`），reward manager 只返回 reward 张量、不改 mask（`verl/workers/reward_manager/naive.py:98-153` 只写 `reward_tensor`）。因此【设计建议】在 `CodeSearchAgentLoop.run()` 装配 AgentLoopOutput 前判定：`finish_call_count == 0` 且（`assistant_turns >= max_assistant_turns` 或 response_length 触顶）→ `agent_data.response_mask` 全置 0，并置 `extra_fields["trajectory_exhausted"]=True`；reward fn 据此返回 0（等价原判据 `code_search_generator.py:324`：无 structured_locations 且 `len(token_messages) >= max_turns`）。这是阻塞点 A-1 的消解方式。

---

## 5. 奖励接入与私有标签数据流

### 5.1 reward 计算路径【源码事实】

- **注册**：`reward.custom_reward_function.path/name`（`verl/trainer/config/reward/reward.yaml:14-19`；`+reward.custom_reward_function.reward_kwargs` 可传额外 kwargs）→ `get_custom_reward_fn`（`verl/trainer/ppo/reward.py:50-86`）`load_extern_object` 动态加载并 `partial` 包装。reward manager 默认 `reward.reward_manager.name=naive`（reward.yaml:23-29）。
- **流式（本项目默认路径）**：无 RM 时 `enable_agent_reward_loop=True`（`verl/trainer/ppo/ray_trainer.py:956-959`：`not self.use_rm or enable_resource_pool`），reward worker handles 传入 AgentLoopManager（ray_trainer.py:963-970；main_ppo.py:127-133）→ AgentLoopWorker 持有 `reward_loop_worker_handles`（agent_loop.py:491-501）→ 每条轨迹 rollout 完立即 `_compute_score`（agent_loop.py:822 调用；实现在 agent_loop.py:967-1029）：**把该样本的全部 dataset kwargs 装进 non_tensor_batch**（`{k: np.array([v]*n) for k,v in kwargs.items()}`，agent_loop.py:1014，即 raw_prompt/extra_info/reward_model/data_source/file_changes 等所有列都在）+ `tool_extra_fields`（= `output.extra_fields`，agent_loop.py:1016）+ num_turns/长度，remote 调 `RewardLoopWorker.compute_score`（agent_loop.py:1025-1028）→ `reward_manager.run_single(data)`（`verl/experimental/reward_loop/reward_loop.py:145-155`；naive 实现在 `verl/experimental/reward_loop/reward_manager/naive.py:34-99`）。
- **`run_single` 给 reward fn 的东西**【源码事实】（reward_loop/reward_manager/naive.py:34-99）：
  - `data_source` = 行的 `data_source` 列（line 42）；
  - `ground_truth` = 行的 `reward_model.ground_truth`（line 43；**硬编码要求该列存在，否则 KeyError**）；
  - `extra_info` = 行的 `extra_info` 列（line 44）∪ `tool_extra_fields`（line 45-47）∪ `num_turns`、`rollout_reward_scores`（line 49-52）——**自定义 agent loop / 工具塞进 `agent_data.extra_fields` 的任意键都能到达 reward fn**；
  - `solution_str` = 最后一条序列 response 的解码文本，**`skip_special_tokens=True`**（line 54-56）——<tool_call>/</tool_call> 标记与 `<|im_start|>`/`<|im_end|>` 均被剥掉，assistant 各轮与工具结果文本连在一起；
  - 签名 `compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)`（line 67-84），返回 float 或 dict（dict 必含 `"score"`，其余键进 `reward_extra_info`，line 89-95 → agent_loop.py:1028 存回 extra_fields → `_postprocess` 展平进 non_tensor_batch，agent_loop.py:1107-1111）。
- **回填**：`reward_score` 进 `AgentLoopOutput.reward_score` → worker `_postprocess` 把它写到 `rm_scores` 的**最后一个有效 response token 位置**（agent_loop.py:1093-1099：`rm_scores[arange, valid_len-1] = scores`；单条 as_dict 路径同样 `rm_scores[-1]=reward`，agent_loop.py:144-148）→ trainer 侧若 batch 已含 rm_scores 则不再调 reward（ray_trainer.py:1563-1565 的 `"rm_scores" not in batch` 条件）。
- 无 async reward handles 时（如独立 RM resource pool），dataset 列经 `_postprocess` 的 `non_tensor_batch.update(input_non_tensor_batch)` 透传（agent_loop.py:1104-1105），trainer 侧 `_compute_reward_colocate`（ray_trainer.py:588-594）走同一 manager。两条路径给 reward fn 的字段一致。

### 5.2 私有标签传递（不进 prompt）【源码事实 + 设计建议】

数据面：parquet 的 `prompt` 列（`data.prompt_key`，默认 "prompt"，legacy_data.yaml:25-26）**只用于 `raw_prompt`**（rl_dataset.py:386-389 的 `_build_messages(row, key=self.prompt_key)`）；其余所有列（含 `reward_model`、`extra_info`、任何 `file_changes` 列）整体经 collate 变 non_tensor_batch（rl_dataset.py:59-71）→ 逐样本 kwargs 进 agent loop（agent_loop.py:645）→ reward 路径（§5.1）。**因此私有标签放 `reward_model.ground_truth`（结构化，naive manager 直接读）或 `extra_info` 内任意子键，均不会进 prompt**；actor prompt 只由 `prompt` 列决定。

【设计建议】不要依赖 `solution_str` 重解析预测（skip_special_tokens 剥标记后各轮文本粘连，脆弱）。推荐数据流：
- `localization_finish` 工具在 `execute()` 中解析参数 JSON，把结构化 `locations` 与解析状态写入 `agent_data.extra_fields["structured_locations"]`（无效 JSON → 写 None + 错误类别）；
- reward fn 从 `extra_info["structured_locations"]` 拿预测、从 `ground_truth` 拿真值，调 `src/rewards/file_localization/file_localization.py` 的 `multilevel_localization_f1_reward`（权重 1/1/1，范围 0-3）与 `parse_structured_outputs`；
- 无 TokenEvent 的原异常占位语义（`code_search_generator.py:484-492`：占位 token、mask 0、reward 0、stop_reason=error）在 verl 对应"工具/环境异常 → structured_locations=None + trajectory_exhausted 标志"，reward 0 但**infra 异常与模型失败分列**（protocol §4：infra 不计 0 分，用 reward_extra_info 记 failure_class，训练前冻结排除规则）。
- sanity check（原 `code_search_generator.py:263-282`：最后响应恰一对 <tool_call>/</tool_call>、恰一个 `<|im_end|>`、</tool_call> 后无正文）：verl 侧在**自定义 loop 的最后 assistant 轮**执行——对最后一轮 `response_ids` 解码后同样计数判罚，失败置 `structured_locations=None`（reward 0）但 mask 不清零（原实现也只清 reward 载体，mask 清零仅用于轮数耗尽路径，`code_search_generator.py:464-468`）。

### 5.3 "agent_loop.reward" 配置键说明

v0.9.1 **没有** `agent_loop.reward` 这一段键；agent-loop 场景的 reward 配置就是全局 `reward.*`（custom_reward_function + reward_manager），另有两个 rollout 内通道：① BaseTool 的 `execute` 每次返回 step reward（`tool_rewards` 收集在 extra_fields，agent_loop.py:212、tool_agent_loop.py:375-376，仅记录不进训练 reward 张量）；② `AgentLoopOutput.reward_score` 已被 agent loop 填好时跳过外部 reward（agent_loop.py:732 注释 "Some AgentLoop may have already computed the reward score, e.g SWE-agent"；reward_loop/reward_manager/naive.py:34-99 走 `run_single`，`_compute_score` 只在 `reward_score is None` 时调外部 reward，agent_loop.py:972）——即自定义 loop 也可以自己算 reward 直接填 `reward_score`，跳过 reward manager。**【设计建议】本项目用标准 reward manager + custom_reward_function 通道**（便于逐任务审计与 infra 失败分列），不用 loop 内自算。

---

## 6. dataset 格式（RLHFDataset）与最小 parquet schema

### 6.1 需要什么列【源码事实】

`__getitem__`（rl_dataset.py:386-411）返回整行 + 派生列：`raw_prompt`（由 `prompt_key` 列的 messages 列表构建，line 389）、`index`/`tools_kwargs`/`interaction_kwargs`（从 `extra_info` 提取，line 402-409）、`dummy_tensor`（line 398）。无 `extra_info` 列时自动补空 dict（line 400-401）。非张量列全部经 collate 成 object 数组进 non_tensor_batch（rl_dataset.py:59-71）。数据集类本身由 `data.dataset_cls` 或默认 RLHFDataset 提供（AgentLoopWorker `self.dataset_cls = get_dataset_class(config.data)`，agent_loop.py:507）。

`interaction_kwargs` 在 v0.9.1 仅有 dataset 侧赋值（rl_dataset.py:404,410），verl 内置 agent loop 不消费——预留给自定义 loop 经 kwargs 读取（对 CodeScout 可用于传 workspace/克隆配置）。

### 6.2 最小 parquet 列 schema【设计建议，字段名与 verl 约束对齐】

```text
data_source          str           常量 "codescout_swe_smith_py"（reward fn 选择键 + 分组统计）
agent_name           str           常量 "codescout_tool_agent"（可省；agent 名也可由 rollout.agent.default_agent_loop 统一指定）
prompt               list<dict>    messages：[{"role":"system","content":<system_prompt_custom_finish.j2 渲染>},
                                                {"role":"user","content":<file_module_custom_finish.j2 渲染>}]
reward_model         dict          {"style":"rule","ground_truth":<私有标签结构化对象>}   # ground_truth 原样到 reward fn
extra_info           dict          {"split","index","instance_id","repo",
                                    "tools_kwargs": {"terminal": {"create_kwargs": {"workspace_root":...}},
                                                     "localization_finish": {"create_kwargs": {...}}},
                                    "interaction_kwargs": {...}}
（可选）file_changes  任意           私有标签冗余列；只供审计工具读，actor 不可见（不进 prompt 列即不进 prompt）
```

硬约束：`reward_model.ground_truth` 必须存在（reward_loop/reward_manager/naive.py:43 直接下标）；`extra_info.index` 必须存在且**同 issue 的 8 条复制样本共享同一 index**——GRPO 组归一化按 `index` 分组（core_algos.py:311-316 `id2score[index[i]]`；verl rollout 用 `rollout.n` 复制时 index 天然相同）。注意 RLHFDataset 不自动加 index，`extra_info.index` 缺失时置 0（rl_dataset.py:402）——**全部样本同组会毁掉 GRPO 分组**，数据构建时必须显式写。

模板渲染在**离线数据构建阶段**完成（jinja2 渲染 `src/prompts/templates/system_prompt_custom_finish.j2` + `file_module_custom_finish.j2` 到 prompt 列），verl 没有"模板文件路径"配置键（见 §7.2）。

---

## 7. prompt 构建

### 7.1 tools 如何进 prompt【源码事实】

`ct_build_initial_tokens` → `apply_chat_template(..., tools=tool_schemas)`（agent_loop.py:222-229 → continuous_token.py:226-241 → chat_template.py:141-168）——tools 以 **HF chat template 的 `tools` 参数**传入，是否渲染成文本由模型模板决定。Qwen3 官方 chat template（tokenizer_config.json 内嵌 jinja）会把工具 JSON schema 渲染成 system 区域的 "# Tools" 文本块；本项目原实现走 vLLM OpenAI API 的 `tools` 参数，vLLM 同样经该模板渲染——**两条路径的 tool 文本由同一模板产生，语义一致；但 schema 字段（description/parameters/strict）的逐字段等价须 smoke 核对**【待核验：加载 Qwen3-4B-Instruct-2507 tokenizer 后 `apply_chat_template(messages, tools=[terminal_schema, finish_schema], add_generation_prompt=True, enable_thinking=False)` 的文本与原 OpenHands/vLLM 路径 diff】。工具 schema 进入 prompt 的内容 = `tool_schema.model_dump(exclude_unset=True, exclude_none=True)`（tool_agent_loop.py:119），因此**localization_finish 的 schema 描述必须从 `src/tools/localization_finish.py:100-121` 的 TOOL_DESCRIPTION 与 CodeLocation 字段 description 逐字迁移**，否则 prompt 漂移。

### 7.2 system/user 模板与 chat_template_kwargs【源码事实 + 设计建议】

- **模板文件路径没有配置键**：messages 来自 parquet `prompt` 列；system/user 内容在离线构建时渲染（§6.2）。`data.apply_chat_template_kwargs` 是唯一影响渲染行为的配置（`+data.apply_chat_template_kwargs.enable_thinking=False`），经 AgentLoopBase 传入 CT builder 的每次渲染（agent_loop.py:242,248-254）——**必须设置**（协议 §1：训练/评测关 thinking；教程同款 `+data.apply_chat_template_kwargs.enable_thinking=False`，examples/tutorial/agent_loop_get_started/agent_loop_tutorial.ipynb cell 34）。注意 RLHFDataset 的长度过滤也会带同一 kwargs + tool schemas（rl_dataset.py:129-151），保证过滤口径与 rollout 一致。
- `multi_turn.use_inference_chat_template`（config/rollout.py:59）是服务端模板开关，与 CT 路径无关。
- `actor_rollout_ref.model.custom_chat_template` 可整体替换 tokenizer/processor 的 chat template（agent_loop.py:540-543）——协议 §1 冻结**不改 chat template**，不用。

---

## 8. 训练配置树与启动脚本

### 8.1 tool_agent 相关配置树【源码事实】

```yaml
actor_rollout_ref:
  rollout:
    mode: async                        # 唯一支持的模式（config/rollout.py:278-292 sync 已移除）
    name: vllm                         # serving backend
    temperature / top_k / top_p        # 训练采样（worker 组装进 sampling_params，agent_loop.py:590-596）
    val_kwargs: {temperature, top_k, top_p, do_sample, n}   # 评测采样（agent_loop.py:604-607）
    prompt_length / response_length     # prompt 左 pad / response 右 pad 上限（agent_loop.py:753-765）
    agent:
      num_workers: 8                   # AgentLoopWorker 数（AgentLoopConfig，config/rollout.py:71-79）
      default_agent_loop: tool_agent   # 无 agent_name 列时的缺省（agent_loop.py:610-612）
      agent_loop_config_path: null      # 自定义 loop 的 hydra 实例化 yaml（agent_loop.py:533-539, 678-693）
      agent_loop_manager_class: null    # 自定义 manager FQN（ray_trainer.py:943-949）
    multi_turn:
      enable: False                    # 仅 meta_info（ray_trainer.py:1329）
      max_assistant_turns: null        # 本项目设 6
      max_user_turns: null
      max_parallel_calls: 1            # 本项目设 5
      max_tool_response_length: 256
      tool_response_truncate_side: middle
      tool_config_path: null            # BaseTool 声明 yaml（tool_registry.py:55-80）
      function_tool_path: null          # @function_tool py 文件（function_tool.py:151-184）
      format: hermes                   # tool parser 选择（tool_agent_loop.py:120）
      use_inference_chat_template: False
      tokenization_sanity_check_mode: strict   # 仅非 agent-loop 路径消费
      num_repeat_rollouts: null
data:
  prompt_key: prompt
  train_files / val_files / train_batch_size / max_prompt_length / max_response_length
  tool_config_path / function_tool_path   # ${oc.select} 引用 rollout 同名键，仅长度过滤用（legacy_data.yaml:46-49）
  apply_chat_template_kwargs: {}       # 本项目 +enable_thinking=False
  filter_overlong_prompts: True        # 超长样本直接剔除（注意：会悄悄改变任务分母，须在 manifest 记录剔除数）
reward:
  custom_reward_function: {path, name}
  reward_manager: {source: register, name: naive}
  reward_model: {enable: False}        # 本项目 False → 流式 reward 路径
```

`trainer.default_local_dir`、`trainer.v1.trainer_mode` 等通用键略；启动入口 `python -m verl.trainer.main_ppo`（TaskRunnerV1，main_ppo.py:100-145）。

### 8.2 ppo run 脚本样例【源码事实】

- **tool_agent 教程**（examples/tutorial/agent_loop_get_started/agent_loop_tutorial.ipynb cell 34，hydra overrides）关键项：`data.return_raw_chat=True`（v0.9.1 RLHFDataset 中此键已无消费者，仅存储，rl_dataset.py:116；可不设）、`actor_rollout_ref.rollout.multi_turn.tool_config_path=...`、`actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent`、`data.train_batch_size=32`、`rollout.n=8`、`+data.apply_chat_template_kwargs.enable_thinking=False`。
- **GSPO 官方脚本**（examples/gspo_trainer/run_qwen3_8b_fsdp.sh:107-141）关键覆盖项：`actor_rollout_ref.actor.policy_loss.loss_mode=gspo`、`actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean`、`actor_rollout_ref.actor.clip_ratio_low=3e-4`、`actor_rollout_ref.actor.clip_ratio_high=4e-4`、`actor_rollout_ref.actor.clip_ratio_c=10.0`、`actor_rollout_ref.actor.optim.lr=1e-6`、`algorithm.adv_estimator=grpo`、`algorithm.use_kl_in_reward=False`、`actor.use_kl_loss=False`、`rollout.n=16`、`model.use_remove_padding=True`。
- **带工具的异步 recipe**（verl/experimental/fully_async_policy/shell/dapo_7b_async_retool.sh:74-110）：示范 `reward.custom_reward_function.path/name` 指向外部文件、`data.custom_cls` 自定义 dataset、clip_ratio_low/high——本项目主实验用官方同步 trainer，此脚本仅作参考。

**【设计建议】** CodeScout 启动脚本（后续在 `scripts/` 落地）在 GSPO 脚本骨架上加：`rollout.agent.default_agent_loop=codescout_tool_agent`、`rollout.agent_loop_config_path=<src/verl_adapter/agent_loop.yaml>`、`rollout.multi_turn.{max_assistant_turns:6, max_parallel_calls:5, max_tool_response_length:16384, tool_response_truncate_side:left, format:hermes}`、`rollout.multi_turn.tool_config_path=<src/verl_adapter/tools.yaml>`、`reward.custom_reward_function.path=<src/verl_adapter/reward.py>`、`algorithm.norm_adv_by_std_in_grpo=False`、`data.train_batch_size=8`、`rollout.n=8`、`rollout.response_length=32768`（6×8192 上限内按 profile）、`rollout.prompt_length=40960`。

---

## 9. policy loss / 优化配置键名（v0.9.1 精确键路径）【源码事实】

数据类定义：`ActorConfig`（`verl/workers/config/actor.py:104-193`）、`PolicyLossConfig`（actor.py:76-101）、`FSDPOptimizerConfig`（`verl/workers/config/optimizer.py:87-124`）、`AlgoConfig`（`verl/trainer/config/algorithm.py:621-684`）。

| 目标 | 键路径 | 默认 | 证据 |
|---|---|---|---|
| GSPO loss | `actor_rollout_ref.actor.policy_loss.loss_mode` | vanilla（合法值含 "gspo"） | actor.py:83-84,94 |
| clip low/high | `actor_rollout_ref.actor.clip_ratio_low` / `clip_ratio_high`（None 时回落 `clip_ratio`） | 0.2 / 0.2；gspo 实现里 fallback 见 core_algos.py:1582-1583 | actor.py:158-160 |
| loss reduction | `actor_rollout_ref.actor.loss_agg_mode`，本项目 `seq-mean-token-mean` | token-mean（合法 5 种，actor.py:212-218） | actor.py:164；实现在 core_algos.py:1194-1201 |
| GRPO 是否除 std | `algorithm.norm_adv_by_std_in_grpo`（False = Dr.GRPO 组中心化） | True → 本项目设 False | algorithm.py:658；core_algos.py:268-316 |
| adv estimator | `algorithm.adv_estimator` | gae → 设 grpo | algorithm.py:657 |
| KL reward / KL loss | `algorithm.use_kl_in_reward`、`algorithm.kl_ctrl.kl_coef`；`actor.use_kl_loss`、`actor.kl_loss_coef`、`actor.kl_loss_type` | 两项均关 | algorithm.py:659-661；actor.py:171,175-176 |
| entropy | `actor_rollout_ref.actor.entropy_coeff` | 0 | actor.py:166 |
| AdamW | `actor_rollout_ref.actor.optim.{optimizer:"AdamW", optimizer_impl:"torch.optim", lr, betas:[0.9,0.999], weight_decay, clip_grad}` | betas (0.9,0.999)、wd 0.01、clip_grad 1.0 | optimizer.py:47-53,106-114；betas 注入 build_optimizer optimizer.py:331-333 |
| **AdamW eps** | 无顶层键；FSDP 用 `actor_rollout_ref.actor.optim.override_optimizer_config: {eps: 1e-8}`（build_optimizer 里 merge 进 kwargs，optimizer.py:335-336,346-351）——不设则 torch 默认 1e-8，与 SkyRL 一致但**显式写死防漂移** | — | optimizer.py:326-351 |
| lr scheduler | `actor_rollout_ref.actor.optim.lr_scheduler_type`（FSDP 合法值 constant/cosine）、`min_lr_ratio`、`num_cycles`、`zero_indexed_step` | constant | optimizer.py:99-124 |
| warmup | `actor.optim.lr_warmup_steps_ratio` / `lr_warmup_steps` | 0.0 / -1（恒定 lr） | optimizer.py:48,51 |
| 更新轮数 | `actor_rollout_ref.actor.ppo_epochs`（= update_epochs_per_batch） | 1 | actor.py:177 |
| mini batch | `actor_rollout_ref.actor.ppo_mini_batch_size`（**单位：prompt 数**；global 序列数 = ×rollout.n，ray_trainer.py:1361）+ `rollout.n` | 256 | actor.py:151；ray_trainer.py:1360-1361 |
| 有效 batch | `data.train_batch_size`（prompt 单位）× `rollout.n`（每 prompt 采样数）= 64 序列 | — | 教程脚本口径；GRPO 按 index 分组自动含 8 条复制 |
| grad clip（FSDP） | `actor_rollout_ref.actor.grad_clip` | 1.0 | actor.py:307 |

**GSPO 实现要点**（core_algos.py:1545-1621）：序列级 importance ratio = mask 内 log-ratio 的**序列均值**再 exp（line 1587-1593），token 级 ratio 用 stop-gradient 合成（line 1596-1598），log 域 clamp max=10.0（line 1598），不对称 clip（line 1582-1583 low/high），`pg_clipfrac_lower` 恒 0 仅兼容（line 1614）。与 SkyRL `gspo_policy_loss`（skyrl-to-verl §2）同公式同 clamp，无需自写 loss。

---

## 10. 与原实现的已知差异清单（verl 聚合/归一化路径）

对齐判据沿用 protocol §6：**同全局 batch 一次 optimizer step 的最终梯度逐张量对照**；逐步中间量对照仅作诊断。`global_batch_size` 来源 = `ppo_mini_batch_size × rollout.n`（ray_trainer.py:1360-1361），`batch_num_tokens` = loss_mask 的 dp 全归约和（`verl/workers/engine/fsdp/transformer_impl.py:730-734`），二者经 `config.global_batch_info` 注入 loss（`verl/workers/utils/losses.py:64-68`）。

| # | 差异点 | 影响 | 处置 |
|---|---|---|---|
| D1 | **全 mask 序列**：verl `seq-mean-token-mean` 把全 0 mask 序列从 sum 剔除（core_algos.py:1194-1198），分母 `global_batch_size` 仍含它；SkyRL `.mean()` 同样含全 mask 序列（贡献 0） | 二者最终梯度口径一致；但 verl 的 `seq_losses` 分母有 `+1e-8`（core_algos.py:1195），SkyRL 无 | **必须固定张量数值对齐验收**：构造含全 mask 序列的 mini batch 对比单步梯度 |
| D2 | global_batch_size 用配置值（mini×n）而非实际非空序列数：若 rollout 产生退化/丢弃样本导致实际序列数 < 配置值，verl 仍按配置值归一 | 与 SkyRL 固定 batch 语义一致；仅当数据丢失时才偏离 | 数值验收时核对实际样本数；只影响梯度幅度不影响方向 |
| D3 | **最后轮 `<|im_end|>` 多 1 个 mask=1 token**（B-1，§1.4） | 每 episode 训练 token 多 1；gsPO 序列均值比率的分母 +1 | 决策项：默认保留 verl 行为并披露；若要逐 token 对齐，在自定义 loop 装配前把最后一个 `<|im_end|>` 从 response 尾部移除或 mask 置 0（须与 6-turn 终止判定兼容）。**数值验收时单独量化此 token 对 loss 的贡献** |
| D4 | 微批/梯度累积组织不同：verl 按 ppo_micro_batch_size_per_gpu 或 dynamic_bsz 切分（actor.py:151-157），SkyRL micro_train_batch_size_per_gpu=1 | 单卡内浮点求和顺序不同 → 位级差异 | **不要求位级一致**；以 fp64 参考或较大容差对齐（只影响精度，不影响语义） |
| D5 | rm_scores 位置：verl 写在最后有效 response token（agent_loop.py:1093-1099）；advantage 对全 mask 序列的 reward 处理由 GRPO 组内均值承担（outcome reward 与位置无关） | 无语义差 | 仅日志/审计注意 |
| D6 | rollout logprobs：`rollout.calculate_log_probs` 默认 False（config/rollout.py:219），训练用 trainer 重算的 old_log_probs（ray_trainer.py:1577-1585） | 与 SkyRL（训练前重算）一致 | 无需动作 |
| D7 | 采样引擎差异：verl 客户端 hermes 解析 vs 原 vLLM 服务端 `enable_auto_tool_choice`；stop token 计入 ids vs `include_stop_str_in_output=False` | 解析失败样本的归类路径不同（verl 丢弃坏 call 而不判罚，tool_parser.py:123-124） | smoke 验证解析等价性；解析失败在 reward 侧用 finish_call_count/locations 解析状态判罚 |
| D8 | `data.filter_overlong_prompts=True` 会静默剔除超长 prompt 样本（rl_dataset.py:206-243） | 任务分母变化 | manifest 记录剔除数与 instance_id，冻结处理规则（protocol §2） |
| D9 | verl 无 per-call 生成上限（B-3，§4） | 长轮可能吃光整 episode 预算 | sampling_params 注入 max_tokens=8192【待核验透传】；只影响资源不影响 loss 语义（被截断轮 mask=1 全保留，与原实现"不截断只记 mask"不同——原实现注释掉的截断分支在 code_search_generator.py:458-462，原本也不截断，故一致） |
| D10 | num_turns 口径 `user_turns+assistant_turns+1`（tool_agent_loop.py:203） | 仅日志/指标 | 无 |

---

## 11. src/verl_adapter/ 模块设计（实现提案）

### 11.1 模块拆分与接口签名草案【设计建议】

```text
src/verl_adapter/
  __init__.py            # 导出 register 入口；确保被 agent_loop_config_path 指向的模块 import
  agent_loop.py          # CodeSearchAgentLoop（A-1/A-2/B-1/B-3 的消解点）
  tools.py               # TerminalTool（BaseTool）+ LocalizationFinishTool（BaseTool）
  tools.yaml             # tool_config_path 声明（class_name: src.verl_adapter.tools.*）
  agent_loop.yaml        # hydra 实例化（name: codescout_tool_agent, _target_: src.verl_adapter.agent_loop.CodeSearchAgentLoop, tools: ${tool_list_ref}）
  reward.py              # compute_score：reward.custom_reward_function 通道
  workspace.py           # 快照准备/克隆/清理（服务器侧执行约定，AGENTS §2 允许根内）
  build_dataset.py       # parquet 构建（离线；渲染 j2 模板、写 extra_info.index/ground_truth）
```

**agent_loop.py 关键接口草案：**

```python
@register("codescout_tool_agent")
class CodeSearchAgentLoop(ToolAgentLoop):
    """语义覆盖点：
    1) localization_finish 执行后立即 TERMINATED（不再发起下一次 LLM 调用）；
    2) 轮数耗尽且无 finish -> response_mask 整条置 0 + trajectory_exhausted 标记；
    3) 最后 assistant 轮 sanity check（<tool_call>/</tool_call> 计数、<|im_end|> 计数、</tool_call> 后正文）；
    4) per-call 生成上限：sampling_params 注入 max_tokens（若 smoke 证明默认不透传）。
    """
    async def run(self, sampling_params, priority=0, **kwargs) -> AgentLoopOutput: ...
    async def _handle_processing_tools_state(self, agent_data) -> AgentState: ...
    def _finalize(self, agent_data) -> AgentLoopOutput: ...   # mask 清零 + extra_fields 落盘
```

**tools.py 关键接口草案：**

```python
class TerminalTool(BaseTool):
    # schema: 从函数签名/docstring 生成（对齐原 TerminalTool 的 bash 语义与 prompt 文本）
    async def create(self, instance_id=None, create_kwargs=None) -> tuple[str, ToolResponse]:
        # create_kwargs: {workspace_root, snapshot_src, timeout_s}
    async def execute(self, instance_id, parameters, agent_data=None) -> tuple[ToolResponse, float, dict]:
        # parameters: {"command": str}；子进程执行 bash，stdout/stderr 合并截断
    async def release(self, instance_id) -> None: ...  # 清理本 episode workspace

class LocalizationFinishTool(BaseTool):
    # schema: 逐字迁移 src/tools/localization_finish.py:100-121 TOOL_DESCRIPTION 与
    #        CodeLocation 字段（file 必填，class_name/function_name 可选）
    async def execute(self, instance_id, parameters, agent_data=None) -> tuple[ToolResponse, float, dict]:
        # 1) 解析 parameters["locations"]（复刻 parse_structured_outputs 的输入形态）；
        # 2) agent_data.extra_fields["structured_locations"] = locations or None + failure_class
        # 3) agent_data.extra_fields["finish_call_count"] += 1
        # 4) 返回 ToolResponse(text="finished")（observation 文本并流，无下一轮）
        # 注意：不返回任何私有标签相关内容
```

**reward.py 关键接口草案：**

```python
def compute_score(data_source: str, solution_str: str, ground_truth: dict,
                  extra_info: dict | None = None, **kwargs) -> dict:
    """返回 {"score": float0-3, "file_f1":…, "module_f1":…, "entity_f1":…,
              "file_p/r":…, "failure_class": None|"no_finish"|"multi_finish"|"parse_error"|"sanity_check"|"exhausted"|"infra", ...}
    - 预测取 extra_info["structured_locations"]（不走 solution_str 重解析）；
    - 真值取 ground_truth（= parquet reward_model.ground_truth，私有标签）；
    - 调 src.rewards.file_localization.file_localization.multilevel_localization_f1_reward（1/1/1）；
    - infra 失败（extra_info["failure_class"]=="infra"）按冻结排除规则上报，不混作 0 分。
    """
```

**workspace.py / build_dataset.py**：接口从简——`prepare_snapshot(instance) -> workspace_root`（无 .git 快照，SWE-smith 应用 mutation patch）、`cleanup(workspace_root)`；build_dataset 产出 §6.2 schema 的 parquet（j2 渲染冻结到 protocol §3 指定版本），并写 split manifest。

### 11.2 阻塞性风险汇总（verl 机制 vs 原语义）

| # | 风险 | 等级 | 消解方式 |
|---|---|---|---|
| A-1 | 轮数耗尽整条 mask=0 无原生开关 | 高（协议 §6 冻结要求） | 自定义 agent loop `run()` 末尾置 0（§4 方案）；验收：mask 全 0 序列的 loss 贡献为 0 且不改其余序列梯度 |
| A-2 | finish 后多一次 LLM 调用（无显式 terminate 接口） | 高 | 自定义 `_handle_processing_tools_state` finish 即 TERMINATED（§2.3 方案）；验收：finish 后 assistant_turns 不再 +1、无新增生成 token |
| B-1 | 最后轮 `<|im_end|>` mask=1 多 1 token | 中 | 默认保留 + 披露；数值验收单独量化；如需逐 token 对齐在 `_finalize` 移除（§10 D3） |
| B-2 | 超过 max_parallel_calls 的调用被静默丢弃而非判罚 | 中 | reward 侧用 extra_fields 的 per-turn tool_calls 计数判罚（§3、§5.2） |
| B-3 | 无 per-call 8192 生成上限 | 中 | sampling_params 注入【待核验透传】；不行则在自定义 loop 的 generate 前改写 sampling_params |
| B-4 | reward fn 拿到的 solution_str 剥掉了 special tokens（skip_special_tokens=True，reward_loop/reward_manager/naive.py:54-56） | 低（已规避） | 结构化预测走 extra_fields 通道，不依赖文本重解析（§5.2） |
| B-5 | Qwen3 模板 tools 渲染与原 OpenHands→vLLM 路径的字段级等价、hermes 解析 + enable_thinking=False 兼容性 | 中 | 部署时 smoke：同一 messages+tools 下 token ids diff（skyrl-to-verl §6.5） |
| B-6 | `interaction_kwargs`/per-sample tool_selection 等新面（agent_loop.py:156-167）与本项目无冲突，但 `extra_info.index` 缺失会毁 GRPO 分组（§6.2） | 高（数据构建） | build_dataset 显式写 index；训练前断言每 index 恰 8 条 |

### 11.3 验收顺序建议（对应 protocol/todo 步骤）

1. **oracle 链路**：build_dataset → 1 条样本 rollout（temp=0）→ reward fn 输出与离线调 `multilevel_localization_f1_reward` 一致（逐级 F1 对账）。
2. **token 流审计**：dump prompt_ids/response_ids/response_mask，逐 token 对比 §1.4 表（含 Qwen 边界换行、generation prompt、finish 轮）；确认 assistant 生成 ids 未被 retokenize。
3. **mask 全 0 路径**：构造一条必然轮数耗尽的样本，核对 mask 全 0、reward 0、loss 贡献 0。
4. **梯度对齐**：固定小 batch（含全 mask 序列 + 含 finish 正常序列），verl 单 step 梯度 vs 手写 GSPO+GRPO(无std)+seq-mean-token-mean 参考实现对照（D1/D3 项）。
5. 之后才进入吞吐 profile 与正式训练日程（protocol §7）。

---

## 附：本文引用的 verl 关键文件索引（v0.9.1）

- `verl/experimental/agent_loop/tool_agent_loop.py`（ToolAgentLoop）
- `verl/experimental/agent_loop/agent_loop.py`（AgentLoopBase/Worker/Manager、ct_*、_compute_score、_postprocess）
- `verl/experimental/agent_loop/tool_parser.py`（HermesToolParser）
- `verl/experimental/reward_loop/reward_loop.py`、`verl/experimental/reward_loop/reward_manager/naive.py`（流式 reward）
- `verl/workers/reward_manager/naive.py`（colocate reward 备用路径）
- `verl/utils/tokenizer/continuous_token.py`、`continuous_token_wiring.py`、`chat_template.py`（CT/mask 核心）
- `verl/utils/dataset/rl_dataset.py`（RLHFDataset）
- `verl/tools/{base_tool,function_tool,schemas,tool_registry}.py`
- `verl/workers/config/{rollout,actor,optimizer}.py`、`verl/trainer/config/algorithm.py`、`verl/trainer/config/reward/reward.yaml`、`verl/trainer/config/data/legacy_data.yaml`
- `verl/trainer/ppo/core_algos.py`（GSPO/agg_loss/GRPO）、`verl/trainer/ppo/ray_trainer.py`、`verl/trainer/ppo/reward.py`、`verl/trainer/main_ppo.py`
- 示例：`examples/gspo_trainer/run_qwen3_8b_fsdp.sh`、`examples/tutorial/agent_loop_get_started/agent_loop_tutorial.ipynb`、`examples/data_preprocess/gsm8k_tool_agent_loop.py`、`verl/experimental/fully_async_policy/shell/dapo_7b_async_retool.sh`
