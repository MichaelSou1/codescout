"""CodeSearchAgentLoop：verl v0.9.1 ToolAgentLoop 的 CodeScout 子类。

协议/设计文档出处
------------------
- docs/reproduction/verl-adapter-design.md §1.4（mask 逐点对比）、§2.3（阻塞风险
  A-2：finish 即终止）、§4（阻塞风险 A-1：轮数耗尽整条 mask 置 0；B-3：per-call
  生成上限）、§5.1-5.2（结构化预测经 ``agent_data.extra_fields`` →
  ``tool_extra_fields`` 到 reward fn）、§10 D7（hermes 坏 JSON 丢弃语义）、
  §11.1（本类即 A-1/A-2/B-1/B-3 的消解点）。
- docs/reproduction/protocol-v1.md §3（6 turns、finish 恰好一次、并行 ≤5）、§4
  （sanity check 失败得 0）、§6（轮数耗尽 mask 置 0 按原语义保留）。

覆盖点（相对 verl 原生 ToolAgentLoop）
--------------------------------------
1. **A-2**：检测到 ``localization_finish`` 工具调用的轮，执行完该轮工具后直接
   ``AgentState.TERMINATED``，不再发起下一次 LLM 调用（原生 loop 执行完任意工具
   都回 GENERATING，会多一次生成且其 token mask=1 进 loss——不等价）。finish 轮
   的 tool 结果**不并入 token 流/prompt**（任务要求；原 OpenHands 语义是 finish
   置 ``ConversationExecutionStatus.FINISHED`` 后 episode 立即结束）。
2. **A-1**：达到 ``max_assistant_turns``（或 response_length 触顶）且无有效
   finish → 整条 ``response_mask`` 置 0（复刻 code_search_generator.py:324 判据
   + :464-468 置 0），并置 ``codesearch_trajectory_exhausted``；全 0 序列在
   ``seq-mean-token-mean`` 下 loss 贡献为 0、不改其余序列梯度（设计文档 §4）。
3. **sanity check**：最后 assistant 轮解码文本做原 ``sanity_check_last_step``
   三项检查（恰一对工具标记、恰一个 ``<|im_end|>``、结束标记后无正文），失败
   置 ``codesearch_failure_class="sanity_check"``（reward 0），mask 不清零
   （原实现仅轮数耗尽路径清 mask，:464-468）。
4. **B-3**：per-call 生成上限——把 ``max_tokens``（默认 8192，协议 §5"每次 LLM
   调用"）注入 sampling_params。verl 的 ``LLMServerClient.generate`` 对
   ``max_tokens`` 有原生支持（llm_server.py:254-273），每次调用按原值生效即
   per-call cap；整 episode 预算由 ``rollout.response_length``（累计 mask 判定，
   tool_agent_loop.py:290,394）另行兜底。
5. **B-2 审计**：逐轮记录解析出的 tool_calls 总数与实际执行数（verl 超过
   ``max_parallel_calls`` 的调用被静默丢弃，与原实现不同；末轮多调用由 sanity
   check 判 0，中间轮不判罚——原实现"Similar checks are not done for previous
   turns"，code_search_generator.py:327）。
6. 轨迹结束清理 TerminalTool 的每轨迹隔离 workspace 副本。

判定逻辑全部抽到 :mod:`src.verl_adapter.semantics`（纯 stdlib），本文件只做
verl 状态机接线——tests 在无 verl 环境下直接测纯函数。

注册：``@register("codesearch_agent")`` + ``src/verl_adapter/agent_loop.yaml``
（agent_loop_config_path；二者等价，见 agent_loop.py:533-539,678-693）。
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from . import semantics
from .semantics import FINISH_TOOL_NAME

try:
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

    _VERL_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover - 本机无 verl 环境
    _VERL_AVAILABLE = False
    _VERL_IMPORT_ERROR = _exc

    def register(name: str):  # type: ignore[misc]
        def decorator(cls):
            return cls

        return decorator

    ToolAgentLoop = object  # type: ignore[assignment,misc]
    AgentState = None  # type: ignore[assignment]
    AgentLoopOutput = None  # type: ignore[assignment]

__all__ = ["CodeSearchAgentLoop"]


@register("codesearch_agent")
class CodeSearchAgentLoop(ToolAgentLoop):
    """CodeScout issue→定位 agent loop（A-1/A-2/sanity/B-3 消解点）。

    Args（经 agent_loop.yaml 由 hydra instantiate 注入）：
        per_call_max_new_tokens: 每次 LLM 调用的生成上限（协议 §5：8192，
            **不是**整 episode 上限）。None 表示不注入（回退 llm_server 的累计
            response_length 预算，llm_server.py:264-273）。
        其余 ``*args/**kwargs`` 原样传给 ``ToolAgentLoop``（tools、tokenizer、
            server_manager 等，见 agent_loop.py:682-693）。
    """

    def __init__(self, *args: Any, per_call_max_new_tokens: Optional[int] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.per_call_max_new_tokens = per_call_max_new_tokens

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    async def run(  # type: ignore[override]
        self,
        sampling_params: dict[str, Any],
        priority: int = 0,
        **kwargs: Any,
    ) -> "AgentLoopOutput":
        """B-3 注入 per-call ``max_tokens`` → 父类状态机 → 终局判定与清理。"""
        if self.per_call_max_new_tokens:
            sampling_params = {
                **sampling_params,
                "max_tokens": int(self.per_call_max_new_tokens),
            }
        output: AgentLoopOutput = await super().run(sampling_params, priority=priority, **kwargs)
        self._finalize_extras(output)
        await self._release_trajectory_workspaces(output.extra_fields)
        return output

    # ------------------------------------------------------------------
    # 状态机覆盖
    # ------------------------------------------------------------------

    async def _handle_generating_state(  # type: ignore[override]
        self,
        agent_data: Any,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> "AgentState":
        """父类完成本轮生成后，记录 sanity check / metrics 所需的每轮信息。

        ``agent_data.response_ids`` 在父类里正是本轮生成的 token ids（:273 赋值，
        下一轮 generate 前被覆盖），``assistant_turns`` 已在 :272 自增——即使本轮
        因轮数/长度终止（终止判定先于 tool 解析，:289-295），记录仍然完整。
        解码用 ``skip_special_tokens=False``，与原 ``sanity_check_last_step``
        （code_search_generator.py:268）口径一致。
        """
        state: AgentState = await super()._handle_generating_state(
            agent_data, sampling_params, ignore_termination=ignore_termination
        )
        extra_fields: dict[str, Any] = agent_data.extra_fields
        try:
            turn_text: str = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False)
        except Exception as exc:  # pragma: no cover - 解码异常按 infra 记录
            turn_text = ""
            extra_fields["codesearch_decode_error"] = repr(exc)
        extra_fields["codesearch_last_turn_text"] = turn_text
        extra_fields["codesearch_assistant_turns"] = agent_data.assistant_turns
        extra_fields.setdefault("codesearch_request_id", agent_data.request_id)
        generated_counts: list[int] = extra_fields.setdefault("codesearch_generated_token_counts", [])
        generated_counts.append(len(agent_data.response_ids))
        return state

    async def _handle_processing_tools_state(self, agent_data: Any) -> "AgentState":
        """A-2：finish 轮特殊处理；其余轮记录并行计数后走父类逻辑。"""
        tool_calls = list(agent_data.tool_calls or [])
        extra_fields: dict[str, Any] = agent_data.extra_fields

        # B-2 审计：verl 只执行前 max_parallel_calls 个（:317 静默丢弃其余），
        # 与原实现"全部执行、末轮多调用 sanity 判 0"不同；逐轮留痕。
        turn_totals: list[int] = extra_fields.setdefault("codesearch_turn_total_tool_calls", [])
        turn_totals.append(len(tool_calls))
        turn_names: list[list[str]] = extra_fields.setdefault("codesearch_turn_tool_names", [])
        turn_names.append([tc.name for tc in tool_calls])
        extra_fields["codesearch_last_turn_executed_tool_calls"] = min(
            len(tool_calls), int(self.max_parallel_calls or 0) or len(tool_calls)
        )

        if any(tc.name == FINISH_TOOL_NAME for tc in tool_calls):
            # finish 轮：执行工具（保留 reward 通道写入），但**不做**
            # ct_merge_context_msg（finish 结果不并入 token 流/prompt）、不增
            # user_turns（无下一轮），直接 TERMINATED（A-2）。
            await self._execute_finish_turn_tools(agent_data)
            return AgentState.TERMINATED

        next_state: AgentState = await super()._handle_processing_tools_state(agent_data)
        # 防御：父类路径理论上不会出现 finish 工具（上面已分流）。
        if extra_fields.get("codesearch_finish_terminated"):
            return AgentState.TERMINATED
        return next_state

    # ------------------------------------------------------------------
    # 终局判定（sanity + A-1 mask 置 0 + metrics）
    # ------------------------------------------------------------------

    def _finalize_extras(self, output: "AgentLoopOutput") -> None:
        """对完成的轨迹复刻原 generator 的 reward 前判定（:322-330、:464-468）。

        顺序与原实现一致：exhausted 判据用 **pre-sanity** 的 structured 有效性
        （原 :324 先算 exhausted，:328 才做 sanity 置 None），mask 清零只发生在
        exhausted 路径。
        """
        extra_fields: dict[str, Any] = output.extra_fields
        finish_call_count = int(extra_fields.get("codesearch_finish_call_count", 0))
        structured: Any = extra_fields.get("codesearch_structured_locations")

        # 原语义：get_structured_locations 仅在恰好一次 finish 时返回非 None
        #（code_search_generator.py:87-90）。sanity 只在这一情形下检查（:328）。
        valid_pre_sanity = finish_call_count == 1 and structured is not None
        if valid_pre_sanity:
            last_turn_text: str = extra_fields.get("codesearch_last_turn_text") or ""
            if not semantics.sanity_check_last_step_text(last_turn_text):
                extra_fields["codesearch_structured_locations"] = None
                extra_fields["codesearch_failure_class"] = "sanity_check"
                valid_pre_sanity = False

        exhausted = semantics.is_trajectory_exhausted(
            finish_call_count=finish_call_count,
            has_valid_structured_locations=valid_pre_sanity,
            assistant_turns=int(extra_fields.get("codesearch_assistant_turns", 0)),
            max_assistant_turns=self.max_assistant_turns,
            response_token_count=len(output.response_mask),
            response_length_budget=self.response_length,
        )
        if exhausted:
            extra_fields["codesearch_trajectory_exhausted"] = True
            # A-1：整条 loss mask 置 0（code_search_generator.py:464-468 原样保留；
            # 全 0 序列在 seq-mean-token-mean 下贡献 0 梯度，设计文档 §4/§10 D1）。
            output.response_mask = [0] * len(output.response_mask)

        generated_counts: list[int] = extra_fields.get("codesearch_generated_token_counts", [])
        extra_fields["codesearch_metrics"] = {
            "assistant_turns": int(extra_fields.get("codesearch_assistant_turns", 0)),
            "user_turns_note": "num_turns 字段口径见 tool_agent_loop.py:203（D10）",
            "generated_tokens": int(sum(generated_counts)),
            "response_tokens": int(len(output.response_mask)),
            "finish_call_count": finish_call_count,
            "turn_total_tool_calls": list(
                extra_fields.get("codesearch_turn_total_tool_calls", [])
            ),
            "trajectory_exhausted": bool(exhausted),
        }

    async def _release_trajectory_workspaces(self, extra_fields: dict[str, Any]) -> None:
        """轨迹结束（含 finish/耗尽/无 tool call 终止）后清理 TerminalTool 隔离副本。

        已知限度：若 ``run()`` 中途抛异常（infra），无 output 可读、request_id
        丢失，本轨迹的隔离目录会残留于 ``<workspace_root>/.trajectories/``，
        由运维在无并行作业时清理（见 workspace 层与 run 脚本的说明）。
        """
        request_id = extra_fields.get("codesearch_request_id")
        if not request_id:
            return
        for tool in getattr(self, "tools", {}).values():
            release = getattr(tool, "release_trajectory", None)
            if release is not None:
                try:
                    await release(request_id)
                except Exception:  # pragma: no cover - 清理失败不阻断轨迹产出
                    pass

    # ------------------------------------------------------------------
    # finish 轮工具执行（不并入 token 流）
    # ------------------------------------------------------------------

    async def _execute_finish_turn_tools(self, agent_data: Any) -> None:
        """执行 finish 轮的前 ``max_parallel_calls`` 个工具调用并记录 reward 通道。

        与父类 ``_handle_processing_tools_state`` 的差别：
        - 不做 ``ct_merge_context_msg`` → 工具结果不进 token 流/prompt（A-2）；
        - 不自增 ``user_turns``、不判 response_length（没有后续生成）；
        - tool 消息仍 append 到 ``agent_data.messages``（仅日志用途）。
        """
        tool_calls = list(agent_data.tool_calls or [])
        cap = self.max_parallel_calls
        capped = tool_calls[:cap] if cap else tool_calls
        tasks = [self._call_tool(tc, agent_data.tools_kwargs, agent_data) for tc in capped]
        responses = await asyncio.gather(*tasks) if tasks else []
        for (tool_response, tool_reward, _metrics), tool_call in zip(responses, capped):
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)
            message: dict[str, Any] = {"role": "tool", "content": tool_response.text or ""}
            if tool_call.tool_call_id is not None:
                message["tool_call_id"] = tool_call.tool_call_id
            agent_data.messages.append(message)
