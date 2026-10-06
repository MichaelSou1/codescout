"""verl v0.9.1 工具定义：terminal（bash）与 localization_finish。

协议/设计文档出处
------------------
- docs/reproduction/protocol-v1.md §3：仅 ``terminal`` 与 ``localization_finish``
  两个工具，schema 逐字复用 ``src/tools/localization_finish.py``；每轮并行 bash
  ≤5；``localization_finish`` 必须恰好一次，多调用/格式错误为有效模型失败。
- docs/reproduction/verl-adapter-design.md §2.1（BaseTool 生命周期：create →
  execute(instance_id, parameters, agent_data=…) → release；``tools_kwargs`` 从
  dataset 行 ``extra_info.tools_kwargs.<tool>.create_kwargs`` 注入）、§2.3（A-2
  finish 即终止）、§2.4（workspace 维度）、§5.2（结构化预测经
  ``agent_data.extra_fields`` 到 reward fn）、§7.1（schema 进 prompt 的形态为
  ``tool_schema.model_dump(exclude_unset=True, exclude_none=True)``）。

schema envelope（关键实现决策）
-------------------------------
verl 的 ``OpenAIFunctionPropertySchema`` 只有 ``type/description/enum`` 三键、
无 ``extra`` 配置（verl/tools/schemas.py:21-29）——嵌套 array-of-object 的
``items``/``$defs`` 会在 pydantic 校验时被**静默丢弃**。而原 OpenHands 路径由
action pydantic 模型导出的**完整嵌套 schema** 经 vLLM ``tools`` 参数进 Qwen chat
template（设计文档 §7.1），为满足协议 §3"schema 逐字复用"，本模块用 duck-typed
:class:`ToolSchemaEnvelope` 直接携带完整 schema dict。verl v0.9.1 对
``tool_schema`` 的全部消费点已逐点核对，均为下述两个接口，无其他用法：

- ``tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)``
  （tool_agent_loop.py:119、rl_dataset.py:137）；
- ``tool.tool_schema.function.name``（base_tool.py:38）。

Hermes parser 收到的 ``tools`` 参数被忽略（tool_parser.py:116-124 只用正则），
不受影响。

terminal 的 workspace 与并行 rollout 隔离
----------------------------------------
verl ``rollout.n=8`` 会把同一 dataset 行复制 8 份并发 rollout，它们共享同一
``extra_info.tools_kwargs``（同一个离线快照路径）。原实现每 episode 独立
``/tmp/testbed/<uuid>``（code_search_generator.py:129-132），为复刻该隔离语义，
``TerminalTool`` 默认按轨迹（``agent_data.request_id``，每条轨迹唯一）把快照
复制到隔离目录后执行 bash；首个 terminal 调用时创建、同轨迹复用，轨迹结束由
``CodeSearchAgentLoop`` 调 ``release_trajectory`` 清理（tools.py 自身无法感知
轨迹结束——BaseTool.release 只拿到 per-call 的 uuid，见 tool_agent_loop.py:493-496）。
复制与 bash 均在 ``asyncio.to_thread`` 中执行，不阻塞事件循环。关闭隔离
（``isolate_per_trajectory=False``）时 8 条轨迹共享同一快照目录，快但有写污染
风险，仅用于 smoke。

私有标签边界（协议 §8）：本模块不读取/不返回任何 file_changes/patch/target；
``create_kwargs`` 只含 workspace 路径与执行参数。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from . import semantics
from .semantics import (
    FINISH_TOOL_NAME,
    TERMINAL_TOOL_NAME,
    normalize_locations,
    truncate_terminal_output,
)

try:  # verl 仅在服务器运行时存在；本机测试环境用等价 fallback（仅接口形态，
    # 不进入 verl 注册/rollout——tests 直接实例化覆盖判定逻辑）。
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import ToolResponse

    _VERL_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover - 本机无 verl 环境
    _VERL_AVAILABLE = False
    _VERL_IMPORT_ERROR = _exc

    class _BaseToolFallback:  # type: ignore[no-redef]
        """最小 BaseTool 形态（verl/tools/base_tool.py:36-38 的构造语义）。"""

        def __init__(self, config: Optional[dict] = None, tool_schema: Any = None, **kwargs: Any):
            self.config = config or {}
            self.tool_schema = tool_schema if tool_schema is not None else self.get_openai_tool_schema()
            assert self.tool_schema is not None, "Tool schema is not set!"
            self.name = self.tool_schema.function.name

        def get_openai_tool_schema(self) -> Any:
            return self.tool_schema

    class _ToolResponseFallback:  # type: ignore[no-redef]
        """最小 ToolResponse 形态（verl/tools/schemas.py:98-127）。"""

        def __init__(self, text: Optional[str] = None, image: Any = None, video: Any = None):
            self.text = text
            self.image = image
            self.video = video

    BaseTool = _BaseToolFallback  # type: ignore[assignment,misc]
    ToolResponse = _ToolResponseFallback  # type: ignore[assignment]

__all__ = [
    "TerminalTool",
    "LocalizationFinishTool",
    "ToolSchemaEnvelope",
]

DEFAULT_TERMINAL_TIMEOUT_S = 120
DEFAULT_TERMINAL_MAX_OUTPUT_BYTES = 131072
_TRAJECTORY_DIR_NAME = ".trajectories"


class ToolSchemaEnvelope:
    """Duck-typed 替代 ``OpenAIFunctionToolSchema``，保留完整嵌套 schema。

    详见模块 docstring"schema envelope"一节。``model_dump`` 的
    ``exclude_unset/exclude_none`` 参数为兼容 verl 调用形态（结果为深拷贝，
    防止调用方改写常量）。
    """

    def __init__(self, schema_dict: dict[str, Any]):
        self._schema = schema_dict
        self.function = _FunctionRef(schema_dict["function"]["name"])

    @property
    def name(self) -> str:
        return self.function.name

    def model_dump(self, exclude_unset: bool = True, exclude_none: bool = True, **kwargs: Any) -> dict[str, Any]:
        return json.loads(json.dumps(self._schema, ensure_ascii=False))

    def __repr__(self) -> str:  # pragma: no cover - 调试便利
        return f"ToolSchemaEnvelope(name={self.function.name!r})"


class _FunctionRef:
    def __init__(self, name: str):
        self.name = name


def _resolve_create_kwargs(agent_data: Any, tool_name: str) -> dict[str, Any]:
    """从 ``agent_data.tools_kwargs[tool].create_kwargs`` 取 per-instance 参数。

    通道：dataset 行 ``extra_info.tools_kwargs``（设计文档 §2.1/§6.2）→
    ``AgentData.tools_kwargs``（tool_agent_loop.py:143,153）→ ``_call_tool``
    （tool_agent_loop.py:485）。工具的 ``create`` 也收到同一 create_kwargs，
    但 ``execute`` 只有经 ``agent_data`` 才能读到（调用点 :486-489）。
    """
    tools_kwargs = getattr(agent_data, "tools_kwargs", None) or {}
    entry = tools_kwargs.get(tool_name, {}) if isinstance(tools_kwargs, dict) else {}
    create_kwargs = entry.get("create_kwargs", {}) if isinstance(entry, dict) else {}
    return create_kwargs if isinstance(create_kwargs, dict) else {}


class TerminalTool(BaseTool):
    """在隔离 workspace 内执行真实 bash 命令的终端工具。

    语义对齐原 OpenHands ``TerminalTool``（code_search_generator.py:129-132 每
    episode 独立 workspace；每次调用独立 shell）。命令失败（非零退出码）把退出
    码与输出作为工具文本返回，**不抛异常**（异常会被 ``_call_tool`` 吞成通用
    错误文本，丢失诊断信息，且原 OpenHands 语义也是把失败输出回给模型）。
    """

    def __init__(self, config: Optional[dict] = None, tool_schema: Any = None, **kwargs: Any):
        # registry 以 config=..., tool_schema=None 实例化（tool_registry.py:65-75）。
        super().__init__(config=config or {}, tool_schema=tool_schema or self.get_openai_tool_schema())
        # 每轨迹隔离目录注册表：request_id -> Path。工具实例每 worker 一个
        #（agent_loop.py:526-531 只 load 一次），单事件循环内访问无需锁。
        self._trajectory_dirs: dict[str, Path] = {}

    def get_openai_tool_schema(self) -> ToolSchemaEnvelope:
        return ToolSchemaEnvelope(semantics.TERMINAL_TOOL_SCHEMA)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def create(self, instance_id: Optional[str] = None, **kwargs: Any) -> tuple[str, "ToolResponse"]:
        """接收 instance 的 workspace 配置并返回 per-call 实例 id。

        ``create`` 在**每次**工具调用时被调（tool_agent_loop.py:486），因此这里
        只做参数存在性校验与 uuid 分配；workspace 解析/隔离复制延后到
        ``execute``（那里才有 ``agent_data``，即轨迹上下文）。
        """
        create_kwargs = kwargs.get("create_kwargs", {})
        if not isinstance(create_kwargs, dict) or not (
            "workspace" in create_kwargs or "workspace_root" in create_kwargs
        ):
            # 缺 workspace 配置属于数据构建错误（infra），抛出让 _call_tool 记为
            # 工具错误文本并进入 reward 侧 infra 分列（协议 §4）。
            raise ValueError(
                "TerminalTool.create_kwargs must contain 'workspace' or ('workspace_root' + 'episode_id'); "
                f"got: {sorted(create_kwargs.keys()) if isinstance(create_kwargs, dict) else type(create_kwargs)}"
            )
        if instance_id is None:
            return str(uuid.uuid4()), ToolResponse()
        return instance_id, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple["ToolResponse", float, dict]:
        """执行 ``parameters["command"]`` 于（可选隔离的）workspace，返回
        ``(ToolResponse, step_reward=0.0, metrics)``。step reward 恒 0（协议 §4：
        outcome-only，唯一奖励是 finish 后的三级 F1）。
        """
        agent_data = kwargs.get("agent_data")
        create_kwargs = _resolve_create_kwargs(agent_data, TERMINAL_TOOL_NAME)
        cwd = await self._resolve_working_dir(agent_data, create_kwargs)

        command = parameters.get("command") if isinstance(parameters, dict) else None
        if not isinstance(command, str) or not command.strip():
            return (
                ToolResponse(text="terminal: missing required string parameter 'command'"),
                0.0,
                {"terminal_error": "missing_command"},
            )

        timeout_s = float(create_kwargs.get("timeout_s", DEFAULT_TERMINAL_TIMEOUT_S))
        max_output_bytes = int(
            create_kwargs.get("max_output_bytes", DEFAULT_TERMINAL_MAX_OUTPUT_BYTES)
        )

        metrics: dict[str, Any] = {}
        stdout_text = await self._run_bash(command, cwd, timeout_s, metrics)
        text = truncate_terminal_output(stdout_text, max_output_bytes)
        return ToolResponse(text=text), 0.0, metrics

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        """per-call 实例清理（无状态，no-op）。轨迹级清理见 :meth:`release_trajectory`。"""
        return None

    async def release_trajectory(self, request_id: str) -> None:
        """删除该轨迹的隔离 workspace 副本（由 agent loop 在轨迹结束时调用）。"""
        path = self._trajectory_dirs.pop(request_id, None)
        if path is None:
            return
        await asyncio.to_thread(shutil.rmtree, path, True)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    async def _resolve_working_dir(self, agent_data: Any, create_kwargs: dict[str, Any]) -> Path:
        """解析本轨迹的 bash 工作目录：快照直用或按轨迹隔离复制。"""
        snapshot = self._resolve_snapshot(create_kwargs)
        isolate = bool(create_kwargs.get("isolate_per_trajectory", True))
        if not isolate:
            return snapshot
        request_id = getattr(agent_data, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            return snapshot  # 拿不到轨迹 id（异常路径）时退化为共享快照
        cached = self._trajectory_dirs.get(request_id)
        if cached is not None and cached.is_dir():
            return cached
        traj_root = snapshot.parent / _TRAJECTORY_DIR_NAME
        traj_dir = traj_root / request_id
        if not traj_dir.is_dir():
            traj_root.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copytree, snapshot, traj_dir, symlinks=True)
        self._trajectory_dirs[request_id] = traj_dir
        return traj_dir

    @staticmethod
    def _resolve_snapshot(create_kwargs: dict[str, Any]) -> Path:
        """离线准备的代码快照路径（build_prompts.py 写入 create_kwargs）。"""
        if "workspace" in create_kwargs:
            snapshot = Path(str(create_kwargs["workspace"]))
        elif "workspace_root" in create_kwargs:
            episode_id = str(create_kwargs.get("episode_id", ""))
            if not episode_id:
                raise ValueError("TerminalTool.create_kwargs has 'workspace_root' but no 'episode_id'")
            snapshot = Path(str(create_kwargs["workspace_root"])) / episode_id
        else:  # create() 已校验；防御性兜底
            raise ValueError("TerminalTool.create_kwargs missing workspace path")
        if not snapshot.is_dir():
            raise FileNotFoundError(f"workspace snapshot not found: {snapshot}")
        return snapshot

    @staticmethod
    async def _run_bash(command: str, cwd: Path, timeout_s: float, metrics: dict[str, Any]) -> str:
        """``bash -c <command>`` 于 cwd；合并 stdout/stderr，失败/超时返回文本。"""

        def _run() -> tuple[bool, int, str, str]:
            import subprocess

            try:
                proc = subprocess.run(
                    ["bash", "-c", command],
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=timeout_s,
                    check=False,
                )
                return False, proc.returncode, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as exc:
                out = exc.stdout or ""
                err = exc.stderr or ""
                if isinstance(out, bytes):
                    out = out.decode("utf-8", errors="replace")
                if isinstance(err, bytes):
                    err = err.decode("utf-8", errors="replace")
                return True, -1, out, err

        timed_out, returncode, stdout, stderr = await asyncio.to_thread(_run)
        combined = ""
        if stdout:
            combined += stdout
        if stderr:
            combined += ("\n" if combined else "") + stderr
        if timed_out:
            combined += (
                ("\n" if combined else "")
                + f"COMMAND TIMED OUT after {timeout_s}s and was terminated"
            )
            metrics["terminal_timeout"] = True
        elif returncode != 0:
            combined += (
                ("\n" if combined else "")
                + f"COMMAND FAILED with exit code {returncode}"
            )
        metrics["terminal_returncode"] = returncode
        return combined


class LocalizationFinishTool(BaseTool):
    """提交定位结果的 finish 工具（A-2 的执行侧）。

    语义对齐 ``src/tools/localization_finish.py``：

    - 参数 schema 逐字复用（TOOL_DESCRIPTION / CodeLocation 字段 description，
      见 semantics.py 常量与 tests 的逐字节合同）；
    - 合法调用把结构化 ``locations`` 写入 ``agent_data.extra_fields``
      （"codesearch_structured_locations"，reward fn 经 tool_extra_fields 读取，
      设计文档 §5.1-5.2），并置 ``codesearch_finish_terminated=True`` 供 agent
      loop 立即终止（不再发起下一次 LLM 调用）；
    - 多调用按原语义 ``cnt != 1 → None``（code_search_generator.py:87-90）：
      第 2 次调用起把结构化预测清空；
    - 格式错误（locations 非 list/条目缺 file 等）置 ``codesearch_failure_class=
      "parse_error"``、不置终止标志（原 OpenHands 在 action 校验失败时回错误
      observation 并继续对话），reward 侧得 0（协议 §3 有效模型失败）。

    返回文本对齐原 executor 的 observation（json.dumps(loc_dict, indent=2)，
    localization_finish.py:93-94）；verl 侧该文本**不并入 prompt**（任务 A-2 要求，
    agent loop 对 finish 轮跳过 token 合并）。绝不返回任何私有标签内容。
    """

    def __init__(self, config: Optional[dict] = None, tool_schema: Any = None, **kwargs: Any):
        super().__init__(config=config or {}, tool_schema=tool_schema or self.get_openai_tool_schema())

    def get_openai_tool_schema(self) -> ToolSchemaEnvelope:
        return ToolSchemaEnvelope(semantics.LOCALIZATION_FINISH_TOOL_SCHEMA)

    async def create(self, instance_id: Optional[str] = None, **kwargs: Any) -> tuple[str, "ToolResponse"]:
        """finish 工具无 per-instance 状态。create_kwargs 接受并忽略（parquet 嵌套
        列要求非空 struct，见 build_prompts；这是 verl 侧管道参数，非模型可见的
        tool schema——原 OpenHands SDK create 的参数限制
        （localization_finish.py:144-145）不约束本接口）。"""
        return instance_id or str(uuid.uuid4()), ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple["ToolResponse", float, dict]:
        agent_data = kwargs.get("agent_data")
        extra_fields = getattr(agent_data, "extra_fields", None)
        if extra_fields is None:
            # 无轨迹上下文（非 agent-loop 调用）：只回 observation，不计分通道。
            ok, locations = normalize_locations(parameters)
            text = json.dumps(locations, indent=2) if ok else "localization_finish: invalid locations"
            return ToolResponse(text=text), 0.0, {}

        count = int(extra_fields.get("codesearch_finish_call_count", 0)) + 1
        extra_fields["codesearch_finish_call_count"] = count
        ok, locations = normalize_locations(parameters)

        if count == 1:
            if ok:
                extra_fields["codesearch_structured_locations"] = locations
                extra_fields["codesearch_finish_terminated"] = True
                text = json.dumps(locations, indent=2)
            else:
                # 唯一一次 finish 但格式错误：置 parse_error，不终止（对齐原
                # FunctionCallValidationError → 错误 observation → 继续对话）。
                extra_fields.setdefault("codesearch_failure_class", "parse_error")
                text = (
                    "localization_finish: invalid arguments — 'locations' must be a "
                    "list of objects each with a string 'file' and optional string "
                    "'class_name'/'function_name'"
                )
        else:
            # 多 finish：cnt != 1 → 结构化预测作废（原 get_structured_locations 语义）。
            extra_fields["codesearch_structured_locations"] = None
            if ok:
                # 本次调用本身合法（合法的并行多 finish 在原实现同样被 sanity
                # check 判 0）；终止标志已在第一次合法调用时置位。
                extra_fields.setdefault("codesearch_finish_terminated", True)
            text = "localization_finish: finish tool called more than once; submission is void"

        return ToolResponse(text=text), 0.0, {"codesearch_finish_call_count": count}

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        """step reward 恒 0：协议 §4 outcome-only（唯一奖励为三级 F1 之和）。"""
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        return None


def _check_verl_available() -> None:
    if not _VERL_AVAILABLE:
        raise ImportError(
            "verl is not importable in this environment; the tools in "
            "src/verl_adapter/tools.py require the verl v0.9.1 runtime"
        ) from _VERL_IMPORT_ERROR
