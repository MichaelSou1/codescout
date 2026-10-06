"""CodeScout 语义纯函数层：冻结协议判定逻辑与工具 schema 常量。

协议/设计文档出处
------------------
- docs/reproduction/protocol-v1.md §3（工具与轮数）、§4（奖励与 sanity check）、
  §5（采样与上下文）、§6（轮数耗尽 mask 置 0）。
- docs/reproduction/verl-adapter-design.md §2.3（A-2 finish 即终止）、§4（A-1
  轮数耗尽 mask 置 0）、§5.2（结构化预测走 extra_fields 通道）、§7.1（schema
  逐字迁移 TOOL_DESCRIPTION 与 CodeLocation 字段）。

本模块**只依赖标准库**（可在本机 Python 3.9/3.12 无 verl/pydantic/jinja2 环境
导入），原因：tests/test_verl_adapter.py 的核心判定/一致性测试必须不依赖 verl
（任务要求"reward/schema/prompts 部分必须不依赖 verl"）。verl/pydantic 相关的
类定义放在 tools.py / agent_loop.py，均从本模块取语义与常量。

原实现语义对齐（源码 commit 9d05a644）
--------------------------------------
- sanity_check_last_step：src/generator/code_search_generator.py:263-282；
- 轮数耗尽整条 mask 置 0：code_search_generator.py:324（判据）与 :464-468（置 0）；
- finish 恰好一次：src/tools/localization_finish.py + code_search_generator.py:87-90
  （cnt != 1 → None）；
- TOOL_DESCRIPTION / CodeLocation 字段 description：src/tools/localization_finish.py
  逐字迁移（不 import 该模块——它依赖 openhands 包，见 verl-adapter-design.md §7.1
  的"逐字迁移"要求；tests 用 ast 解析原文件源码做逐字节合同校验）。
"""

from __future__ import annotations

import re
from typing import Any, Optional

__all__ = [
    "CODESEARCH_DATA_SOURCE",
    "FINISH_TOOL_NAME",
    "TERMINAL_TOOL_NAME",
    "TOOL_CALL_START_MARKER",
    "TOOL_CALL_END_MARKER",
    "IM_END_TOKEN",
    "TOOL_DESCRIPTION",
    "LOCATIONS_FIELD_DESCRIPTION",
    "LOCALIZATION_FINISH_PARAMETERS_SCHEMA",
    "LOCALIZATION_FINISH_TOOL_SCHEMA",
    "TERMINAL_TOOL_SCHEMA",
    "sanity_check_last_step_text",
    "is_trajectory_exhausted",
    "normalize_locations",
    "resolve_prediction",
    "truncate_terminal_output",
    "FailureClass",
]

# ---------------------------------------------------------------------------
# 常量（协议 §3：仅 terminal 与 localization_finish 两个工具）
# ---------------------------------------------------------------------------

#: parquet data_source 常量（设计文档 §6.2；reward fn 选择键 + 分组统计）。
CODESEARCH_DATA_SOURCE = "codescout_swe_smith_py"

FINISH_TOOL_NAME = "localization_finish"
TERMINAL_TOOL_NAME = "terminal"

#: Hermes 工具调用起/止标记（verl/experimental/agent_loop/tool_parser.py:103-104，
#: 设计文档 §头注）。sanity check 按字符串计数，与原实现一致。
TOOL_CALL_START_MARKER = "<tool_call>"
TOOL_CALL_END_MARKER = "</tool_call>"
IM_END_TOKEN = "<|im_end|>"

#: FailureClass 合法值（reward fn 分级上报；协议 §4 有效模型失败/infra 分列）。
FailureClass = Optional[str]
_VALID_FAILURE_CLASSES = (
    None,
    "no_finish",
    "multi_finish",
    "parse_error",
    "sanity_check",
    "exhausted",
    "infra",
)

# ---------------------------------------------------------------------------
# localization_finish schema：逐字迁移 src/tools/localization_finish.py
# ---------------------------------------------------------------------------

#: TOOL_DESCRIPTION 逐字复制自 src/tools/localization_finish.py:100-121。
#: tests/test_verl_adapter.py 用 ast 解析原文件源码校验逐字节一致。
TOOL_DESCRIPTION = """Submit your final code localization results.

Use this tool when you have identified all relevant files, classes, and functions that need to be modified to address the issue described in the problem statement.

Provide a structured list of locations. Each location must have:
- file: Path to the file relative to the root of the repository (required)
- class_name: Class name (optional)
- function_name: Function/method name (optional)

You must submit a list of locations that require modification and for each location you must follow the below rules in your output:
1. If the required modifications belong to a specific function that belongs to a class, provide the file path, class name, and function name.
2. If the required modification belongs to a function that is not part of any class, provide the file path and function name.
3. If the required modification does not belong to any specific class or a function (e.g. global variables, imports, new class, new global function etc.), it is sufficient to provide only the file path.
4. If the required modification belongs to a class (e.g. adding a new method to a class, changing the class inheritance), provide the file path and class name. If you are modifying the __init__ method of a class, you should provide the function name as well.

IMPORTANT:
1. If multiple different edits need to be edited in the same file, you should create separate entries for each edit, specifying the same file path but different class/function names as applicable. Each entry should compulsorily include the file path.
2. Do NOT include duplicate entries in your output for which the file, class, and function names are all identical.
3. Ensure that the file paths are accurate and relative to the root of the repository without any leading "./" or "/". All locations must be valid and exist in the codebase and this applies to class and function names as well.
4. Aim for high precision (all returned locations are relevant) and high recall (no relevant locations missed).
5. The agent will terminate its execution after you call this tool.
"""

#: LocalizationFinishAction.locations 字段描述，逐字复制自
#: src/tools/localization_finish.py:37-43。
LOCATIONS_FIELD_DESCRIPTION = """List of code locations to modify. Each location in this list must have:
- file: Path to the file relative to the repository root (required)
- class_name: Class name (optional, omit for changes to imports, global variables, and global functions)
- function_name: Function/method name (optional, omit for changes that edit parts of a file outside of any particular function)
"""

#: CodeLocation / LocalizationFinishAction 的 JSON Schema，按 pydantic v2
#: ``model_json_schema()`` 的等价形态手写（纯 pydantic 重写模型的导出结果见
#: tools.py::export_localization_finish_pydantic_schema，服务器端有 pydantic 时
#: 逐字段对账）。原 OpenHands 路径由 action pydantic 模型导出后经 vLLM ``tools``
#: 参数进 Qwen chat template（设计文档 §7.1），因此嵌套 ``$defs``/``items`` 结构
#: 必须原样进入 prompt。**注意**：verl 的 ``OpenAIFunctionPropertySchema`` 只有
#: type/description/enum 三键且 extra=ignore（verl/tools/schemas.py:21-29），嵌套
#: items 会在 pydantic 校验时被静默丢弃——这正是 tools.py 用 duck-typed
#: envelope 绕过的原因（见该文件 docstring）。
LOCALIZATION_FINISH_PARAMETERS_SCHEMA: dict[str, Any] = {
    "$defs": {
        "CodeLocation": {
            "description": "A single code location with optional class and function.",
            "properties": {
                "file": {
                    "description": "Path to the file (required)",
                    "title": "File",
                    "type": "string",
                },
                "class_name": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "Class name (optional)",
                    "title": "Class Name",
                },
                "function_name": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "Function/method name (optional)",
                    "title": "Function Name",
                },
            },
            "required": ["file"],
            "title": "CodeLocation",
            "type": "object",
        }
    },
    "description": "Action for submitting final localization results.",
    "properties": {
        "locations": {
            "description": LOCATIONS_FIELD_DESCRIPTION,
            "items": {"$ref": "#/$defs/CodeLocation"},
            "title": "Locations",
            "type": "array",
        }
    },
    "required": ["locations"],
    "title": "LocalizationFinishAction",
    "type": "object",
}

#: localization_finish 的完整 OpenAI 工具 schema（进 prompt 的最终形态；
#: ``type/strict`` 外层与 verl 原生工具的 model_dump 对齐——``strict`` 为
#: OpenAIFunctionSchema 的 unset 默认，不出现在 exclude_unset 的 dump 中）。
LOCALIZATION_FINISH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": FINISH_TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": LOCALIZATION_FINISH_PARAMETERS_SCHEMA,
    },
}

#: terminal（bash）工具 schema。原 OpenHands ``TerminalTool`` 的描述文本不可在
#: 本机核验（依赖 openhands 包），此处描述按 system prompt 的 bash 语义撰写；
#: 与原描述的逐字段 diff 列入服务器 smoke 验收（设计文档 B-5 同类项）。
TERMINAL_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TERMINAL_TOOL_NAME,
        "description": (
            "Execute a bash command in the isolated code repository workspace to "
            "search and explore the codebase. Use commands like rg, grep, find, ls, "
            "cat, head, tail, sed, and wc. Each invocation runs in the repository "
            "root with a fresh shell; cd does not persist across invocations. "
            "Combine commands with && or ; when needed. The output may be truncated "
            "for very long results."
        ),
        "parameters": {
            "properties": {
                "command": {
                    "description": "The bash command to execute",
                    "title": "Command",
                    "type": "string",
                }
            },
            "required": ["command"],
            "title": "TerminalAction",
            "type": "object",
        },
    },
}


# ---------------------------------------------------------------------------
# sanity_check_last_step（原 code_search_generator.py:263-282 逐点复刻）
# ---------------------------------------------------------------------------


def sanity_check_last_step_text(last_response_str: str) -> bool:
    """复刻原 ``sanity_check_last_step`` 的三项检查（协议 §4）。

    输入是**最后一个 assistant 轮**的 ``skip_special_tokens=False`` 解码文本
    （verl 侧 stop token 计入生成 ids，设计文档 §1.4/B-1）。三项检查与原实现
    逐点对应（code_search_generator.py:269-281）：

    1. 恰好一对工具调用标记（多 tool call 一律 0）；
    2. 恰好一个 ``<|im_end|>``；
    3. 结束标记（``TOOL_CALL_END_MARKER``，与起始标记成对）之后、``<|im_end|>``
       之前无非空白正文（原 :279-281 split 语义）。

    空字符串与 ``None`` 视为失败（原实现 len==0 → False）。
    """
    if not last_response_str:
        return False
    cnt_tool_call = last_response_str.count(TOOL_CALL_START_MARKER)
    cnt_tool_end = last_response_str.count(TOOL_CALL_END_MARKER)
    if cnt_tool_call != 1 or cnt_tool_end != 1:
        return False
    if last_response_str.count(IM_END_TOKEN) != 1:
        return False
    portion = last_response_str.split(TOOL_CALL_END_MARKER)[1].split(IM_END_TOKEN)[0]
    if portion.strip() != "":
        return False
    return True


# ---------------------------------------------------------------------------
# A-1：轮数耗尽判定（原 code_search_generator.py:324 + :464-468）
# ---------------------------------------------------------------------------


def is_trajectory_exhausted(
    finish_call_count: int,
    has_valid_structured_locations: bool,
    assistant_turns: int,
    max_assistant_turns: Optional[int],
    response_token_count: int,
    response_length_budget: Optional[int],
) -> bool:
    """原判据：``structured_locations is None and len(token_messages) >= max_turns``。

    - ``structured_locations is None`` 的等价条件：finish 恰好一次且参数可解析
      才算有效（code_search_generator.py:87-90 cnt != 1 → None；本参数即
      "get_structured_locations 返回非 None" 的布尔化）。**注意**：sanity check
      失败在原实现里发生在 exhausted 判定**之后**（:324 先算、:328 后置 None），
      因此这里不看 sanity——传 has_valid_structured_locations（pre-sanity）。
    - ``len(token_messages) >= max_turns``：token_messages 即 LLM 调用数 ==
      verl ``assistant_turns``（每次 generate +1，tool_agent_loop.py:272）。
    - 追加 verl 侧条件（设计文档 §4）：response_length 预算触顶同样视为"没
      finish 就耗尽"（tool_agent_loop.py:290,394 的终止来源）。
    """
    structured_is_none = not (finish_call_count == 1 and has_valid_structured_locations)
    turn_exhausted = max_assistant_turns is not None and assistant_turns >= max_assistant_turns
    budget_exhausted = (
        response_length_budget is not None and response_token_count >= response_length_budget
    )
    return structured_is_none and (turn_exhausted or budget_exhausted)


# ---------------------------------------------------------------------------
# localization_finish 参数解析/归一（喂给冻结 scorer 的结构化预测）
# ---------------------------------------------------------------------------

_LOCATIONS_KEY_RE = re.compile(r"^locations$")


def normalize_locations(parameters: Any) -> tuple[bool, Optional[list[dict[str, Any]]]]:
    """把 finish 工具调用参数归一为 ``[{"file","class_name","function_name"}, ...]``。

    返回 ``(ok, locations)``：
    - ``parameters`` 不是 dict、无 ``locations`` 键、或其值不是 list → ``(False, None)``
      （格式错误，按有效模型失败 reward 0，协议 §3）；
    - 每个条目必须是 dict 且含字符串 ``file``；``class_name``/``function_name``
      缺省补 None（与 parse_structured_outputs 的 ``.get(..., None)`` 输入形态一致，
      module_rewards.py:155-158）。条目非法 → ``(False, None)``；
    - 空列表合法（原语义：有效 finish 但零定位 → scorer 走三级空预测，
      tests/test_reward_contract.py::test_empty_location_list_finish_zero）。

    注意：这里**不做** file 空白校验——原实现把空文件名的惩罚留给 scorer 的
    parse_structured_outputs（file_path None/空白 → 整体清空得 0），归一层不重复
    实现以免语义漂移。
    """
    if not isinstance(parameters, dict):
        return False, None
    if "locations" not in parameters:
        return False, None
    raw = parameters["locations"]
    if not isinstance(raw, list):
        return False, None
    locations: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            return False, None
        file_path = item.get("file")
        if not isinstance(file_path, str):
            return False, None
        class_name = item.get("class_name")
        function_name = item.get("function_name")
        if class_name is not None and not isinstance(class_name, str):
            return False, None
        if function_name is not None and not isinstance(function_name, str):
            return False, None
        locations.append(
            {
                "file": file_path,
                "class_name": class_name,
                "function_name": function_name,
            }
        )
    return True, locations


# ---------------------------------------------------------------------------
# 预测解析（agent loop 与 reward fn 共用的单一判定入口）
# ---------------------------------------------------------------------------


def resolve_prediction(extra_fields: dict[str, Any]) -> tuple[Optional[list[dict[str, Any]]], FailureClass]:
    """从轨迹 extra_fields 还原"原 get_structured_locations + sanity 判罚"结果。

    输入约定（agent_loop.py 写入，reward.py 读取；通道见设计文档 §5.1-5.2，
    naive reward manager 把 ``tool_extra_fields`` 并进 ``extra_info``）：

    - ``codesearch_finish_call_count``：finish 工具调用总次数（整条轨迹）；
    - ``codesearch_structured_locations``：唯一一次合法 finish 的结构化预测；
      无合法 finish 或解析失败时为 None/缺失；
    - ``codesearch_failure_class``：agent loop 预置的判罚类别（"sanity_check"）
      或 infra 标记；
    - ``codesearch_trajectory_exhausted``：A-1 标志。

    返回 ``(structured_locations, failure_class)``；``structured_locations`` 非
    None 且 ``failure_class`` 为 None 时才计分，否则按类别的原语义得 0。
    判定顺序复刻原实现：cnt != 1 → None（code_search_generator.py:87-90）；
    sanity 只在 cnt==1 且有 structured 时检查（:328）。
    """
    failure_class = extra_fields.get("codesearch_failure_class")
    if failure_class == "infra":
        return None, "infra"

    count = extra_fields.get("codesearch_finish_call_count", 0)
    if not isinstance(count, int) or count < 0:
        return None, "parse_error"

    if count == 0:
        if extra_fields.get("codesearch_trajectory_exhausted"):
            return None, "exhausted"
        return None, "no_finish"
    if count > 1:
        return None, "multi_finish"

    # count == 1：唯一一次 finish。sanity 失败由 agent loop 预置（原实现在
    # reward 前置 None，code_search_generator.py:328-330）。
    if failure_class == "sanity_check":
        return None, "sanity_check"
    if failure_class is not None:
        return None, failure_class
    locations = extra_fields.get("codesearch_structured_locations")
    if locations is None:
        return None, "parse_error"
    return locations, None


# ---------------------------------------------------------------------------
# terminal 输出截断（纯函数；verl 侧另有 max_tool_response_length 二次截断）
# ---------------------------------------------------------------------------


def truncate_terminal_output(
    text: str,
    max_output_bytes: int,
    head_keep_bytes: int = 1024,
) -> str:
    """按字节预算截断终端输出：保留头部 + 尾部，中段以标记替换。

    参数化要求见任务交付物 1（"超时与输出截断参数化"）。尾部优先（搜索命令的
    报错/文件尾更有信息量），头部保留少量上下文。``max_output_bytes <= 0`` 时
    不截断。verl 的 ``max_tool_response_length`` 截断（tool_agent_loop.py:499-506）
    作用在工具返回之后、由 loop 配置控制，两者叠加不冲突。
    """
    if max_output_bytes is None or max_output_bytes <= 0:
        return text
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_output_bytes:
        return text
    if head_keep_bytes >= max_output_bytes:
        head_keep_bytes = max_output_bytes // 4
    head = encoded[:head_keep_bytes].decode("utf-8", errors="replace")
    tail = encoded[-(max_output_bytes - head_keep_bytes):].decode("utf-8", errors="replace")
    marker = "\n...(truncated %d bytes)...\n" % (len(encoded) - max_output_bytes)
    return head + marker + tail
