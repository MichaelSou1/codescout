# -*- coding: utf-8 -*-
"""verl_adapter 单元测试（分层 skip：核心层不依赖 verl/GPU/pydantic/jinja2）。

分层结构（任务要求：reward/schema/prompts 与纯判定逻辑必须不依赖 verl；
import verl/pydantic/pandas/jinja2 失败则对应层 skip）：

  1. 语义纯函数（semantics.py）：sanity check / 轮数耗尽 / locations 归一 /
     预测判罚 / 截断 —— 纯标准库，任何环境必跑。
  2. schema 一致性：semantics 的 localization_finish schema 与
     src/tools/localization_finish.py **源码文本**（ast 解析，不 import——该模块
     依赖 openhands）逐字节合同；pydantic 可用时再与纯 pydantic 模型导出 schema
     逐字段对账。
  3. tools.py：verl 缺失时用最小 stub（BaseTool/ToolResponse）实例化两个工具，
     覆盖 finish 计数/多 finish/parse_error 与 terminal 的 bash 执行/超时/截断/
     每轨迹隔离。
  4. reward.py：≥50 个合成边界 case 上 adapter 与冻结 scorer 逐项差 ≤1e-8；
     借用 tests/test_reward_contract.py 的 10+ 真实语义 case；失败类别分列。
     （需要 Python ≥ 3.10：冻结 scorer 用 PEP 604 注解。）
  5. build_prompts.py：模板渲染与手工期望一致（含 working_dir 替换）；训练行
     schema 符合设计文档 §6.2；CLI 端到端（pandas 可用时）。

运行方式（本机无 pytest，自带 __main__ 运行器；同一批 test_* 函数）：

    ~/.local/share/uv/python/cpython-3.12.14-macos-aarch64-none/bin/python3 \\
        tests/test_verl_adapter.py

冻结协议引用：docs/reproduction/protocol-v1.md §3/§4/§5；
设计文档：docs/reproduction/verl-adapter-design.md（§5/§6.2/§11）。
"""

import ast
import asyncio
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

TOL = 1e-8  # 任务要求：adapter 与冻结 scorer 逐项差 ≤ 1e-8

_TEMPLATE_DIR = Path(_REPO_ROOT) / "src" / "prompts" / "templates"
_ORIG_FINISH = Path(_REPO_ROOT) / "src" / "tools" / "localization_finish.py"


class SkipTest(Exception):
    """分层 skip：对应依赖在本环境不可用。"""


def _skip(msg: str):
    """pytest 下走 pytest.skip（unittest 式 SkipTest 在普通 pytest 函数中会被记为失败）；
    __main__ 直跑时抛本模块 SkipTest 由运行器捕获。"""
    if "pytest" in sys.modules:
        import pytest

        pytest.skip(msg)
    raise SkipTest(msg)


def _import_scorer():
    """冻结 scorer 需要 PEP 604（Python ≥ 3.10），不可用时跳过 reward 层。"""
    try:
        from src.rewards.file_localization.file_localization import (
            multilevel_localization_f1_reward,
        )
        from src.rewards.file_localization.module_rewards import parse_structured_outputs
    except TypeError as exc:
        _skip(
            "frozen scorer needs Python >= 3.10 (PEP 604 annotations): %r" % exc
        )
    return multilevel_localization_f1_reward, parse_structured_outputs


# ---------------------------------------------------------------------------
# 原实现源码合同（ast 解析 src/tools/localization_finish.py，不 import）
# ---------------------------------------------------------------------------


def _original_finish_source_contracts():
    """从原文件源码提取 TOOL_DESCRIPTION、字段 description 与类 docstring。"""
    tree = ast.parse(_ORIG_FINISH.read_text(encoding="utf-8"))
    out: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TOOL_DESCRIPTION":
                    if isinstance(node.value, ast.Constant):
                        out["TOOL_DESCRIPTION"] = node.value.value
        if isinstance(node, ast.ClassDef) and node.name == "CodeLocation":
            out["CodeLocation_doc"] = ast.get_docstring(node)
            fields = {}
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.value, ast.Call)
                    and getattr(stmt.value.func, "id", "") == "Field"
                ):
                    for kw in stmt.value.keywords:
                        if kw.arg == "description" and isinstance(kw.value, ast.Constant):
                            fields[stmt.target.id] = kw.value.value  # type: ignore[attr-defined]
            out["CodeLocation_fields"] = fields
        if isinstance(node, ast.ClassDef) and node.name == "LocalizationFinishAction":
            out["Action_doc"] = ast.get_docstring(node)
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.value, ast.Call)
                    and getattr(stmt.value.func, "id", "") == "Field"
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == "locations"
                ):
                    for kw in stmt.value.keywords:
                        if kw.arg == "description" and isinstance(kw.value, ast.Constant):
                            out["locations_description"] = kw.value.value
    return out


def test_schema_matches_original_source_bytes():
    """semantics 的 schema 文本与原文件逐字节一致（协议 §3"schema 逐字复用"）。"""
    from src.verl_adapter import semantics

    contracts = _original_finish_source_contracts()
    assert semantics.TOOL_DESCRIPTION == contracts["TOOL_DESCRIPTION"], "TOOL_DESCRIPTION 漂移"
    assert semantics.LOCATIONS_FIELD_DESCRIPTION == contracts["locations_description"], (
        "locations 字段 description 漂移"
    )
    params = semantics.LOCALIZATION_FINISH_PARAMETERS_SCHEMA
    code_loc = params["$defs"]["CodeLocation"]
    assert code_loc["description"] == contracts["CodeLocation_doc"]
    fields = contracts["CodeLocation_fields"]
    assert code_loc["properties"]["file"]["description"] == fields["file"]
    assert code_loc["properties"]["class_name"]["description"] == fields["class_name"]
    assert code_loc["properties"]["function_name"]["description"] == fields["function_name"]
    assert params["description"] == contracts["Action_doc"]


def test_schema_structure_contract():
    """schema 结构（手写 JSON dict 断言；设计文档 §7.1 逐字段迁移要求）。"""
    from src.verl_adapter import semantics

    schema = semantics.LOCALIZATION_FINISH_TOOL_SCHEMA
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "localization_finish"
    assert isinstance(fn["description"], str) and fn["description"].startswith(
        "Submit your final code localization results."
    )
    params = fn["parameters"]
    assert params["type"] == "object"
    assert set(params["properties"].keys()) == {"locations"}
    assert params["required"] == ["locations"]
    loc = params["properties"]["locations"]
    assert loc["type"] == "array"
    assert loc["items"] == {"$ref": "#/$defs/CodeLocation"}
    code = params["$defs"]["CodeLocation"]
    assert code["required"] == ["file"]
    assert set(code["properties"].keys()) == {"file", "class_name", "function_name"}
    assert code["properties"]["file"]["type"] == "string"
    assert code["properties"]["class_name"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert code["properties"]["class_name"]["default"] is None
    # terminal schema 基本形态
    t = semantics.TERMINAL_TOOL_SCHEMA
    assert t["function"]["name"] == "terminal"
    assert t["function"]["parameters"]["required"] == ["command"]
    assert t["function"]["parameters"]["properties"]["command"]["type"] == "string"


def test_schema_pydantic_equivalent():
    """pydantic 可用时：纯 pydantic 模型导出 schema 与 semantics 常量逐字段一致
    （任务交付物 1："JSON schema 从 pydantic 模型精确导出"的服务器侧对账）。"""
    try:
        from pydantic import BaseModel, Field
    except ImportError as exc:
        _skip("pydantic not available locally (server-side check): %r" % exc)
    from src.verl_adapter import semantics

    class CodeLocation(BaseModel):
        """A single code location with optional class and function."""

        file: str = Field(description="Path to the file (required)")
        class_name: str | None = Field(default=None, description="Class name (optional)")
        function_name: str | None = Field(default=None, description="Function/method name (optional)")

    class LocalizationFinishAction(BaseModel):
        """Action for submitting final localization results."""

        locations: list[CodeLocation] = Field(description=semantics.LOCATIONS_FIELD_DESCRIPTION)

    exported = LocalizationFinishAction.model_json_schema()
    expected = semantics.LOCALIZATION_FINISH_PARAMETERS_SCHEMA
    assert exported == expected, (
        "pydantic export differs from semantics schema:\n%r\n!=\n%r" % (exported, expected)
    )


def test_schema_matches_openhands_original_models():
    """openhands 可用时（服务器）：与原 pydantic 模型导出 schema 的受控字段对账。

    只对账我们冻结迁移的部分（locations 字段与 CodeLocation $defs）；原
    ``LocalizationFinishAction(Action)`` 的基类额外字段（若有）属于待 smoke 项
    （设计文档 B-5），不在此断言。
    """
    try:
        from src.tools.localization_finish import (  # noqa: F401
            CodeLocation,
            LocalizationFinishAction,
        )
    except Exception as exc:
        _skip("openhands runtime not available locally: %r" % exc)
    from src.verl_adapter import semantics

    orig_loc = LocalizationFinishAction.model_json_schema()
    ours = semantics.LOCALIZATION_FINISH_PARAMETERS_SCHEMA
    assert orig_loc["properties"]["locations"] == ours["properties"]["locations"]
    assert orig_loc["$defs"]["CodeLocation"] == ours["$defs"]["CodeLocation"]


# ---------------------------------------------------------------------------
# semantics 纯函数（sanity / exhaustion / normalize / resolve / truncate）
# ---------------------------------------------------------------------------


def test_sanity_check_last_step_text():
    from src.verl_adapter import semantics as S

    START = S.TOOL_CALL_START_MARKER
    END = S.TOOL_CALL_END_MARKER
    IM = "<|im_end|>"

    def call(name="localization_finish", arguments='{"locations": []}'):
        return START + json.dumps({"name": name, "arguments": json.loads(arguments)}) + END

    # 原实现三项检查的逐点复刻（code_search_generator.py:269-281）
    assert S.sanity_check_last_step_text(call() + IM) is True
    assert S.sanity_check_last_step_text("") is False
    assert S.sanity_check_last_step_text("plain text without tool call") is False
    # 检查 1：恰一对工具调用标记（多调用/多标记一律 False）
    assert S.sanity_check_last_step_text(call() + call("terminal", '{"command": "ls"}') + IM) is False
    assert S.sanity_check_last_step_text(call() + START + IM) is False
    assert S.sanity_check_last_step_text(call() + IM + END + IM) is False
    # 检查 2：恰一个 <|im_end|>
    assert S.sanity_check_last_step_text(call() + IM + IM) is False
    assert S.sanity_check_last_step_text(call()) is False
    # 检查 3：结束后无非空白正文
    assert S.sanity_check_last_step_text(call() + "leftover text" + IM) is False
    assert S.sanity_check_last_step_text(call() + "   \n  " + IM) is True  # 纯空白合法
    # 工具调用前的正文不影响（原实现只查标记计数与后置正文）
    assert S.sanity_check_last_step_text("thinking aloud... " + call() + IM) is True


def test_is_trajectory_exhausted():
    from src.verl_adapter import semantics as S

    # 原 code_search_generator.py:324 判据：structured None 且 LLM 调用数达上限
    assert S.is_trajectory_exhausted(0, False, 6, 6, 100, 32768) is True  # 无 finish + 轮数耗尽
    assert S.is_trajectory_exhausted(0, False, 5, 6, 100, 32768) is False
    assert S.is_trajectory_exhausted(1, True, 6, 6, 100, 32768) is False  # 末轮合法 finish
    assert S.is_trajectory_exhausted(2, False, 6, 6, 100, 32768) is True  # 多 finish + 轮数耗尽
    assert S.is_trajectory_exhausted(2, False, 3, 6, 100, 32768) is False  # 多 finish 早停：mask 保留（原语义）
    assert S.is_trajectory_exhausted(1, False, 6, 6, 100, 32768) is True  # 唯一 finish 但解析失败于末轮
    assert S.is_trajectory_exhausted(1, False, 2, 6, 100, 32768) is False
    # response_length 预算触顶（设计文档 §4 追加的 verl 侧条件）
    assert S.is_trajectory_exhausted(0, False, 2, 6, 32768, 32768) is True
    assert S.is_trajectory_exhausted(1, True, 2, 6, 32768, 32768) is False
    # max_assistant_turns=None（未配置）：只看预算
    assert S.is_trajectory_exhausted(0, False, 99, None, 100, 32768) is False
    assert S.is_trajectory_exhausted(0, False, 99, None, 32768, 32768) is True


def test_normalize_locations():
    from src.verl_adapter import semantics as S

    ok, locs = S.normalize_locations(
        {"locations": [{"file": "a.py", "class_name": "C", "function_name": "f"}]}
    )
    assert ok and locs == [{"file": "a.py", "class_name": "C", "function_name": "f"}]
    ok, locs = S.normalize_locations({"locations": [{"file": "a.py"}]})
    assert ok and locs == [{"file": "a.py", "class_name": None, "function_name": None}]
    ok, locs = S.normalize_locations({"locations": []})
    assert ok and locs == []
    # file 空白/None 不在此层拒绝（留给冻结 scorer 的 parse_structured_outputs）
    ok, locs = S.normalize_locations({"locations": [{"file": "   "}]} )
    assert ok and locs == [{"file": "   ", "class_name": None, "function_name": None}]
    # 格式错误
    for bad in (
        None, "x", 42, {}, {"locations": None}, {"locations": "x"},
        {"locations": [1]}, {"locations": [{}]}, {"locations": [{"file": 1}]},
        {"locations": [{"file": "a", "class_name": 2}]},
        {"locations": [{"file": "a", "function_name": []}]},
    ):
        ok, _ = S.normalize_locations(bad)
        assert ok is False, "should reject %r" % (bad,)


def test_resolve_prediction():
    from src.verl_adapter import semantics as S

    locs = [{"file": "a.py", "class_name": None, "function_name": None}]
    assert S.resolve_prediction({"codesearch_finish_call_count": 1, "codesearch_structured_locations": locs}) == (locs, None)
    assert S.resolve_prediction({}) == (None, "no_finish")
    assert S.resolve_prediction({"codesearch_finish_call_count": 0}) == (None, "no_finish")
    assert S.resolve_prediction(
        {"codesearch_finish_call_count": 0, "codesearch_trajectory_exhausted": True}
    ) == (None, "exhausted")
    assert S.resolve_prediction({"codesearch_finish_call_count": 2}) == (None, "multi_finish")
    assert S.resolve_prediction({"codesearch_finish_call_count": 3}) == (None, "multi_finish")
    assert S.resolve_prediction({"codesearch_finish_call_count": 1}) == (None, "parse_error")
    assert S.resolve_prediction(
        {"codesearch_finish_call_count": 1, "codesearch_failure_class": "sanity_check"}
    ) == (None, "sanity_check")
    assert S.resolve_prediction({"codesearch_failure_class": "infra"}) == (None, "infra")
    assert S.resolve_prediction({"codesearch_finish_call_count": -1}) == (None, "parse_error")
    assert S.resolve_prediction({"codesearch_finish_call_count": "x"}) == (None, "parse_error")


def test_truncate_terminal_output():
    from src.verl_adapter import semantics as S

    assert S.truncate_terminal_output("short", 1000) == "short"
    out = S.truncate_terminal_output("A" * 100 + "B" * 100, 50)
    assert "truncated" in out and out.startswith("A") and "B" in out
    assert S.truncate_terminal_output("x" * 10, 0) == "x" * 10  # 0 = 不截断
    # 多字节 UTF-8 安全（errors="replace" 不抛错）
    S.truncate_terminal_output("中" * 100, 10)


# ---------------------------------------------------------------------------
# tools.py（verl 缺失时用 stub；真实 subprocess bash 本机可测）
# ---------------------------------------------------------------------------


def _load_tools_module():
    """导入 tools 模块。verl 可用时用真实 BaseTool；缺失时模块自带最小 fallback
    （tools.py 的 _BaseToolFallback/_ToolResponseFallback，接口形态对齐
    verl/tools/base_tool.py:36-38 与 schemas.py:98-127），判定逻辑可全量本机测试。"""
    import src.verl_adapter.tools as tools_mod

    return tools_mod


def test_tools_schema_envelope():
    """ToolSchemaEnvelope 的 model_dump 输出 == semantics 完整嵌套 schema，
    function.name 正确（verl 消费点：tool_agent_loop.py:119/299、rl_dataset.py:137、
    base_tool.py:38）。"""
    tools_mod = _load_tools_module()
    from src.verl_adapter import semantics

    for cls, expected in (
        (tools_mod.TerminalTool, semantics.TERMINAL_TOOL_SCHEMA),
        (tools_mod.LocalizationFinishTool, semantics.LOCALIZATION_FINISH_TOOL_SCHEMA),
    ):
        tool = cls()
        assert tool.tool_schema.function.name == expected["function"]["name"]
        dumped = tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
        assert dumped == expected, "%s schema envelope mismatch" % cls.__name__
        # dump 是深拷贝：改写不影响常量
        dumped["function"]["name"] = "tampered"
        assert tool.tool_schema.model_dump()["function"]["name"] == expected["function"]["name"]


def test_tools_finish_execute_semantics():
    """LocalizationFinishTool.execute：计数/多 finish/parse_error/终止标志
    （原语义 cnt != 1 → None，code_search_generator.py:87-90；任务 A-2）。"""
    tools_mod = _load_tools_module()
    tool = tools_mod.LocalizationFinishTool()

    class _AgentData:
        def __init__(self):
            self.extra_fields = {}

    # 1) 合法唯一 finish：structured 落 channel、终止标志置位、observation 为 json
    ag = _AgentData()
    params = {"locations": [{"file": "a.py", "class_name": "C", "function_name": "f"}]}
    resp, rew, metrics = asyncio.run(tool.execute("inst-1", params, agent_data=ag))
    assert ag.extra_fields["codesearch_finish_call_count"] == 1
    assert ag.extra_fields["codesearch_structured_locations"] == params["locations"]
    assert ag.extra_fields["codesearch_finish_terminated"] is True
    assert rew == 0.0 and metrics["codesearch_finish_call_count"] == 1
    assert json.loads(resp.text) == params["locations"]

    # 2) 第二次 finish：多 finish → structured 置 None
    resp, _, _ = asyncio.run(tool.execute("inst-1", params, agent_data=ag))
    assert ag.extra_fields["codesearch_finish_call_count"] == 2
    assert ag.extra_fields["codesearch_structured_locations"] is None
    assert ag.extra_fields["codesearch_finish_terminated"] is True  # 首次合法调用已置位

    # 3) 唯一 finish 但格式错误：parse_error、不终止（原 FunctionCallValidationError → 继续）
    ag = _AgentData()
    resp, _, _ = asyncio.run(tool.execute("inst-1", {"locations": "bad"}, agent_data=ag))
    assert ag.extra_fields["codesearch_finish_call_count"] == 1
    assert "codesearch_structured_locations" not in ag.extra_fields
    assert ag.extra_fields["codesearch_failure_class"] == "parse_error"
    assert "codesearch_finish_terminated" not in ag.extra_fields
    assert "invalid" in resp.text.lower()

    # 4) 空 locations 列表合法（原语义：有效 finish、零定位、scorer 走空预测）
    ag = _AgentData()
    resp, _, _ = asyncio.run(tool.execute("inst-1", {"locations": []}, agent_data=ag))
    assert ag.extra_fields["codesearch_structured_locations"] == []
    assert ag.extra_fields["codesearch_finish_terminated"] is True

    # 5) create：接受并忽略 create_kwargs（parquet 嵌套 struct 非空管道参数，
    #    非 OpenHands SDK 模型可见接口，见 tools.py docstring）
    inst_id, _ = asyncio.run(tool.create(create_kwargs={"instance_id": "ep-1"}))
    assert isinstance(inst_id, str) and inst_id
    inst_id, _ = asyncio.run(tool.create())
    assert isinstance(inst_id, str) and inst_id


def test_tools_terminal_execute_local():
    """TerminalTool：真实 bash 于 workspace、失败返回退出码文本、每轨迹隔离、
    create 校验 workspace 路径。"""
    tools_mod = _load_tools_module()
    tool = tools_mod.TerminalTool()

    tmp = Path(tempfile.mkdtemp(prefix="codescout-terminal-test-"))
    try:
        snapshot = tmp / "ep1"
        snapshot.mkdir()
        (snapshot / "flag.txt").write_text("hello", encoding="utf-8")

        class _AgentData:
            def __init__(self, request_id, tools_kwargs):
                self.request_id = request_id
                self.tools_kwargs = tools_kwargs
                self.extra_fields = {}

        def make_agent(request_id, isolate=True):
            return _AgentData(
                request_id,
                {
                    "terminal": {
                        "create_kwargs": {
                            "workspace_root": str(tmp),
                            "episode_id": "ep1",
                            "isolate_per_trajectory": isolate,
                            "timeout_s": 10,
                            "max_output_bytes": 0,
                        }
                    }
                },
            )

        # create：缺 workspace 配置 → ValueError（数据构建错误显式暴露）
        try:
            asyncio.run(tool.create(create_kwargs={}))
        except ValueError:
            pass
        else:
            raise AssertionError("TerminalTool.create should require workspace config")

        # 每轨迹隔离：cwd 在 .trajectories/<request_id>/ 的副本里
        ag = make_agent("req-1")
        resp, rew, _ = asyncio.run(tool.execute("i", {"command": "pwd"}, agent_data=ag))
        assert ".trajectories" in resp.text and "req-1" in resp.text
        traj_dir = tmp / ".trajectories" / "req-1"
        assert traj_dir.is_dir() and (traj_dir / "flag.txt").exists()
        # 同轨迹复用同一副本
        resp2, _, _ = asyncio.run(tool.execute("i", {"command": "pwd"}, agent_data=ag))
        assert resp2.text == resp.text
        # 另一轨迹独立副本
        ag2 = make_agent("req-2")
        resp3, _, _ = asyncio.run(tool.execute("i", {"command": "pwd"}, agent_data=ag2))
        assert "req-2" in resp3.text and resp3.text != resp.text
        # 命令失败：返回退出码文本而非异常
        resp4, _, _ = asyncio.run(tool.execute("i", {"command": "exit 3"}, agent_data=ag))
        assert "COMMAND FAILED with exit code 3" in resp4.text
        # 缺 command 参数
        resp5, _, _ = asyncio.run(tool.execute("i", {}, agent_data=ag))
        assert "missing" in resp5.text
        # 轨迹清理
        asyncio.run(tool.release_trajectory("req-1"))
        assert not traj_dir.exists()
        asyncio.run(tool.release_trajectory("req-unknown"))  # no-op 不抛

        # 关闭隔离：直接在快照目录执行
        ag3 = make_agent("req-3", isolate=False)
        resp6, _, _ = asyncio.run(tool.execute("i", {"command": "pwd"}, agent_data=ag3))
        assert str(snapshot) in resp6.text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# build_prompts.py：渲染与行 schema（设计文档 §6.2）
# ---------------------------------------------------------------------------


def test_render_user_template_manual_expectation():
    """用户模板渲染输出与手工期望逐字节一致（含 working_dir 替换；
    渲染 context 与原 prompt_builder.get_instruction 完全相同）。"""
    try:
        from src.verl_adapter import build_prompts as bp
    except Exception as exc:  # pragma: no cover
        _skip("cannot import build_prompts: %r" % exc)

    working_dir = "/mmu_vlm_hdd/home/rhsu/playground/codescout-data/workspaces/run-x/ep1"
    issue = "Bug: search crashes on empty query.\nSteps: run rg."
    context = {
        "instance": {"repo": "owner/myrepo", "problem_statement": issue},
        "workspace_dir_name": "myrepo",
        "working_dir": working_dir,
        "test_instructions": "",
    }
    template_text = (_TEMPLATE_DIR / "file_module_custom_finish.j2").read_text(encoding="utf-8")
    rendered = bp.render_template(template_text, context)

    expected = (
        "I have access to a python code repository in the directory "
        + working_dir
        + " . Consider the following issue description:\n"
        "\n"
        "<issue_description>\n"
        + issue
        + "\n"
        "</issue_description>\n"
        "\n"
        "Act as a code search agent and localize the specific files, classes or functions "
        "of code that need modification to resolve the issue in <issue_description>.\n"
        "\n"
        "NOTE: You do not need to solve the issue, all you need to do is localize relevant "
        "code from the repository. Your output will be used to guide another agent to solve the issue.\n"
        "\n"
        "IMPORTANT: Your output MUST follow the below rules:\n"
        '1. The final output must be a tool call to the "localization_finish" tool containing relevant code locations.\n'
        "2. The locations of the file path must be RELATIVE to the "
        + working_dir
        + ' directory WITHOUT any leading "./" in the output.\n'
        "3. Only include those locations in your output that need modification to resolve the issue "
        "in <issue_description>. Do NOT include any locations that do not need modification."
    )
    assert rendered == expected, "rendered user prompt drifts from manual expectation:\n%r" % rendered

    # jinja2 优先路径与最小渲染路径一致（服务器有 jinja2，本机最小渲染器）
    try:
        import jinja2  # noqa: F401

        env = jinja2.Environment()
        jinja_rendered = env.from_string(
            template_text[:-1] if template_text.endswith("\n") else template_text
        ).render(**context)
        assert jinja_rendered == rendered
    except ImportError:
        pass  # 本机无 jinja2：最小渲染器即为被测路径

    # 未知变量/复杂语法：最小渲染器显式失败（不静默产出错误 prompt）
    if "jinja2" not in sys.modules:
        try:
            bp.render_template("{{ unknown_var }}", context)
        except ValueError:
            pass
        else:
            raise AssertionError("unknown variable should fail loudly")
        try:
            bp.render_template("{% if x %}y{% endif %}", context)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported syntax should fail loudly")


def test_build_training_row_schema():
    """训练行 schema 符合设计文档 §6.2（prompt/data_source/reward_model/extra_info；
    ground_truth 必含 file_changes；extra_info.index 显式写入（B-6））。"""
    from src.verl_adapter import build_prompts as bp
    from src.verl_adapter import semantics

    instance = {
        "instance_id": "owner__repo-1234",
        "repo": "owner/repo",
        "base_commit": None,
        "problem_statement": "some issue text",
        "use_patch": True,
    }
    labels = {
        "instance_id": "owner__repo-1234",
        "file_changes": [{"file": "a.py", "changes": {"edited_modules": ["a.py:F"]}}],
        "patch": "PRIVATE",
    }
    row = bp.build_training_row(
        index=7,
        instance_row=instance,
        labels_entry=labels,
        workspace_root="/ws-root",
        split="train",
    )
    # 列集 == 设计文档 §6.2
    assert set(row.keys()) == {"prompt", "data_source", "reward_model", "extra_info"}
    assert row["data_source"] == semantics.CODESEARCH_DATA_SOURCE
    # prompt：system + user
    messages = row["prompt"]
    assert [m["role"] for m in messages] == ["system", "user"]
    system_text = (_TEMPLATE_DIR / "system_prompt_custom_finish.j2").read_text(encoding="utf-8")
    assert messages[0]["content"] == system_text  # 无变量模板：原样（含逐字 system prompt）
    assert "/ws-root/owner__repo-1234" in messages[1]["content"]
    # reward_model：ground_truth 即私有标签 file_changes（不进 prompt 列）
    assert row["reward_model"]["style"] == "rule"
    assert row["reward_model"]["ground_truth"] == {"file_changes": labels["file_changes"]}
    # extra_info：B-6 分组键 + tools_kwargs 通道
    ei = row["extra_info"]
    assert set(ei.keys()) == {
        "split", "index", "instance_id", "repo", "episode_id",
        "tools_kwargs", "interaction_kwargs",
    }
    assert ei["index"] == 7 and isinstance(ei["index"], int)
    assert ei["split"] == "train" and ei["instance_id"] == "owner__repo-1234"
    tk = ei["tools_kwargs"]
    assert set(tk.keys()) == {"terminal", "localization_finish"}
    ck = tk["terminal"]["create_kwargs"]
    assert ck["workspace_root"] == "/ws-root" and ck["episode_id"] == "owner__repo-1234"
    assert ck["isolate_per_trajectory"] is True and ck["timeout_s"] == 120.0
    assert tk["localization_finish"]["create_kwargs"] == {"instance_id": "owner__repo-1234"}
    assert ei["interaction_kwargs"] == {}
    # 私有标签不出现在 prompt 或 extra_info 的任何字符串值里
    assert "PRIVATE" not in json.dumps(row["prompt"])
    assert "PRIVATE" not in json.dumps(ei)
    # index 唯一性（B-6）：多行不重
    rows = [
        bp.build_training_row(i, instance, labels, workspace_root="/ws-root", split="train")
        for i in range(3)
    ]
    assert len({r["extra_info"]["index"] for r in rows}) == 3


def test_episode_id_safety():
    from src.verl_adapter import build_prompts as bp

    safe_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    assert bp.episode_id_for("owner__repo-1234") == "owner__repo-1234"
    for raw in ("..", ".", "a/b", "a b", "a:b", "x" * 200, "owner__repo-1234"):
        eid = bp.episode_id_for(raw)
        assert safe_re.match(eid), "unsafe episode_id %r for %r" % (eid, raw)
        # 目录分量安全：不是 "." / ".."、无路径分隔符
        assert eid not in (".", "..") and "/" not in eid and eid == eid.strip()
        assert len(eid) <= 128


def test_cli_end_to_end():
    """CLI：actor-safe parquet + labels jsonl → 训练 parquet（pandas 可用时）。"""
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        _skip("pandas not available locally (server-side check): %r" % exc)
    from src.verl_adapter import build_prompts as bp

    tmp = Path(tempfile.mkdtemp(prefix="codescout-cli-test-"))
    try:
        import pandas as pd

        df = pd.DataFrame(
            [
                {"instance_id": "o__r-1", "repo": "o/r", "base_commit": None,
                 "problem_statement": "issue one", "use_patch": True},
                {"instance_id": "o__r-2", "repo": "o/r", "base_commit": None,
                 "problem_statement": "issue two", "use_patch": True},
            ]
        )
        in_parquet = tmp / "input.parquet"
        df.to_parquet(in_parquet)
        labels_path = tmp / "labels.jsonl"
        with open(labels_path, "w", encoding="utf-8") as f:
            for iid in ("o__r-1", "o__r-2"):
                f.write(json.dumps({"instance_id": iid, "file_changes": [{"file": "a.py"}], "patch": "p"}) + "\n")

        out = tmp / "out" / "train.parquet"
        rc = bp.main(
            [
                "--input-parquet", str(in_parquet),
                "--labels", str(labels_path),
                "--workspace-root", str(tmp / "ws"),
                "--output", str(out),
                "--split", "train",
            ]
        )
        assert rc == 0 and out.exists()
        result = pd.read_parquet(out)
        assert list(result.columns) == ["prompt", "data_source", "reward_model", "extra_info"]
        assert len(result) == 2
        # 缺 labels → 显式失败
        bad_labels = tmp / "labels-missing.jsonl"
        bad_labels.write_text(
            json.dumps({"instance_id": "o__r-1", "file_changes": [], "patch": "p"}) + "\n",
            encoding="utf-8",
        )
        try:
            bp.main(["--input-parquet", str(in_parquet), "--labels", str(bad_labels),
                     "--workspace-root", str(tmp / "ws"), "--output", str(tmp / "bad.parquet"),
                     "--split", "train"])
        except SystemExit:
            pass
        else:
            raise AssertionError("missing labels should fail loudly")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# reward.py：adapter vs 冻结 scorer（≥50 合成 case）+ 合同语义 case + 失败类别
# ---------------------------------------------------------------------------


def _loc(file, class_name=None, function_name=None):
    return {"file": file, "class_name": class_name, "function_name": function_name}


def _adapter_score(scorer_fn, pred, gold, *, extra_info=None):
    """adapter 路径：经 compute_score（带合法单次 finish 的 extra_info）。"""
    from src.verl_adapter.reward import compute_score

    info = {"codesearch_finish_call_count": 1, "codesearch_structured_locations": pred}
    if extra_info:
        info.update(extra_info)
    return compute_score(
        data_source="codescout_swe_smith_py",
        solution_str="",
        ground_truth={"file_changes": gold},
        extra_info=info,
    ), scorer_fn


def _scorer_direct(scorer_fn, pred, gold):
    return scorer_fn("", {"file_changes": gold}, structured_locations=pred)


GOLDS = [
    [{"file": "src/a.py", "changes": {"edited_modules": ["src/a.py:Foo"], "edited_entities": ["src/a.py:Foo.bar"]}}],
    [{"file": "src/a.py", "changes": {"edited_modules": ["src/a.py:Foo", "src/a.py:run"],
                                     "edited_entities": ["src/a.py:Foo.bar", "src/a.py:run"]}}],
    [
        {"file": "src/a.py", "changes": {"edited_modules": ["src/a.py:Foo"], "edited_entities": ["src/a.py:Foo.bar", "src/a.py:Foo.baz"]}},
        {"file": "src/b.py", "changes": {"edited_modules": ["src/b.py:Util"], "edited_entities": ["src/b.py:Util.run"]}},
    ],
    [],                                        # 空真值列表
    [{"file": "src/a.py"}],                    # 只有 file 级真值
    [{"file": "src/a.py", "changes": {"edited_modules": None, "edited_entities": None}}],
    [{"file": "src/a.py", "changes": {}}],
    [{"changes": {"edited_modules": ["src/a.py:Foo"], "edited_entities": ["src/a.py:Foo.bar"]}}],  # 无 file 键
    [
        {"file": "src/a.py", "changes": {"edited_modules": ["src/a.py:Foo", "src/a.py:Bar", "src/a.py:top", "src/a.py:shared"],
                                        "edited_entities": ["src/a.py:Foo.__init__", "src/a.py:Bar.run", "src/a.py:top", "src/a.py:shared"]}},
        {"file": "src/b.py", "changes": {"edited_modules": ["src/b.py:Foo", "src/b.py:shared"],
                                        "edited_entities": ["src/b.py:Foo.helper", "src/b.py:shared"]}},
    ],
    [{"file": "src/a.py", "changes": {"edited_modules": ["src/a.py:Foo"]}}],
    [{"file": "src/a.py", "changes": {"edited_entities": ["src/a.py:Foo.bar"]}}],
    [dict(g) for g in [{"file": "src/a.py", "changes": {"edited_modules": ["src/a.py:Foo"]}}] for _ in range(3)],  # 重复真值
]

PREDS = [
    [],
    [_loc("src/a.py", "Foo", "bar")],
    [_loc("src/a.py", None, "run")],
    [_loc("src/a.py", "Foo", None)],
    [_loc("src/a.py", "Foo", "__init__")],
    [_loc("src/a.py", None, "top")],
    [_loc("src/b.py", None, "shared")],
    [_loc("src/a.py", "Foo", "bar"), _loc("src/b.py", "Util", "run"), _loc("src/c.py", None, "top")],
    [_loc("src/a.py", "Foo", "bar")] * 5,      # 重复预测
    [_loc(None, "Foo", "bar")],                # 空文件名 → 三级全空
    [_loc("   ", "Foo", "bar")],
    [_loc(" src/a.py ", None, "run")],          # 不做 strip（原语义）
    [_loc("src/a.py", "Foo", "bar"), _loc(None, "Foo", "bar")],  # 混入坏条目整体清空
    [_loc("src/x.py", "Zed", "why")],            # 完全不相交
]


def test_reward_adapter_matches_frozen_scorer_50plus():
    """≥50 个合成边界 case：adapter compute_score 与冻结 scorer 逐项差 ≤ 1e-8
    （任务交付物 7 的数值合同）。"""
    multilevel, _ = _import_scorer()
    from src.verl_adapter.reward import compute_score

    n_cases = 0
    for gold in GOLDS:
        for pred in PREDS:
            adapter = compute_score(
                data_source="codescout_swe_smith_py",
                solution_str="",
                ground_truth={"file_changes": gold},
                extra_info={
                    "codesearch_finish_call_count": 1,
                    "codesearch_structured_locations": pred,
                },
            )
            ref_reward, ref_dict = _scorer_direct(multilevel, pred, gold)
            assert abs(adapter["score"] - float(ref_reward)) <= TOL, (
                "score mismatch for pred=%r gold=%r: %r vs %r" % (pred, gold, adapter["score"], ref_reward)
            )
            for key in ("file_reward", "module_reward", "entity_reward"):
                assert abs(adapter[key] - float(ref_dict[key])) <= TOL, (
                    "%s mismatch for pred=%r gold=%r" % (key, pred, gold)
                )
            # 分母一致的 P/R 审计指标必须存在且落在 [0,1]
            for key in ("file_precision", "file_recall", "module_precision",
                        "module_recall", "entity_precision", "entity_recall"):
                v = adapter[key]
                assert 0.0 <= v <= 1.0, "%s out of range: %r" % (key, v)
            assert adapter["failure_class"] is None
            n_cases += 1
    assert n_cases >= 50, "需要 ≥50 个 case，实际 %d" % n_cases
    print("    [info] adapter vs scorer cases: %d" % n_cases)


def test_reward_failure_classes():
    """空预测/无 finish/多 finish/格式错/sanity/轮数耗尽/infra → 0 且分类正确。"""
    _ = _import_scorer()
    from src.verl_adapter.reward import compute_score

    gold = [{"file": "src/a.py", "changes": {"edited_modules": ["src/a.py:Foo"],
                                             "edited_entities": ["src/a.py:Foo.bar"]}}]
    gt = {"file_changes": gold}
    ds = "codescout_swe_smith_py"

    def score(info):
        return compute_score(data_source=ds, solution_str="", ground_truth=gt, extra_info=info)

    # 无 finish
    r = score({})
    assert r["score"] == 0.0 and r["failure_class"] == "no_finish"
    # 轮数耗尽（A-1 标志由 agent loop 写入）
    r = score({"codesearch_finish_call_count": 0, "codesearch_trajectory_exhausted": True})
    assert r["score"] == 0.0 and r["failure_class"] == "exhausted"
    # 多 finish
    r = score({"codesearch_finish_call_count": 2,
               "codesearch_structured_locations": [_loc("src/a.py", "Foo", "bar")]})
    assert r["score"] == 0.0 and r["failure_class"] == "multi_finish"
    # 格式错误（唯一一次 finish 解析失败）
    r = score({"codesearch_finish_call_count": 1, "codesearch_failure_class": "parse_error"})
    assert r["score"] == 0.0 and r["failure_class"] == "parse_error"
    # sanity 失败
    r = score({"codesearch_finish_call_count": 1,
               "codesearch_structured_locations": [_loc("src/a.py", "Foo", "bar")],
               "codesearch_failure_class": "sanity_check"})
    assert r["score"] == 0.0 and r["failure_class"] == "sanity_check"
    # infra 分列：score 0 但单独标记（协议 §4）
    r = score({"codesearch_failure_class": "infra"})
    assert r["score"] == 0.0 and r["failure_class"] == "infra"
    # 三级 F1 字段在 0 分路径也分列输出（no_finish 路径）
    r = score({})
    assert (r["file_reward"], r["module_reward"], r["entity_reward"]) == (0.0, 0.0, 0.0)
    # 合法空 locations 列表（有效 finish、零预测）：gold 非空 → P=R=0 → score 0
    r = score({"codesearch_finish_call_count": 1, "codesearch_structured_locations": []})
    assert r["score"] == 0.0 and r["failure_class"] is None
    assert (r["file_precision"], r["file_recall"]) == (0.0, 0.0)

    # ground_truth 合同：缺 file_changes 或非 list → 显式 ValueError（数据构建
    # 错误，不静默 0 分；原 instance.get("file_changes", []) 的空缺省语义由
    # build_prompts 保证写入 []，不依赖 reward 侧兜底）
    for bad_gt in ({"nope": 1}, {"file_changes": None}, {"file_changes": "x"}):
        try:
            compute_score(data_source=ds, solution_str="", ground_truth=bad_gt, extra_info={
                "codesearch_finish_call_count": 1, "codesearch_structured_locations": []})
        except ValueError:
            pass
        else:
            raise AssertionError("malformed ground_truth %r must raise" % (bad_gt,))


def test_reward_contract_semantics_borrowed():
    """借用 tests/test_reward_contract.py 的真实语义 case（class/顶层函数/空标签/
    无 finish/混合边界），经 adapter 的期望值逐项 ≤ 1e-8。"""
    _ = _import_scorer()
    from src.verl_adapter.reward import compute_score

    def score(pred, gold):
        return compute_score(
            data_source="codescout_swe_smith_py", solution_str="",
            ground_truth={"file_changes": gold},
            extra_info={"codesearch_finish_call_count": 1,
                        "codesearch_structured_locations": pred},
        )

    def close(a, b, label):
        assert abs(a - b) <= 1e-8, "%s: %r != %r" % (label, a, b)

    # 合同 case 1：gold 精确预测 → 3.0（test_exact_gold_prediction_scores_three）
    gold = [{"file": "src/a.py", "changes": {"edited_modules": ["src/a.py:Foo", "src/a.py:run"],
                                             "edited_entities": ["src/a.py:Foo.bar", "src/a.py:run"]}}]
    pred = [_loc("src/a.py", "Foo", "bar"), _loc("src/a.py", None, "run")]
    r = score(pred, gold)
    close(r["score"], 3.0, "exact")
    assert (r["file_reward"], r["module_reward"], r["entity_reward"]) == (1.0, 1.0, 1.0)

    # 合同 case 2：部分遗漏 → 11/6（test_partial_miss_precision_one_recall_less）
    gold2 = [
        {"file": "src/a.py", "changes": {"edited_modules": ["src/a.py:Foo"],
                                         "edited_entities": ["src/a.py:Foo.bar", "src/a.py:Foo.baz"]}},
        {"file": "src/b.py", "changes": {"edited_modules": ["src/b.py:Util"],
                                         "edited_entities": ["src/b.py:Util.run"]}},
    ]
    r = score([_loc("src/a.py", "Foo", "bar")], gold2)
    close(r["score"], 11 / 6, "partial")
    close(r["file_reward"], 2 / 3, "partial file")
    close(r["module_reward"], 2 / 3, "partial module")
    close(r["entity_reward"], 0.5, "partial entity")

    # 合同 case 3：过报 → 1.5（test_overprediction_recall_one_precision_less）
    gold3 = [{"file": "src/a.py", "changes": {"edited_modules": ["src/a.py:Foo"],
                                              "edited_entities": ["src/a.py:Foo.bar"]}}]
    pred3 = [_loc("src/a.py", "Foo", "bar"), _loc("src/b.py", "Util", "run"), _loc("src/c.py", None, "top")]
    r = score(pred3, gold3)
    close(r["score"], 1.5, "over")

    # 合同 case 4：重复预测 == 去重（test_duplicate_predictions_not_penalized_nor_rewarded）
    r_dup = score([_loc("src/a.py", "Foo", "bar")] * 6, gold3)
    r_uniq = score([_loc("src/a.py", "Foo", "bar")], gold3)
    close(r_dup["score"], r_uniq["score"], "duplicate == unique")

    # 合同 case 5：class-only → 2.0（entity 级 0；test_class_method_prediction_rules）
    gold5 = [{"file": "src/a.py", "changes": {"edited_modules": ["src/a.py:Foo"],
                                              "edited_entities": ["src/a.py:Foo.__init__"]}}]
    r = score([_loc("src/a.py", "Foo", None)], gold5)
    close(r["score"], 2.0, "class-only")
    assert r["entity_reward"] == 0.0

    # 合同 case 6：顶层函数 module==entity → 3.0（test_top_level_function_module_equals_entity）
    gold6 = [{"file": "src/a.py", "changes": {"edited_modules": ["src/a.py:run"],
                                              "edited_entities": ["src/a.py:run"]}}]
    r = score([_loc("src/a.py", None, "run")], gold6)
    close(r["score"], 3.0, "top-level")

    # 合同 case 7：空真值变体（test_empty_ground_truth_zero_f1 的 b-f）。
    # 注：原 case (a)"instance 完全没有 file_changes 键"在 adapter 侧不可表达——
    # adapter 总是显式传 {"file_changes": [...]}；空缺省语义由 build_prompts
    # 写入 [] 承担。file_changes=None 属 malformed labels，在
    # test_reward_failure_classes 中要求显式 ValueError。
    pred7 = [_loc("src/a.py", "Foo", "bar")]
    for gold7, expected in (
        ([], 0.0),
        ([{"file": "src/a.py"}], 1.0),
        ([{"file": "src/a.py", "changes": {"edited_modules": None, "edited_entities": None}}], 1.0),
        ([{"file": "src/a.py", "changes": {}}], 1.0),
    ):
        r = score(pred7, gold7)
        close(r["score"], expected, "empty-gold variant %r" % (gold7,))

    # 合同 case 8：条目无 file 键 → 2.0（test_change_without_file_key_counts_modules_only）
    gold8 = [{"changes": {"edited_modules": ["src/a.py:Foo"], "edited_entities": ["src/a.py:Foo.bar"]}}]
    r = score([_loc("src/a.py", "Foo", "bar")], gold8)
    close(r["score"], 2.0, "no-file-key")

    # 合同 case 9：混合边界 → 31/11（test_boundary_mixed_gold_multi_file_multi_class）
    gold9 = GOLDS[8]
    pred9 = [
        _loc("src/a.py", "Foo", "__init__"),
        _loc("src/a.py", "Bar", "run"),
        _loc("src/a.py", None, "top"),
        _loc("src/b.py", "Foo", "helper"),
        _loc("src/b.py", None, "shared"),
    ]
    r = score(pred9, gold9)
    close(r["score"], 31 / 11, "boundary mixed")
    close(r["file_reward"], 1.0, "boundary file")
    close(r["module_reward"], 10 / 11, "boundary module")
    close(r["entity_reward"], 10 / 11, "boundary entity")

    # 合同 case 10：无 finish → 0（test_no_finish_structured_locations_none_zero）
    r = compute_score(data_source="codescout_swe_smith_py", solution_str="ignored",
                      ground_truth={"file_changes": gold3},
                      extra_info={})
    assert r["score"] == 0.0 and r["failure_class"] == "no_finish"

    # 合同 case 11：file=None/空白 → 三级全 0（test_none_or_blank_file_zeroes_all_levels）
    r = score([_loc("src/a.py", "Foo", "bar"), _loc(None, "Foo", "bar")], gold3)
    close(r["score"], 0.0, "none-file nukes")

    # 合同 case 12：空 locations 列表 → 0（test_empty_location_list_finish_zero）
    r = score([], gold3)
    close(r["score"], 0.0, "empty locations")


# ---------------------------------------------------------------------------
# agent_loop：verl 运行时层（本机无 verl → skip；类可导入性/注册名由服务器验收）
# ---------------------------------------------------------------------------


def test_agent_loop_import_guards():
    try:
        import src.verl_adapter.agent_loop as al  # noqa: F401
    except Exception as exc:
        _skip("agent_loop import failed: %r" % exc)
    assert hasattr(al, "CodeSearchAgentLoop")


# ---------------------------------------------------------------------------
# __main__ 运行器（本机无 pytest；与 pytest 执行同一批 test_* 函数）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    _failed = 0
    _skipped = 0
    for _name, _fn in _tests:
        try:
            _fn()
        except SkipTest as _e:
            _skipped += 1
            print("SKIP  %s: %s" % (_name, _e))
        except AssertionError as _e:
            _failed += 1
            print("FAIL  %s: %s" % (_name, _e))
        except Exception as _e:  # noqa: BLE001 - 直跑运行器需完整暴露非断言异常
            _failed += 1
            print("ERROR %s: %s: %s" % (_name, type(_e).__name__, _e))
        else:
            print("PASS  %s" % _name)
    print("\n%d/%d tests passed, %d skipped" % (len(_tests) - _failed - _skipped, len(_tests), _skipped))
    if _failed:
        print("FAILED tests: %d" % _failed)
    sys.exit(1 if _failed else 0)
