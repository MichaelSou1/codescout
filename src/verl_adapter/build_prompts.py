"""离线构建 verl RLHFDataset 训练 parquet：模板渲染 + 私有标签入列。

协议/设计文档出处
------------------
- docs/reproduction/protocol-v1.md §3（system/user 模板版本冻结）、§8（私有
  标签 file_changes/patch 不进 prompt、不进 actor 可读路径）。
- docs/reproduction/verl-adapter-design.md §6.2（最小 parquet 列 schema：
  ``prompt``（messages 列表）、``data_source``、``reward_model{style,ground_truth}``、
  ``extra_info{split,index,instance_id,repo,tools_kwargs,interaction_kwargs}``；
  硬约束 ``reward_model.ground_truth`` 必须存在（reward_loop/reward_manager/
  naive.py:43 直接下标）；``extra_info.index`` 必须显式写——RLHFDataset 缺失时
  置 0 会让全部样本同一 GRPO 组（B-6））、§7.2（verl 无模板文件路径配置键，
  system/user 内容必须在离线构建阶段渲染进 ``prompt`` 列）。

语义对齐原实现
--------------
- user 模板渲染复刻 ``src/prompts/prompt_builder.py::get_instruction``：context
  为 ``{"instance": instance, "working_dir": workspace_path,
  "workspace_dir_name": repo.split("/")[-1], "test_instructions": ""}``；有
  jinja2 时用 jinja2 渲染（与原实现同库同语义，含 jinja2 默认去掉模板末尾单个
  换行）；本机无 jinja2 时退化为严格 ``{{ name.attr }}`` 最小渲染器（对这两个
  模板逐字等价，见 tests；不支持复杂语法会显式抛错）。
- system 模板（``system_prompt_custom_finish.j2``）无变量，原样渲染（OpenHands
  CustomAgent 同样把它作为模板文本处理；其末尾换行处理列入服务器 smoke 字节
  对账，见设计文档 B-5）。
- workspace 路径 = ``workspace_root/<episode_id>``（协议 §3：替代原
  ``/tmp/testbed/<uuid>``；快照由 workspace.build_workspace 离线准备）。该路径
  写进 user prompt 的 ``{{ working_dir }}``，同时进 terminal 的
  ``create_kwargs``——prompt 文本与工具实际 cwd 必须同源。

CLI 输入输出
------------
输入：``scripts/prepare_verl_data.py`` 的 actor-safe parquet
（instance_id/repo/base_commit/problem_statement/use_patch）+ 私有 labels jsonl
（instance_id → file_changes/patch）+ workspace 根。
输出：训练 parquet（列：prompt/data_source/reward_model/extra_info）+ manifest
摘要（stdout）。**labels 文件与输出中的 reward_model 列都是私有标签**，只能存
actor 不可读的运行时位置（协议 §8）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(_REPO_ROOT))

from src.verl_adapter import semantics  # noqa: E402
from src.verl_adapter.workspace import _EPISODE_RE as _WS_EPISODE_RE  # noqa: E402

_TEMPLATE_DIR = _REPO_ROOT / "src" / "prompts" / "templates"
_SYSTEM_TEMPLATE = "system_prompt_custom_finish.j2"
_USER_TEMPLATE = "file_module_custom_finish.j2"

#: 与原 get_instruction 相同的渲染上下文键（prompt_builder.py:31-37）。
_VAR_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\}\}")
_MARKER_PATTERN = re.compile(r"\{\{.*?\}\}|\{%.*?%\}")


# ---------------------------------------------------------------------------
# 模板渲染（jinja2 优先；无 jinja2 环境的严格最小渲染器）
# ---------------------------------------------------------------------------


def render_template(template_text: str, context: dict[str, Any]) -> str:
    """渲染模板文本；与原 jinja2 ``Environment`` 语义对齐。

    - 有 jinja2：``jinja2.Environment``（无 loader、无 autoescape，同原实现），
      并按 jinja2 默认 ``keep_trailing_newline=False`` 去掉模板末尾单个换行。
    - 无 jinja2（本机测试环境）：仅支持 ``{{ name }}`` / ``{{ name.attr }}``
      变量替换，遇到任何其他 jinja 语法（``{% %}`` 等）抛 ``ValueError``；同样
      去掉末尾单个换行，保证两条路径对这两个模板字节一致。
    """
    text = template_text
    if text.endswith("\n"):
        text = text[:-1]  # jinja2 keep_trailing_newline=False 默认语义

    try:
        from jinja2 import Environment
    except ImportError:
        return _render_minimal(text, context)

    env = Environment()  # autoescape=False，同原实现（无 HTML 转义）
    return env.from_string(text).render(**context)


def _render_minimal(text: str, context: dict[str, Any]) -> str:
    def _lookup(match: "re.Match[str]") -> str:
        expr = match.group(1)
        value: Any = context
        for part in expr.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(
                    f"minimal jinja renderer: unknown variable {expr!r} "
                    f"(context keys: {sorted(context.keys())})"
                )
            value = value[part]
        if not isinstance(value, str):
            raise ValueError(f"minimal jinja renderer: {expr!r} is not a string: {type(value).__name__}")
        return value

    unsupported = [m.group(0) for m in _MARKER_PATTERN.finditer(text) if not _VAR_PATTERN.fullmatch(m.group(0).strip())]
    if unsupported:
        raise ValueError(
            "minimal jinja renderer only supports {{ name.attr }} substitution; "
            f"unsupported constructs: {unsupported[:3]}"
        )
    return _VAR_PATTERN.sub(_lookup, text)


# ---------------------------------------------------------------------------
# 行构建（纯函数，测试不依赖 pandas）
# ---------------------------------------------------------------------------


def episode_id_for(instance_id: str) -> str:
    """instance_id → 安全 episode 目录名（workspace.build_workspace 的
    episode_id 约束：字母数字开头的 [A-Za-z0-9._-]，≤128 字符）。"""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(instance_id))
    if not safe or not _WS_EPISODE_RE.match(safe):
        safe = "ep-" + safe
    return safe[:128]


def build_training_row(
    index: int,
    instance_row: dict[str, Any],
    labels_entry: dict[str, Any],
    *,
    workspace_root: str,
    split: str,
    data_source: str = semantics.CODESEARCH_DATA_SOURCE,
    system_template_text: Optional[str] = None,
    user_template_text: Optional[str] = None,
    terminal_timeout_s: float = 120.0,
    terminal_max_output_bytes: int = 131072,
    isolate_per_trajectory: bool = True,
) -> dict[str, Any]:
    """构建单条 RLHFDataset 行（设计文档 §6.2 schema）。

    Args:
        index: 行序号（0 起）。写入 ``extra_info.index``——GRPO 组归一化键
            （core_algos.py:311-316），**必须显式写**：缺失时 RLHFDataset 置 0，
            全部样本会落进同一组毁掉分组（B-6）。verl ``rollout.n=8`` 复制样本
            时 index 原样复制，同 issue 8 条共享同一 index。
        instance_row: actor-safe parquet 行（instance_id/repo/problem_statement…）。
        labels_entry: 私有标签（instance_id/file_changes/patch）。
        workspace_root: 协议 §3 的 workspace 根（快照父目录）。
        split: "train" | "validation"。
        data_source: parquet ``data_source`` 列常量。
        terminal_timeout_s / terminal_max_output_bytes / isolate_per_trajectory:
            见 tools.py::TerminalTool（超时与输出截断参数化）。

    Returns:
        dict：``{"prompt": [system, user], "data_source", "reward_model",
        "extra_info"}``。
    """
    instance_id = str(instance_row["instance_id"])
    repo = str(instance_row["repo"])
    episode_id = episode_id_for(instance_id)
    working_dir = str(Path(workspace_root) / episode_id)

    # 与原 get_instruction 完全相同的 context（prompt_builder.py:31-37）
    context: dict[str, Any] = {
        "instance": instance_row,
        "workspace_dir_name": repo.split("/")[-1],
        "working_dir": working_dir,
        "test_instructions": "",
    }

    system_text = (
        system_template_text
        if system_template_text is not None
        else (_TEMPLATE_DIR / _SYSTEM_TEMPLATE).read_text(encoding="utf-8")
    )
    user_text = (
        user_template_text
        if user_template_text is not None
        else (_TEMPLATE_DIR / _USER_TEMPLATE).read_text(encoding="utf-8")
    )
    messages = [
        {"role": "system", "content": render_template(system_text, context)},
        {"role": "user", "content": render_template(user_text, context)},
    ]

    ground_truth = {"file_changes": labels_entry["file_changes"]}
    extra_info = {
        "split": split,
        "index": int(index),
        "instance_id": instance_id,
        "repo": repo,
        "episode_id": episode_id,
        "tools_kwargs": {
            semantics.TERMINAL_TOOL_NAME: {
                "create_kwargs": {
                    "workspace_root": str(workspace_root),
                    "episode_id": episode_id,
                    "isolate_per_trajectory": bool(isolate_per_trajectory),
                    "timeout_s": float(terminal_timeout_s),
                    "max_output_bytes": int(terminal_max_output_bytes),
                }
            },
            # finish 工具无状态；instance_id 仅为满足 parquet 嵌套 struct 非空要求
            # （空 dict 会触发 pyarrow "struct with no child field"），create 侧忽略。
            semantics.FINISH_TOOL_NAME: {"create_kwargs": {"instance_id": episode_id}},
        },
        "interaction_kwargs": {},
    }
    return {
        "prompt": messages,
        "data_source": data_source,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": extra_info,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_labels(path: str) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "instance_id" not in entry or "file_changes" not in entry:
                raise SystemExit(f"ERROR: labels line {line_no} missing instance_id/file_changes")
            labels[str(entry["instance_id"])] = entry
    return labels


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="构建 verl RLHFDataset 训练 parquet（详见模块 docstring）")
    ap.add_argument("--input-parquet", required=True,
                    help="scripts/prepare_verl_data.py 输出的 actor-safe parquet（train/validation）")
    ap.add_argument("--labels", required=True, help="私有 labels jsonl（actor 不可读位置）")
    ap.add_argument("--workspace-root", required=True,
                    help="workspace 根（协议 §3：codescout-data/workspaces/<run_id>，快照父目录）")
    ap.add_argument("--output", required=True, help="输出训练 parquet 路径")
    ap.add_argument("--split", required=True, choices=["train", "validation"])
    ap.add_argument("--data-source", default=semantics.CODESEARCH_DATA_SOURCE)
    ap.add_argument("--terminal-timeout-s", type=float, default=120.0)
    ap.add_argument("--terminal-max-output-bytes", type=int, default=131072)
    ap.add_argument("--no-isolate-per-trajectory", action="store_true",
                    help="8 条并行 rollout 共享同一快照目录（有写污染风险，仅 smoke 用）")
    args = ap.parse_args(argv)

    import pandas as pd  # 服务器运行时依赖；延迟导入让本模块核心可脱离 pandas 测试

    df = pd.read_parquet(args.input_parquet)
    required_cols = {"instance_id", "repo", "problem_statement"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise SystemExit(f"ERROR: input parquet missing columns: {missing}; available: {list(df.columns)}")

    labels = _load_labels(args.labels)
    missing_labels = sorted(set(df["instance_id"].astype(str)) - set(labels))
    if missing_labels:
        raise SystemExit(f"ERROR: {len(missing_labels)} instance(s) missing labels, e.g. {missing_labels[:5]}")

    started = datetime.now(timezone.utc).isoformat()
    rows = []
    for index, record in enumerate(df.to_dict(orient="records")):
        instance_id = str(record["instance_id"])
        rows.append(
            build_training_row(
                index=index,
                instance_row=record,
                labels_entry=labels[instance_id],
                workspace_root=args.workspace_root,
                split=args.split,
                data_source=args.data_source,
                terminal_timeout_s=args.terminal_timeout_s,
                terminal_max_output_bytes=args.terminal_max_output_bytes,
                isolate_per_trajectory=not args.no_isolate_per_trajectory,
            )
        )

    if len({row["extra_info"]["index"] for row in rows}) != len(rows):
        raise SystemExit("ERROR: extra_info.index not unique — GRPO 分组键被破坏（B-6）")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path)

    manifest = {
        "created_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "input_parquet": args.input_parquet,
        "labels": args.labels,
        "workspace_root": args.workspace_root,
        "output": str(out_path),
        "split": args.split,
        "data_source": args.data_source,
        "num_rows": len(rows),
        "templates": {"system": _SYSTEM_TEMPLATE, "user": _USER_TEMPLATE},
        "semantics": "prompt_builder.get_instruction context; reward_model.ground_truth=file_changes; extra_info.index explicit (B-6)",
        "expected_rollout_n": 8,
        "note": "rollout.n=8 在训练时复制样本；同 index 恰 8 条的运行期断言在服务器验收（设计文档 B-6）",
    }
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
