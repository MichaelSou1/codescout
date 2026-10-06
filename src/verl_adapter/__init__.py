"""verl_adapter：CodeScout verl 迁移的适配层。

协议出处：docs/reproduction/protocol-v1.md（冻结协议）与
docs/reproduction/verl-adapter-design.md（设计研究）。

模块一览（设计文档 §11.1）：

- ``workspace``：隔离 workspace 构建（无 ``.git`` 快照）；
- ``semantics``：纯 stdlib 语义层（冻结判定逻辑 + 工具 schema 常量，
  tests 的核心层不依赖 verl/pydantic/jinja2）；
- ``tools``：verl BaseTool 工具（terminal / localization_finish）；
- ``agent_loop``：``CodeSearchAgentLoop``（A-1/A-2/sanity/B-3 消解点，
  注册名 ``codesearch_agent``）；
- ``reward``：verl custom reward function（调冻结 scorer）；
- ``build_prompts``：离线渲染模板并产出 RLHFDataset 训练 parquet；
- ``tools.yaml`` / ``agent_loop.yaml``：verl 工具与 agent loop 的声明文件
  （``multi_turn.tool_config_path`` / ``agent.agent_loop_config_path``）。

注意：本 ``__init__`` 只导入纯 stdlib 的 ``workspace``，保证本机无 verl/pydantic
环境可导入 ``src.verl_adapter.semantics`` 等核心层；verl 依赖模块请按需
单独 import。
"""

from .workspace import (
    ApplyError,
    CheckoutError,
    CloneError,
    IntegrityError,
    WorkspaceError,
    build_workspace,
    workspace_tree_fingerprint,
)

__all__ = [
    "ApplyError",
    "CheckoutError",
    "CloneError",
    "IntegrityError",
    "WorkspaceError",
    "build_workspace",
    "workspace_tree_fingerprint",
]
