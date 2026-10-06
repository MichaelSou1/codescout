"""verl_adapter：CodeScout verl 迁移的 workspace 构建适配层。

协议出处：docs/reproduction/protocol-v1.md §3（workspace 路径与无 ``.git`` 快照）、
§8（私有标签隔离）。入口见 ``workspace`` 模块 docstring。
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
