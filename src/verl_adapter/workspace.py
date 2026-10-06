"""隔离 workspace 构建：verl rollout 的仓库快照准备层。

协议出处
--------
``docs/reproduction/protocol-v1.md`` §3（冻结协议）：

  - workspace 路径为 ``codescout-data/workspaces/<run_id>/<episode_id>``，替代原实现的
    ``/tmp/testbed/<uuid>``；
  - SWE-smith（``use_patch=True``）必须先在**含 ``.git`` 的克隆目录**内应用 mutation
    patch，再向 workspace 导出**不含 ``.git``** 的代码树快照，防止 agent 用
    ``git diff`` 等命令泄漏 mutation 位置（协议 §8 私有标签隔离）。

与上游 ``src/utils/instance.py::clone_instance``（源码 commit 9d05a644）的语义对齐
---------------------------------------------------------------------------
  - 从 ``https://github.com/{repo}.git`` 克隆；
  - ``use_patch=True``  → 不 checkout（SWE-smith ``base_commit=None``），
    ``git apply`` mutation patch（patch 引入 bug，是私有评分标签）；
  - ``use_patch=False`` → ``git checkout base_commit``，不应用任何 patch。labels 中的
    ``patch`` 对 Verified/Pro/Lite 是 gold 修复 patch，**绝不能应用**；本模块仅当
    ``use_patch=True`` 时读取 ``instance["patch"]``，其余情况完全忽略该字段。

与原实现的关键差异（协议要求）
------------------------------
  1. 克隆/checkout/apply 全部发生在 ``workspace_root`` 下的隐藏临时目录
     （``.build-<episode_id>-*``）；完成后把工作树复制到 workspace（**无 ``.git``**），
     临时克隆目录随即删除。正常路径下临时目录在 finally 中清理；若进程被强杀可能残留
     （内含 ``.git``），应在无并行作业时人工清理。
  2. 失败显式抛出 ``CloneError``/``CheckoutError``/``ApplyError``/``IntegrityError``，
     绝不静默返回 False；属于 infra 异常，训练侧不得计为模型 0 分（协议 §4）。
  3. 可选 ``clone_cache``：同一 repo+commit 复用克隆模板；每个 episode 从模板做本地
     ``git clone`` 后独立 checkout/apply，快照绝不共享工作树。``base_commit=None``
     的 SWE-smith 任务缓存键为 repo 的 default HEAD（克隆时刻钉定，记入 sidecar 的
     ``resolved_head``，树指纹可审计）。
  4. 幂等：workspace 已存在且树指纹与 sidecar（``workspace_root/.workspace_meta/
     <episode_id>.json``）一致时直接返回；不一致则重建覆盖。

私有标签边界（协议 §8）
----------------------
  - mutation patch 只通过 stdin / 系统临时文件（OS tmp 目录，finally 删除）传递，绝不
    写入 ``workspace_root`` 下任何位置，绝不进入异常消息；
  - ``ApplyError`` 的消息不含 ``git apply`` 的 stderr（stderr 可能引用 patch 上下文行，
    属泄漏向量）；``CloneError``/``CheckoutError`` 的 stderr 不含 patch 内容，截断后保留；
  - sidecar 只保存摘要与 digest（``patch_sha256`` 为 patch 文本的 SHA-256，不含原文）。

依赖：仅标准库。pyarrow/pandas 由上层（如 smoke 脚本）按需延迟导入。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "ApplyError",
    "CheckoutError",
    "CloneError",
    "IntegrityError",
    "WorkspaceError",
    "build_workspace",
    "workspace_tree_fingerprint",
]

_GIT = "git"
_META_DIR_NAME = ".workspace_meta"
_CACHE_META_NAME = ".clone_cache_meta.json"
_BUILD_TMP_PREFIX = ".build-"
# owner/repo（GitHub 组织/用户名与仓库名的合法字符）
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
# episode_id 必须是安全路径分量：字母数字开头，仅字母数字/./_/-
_EPISODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# ---------------------------------------------------------------------------
# 错误分类（infra 异常，与有效模型失败分列，协议 §4/§8）
# ---------------------------------------------------------------------------


class WorkspaceError(Exception):
    """workspace 构建失败的基类（infra 异常，不计为模型 0 分）。"""


class CloneError(WorkspaceError):
    """git 不可用，https 克隆或本地克隆失败，克隆缓存条目损坏。"""


class CheckoutError(WorkspaceError):
    """base_commit checkout 失败，或 HEAD 与期望 commit 不符。"""


class ApplyError(WorkspaceError):
    """mutation patch 缺失/为空、``git apply`` 失败、应用后无变更、快照反向校验失败。"""


class IntegrityError(WorkspaceError):
    """快照导出不完整（含 ``.git``、文件数不一致、指纹异常）。"""


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail(text: str, limit: int = 400) -> str:
    """截断长文本，避免异常消息吞没日志。"""
    text = (text or "").strip()
    return text if len(text) <= limit else text[-limit:]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git(args: list[str], *, cwd: Path | None = None, input_text: str | None = None) -> tuple[int, str, str]:
    """运行 git 子命令，返回 ``(returncode, stdout, stderr)``；不抛 CalledProcessError。

    git 可执行文件缺失时抛 ``CloneError``。
    """
    try:
        proc = subprocess.run(
            [_GIT, *args],
            cwd=None if cwd is None else str(cwd),
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise CloneError("git executable not found on PATH") from exc
    return proc.returncode, proc.stdout, proc.stderr


def _rev_parse(git_dir: Path) -> str:
    rc, out, err = _git(["rev-parse", "HEAD"], cwd=git_dir)
    if rc != 0:
        raise CheckoutError(f"git rev-parse HEAD failed in {git_dir} (rc={rc}): {_tail(err)}")
    return out.strip()


def _count_tree_entries(root: Path, *, skip_git: bool) -> int:
    """统计文件与符号链接条目数（协议要求不忽略无关目录，保持简单忠实）。

    ``skip_git=True`` 时跳过任何路径分量名为 ``.git`` 的条目（用于克隆目录计数）。
    """
    n = 0
    for path in root.rglob("*"):
        if skip_git and ".git" in path.relative_to(root).parts:
            continue
        if path.is_symlink() or path.is_file():
            n += 1
    return n


# ---------------------------------------------------------------------------
# 克隆缓存（可选；同一 repo+commit 复用模板，绝不共享工作树）
# ---------------------------------------------------------------------------


def _cache_key(repo: str, commit: str | None) -> str:
    safe_repo = re.sub(r"[^A-Za-z0-9._-]", "_", repo)
    if commit is None:
        return f"{safe_repo}__default-head"
    safe_commit = re.sub(r"[^A-Za-z0-9._-]", "_", commit)
    return f"{safe_repo}__commit-{safe_commit}"


def _check_cache_entry(cache_dir: Path, repo: str, commit: str | None) -> tuple[bool, str]:
    """校验缓存条目与请求的 repo+commit 一致且 git 状态未被改动。"""
    meta_path = cache_dir / _CACHE_META_NAME
    try:
        meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable cache meta: {exc}"
    if not isinstance(meta, dict) or meta.get("repo") != repo:
        return False, "repo mismatch"
    if commit is not None and meta.get("commit") != commit:
        return False, "commit mismatch"
    rc, out, _ = _git(["rev-parse", "HEAD"], cwd=cache_dir)
    if rc != 0:
        return False, "rev-parse failed"
    head = out.strip()
    if head != (meta.get("resolved_head") or ""):
        return False, "HEAD moved since cache creation"
    if commit is not None and head.lower() != commit.lower():
        return False, f"HEAD {head} != requested commit"
    return True, "ok"


def _acquire_cached_clone(repo: str, commit: str | None, cache_root: Path) -> tuple[Path, str]:
    """取回或构建 repo+commit 的克隆模板，返回 ``(缓存目录, cache_key)``。

    构建过程在缓存根下的 staging 目录完成后经原子 ``os.rename`` 发布，保证并发读者
    只能看到完整条目；条目损坏时抛 ``CloneError``（提示人工处理），不自动覆盖——
    避免与正在本地克隆该条目的并发进程竞争。
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    key = _cache_key(repo, commit)
    cache_dir = cache_root / key
    if (cache_dir / ".git").exists():
        valid, reason = _check_cache_entry(cache_dir, repo, commit)
        if valid:
            return cache_dir, key
        raise CloneError(
            f"clone cache entry {cache_dir} invalid ({reason}); remove it or point clone_cache elsewhere"
        )

    staging = Path(tempfile.mkdtemp(prefix=f".staging-{key}-", dir=cache_root))
    try:
        rc, _, err = _git(["clone", f"https://github.com/{repo}.git", str(staging)])
        if rc != 0:
            raise CloneError(f"https clone failed for {repo} (rc={rc}): {_tail(err)}")
        if commit is not None:
            rc, _, err = _git(["checkout", commit], cwd=staging)
            if rc != 0:
                raise CheckoutError(f"checkout {commit} failed for {repo} (rc={rc}): {_tail(err)}")
        head = _rev_parse(staging)
        meta = {
            "repo": repo,
            "commit": commit,
            "resolved_head": head,
            "created_utc": _utcnow(),
            "clone_url": f"https://github.com/{repo}.git",
        }
        (staging / _CACHE_META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        try:
            os.rename(staging, cache_dir)
        except OSError:
            # 并发方抢先发布了同一 key：校验后复用，否则报错
            if (cache_dir / ".git").exists():
                valid, reason = _check_cache_entry(cache_dir, repo, commit)
                if valid:
                    return cache_dir, key
                raise CloneError(f"concurrently published cache entry {cache_dir} invalid ({reason})")
            raise CloneError(f"failed to publish clone cache entry {cache_dir}")
        return cache_dir, key
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# sidecar（幂等校验记录；只存摘要与 digest，绝不存 patch 文本）
# ---------------------------------------------------------------------------


def _sidecar_path(root: Path, episode_id: str) -> Path:
    return root / _META_DIR_NAME / f"{episode_id}.json"


def _write_sidecar(root: Path, episode_id: str, payload: dict[str, Any]) -> None:
    meta_dir = root / _META_DIR_NAME
    meta_dir.mkdir(parents=True, exist_ok=True)
    final = _sidecar_path(root, episode_id)
    tmp = final.with_name(f".{episode_id}.json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, final)


def _read_sidecar(root: Path, episode_id: str) -> dict[str, Any] | None:
    try:
        data = json.loads(_sidecar_path(root, episode_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _sidecar_matches_request(
    sidecar: dict[str, Any],
    repo: str,
    use_patch: bool,
    base_commit: str | None,
    patch_sha256: str | None,
) -> bool:
    """sidecar 与本次请求的任务身份一致（repo/语义/base_commit/patch 摘要）。"""
    return (
        sidecar.get("repo") == repo
        and bool(sidecar.get("use_patch")) == use_patch
        and sidecar.get("base_commit") == base_commit
        and sidecar.get("patch_sha256") == patch_sha256
    )


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------


def workspace_tree_fingerprint(workspace: str | os.PathLike[str]) -> dict[str, Any]:
    """计算目录树指纹，用于 manifest 与可复现验收（协议 §3/§10 实验账本）。

    语义：递归遍历全部条目（协议要求不忽略常见无关目录，保持简单忠实）；按相对路径
    排序，文件按内容 SHA-256，符号链接按目标字符串 SHA-256；``tree_sha256`` 为
    ``"rel\\0type\\0bytes\\0sha256\\n"`` 行序列的累积 SHA-256，对同一棵树可复现。

    注意：若目录中存在 ``.git``（不应发生），也会被如实计入指纹——冒烟脚本单独
    审计 ``.git`` 的存在性。

    Args:
        workspace: 待指纹的目录路径。

    Returns:
        dict：``{"file_count": int, "total_bytes": int, "tree_sha256": str,
        "files": [{"path", "type", "bytes", "sha256"}, ...]}``。

    Raises:
        WorkspaceError: 目标不是目录。
    """
    root = Path(workspace)
    if not root.is_dir():
        raise WorkspaceError(f"fingerprint target is not a directory: {root}")

    tree_hash = hashlib.sha256()
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            entry_type = "symlink"
            digest = _sha256_text(f"symlink:{os.readlink(path)}")
            entry_bytes = 0
        elif path.is_file():
            entry_type = "file"
            digest = _sha256_file(path)
            entry_bytes = path.stat().st_size
        else:
            continue  # 目录不单独计入
        total_bytes += entry_bytes
        tree_hash.update(f"{rel}\0{entry_type}\0{entry_bytes}\0{digest}\n".encode("utf-8"))
        files.append({"path": rel, "type": entry_type, "bytes": entry_bytes, "sha256": digest})

    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": tree_hash.hexdigest(),
        "files": files,
    }


def build_workspace(
    instance: dict[str, Any],
    workspace_root: str | os.PathLike[str],
    episode_id: str,
    clone_cache: str | os.PathLike[str] | None = None,
) -> Path:
    """为单个 episode 构建隔离的真实仓库快照 workspace。

    协议出处：``docs/reproduction/protocol-v1.md`` §3。语义对齐上游
    ``src/utils/instance.py::clone_instance``（详见模块 docstring）：

      - ``use_patch=True``  → 不 checkout、在含 ``.git`` 的克隆内 ``git apply``
        mutation patch（SWE-smith：``base_commit=None``）；
      - ``use_patch=False`` → ``git checkout base_commit``、不应用任何 patch
        （Verified/Pro/Lite 语义；labels 的 gold patch 绝不应用）。

    关键流程（协议要求）：克隆/checkout/apply 全部发生在临时目录；完成后将工作树
    复制导出（**无 ``.git``**，防止 ``git diff`` 泄漏 mutation 位置）到
    ``workspace_root/<episode_id>``，再删除临时克隆。``use_patch=True`` 时对导出树执行
    ``git apply --check --reverse`` 校验（只读、不修改快照），确认 mutation 已包含在
    导出快照中。

    并发安全：episode_id 唯一保证不同 episode 互不覆盖；启用 ``clone_cache`` 时缓存
    模板经原子发布，每 episode 仍独立本地克隆/独立导出，绝不共享工作树。

    幂等：目标已存在、sidecar 记录的任务身份一致且树指纹校验一致时直接返回；否则
    重建覆盖（重建失败时保留旧目标，不先删后建）。

    Args:
        instance: 任务字段。必需键 ``instance_id``/``repo``/``use_patch``；
            ``use_patch=True`` 时必需非空 ``patch``（mutation patch，私有标签）；
            ``use_patch=False`` 时必需非空 ``base_commit``。其余键（如
            ``problem_statement``）被忽略。
        workspace_root: workspace 根目录（如
            ``codescout-data/workspaces/<run_id>``），不存在则创建。
        episode_id: 唯一 episode 标识，必须为安全路径分量（字母数字开头的
            ``[A-Za-z0-9._-]`` 序列），作为 workspace 子目录名。
        clone_cache: 可选克隆缓存目录；同一 repo+commit 复用克隆模板以减少重复
            https 克隆。``base_commit=None`` 的任务按 repo default HEAD 缓存（克隆
            时刻钉定，见 ``resolved_head``）。

    Returns:
        构建好的 workspace 目录路径（``workspace_root/<episode_id>``，无 ``.git``）。

    Raises:
        WorkspaceError: episode_id 或 instance 字段非法。
        CloneError: git 不可用 / https 或本地克隆失败 / 缓存条目损坏。
        CheckoutError: checkout 失败或 HEAD 校验不符。
        ApplyError: patch 缺失或为空 / ``git apply`` 失败 / 应用后无变更 /
            导出快照反向校验失败（消息不含 patch 内容）。
        IntegrityError: 导出含 ``.git``、文件数不一致等快照完整性问题。
    """
    # ---- 参数与任务字段校验 ----
    if not isinstance(episode_id, str) or not _EPISODE_RE.match(episode_id):
        raise WorkspaceError(
            f"invalid episode_id: {episode_id!r} (must be alnum-started [A-Za-z0-9._-], not '.'/'..')"
        )

    instance_id = instance.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise WorkspaceError(f"instance_id missing/empty for episode {episode_id}")
    repo = instance.get("repo")
    if not isinstance(repo, str) or not _REPO_RE.match(repo):
        raise WorkspaceError(f"invalid repo for instance {instance_id}: {repo!r}")
    if instance.get("use_patch") is None:
        raise WorkspaceError(f"use_patch missing for instance {instance_id}")

    use_patch = bool(instance["use_patch"])
    base_commit: str | None = None
    patch: str | None = None
    if use_patch:
        patch = instance.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            raise ApplyError(
                f"instance {instance_id}: use_patch=True but mutation patch is missing/empty"
            )
    else:
        base = instance.get("base_commit")
        if not isinstance(base, str) or not base.strip():
            raise CheckoutError(
                f"instance {instance_id}: use_patch=False but base_commit is missing/empty"
            )
        base_commit = base.strip()

    root = Path(workspace_root)
    target = root / episode_id
    patch_sha256 = _sha256_text(patch) if use_patch else None

    # ---- 幂等快速路径：目标存在且身份/指纹校验一致则直接返回 ----
    if target.is_dir():
        sidecar = _read_sidecar(root, episode_id)
        if sidecar is not None and _sidecar_matches_request(
            sidecar, repo, use_patch, base_commit, patch_sha256
        ):
            try:
                current = workspace_tree_fingerprint(target)
            except (OSError, WorkspaceError):
                current = None  # 指纹失败（如部分文件不可读）→ 重建
            if current is not None and (
                sidecar.get("file_count") == current["file_count"]
                and sidecar.get("total_bytes") == current["total_bytes"]
                and sidecar.get("tree_sha256") == current["tree_sha256"]
            ):
                return target
        # sidecar 缺失/身份不符/指纹不一致 → 继续重建覆盖

    root.mkdir(parents=True, exist_ok=True)
    build_tmp = Path(tempfile.mkdtemp(prefix=f"{_BUILD_TMP_PREFIX}{episode_id}-", dir=root))
    patch_tmp: Path | None = None
    try:
        clone_dir = build_tmp / "clone"

        # ---- 步骤 1：获得克隆（缓存模板的本地克隆，或直接 https 克隆）----
        cache_key_used: str | None = None
        template_head: str | None = None
        if clone_cache is not None:
            template, cache_key_used = _acquire_cached_clone(repo, base_commit, Path(clone_cache))
            rc, _, err = _git(["clone", str(template), str(clone_dir)])
            if rc != 0:
                raise CloneError(
                    f"local clone from cache {cache_key_used} failed for {repo} (rc={rc}): {_tail(err)}"
                )
            template_head = _rev_parse(template)
        else:
            rc, _, err = _git(["clone", f"https://github.com/{repo}.git", str(clone_dir)])
            if rc != 0:
                raise CloneError(f"https clone failed for {repo} (rc={rc}): {_tail(err)}")

        # ---- 步骤 2：按任务语义设置克隆状态（对齐 clone_instance）----
        if use_patch:
            head = _rev_parse(clone_dir)
            if template_head is not None and head.lower() != template_head.lower():
                raise CheckoutError(
                    f"episode clone HEAD {head} != cache template HEAD {template_head} for {repo}"
                )
            # patch 经 stdin 传入，不落盘到 workspace_root（git apply 只改工作树）
            rc, _, _apply_err = _git(["apply"], cwd=clone_dir, input_text=patch)
            if rc != 0:
                # stderr 可能引用 patch 上下文（私有标签），不进入异常消息/日志
                raise ApplyError(
                    f"git apply failed for instance {instance_id} (rc={rc}); "
                    "stderr suppressed (may quote private patch content)"
                )
            rc, out, _ = _git(["status", "--porcelain"], cwd=clone_dir)
            if rc != 0:
                raise ApplyError(f"git status failed for instance {instance_id} (rc={rc})")
            if not out.strip():
                raise ApplyError(
                    f"mutation patch for instance {instance_id} produced no working-tree changes"
                )
            # 反向校验用的 patch 副本放系统 tmp（workspace_root 之外），finally 删除
            fd, patch_tmp_name = tempfile.mkstemp(prefix="codescout-mutation-", suffix=".patch")
            patch_tmp = Path(patch_tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(patch)
        else:
            rc, _, err = _git(["checkout", base_commit], cwd=clone_dir)
            if rc != 0:
                raise CheckoutError(
                    f"checkout {base_commit} failed for {repo} (rc={rc}): {_tail(err)}"
                )
            head = _rev_parse(clone_dir)
            if head.lower() != base_commit.lower():
                raise CheckoutError(f"HEAD {head} != base_commit {base_commit} for {repo}")

        # ---- 步骤 3：导出无 .git 的代码树快照（协议 §3 核心要求）----
        export_dir = build_tmp / "export"
        shutil.copytree(
            clone_dir, export_dir, symlinks=True, ignore=shutil.ignore_patterns(".git")
        )

        # ---- 步骤 4：完整性校验 ----
        if (export_dir / ".git").exists():
            raise IntegrityError(f"export for episode {episode_id} unexpectedly contains .git")
        n_clone = _count_tree_entries(clone_dir, skip_git=True)
        n_export = _count_tree_entries(export_dir, skip_git=False)
        if n_clone != n_export:
            raise IntegrityError(
                f"export entry-count mismatch for episode {episode_id}: "
                f"clone {n_clone} vs export {n_export}"
            )
        if use_patch and patch_tmp is not None:
            # 只读校验（--check），不修改快照：反向应用成功 == 导出树包含已应用的 mutation
            rc, _, _rev_err = _git(
                ["apply", "--check", "--reverse", str(patch_tmp)], cwd=export_dir
            )
            if rc != 0:
                raise ApplyError(
                    f"snapshot reverse-apply verification failed for instance {instance_id} "
                    f"(rc={rc}); export does not contain the applied mutation"
                )

        fingerprint = workspace_tree_fingerprint(export_dir)

        # ---- 步骤 5：发布（成功后才替换旧目标，失败保留旧内容）----
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        os.rename(export_dir, target)

        sidecar = {
            "episode_id": episode_id,
            "instance_id": instance_id,
            "repo": repo,
            "use_patch": use_patch,
            "base_commit": base_commit,
            "resolved_head": head,
            "patch_sha256": patch_sha256,
            "clone_cache_key": cache_key_used,
            "file_count": fingerprint["file_count"],
            "total_bytes": fingerprint["total_bytes"],
            "tree_sha256": fingerprint["tree_sha256"],
            "built_utc": _utcnow(),
        }
        _write_sidecar(root, episode_id, sidecar)
        return target
    finally:
        shutil.rmtree(build_tmp, ignore_errors=True)
        if patch_tmp is not None:
            try:
                patch_tmp.unlink()
            except FileNotFoundError:
                pass
