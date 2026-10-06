#!/usr/bin/env python3
"""workspace 构建冒烟测试：随机抽样任务，逐个 build_workspace 并审计协议约束。

协议出处：docs/reproduction/protocol-v1.md §3（workspace 路径、无 ``.git`` 快照）、
§8（私有标签隔离）。**在服务器 verl 环境运行**（需 pandas/pyarrow）；本机只做
``python3 -m py_compile`` 语法检查，不执行真实构建。

输入（均为 prepare_verl_data.py 的产物）：
  --instances-parquet  actor-safe parquet：instance_id/repo/base_commit/problem_statement/use_patch
  --labels-jsonl       私有标签：instance_id/file_changes/patch。只能在 actor 不可读的
                       私有位置运行本脚本；patch 全文不进入报告与 stdout。
  --workspace-root     workspace 根目录（如 codescout-data/workspaces/smoke-<date>）
  --count N            随机抽样的任务数（默认 20）
  --seed               抽样随机种子（与 RL 训练 seed 无关，仅决定抽样）
  --clone-cache        可选克隆缓存目录（复用 repo+commit 克隆模板）
  --output             JSON 报告输出路径（小结果，可入 results/）

每个任务记录：成功/失败及错误分类（CloneError/CheckoutError/ApplyError/
IntegrityError/MissingPrivateLabel/Unexpected.*）、耗时、workspace 体积与树指纹、
workspace 内是否残留 ``.git``（必须为否）、文件名含 patch/gold 的泄漏嫌疑清单
（信息性：真实仓库可能合法包含此类文件名，需人工核对是否为仓库原生文件）。

退出码：全部成功且无 ``.git`` 残留为 0，否则 1；泄漏嫌疑为信息项，不影响退出码。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.verl_adapter.workspace import (  # noqa: E402
    WorkspaceError,
    build_workspace,
    workspace_tree_fingerprint,
)

_PROTOCOL_REF = "docs/reproduction/protocol-v1.md §3/§8"
_FINGERPRINT_KEYS = ("file_count", "total_bytes", "tree_sha256")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--labels-jsonl", required=True, help="私有标签 jsonl（actor 不可读位置）")
    ap.add_argument("--instances-parquet", required=True, help="actor-safe parquet")
    ap.add_argument("--workspace-root", required=True, help="workspace 根目录")
    ap.add_argument("--count", type=int, default=20, help="随机抽样的任务数（默认 20）")
    ap.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    ap.add_argument("--clone-cache", default=None, help="可选克隆缓存目录")
    ap.add_argument("--output", required=True, help="JSON 报告输出路径")
    return ap.parse_args(argv)


def _safe_message(exc: BaseException, limit: int = 400) -> str:
    """异常消息净化：压平换行并截断；workspace 模块已保证不泄漏 patch 内容。"""
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[:limit] + "..."


def _load_labels(path: Path) -> dict[str, dict[str, Any]]:
    """读私有标签 jsonl → {instance_id: record}。不做任何打印，避免标签进入 stdout。"""
    labels: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"ERROR: labels jsonl line {lineno} invalid JSON: {exc}")
            iid = record.get("instance_id")
            if not iid:
                sys.exit(f"ERROR: labels jsonl line {lineno} missing instance_id")
            labels[str(iid)] = record
    return labels


def _code_commit() -> str | None:
    """记录当前代码 commit，便于实验账本追溯；失败返回 None。"""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), capture_output=True, text=True, check=False
        )
        return proc.stdout.strip() if proc.returncode == 0 else None
    except OSError:
        return None


def _clean_base_commit(raw: Any) -> str | None:
    """parquet 的 base_commit 列可能是 None/NaN/str；统一为 str | None。"""
    try:
        import pandas as pd

        if pd.isna(raw):
            return None
    except (ImportError, TypeError, ValueError):
        if raw is None:
            return None
    text = str(raw).strip()
    return text or None


def _leak_suspect_names(workspace: Path) -> list[str]:
    """workspace 内文件名含 patch/gold 的去重清单（信息性泄漏检查）。"""
    suspects = {
        path.name
        for path in workspace.rglob("*")
        if ("patch" in path.name.lower() or "gold" in path.name.lower())
    }
    return sorted(suspects)


def main() -> int:
    args = _parse_args()

    try:
        import pandas as pd  # 服务器 verl 环境延迟导入；本机不执行本脚本
    except ImportError as exc:
        sys.exit(f"ERROR: pandas 不可用（本脚本应在服务器 verl 环境运行）: {exc}")

    started_utc = datetime.now(timezone.utc).isoformat()

    frame = pd.read_parquet(args.instances_parquet)
    required_cols = ["instance_id", "repo", "base_commit", "problem_statement", "use_patch"]
    missing = [c for c in required_cols if c not in frame.columns]
    if missing:
        sys.exit(f"ERROR: parquet missing columns {missing}; available: {list(frame.columns)}")

    labels = _load_labels(Path(args.labels_jsonl))

    total_rows = len(frame)
    indices = list(range(total_rows))
    random.Random(args.seed).shuffle(indices)
    selected = indices[: args.count]

    results: list[dict[str, Any]] = []
    for pos, row_idx in enumerate(selected, 1):
        row = frame.iloc[row_idx]
        instance_id = str(row["instance_id"])
        use_patch = bool(row["use_patch"])
        repo = str(row["repo"])
        workspace_path = Path(args.workspace_root) / instance_id
        already_present = workspace_path.exists()

        record: dict[str, Any] = {
            "instance_id": instance_id,
            "repo": repo,
            "use_patch": use_patch,
            "already_present_before": already_present,
            "status": None,
            "error_class": None,
            "error_message": None,
            "duration_sec": None,
            "workspace": None,
            "fingerprint": None,
            "dot_git_present": None,
            "leak_suspect_files": None,
        }

        if use_patch and instance_id not in labels:
            record["status"] = "error"
            record["error_class"] = "MissingPrivateLabel"
            record["error_message"] = "use_patch=True but instance_id not found in labels jsonl"
            results.append(record)
            print(f"[{pos}/{len(selected)}] {instance_id}: ERROR MissingPrivateLabel")
            continue

        instance = {
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": _clean_base_commit(row["base_commit"]),
            "use_patch": use_patch,
        }
        if instance_id in labels:
            # gold/mutation patch 是私有标签：只传给构建器，不打印、不写入报告
            instance["patch"] = labels[instance_id].get("patch")

        t0 = time.perf_counter()
        try:
            ws = build_workspace(
                instance, args.workspace_root, instance_id, clone_cache=args.clone_cache
            )
            duration = time.perf_counter() - t0
            fingerprint = workspace_tree_fingerprint(ws)
            record.update(
                {
                    "status": "ok",
                    "duration_sec": round(duration, 3),
                    "workspace": str(ws),
                    "fingerprint": {k: fingerprint[k] for k in _FINGERPRINT_KEYS},
                    "dot_git_present": (ws / ".git").exists(),
                    "leak_suspect_files": _leak_suspect_names(ws),
                }
            )
            mib = fingerprint["total_bytes"] / (1024 * 1024)
            reused = " (idempotent reuse)" if already_present else ""
            print(
                f"[{pos}/{len(selected)}] {instance_id}: ok{reused} "
                f"({duration:.1f}s, {fingerprint['file_count']} files, {mib:.1f} MiB)"
            )
        except WorkspaceError as exc:
            duration = time.perf_counter() - t0
            record.update(
                {
                    "status": "error",
                    "error_class": type(exc).__name__,
                    "error_message": _safe_message(exc),
                    "duration_sec": round(duration, 3),
                }
            )
            print(f"[{pos}/{len(selected)}] {instance_id}: ERROR {type(exc).__name__}")
        except Exception as exc:  # noqa: BLE001 —— 冒烟必须记录意外失败而不是中断
            duration = time.perf_counter() - t0
            record.update(
                {
                    "status": "error",
                    "error_class": f"Unexpected.{type(exc).__name__}",
                    "error_message": _safe_message(exc),
                    "duration_sec": round(duration, 3),
                }
            )
            print(f"[{pos}/{len(selected)}] {instance_id}: ERROR Unexpected.{type(exc).__name__}")
        results.append(record)

    # ---- 汇总 ----
    ok_records = [r for r in results if r["status"] == "ok"]
    failed_records = [r for r in results if r["status"] == "error"]
    failure_by_class: dict[str, int] = {}
    for r in failed_records:
        failure_by_class[r["error_class"]] = failure_by_class.get(r["error_class"], 0) + 1
    durations = [r["duration_sec"] for r in ok_records if r["duration_sec"] is not None]
    dot_git_violations = [
        r["instance_id"] for r in ok_records if r.get("dot_git_present")
    ]
    leak_tasks = {
        r["instance_id"]: len(r["leak_suspect_files"])
        for r in ok_records
        if r.get("leak_suspect_files")
    }
    reused = sum(1 for r in ok_records if r["already_present_before"])
    total_bytes = sum(
        r["fingerprint"]["total_bytes"] for r in ok_records if r.get("fingerprint")
    )
    total_files = sum(
        r["fingerprint"]["file_count"] for r in ok_records if r.get("fingerprint")
    )

    root_path = Path(args.workspace_root)
    leftover_build_dirs = (
        sorted(p.name for p in root_path.iterdir() if p.name.startswith(".build-"))
        if root_path.is_dir()
        else []
    )

    summary = {
        "requested_count": args.count,
        "attempted": len(results),
        "ok": len(ok_records),
        "failed": len(failed_records),
        "failure_by_class": failure_by_class,
        "reused_existing": reused,
        "duration_sec": {
            "mean": round(statistics.fmean(durations), 3) if durations else None,
            "median": round(statistics.median(durations), 3) if durations else None,
            "max": round(max(durations), 3) if durations else None,
        },
        "workspace_bytes_total": total_bytes,
        "workspace_files_total": total_files,
        "dot_git_violations": dot_git_violations,
        "leak_suspect_tasks": leak_tasks,
        "leftover_build_dirs": leftover_build_dirs,
    }
    report = {
        "meta": {
            "created_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "timezone_note": "本机/服务器操作时区按 AGENTS.md 记 Asia/Shanghai；报告时间为 UTC",
            "protocol_ref": _PROTOCOL_REF,
            "code_commit": _code_commit(),
            "python_version": sys.version.split()[0],
            "seed": args.seed,
            "count": args.count,
            "labels_path": str(Path(args.labels_jsonl).resolve()),
            "instances_parquet": str(Path(args.instances_parquet).resolve()),
            "workspace_root": str(root_path.resolve()),
            "clone_cache": str(Path(args.clone_cache).resolve()) if args.clone_cache else None,
            "parquet_rows_total": total_rows,
        },
        "results": results,
        "summary": summary,
    }

    output_path = Path(args.output)
    if output_path.parent != Path(""):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- 人类可读摘要（不含任何私有标签内容）----
    print()
    print("==== smoke_workspace 摘要 ====")
    print(
        f"任务: 请求 {args.count}, 尝试 {len(results)}, 成功 {len(ok_records)}, "
        f"失败 {len(failed_records)}"
    )
    if failure_by_class:
        print(f"失败分类: {json.dumps(failure_by_class, ensure_ascii=False)}")
    if durations:
        print(
            f"成功任务耗时: mean {summary['duration_sec']['mean']}s / "
            f"median {summary['duration_sec']['median']}s / max {summary['duration_sec']['max']}s"
        )
    print(
        f"workspace 总体积: {total_bytes / (1024 * 1024):.1f} MiB, 文件总数: {total_files}, "
        f"幂等复用: {reused}"
    )
    if dot_git_violations:
        print(f"违规: 以下 workspace 残留 .git（协议 §3 禁止）: {dot_git_violations}")
    else:
        print(".git 残留: 0（符合协议 §3）")
    if leak_tasks:
        print(
            f"泄漏嫌疑（文件名含 patch/gold，信息性，需人工核对是否为仓库原生文件）: "
            f"{json.dumps(leak_tasks, ensure_ascii=False)}"
        )
    else:
        print("泄漏嫌疑文件名: 0")
    if leftover_build_dirs:
        print(f"警告: 残留构建临时目录（应人工检查清理）: {leftover_build_dirs}")
    print(f"报告: {output_path}")

    clean = not failed_records and not dot_git_violations
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
