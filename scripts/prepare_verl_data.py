#!/usr/bin/env python3
"""准备 CodeScout 训练数据（actor-safe parquet + 私有标签索引）。

语义对齐 src/build_dataset.py（冻结，见 docs/reproduction/protocol-v1.md §2）：
  - 删除空 problem_statement
  - random_state=42 全量打乱（pandas sample 语义）
  - 末 100 条为 validation，其余为 train
  - use_patch=True 语义（SWE-smith mutation patch；base_commit 置 None）

输出：
  <output>/train.parquet        # actor-safe：issue、仓库标识、episode 配置
  <output>/validation.parquet   # 同上（冻结 dev 100 条）
  <output>/labels.jsonl         # 私有：instance_id → file_changes/patch（不进 Git、actor 不可读位置）

用法（服务器）：
  "$CODESCOUT_PYTHON" scripts/prepare_verl_data.py \
      --dataset OpenHands/SWE-smith-py-code-search \
      --revision b1e6864f1c847455ea64c7ebdb4164da2b902bb6 \
      --output /mmu_vlm_hdd/home/rhsu/playground/codescout-data/datasets/swe_smith \
      --labels-output /mmu_vlm_hdd/home/rhsu/playground/codescout-data/datasets/private-labels/swe_smith_labels.jsonl

注意：labels 输出路径必须位于 actor 不可读的私有目录（协议 §8），不要放 Git 内。
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone


def _to_jsonable(obj):
    """pandas/numpy 对象 → 纯 Python JSON 可序列化结构（保序、保字符串）。"""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="OpenHands/SWE-smith-py-code-search")
    ap.add_argument("--revision", required=True, help="HF revision SHA（下载前固定）")
    ap.add_argument("--parquet-file", default=None,
                    help="直接读本地 parquet（离线模式），如 data/swe_smith/train.parquet；与 --dataset 二选一")
    ap.add_argument("--output", required=True, help="输出目录（actor-safe parquet）")
    ap.add_argument("--labels-output", required=True, help="私有标签 jsonl 路径（actor 不可读）")
    ap.add_argument("--validation-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pandas as pd

    started = datetime.now(timezone.utc).isoformat()
    if args.parquet_file:
        df = pd.read_parquet(args.parquet_file)
        source = {"local_parquet": args.parquet_file}
    else:
        from datasets import load_dataset
        ds = load_dataset(args.dataset, split="train", revision=args.revision)
        df = ds.to_pandas()
        source = {"dataset": args.dataset, "revision": args.revision,
                  "num_rows": len(ds), "features": {k: str(v) for k, v in ds.features.items()}}

    n_raw = len(df)
    # 冻结语义：删除空 problem_statement（str.strip 布尔过滤）
    df = df[df["problem_statement"].str.strip().astype(bool)]
    n_kept = len(df)
    dropped = n_raw - n_kept

    # use_patch=True 语义
    df = df.copy()
    df["use_patch"] = True
    df["base_commit"] = None

    # 冻结语义：seed42 打乱，末 validation-size 条为 dev
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    val = df.iloc[-args.validation_size:]
    train = df.iloc[:-args.validation_size]

    # actor-safe 列：只保留任务可见信息 + episode 配置；file_changes/patch 进私有标签
    actor_safe_cols = ["instance_id", "repo", "base_commit", "problem_statement", "use_patch"]
    private_cols = ["file_changes", "patch"]

    missing = [c for c in actor_safe_cols + private_cols if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: dataset missing columns: {missing}; available: {list(df.columns)}")

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.dirname(args.labels_output), exist_ok=True)

    def emit(frame, name):
        out = os.path.join(args.output, f"{name}.parquet")
        safe = frame[actor_safe_cols].reset_index(drop=True)
        safe.to_parquet(out)
        return out, len(safe)

    train_path, n_train = emit(train, "train")
    val_path, n_val = emit(val, "validation")

    with open(args.labels_output, "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(_to_jsonable({
                "instance_id": row["instance_id"],
                "file_changes": row["file_changes"],
                "patch": row["patch"],
            }), ensure_ascii=False) + "\n")

    # manifest 摘要（可入 Git 的小文件）
    manifest = {
        "created_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "semantics": "src/build_dataset.py: drop empty problem_statement, sample(frac=1, random_state=42), last 100 = validation",
        "rows": {"raw": int(n_raw), "kept": int(n_kept), "dropped_empty_issue": int(dropped),
                 "train": int(n_train), "validation": int(n_val)},
        "seed": args.seed,
        "outputs": {"train": train_path, "validation": val_path, "labels": args.labels_output},
        "labels_sha256": hashlib.sha256(open(args.labels_output, "rb").read()).hexdigest(),
        "instance_id_overlap_train_val": sorted(set(train["instance_id"]) & set(val["instance_id"])),
    }
    manifest_path = os.path.join(args.output, "prepare-manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    if manifest["instance_id_overlap_train_val"]:
        sys.exit("ERROR: train/validation instance_id overlap — 不得继续，先审计划分")
    print("OK")


if __name__ == "__main__":
    main()
