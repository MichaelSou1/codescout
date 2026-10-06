#!/usr/bin/env python3
"""汇总 reward 审计 JSONL → 评测 summary.json（步骤 5/8 的指标口径）。

口径（docs/reproduction/protocol-v1.md §7）：
- 主指标：每任务 file_F1 + module_F1 + entity_F1 之和的宏平均（0–3）；
  三级 F1 分别宏平均另报。
- 分母：出现的 instance_id 数（去重；同 instance 多条 = infra 重复，报 warning）。
- failure_class 分列；infra 不计入模型 0 分（协议 §4），但摘要中单列并保留。
- precision/recall 宏平均同表（P/R 只在有非空预测的任务上有意义，另报非空预测口径）。
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True, help="reward fn 审计 JSONL 路径")
    ap.add_argument("--output", required=True, help="输出 summary.json 路径")
    ap.add_argument("--filter-parquet", default=None,
                    help="可选：只统计该 parquet（actor-safe 或 RL parquet）中出现"
                         "的 instance_id——评测 run 的审计会混入占位训练任务，必须过滤")
    args = ap.parse_args()

    rows = []
    with open(args.audit, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if args.filter_parquet:
        import pandas as pd

        df = pd.read_parquet(args.filter_parquet)
        col = "extra_info" if "extra_info" in df.columns else None
        if col is not None:
            keep = {e["instance_id"] for e in df[col]}
        elif "instance_id" in df.columns:
            keep = set(df["instance_id"])
        else:
            raise SystemExit(f"ERROR: {args.filter_parquet} has neither extra_info nor instance_id")
        rows = [r for r in rows if r.get("instance_id") in keep]

    n_lines = len(rows)
    by_inst: dict[str, list[dict]] = {}
    for r in rows:
        by_inst.setdefault(r.get("instance_id"), []).append(r)
    dup = {k: len(v) for k, v in by_inst.items() if len(v) > 1}

    # 每 instance 取最后一条（重复时前条视为重试残留）
    tasks = [v[-1] for v in by_inst.values()]
    n = len(tasks)

    def macro(key: str) -> float:
        return sum(t.get(key) or 0.0 for t in tasks) / n if n else 0.0

    fc_counter = Counter(t.get("failure_class") for t in tasks)
    with_pred = [t for t in tasks if t.get("failure_class") is None]

    summary = {
        "num_audit_lines": n_lines,
        "num_tasks": n,
        "duplicate_instance_warning": dup,
        "macro": {
            "sum_f1": macro("score"),
            "file_f1": macro("file_reward"),
            "module_f1": macro("module_reward"),
            "entity_f1": macro("entity_reward"),
            "file_precision": macro("file_precision"),
            "file_recall": macro("file_recall"),
            "module_precision": macro("module_precision"),
            "module_recall": macro("module_recall"),
            "entity_precision": macro("entity_precision"),
            "entity_recall": macro("entity_recall"),
            "num_turns": macro("num_turns"),
        },
        "failure_class_counts": dict(fc_counter),
        "valid_prediction_tasks": len(with_pred),
        "note": "score range 0-3 (file+module+entity F1, weights 1/1/1); infra rows listed separately",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["macro"], indent=2))


if __name__ == "__main__":
    main()
