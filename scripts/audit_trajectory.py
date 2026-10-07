#!/usr/bin/env python3
"""轨迹/训练语义审计（todo.md 步骤 4 的 audit_trajectory.py）。

输入一个或多个 run 的审计产物（reward fn 逐任务 JSONL + verl trainer 日志），
输出该 run 的训练/评测语义摘要 JSON：轮数与 token 统计、failure_class 分列、
全同 reward（GRPO advantage 全零）组比例、轮数耗尽（整条 mask=0，A-1）比例、
有效预测比例、三级 F1/P-R 宏平均。摘要入 Git；完整输入保存在 run 目录（协议约定）。

用法（服务器或本机，纯标准库）：
  python3 scripts/audit_trajectory.py --audit runs/cs4b-test-base-verified/reward_audit_filtered.jsonl \
      [--log logs/rl_s17.log --batch-prompts 8 --rollout-n 8] \
      --output results/experiments/.../audit-<name>.json

--log 提供时额外提取训练语义指标（全同组比例需 reward fn 审计行 + 分组重构；
verl 日志行含 critic/advantages 与训练 reward 序列）。
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def audit_reward_jsonl(path: str) -> dict:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by_inst: dict[str, dict] = {}
    for r in rows:
        by_inst[r.get("instance_id")] = r  # 同实例取最后一条（epoch/重试覆盖）
    tasks = list(by_inst.values())
    n = len(tasks)

    def macro(key: str) -> float:
        return sum((t.get(key) or 0.0) for t in tasks) / n if n else 0.0

    fc = Counter(t.get("failure_class") for t in tasks)
    exhausted = sum(1 for t in tasks if t.get("trajectory_exhausted"))
    no_finish = fc.get("no_finish", 0)
    infra = fc.get("infra", 0)
    valid = sum(v for k, v in fc.items() if k is None)
    multi_finish = fc.get("multi_finish", 0)
    parse_err = fc.get("parse_error", 0)
    sanity = fc.get("sanity_check", 0)
    turns = [t.get("num_turns") for t in tasks if isinstance(t.get("num_turns"), (int, float))]
    out = {
        "audit_source": path,
        "num_tasks": n,
        "macro": {
            "sum_f1": macro("score"),
            "file_f1": macro("file_reward"),
            "module_f1": macro("module_reward"),
            "entity_f1": macro("entity_reward"),
            "file_precision": macro("file_precision"),
            "file_recall": macro("file_recall"),
            "module_precision": macro("module_precision"),
            "entity_precision": macro("entity_precision"),
            "num_turns": sum(turns) / len(turns) if turns else None,
        },
        "failure_class": {
            "valid_prediction": valid,
            "no_finish": no_finish,
            "multi_finish": multi_finish,
            "parse_error": parse_err,
            "sanity_check": sanity,
            "infra": infra,
        },
        "exhausted_turns_masked_out": exhausted,
        "turns_min": min(turns) if turns else None,
        "turns_max": max(turns) if turns else None,
    }
    return out


def audit_trainer_log(path: str, batch_prompts: int, rollout_n: int) -> dict:
    """训练日志语义审计：逐步 reward 序列 → 全同 reward 组比例（GRPO advantage 全零组的下界）。

    verl 同步 trainer 每 step 输出一次 critic/score/mean；逐步 mean 无法重构组内方差，
    因此这里提取逐步 reward mean/min/max 序列与 update 计数；组级全同比例需
    rollout 轨迹文件（若 run 保存了 per-episode JSONL 才能精确算）——日志层给出
    score 序列与 step 数一致性。
    """
    scores = []
    steps = []
    adv_max = []
    adv_min = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"critic/score/mean:([0-9.eE+-]+)", line)
            am = re.search(r"critic/advantages/max:([0-9.eE+-]+)", line)
            an = re.search(r"critic/advantages/min:([0-9.eE+-]+)", line)
            s = re.search(r"step:(\d+) -", line)
            if s:
                steps.append(int(s.group(1)))
            if m:
                scores.append(float(m.group(1)))
            if am:
                adv_max.append(float(am.group(1)))
            if an:
                adv_min.append(float(an.group(1)))
    zero_adv_steps = sum(1 for a, b in zip(adv_max, adv_min) if a == 0.0 and b == 0.0)
    nonzero = [(a, b) for a, b in zip(adv_max, adv_min) if not (a == 0.0 and b == 0.0)]
    out = {
        "log_source": path,
        "num_step_lines": len(set(steps)),
        "max_step": max(steps) if steps else None,
        "num_score_means": len(scores),
        "score_mean_first5": scores[:5],
        "score_mean_last5": scores[-5:],
        "grpo_zero_advantage_steps": zero_adv_steps,
        "grpo_nonzero_advantage_steps": len(nonzero),
        "grpo_zero_advantage_step_ratio": (
            zero_adv_steps / (zero_adv_steps + len(nonzero)) if (zero_adv_steps + len(nonzero)) else None
        ),
        "note": (
            "全同 reward 组比例的 step 级下界：advantages max==min==0 的 step 即该 step "
            "所有 GRPO 组全同（advantage 全零）。组级精确比例需逐 episode 审计（训练时未开启 "
            "CODESCOUT_REWARD_AUDIT_PATH，此为已知数据缺口，如实记录）。"
        ),
    }
    return out


def group_analysis(path: str, batch_prompts: int, rollout_n: int) -> dict:
    """组级审计：从训练 run 的 reward 审计 JSONL 重构 GRPO 组（按 extra_info index 分组）。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if rows and rows[0].get("index") is None:
        return {"group_analysis": "skipped: audit rows lack group index"}
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("index")].append(float(r.get("score") or 0.0))
    n_groups = len(groups)
    const_groups = sum(1 for v in groups.values() if len(set(v)) == 1)
    bad_size = sum(1 for v in groups.values() if len(v) != rollout_n)
    return {
        "num_groups": n_groups,
        "expected_group_size": rollout_n,
        "wrong_size_groups": bad_size,
        "constant_reward_groups": const_groups,
        "constant_reward_ratio": (const_groups / n_groups) if n_groups else None,
        "note": "advantage 全零组 = 全同 reward 组（GRPO 中心化，norm_by_std=False）",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--log", default=None)
    ap.add_argument("--grouped-audit", default=None,
                    help="训练 run 的含 index 审计 JSONL（组级全同 reward 分析）")
    ap.add_argument("--batch-prompts", type=int, default=8)
    ap.add_argument("--rollout-n", type=int, default=8)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = {"generated": None, "audits": {}}
    import datetime

    out["generated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for spec in args.audit:
        name, path = spec.split("=", 1)
        out["audits"][name] = audit_reward_jsonl(path)
    if args.log:
        out["trainer_log"] = audit_trainer_log(args.log, args.batch_prompts, args.rollout_n)
    if args.grouped_audit:
        out["group_analysis"] = group_analysis(args.grouped_audit, args.batch_prompts, args.rollout_n)

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"wrote {dest}")
    for name, a in out["audits"].items():
        print(f"[{name}] tasks={a['num_tasks']} sum_f1={a['macro']['sum_f1']:.4f} "
              f"failure={a['failure_class']}")


if __name__ == "__main__":
    main()
