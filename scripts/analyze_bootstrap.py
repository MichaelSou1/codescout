#!/usr/bin/env python3
"""最终评测的统计分析（步骤 8）：任务配对 bootstrap + repo 聚类敏感性。

口径（docs/reproduction/protocol-v1.md §7/§8；todo.md 步骤 8）：
- 主指标：每任务 sum_F1（file+module+entity，0–3）；
- 差值：RL(seed) − base 的任务配对差（同任务同分母）；两 seed 分别 + 平均；
- 任务配对 bootstrap：对任务差值重采样（10000 次、固定 bootstrap seed）→ 95% CI；
- repo 聚类 bootstrap：按 repo 分组重采样（仓库少时明确不确定性标注）；
- 收益门槛（预设）：宏平均提升 ≥0.10，两 seed 方向同为正，平均差值 CI 下界>0；
- infra 任务按冻结规则剔除（failure_class=="infra"），其余模型失败计 0（协议 §8）。

输入：多个模型的 reward 审计 JSONL（--audit name=path ...）。
输出：JSON 摘要（stdout + --output）。

纯标准库；bootstrap 10000×500 任务约数秒。
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_scores(path: str) -> dict[str, float]:
    """audit JSONL → {instance_id: sum_F1}（最后一条为准；infra 剔除）。"""
    scores: dict[str, float] = {}
    infra: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            iid = r.get("instance_id")
            if iid is None:
                continue
            if r.get("failure_class") == "infra":
                infra.add(iid)
                continue
            scores[iid] = float(r.get("score") or 0.0)
    for iid in infra:
        scores.pop(iid, None)
    return scores


def paired_diffs(a: dict[str, float], b: dict[str, float]):
    """共同任务上的配对差 a-b。分母对齐：以两模型共同任务为准（评测分母冻结后调用）。"""
    common = sorted(set(a) & set(b))
    if len(common) != len(a) or len(common) != len(b):
        print(f"WARNING: denominator mismatch: a={len(a)} b={len(b)} common={len(common)}")
    return common, [a[i] - b[i] for i in common]


def bootstrap_ci(diffs: list[float], n_boot: int, seed: int, alpha: float = 0.05):
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return sum(diffs) / n, lo, hi


def repo_bootstrap_ci(diffs_by_repo: dict[str, list[float]], n_boot: int, seed: int, alpha: float = 0.05):
    """repo 聚类 bootstrap：整仓重采样（仓库数少时 CI 宽——明确报告仓库数）。"""
    rng = random.Random(seed)
    repos = sorted(diffs_by_repo)
    repo_means = [sum(diffs_by_repo[r]) / len(diffs_by_repo[r]) for r in repos]
    n = len(repos)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        picked = [repo_means[rng.randrange(n)] for _ in range(n)]
        means.append(sum(picked) / n)
    means.sort()
    overall = sum(sum(v) for v in diffs_by_repo.values()) / sum(len(v) for v in diffs_by_repo.values())
    return overall, means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="append", required=True, metavar="NAME=PATH",
                    help="模型审计 JSONL（如 base=.../audit.jsonl rl_s17=...）")
    ap.add_argument("--baseline", default="base", help="基线名称（默认 base）")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261007)
    ap.add_argument("--output", default=None)
    ap.add_argument("--repo-map", default=None,
                    help="可选 JSONL：instance_id→repo（审计行没有 repo 时用于聚类）")
    args = ap.parse_args()

    audits: dict[str, dict[str, float]] = {}
    for spec in args.audit:
        name, path = spec.split("=", 1)
        audits[name] = load_scores(path)

    repo_of: dict[str, str] = {}
    if args.repo_map:
        with open(args.repo_map, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    repo_of[r["instance_id"]] = r.get("repo") or r.get("extra_info", {}).get("repo")

    base = audits[args.baseline]
    seeds = [k for k in audits if k != args.baseline]
    out: dict = {
        "n_boot": args.n_boot,
        "bootstrap_seed": args.bootstrap_seed,
        "macro": {},
        "threshold": {"sum_f1_gain": 0.10, "both_seeds_positive": None, "avg_ci_lower_gt_0": None},
    }
    avg_seed_diffs: list[float] = []
    all_positive = True
    for name in seeds:
        tasks, diffs = paired_diffs(audits[name], base)
        mean, lo, hi = bootstrap_ci(diffs, args.n_boot, args.bootstrap_seed)
        out["macro"][name] = {
            "n_tasks": len(tasks),
            "mean_diff": round(mean, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        }
        if repo_of:
            by_repo: dict[str, list[float]] = defaultdict(list)
            for t, d in zip(tasks, diffs):
                by_repo[repo_of.get(t, "?")].append(d)
            rmean, rlo, rhi = repo_bootstrap_ci(by_repo, args.n_boot, args.bootstrap_seed)
            out["macro"][name]["repo_cluster"] = {
                "n_repos": len(by_repo),
                "mean_diff": round(rmean, 4),
                "ci95": [round(rlo, 4), round(rhi, 4)],
                "note": "repo-cluster bootstrap; few repos → wide CI is expected",
            }
        avg_seed_diffs.append((name, tasks, diffs))
        all_positive = all_positive and mean > 0

    # 两 seed 平均差值（任务配对）+ bootstrap
    if len(avg_seed_diffs) >= 2:
        common = set(avg_seed_diffs[0][1])
        for _, tasks, _ in avg_seed_diffs[1:]:
            common &= set(tasks)
        common = sorted(common)
        idx = {t: i for i, t in enumerate(avg_seed_diffs[0][1])}
        avg = []
        for t in common:
            vals = []
            for _, tasks, diffs in avg_seed_diffs:
                j = {tt: ii for ii, tt in enumerate(tasks)}[t]
                vals.append(diffs[j])
            avg.append(sum(vals) / len(vals))
        mean, lo, hi = bootstrap_ci(avg, args.n_boot, args.bootstrap_seed)
        out["macro"]["avg_seeds"] = {"n_tasks": len(common), "mean_diff": round(mean, 4),
                                     "ci95": [round(lo, 4), round(hi, 4)]}
        out["threshold"]["avg_ci_lower_gt_0"] = lo > 0
        out["threshold"]["gain_ge_0.10"] = mean >= 0.10
    out["threshold"]["both_seeds_positive"] = all_positive

    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
