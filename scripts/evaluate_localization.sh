#!/usr/bin/env bash
# CodeScout 定位评测入口（步骤 5/8）：模型 + split + 解码配置 → 逐任务结构预测指标。
#
# 实现方式：verl v0.9.1 无独立 rollout-only 入口（run_agent_generate 已移除），
# 用官方 trainer 的 validation 流程承载评测：val_before_train=True 先跑全量 val
# rollout 再做 1 个无意义的最小 update（train_files 指向 8 行占位集）后退出。
# 逐任务指标经 reward fn 的审计通道输出：
#   CODESCOUT_REWARD_AUDIT_PATH=<run>/reward_audit.jsonl（src/verl_adapter/reward.py
#   逐条 append：instance_id、三级 F1/P-R、failure_class、轮数）。
#
# 协议出处：docs/reproduction/protocol-v1.md §5（正式评测温度 0.7/top_k20/top_p0.8、
# 上下文 132K、每任务单 episode、actor-only）。
#
# 用法（服务器）：
#   scripts/evaluate_localization.sh <MODEL_PATH> <VAL_PARQUET> <RUN_ID> \
#       [额外 hydra overrides...]
# 输出：$CODESCOUT_DATA_ROOT/runs/<RUN_ID>/reward_audit.jsonl + summary.json

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 MODEL_PATH VAL_PARQUET RUN_ID [extra hydra overrides ...]" >&2
    exit 2
fi

MODEL_PATH="$1"
VAL_PARQUET="$2"
RUN_ID="$3"
shift 3

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/project_env.sh"
DATA_ROOT="${CODESCOUT_DATA_ROOT:?CODESCOUT_DATA_ROOT not set}"
RUN_DIR="$DATA_ROOT/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

export CODESCOUT_REWARD_AUDIT_PATH="$RUN_DIR/reward_audit.jsonl"
rm -f "$CODESCOUT_REWARD_AUDIT_PATH"

# 8 行占位训练集（避免 val 前训练；profile_train 的前 8 行即可）
PLACEHOLDER_TRAIN="${PLACEHOLDER_TRAIN:-$DATA_ROOT/datasets/swe_smith_prepared/smoke_train_rl.parquet}"

# 正式评测参数（协议 §5）：132K 窗口、单 episode、温度 0.7/top_k20/top_p0.8
# （val_kwargs 已在权威配置中）。132K 窗口：prompt 上限维持 40960（任务 prompt 实测
# ~2.8K），episode 累计窗口放大到 132K。
bash "$REPO_ROOT/scripts/run_verl_4b.sh" \
    "$MODEL_PATH" \
    "$PLACEHOLDER_TRAIN" \
    "$VAL_PARQUET" \
    "$RUN_ID" \
    trainer.val_before_train=True \
    trainer.total_training_steps=1 \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    "actor_rollout_ref.rollout.max_model_len=131072" \
    "actor_rollout_ref.rollout.response_length=98304" \
    "$@"

# ---- 汇总：从逐任务审计计算宏平均指标（过滤占位训练任务，只统计被评测集） ----
"$CODESCOUT_PYTHON" "$REPO_ROOT/scripts/eval_summary.py" \
    --audit "$CODESCOUT_REWARD_AUDIT_PATH" --output "$RUN_DIR/summary.json" \
    --filter-parquet "$VAL_PARQUET"

echo
echo "== 评测汇总（$RUN_ID）=="
cat "$RUN_DIR/summary.json"
