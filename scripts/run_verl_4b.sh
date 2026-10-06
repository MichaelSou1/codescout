#!/usr/bin/env bash
# CodeScout 4B verl 训练启动脚本（configs/verl/codescout_4b.yaml 的运行入口）。
#
# 协议出处：docs/reproduction/protocol-v1.md §5/§6（采样与训练语义）；
# docs/reproduction/verl-adapter-design.md §8.2（启动形态：官方同步 trainer +
# hydra overrides，等价 examples/gspo_trainer/run_qwen3_8b_fsdp.sh 的命令行）。
#
# 用法（服务器，KML 实例；先完成 scripts/project_env.sh 环境验收）：
#   cd /mmu_vlm_hdd/home/rhsu/playground/codescout
#   source scripts/project_env.sh
#   scripts/run_verl_4b.sh \
#       "$CODESCOUT_MODEL_DIR/Qwen3-4B-Instruct-2507" \
#       "$CODESCOUT_DATA/datasets/swe_smith_verl/train.parquet" \
#       "$CODESCOUT_DATA/datasets/swe_smith_verl/validation.parquet" \
#       run-20261007-a \
#       [额外 hydra overrides ...]
#
# 环境变量：
#   ENABLE_WANDB=1             追加 wandb logger（凭据经 WANDB_API_KEY 注入，
#                              不写入本脚本/日志；AGENTS §3 秘密不入库）
#   额外 overrides 直接跟在 run_id 之后（hydra 后者覆盖前者，可用来改
#   trainer.nnodes / rollout.agent.num_workers 等）。
#
# 注意（AGENTS §1/§5）：
#   - 本脚本不在本机执行训练；服务器侧运行须先核验 GPU/驱动/挂载。
#   - 无 set -x（原上游脚本带 set -x，凭据风险 + AGENTS §3 要求）。
#   - PYTHONPATH 导出仓库根：ray worker 需要导入 src.verl_adapter（tools.yaml
#     的 class_name 与 agent_loop.yaml 的 _target_ 都是绝对包路径）。

set -euo pipefail

# --------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------
if [ "$#" -lt 4 ]; then
    echo "usage: $0 MODEL_PATH TRAIN_PARQUET VAL_PARQUET RUN_ID [extra hydra overrides ...]" >&2
    exit 2
fi

MODEL_PATH="$1"
TRAIN_PARQUET="$2"
VAL_PARQUET="$3"
RUN_ID="$4"
shift 4
EXTRA_OVERRIDES=("$@")

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --------------------------------------------------------------------------
# 环境（AGENTS §1/§2：服务器入口脚本；本机仅语法检查）
# --------------------------------------------------------------------------
if [ -f "$REPO_ROOT/scripts/project_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/scripts/project_env.sh"
fi
if [ -z "${CODESCOUT_PYTHON:-}" ]; then
    echo "ERROR: CODESCOUT_PYTHON is not set — source scripts/project_env.sh first (AGENTS §2: 不得用系统 Python 训练)" >&2
    exit 1
fi

for required in "$MODEL_PATH" "$TRAIN_PARQUET" "$VAL_PARQUET"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: path does not exist: $required" >&2
        exit 1
    fi
done

# project_env.sh 导出的是 CODESCOUT_DATA_ROOT；兼容旧名 CODESCOUT_DATA，
# 绝不回退到 $REPO_ROOT/codescout-data（2026-10-07 smoke 实测写进 checkout）。
DATA_ROOT="${CODESCOUT_DATA_ROOT:-${CODESCOUT_DATA:-}}"
if [ -z "$DATA_ROOT" ]; then
    echo "ERROR: CODESCOUT_DATA_ROOT is not set — source scripts/project_env.sh first" >&2
    exit 1
fi
RUN_DIR="$DATA_ROOT/runs/$RUN_ID"
mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/logs" "$RUN_DIR/hydra"

# ray worker 侧可导入 src.verl_adapter（tools.yaml class_name / agent_loop.yaml _target_）
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# tensorboard 落 run 目录（verl tracking.py:557 读 TENSORBOARD_DIR；默认写 checkout）
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"

CONFIG_YAML="$REPO_ROOT/configs/verl/codescout_4b.yaml"

# --------------------------------------------------------------------------
# 读取权威配置清单并替换占位符（yaml 由 $CODESCOUT_PYTHON 解析，PyYAML 随
# omegaconf/verl 依赖可用）
# --------------------------------------------------------------------------
OVERRIDES=$("$CODESCOUT_PYTHON" - "$CONFIG_YAML" \
    "$MODEL_PATH" "$TRAIN_PARQUET" "$VAL_PARQUET" "$RUN_ID" "$REPO_ROOT" "$DATA_ROOT" <<'PYEOF'
import sys
import yaml

config_path, model_path, train_parquet, val_parquet, run_id, repo_root, data_root = sys.argv[1:8]
subs = {
    "${MODEL_PATH}": model_path,
    "${TRAIN_PARQUET}": train_parquet,
    "${VAL_PARQUET}": val_parquet,
    "${RUN_ID}": run_id,
    "${REPO_ROOT}": repo_root,
    "${DATA_ROOT}": data_root,
}
with open(config_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
overrides = cfg["overrides"]
if not isinstance(overrides, list) or not overrides:
    raise SystemExit(f"ERROR: {config_path} must contain a non-empty 'overrides' list")
for raw in overrides:
    line = raw
    for key, value in subs.items():
        line = line.replace(key, value)
    print(line)
PYEOF
)

# --------------------------------------------------------------------------
# WANDB 可选（凭据经环境变量注入，不落盘）
# --------------------------------------------------------------------------
LOGGERS='["console","tensorboard"]'
if [ "${ENABLE_WANDB:-0}" = "1" ]; then
    LOGGERS='["console","tensorboard","wandb"]'
    export WANDB_PROJECT="${WANDB_PROJECT:-codescout}"
fi
OVERRIDES="$OVERRIDES
trainer.logger=$LOGGERS
hydra.run.dir=$RUN_DIR/hydra"

# --------------------------------------------------------------------------
# 启动（verl v0.9.1 同步 trainer 主入口；额外 overrides 追加在后 → hydra 后者生效）
# --------------------------------------------------------------------------
cd "$REPO_ROOT"
# shellcheck disable=SC2086  # OVERRIDES 按行分词是有意的（每行一个 hydra override）
"$CODESCOUT_PYTHON" -m verl.trainer.main_ppo \
    $OVERRIDES \
    "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}" \
    2>&1 | tee "$RUN_DIR/logs/train_$(date -u +%Y%m%dT%H%M%SZ).log"
