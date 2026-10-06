#!/bin/bash
# CodeScout 项目环境入口（服务器端）。
# 约定见 AGENTS.md §2：所有缓存/环境/数据写允许根，不写 HOME/系统 Python/根盘。
# 用法：cd /mmu_vlm_hdd/home/rhsu/playground/codescout && source scripts/project_env.sh

# 允许根（部署记录见 docs/deployment/kml-1005.md）
export CODESCOUT_ROOT="${CODESCOUT_ROOT:-/mmu_vlm_hdd/home/rhsu/playground/codescout}"
export CODESCOUT_DATA_ROOT="${CODESCOUT_DATA_ROOT:-/mmu_vlm_hdd/home/rhsu/playground/codescout-data}"

# 项目 Python（verl 训练/推理环境；入口不存在时由 setup_conda_verl.sh 创建）
export CODESCOUT_PYTHON="${CODESCOUT_PYTHON:-$CODESCOUT_DATA_ROOT/envs/verl-vllm/bin/python}"
# OpenHands 工具执行环境（CPU，步骤3建立；不存在不影响主环境）
export CODESCOUT_OPENHANDS_PYTHON="${CODESCOUT_OPENHANDS_PYTHON:-$CODESCOUT_DATA_ROOT/envs/openhands/bin/python}"

# conda 发行版与项目级 conda 配置（关闭用户环境注册，见 .condarc）
export CONDA_DIST="$CODESCOUT_DATA_ROOT/tools/conda-dist"
export CONDARC="$CODESCOUT_DATA_ROOT/envs/.condarc"
export CONDA_PKGS_DIRS="$CODESCOUT_DATA_ROOT/cache/conda-pkgs"
export CONDA_ENVS_DIRS="$CODESCOUT_DATA_ROOT/envs"
# conda 可执行入口（若已安装）
if [ -x "$CONDA_DIST/bin/conda" ]; then
  export CONDA_EXE="$CONDA_DIST/bin/conda"
  export CONDA_PYTHON_EXE="$CONDA_DIST/bin/python"
  case ":$PATH:" in
    *":$CONDA_DIST/bin:"*) : ;;
    *) export PATH="$CONDA_DIST/bin:$PATH" ;;
  esac
fi

# 缓存与临时目录全部落允许根
export XDG_CACHE_HOME="$CODESCOUT_DATA_ROOT/cache/xdg"
export PIP_CACHE_DIR="$CODESCOUT_DATA_ROOT/cache/pip"
export UV_CACHE_DIR="$CODESCOUT_DATA_ROOT/cache/uv"
export PIP_CONFIG_FILE="${PIP_CONFIG_FILE:-$CODESCOUT_DATA_ROOT/envs/.pip.conf}"  # 由 setup 脚本生成
export HF_HOME="$CODESCOUT_DATA_ROOT/cache/hf"
export HF_HUB_CACHE="$CODESCOUT_DATA_ROOT/cache/hf/hub"
export TORCH_HOME="$CODESCOUT_DATA_ROOT/cache/torch"
export TRITON_CACHE_DIR="$CODESCOUT_DATA_ROOT/cache/triton"
export CUDA_CACHE_PATH="$CODESCOUT_DATA_ROOT/cache/cuda"
export RAY_TMPDIR="$CODESCOUT_DATA_ROOT/tmp/ray"
export RAY_TMPDIR_PATH_UNUSED=1  # 只用 RAY_TMPDIR；防止误写 /tmp/ray
export TMPDIR="$CODESCOUT_DATA_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CODESCOUT_DATA_ROOT/cache/vllm}"
export WANDB_DIR="$CODESCOUT_DATA_ROOT/logs/wandb"
export NODOCKER_LOGS_DIR_UNUSED=1
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$CUDA_CACHE_PATH" "$RAY_TMPDIR" "$TMPDIR" "$WANDB_DIR" 2>/dev/null || true

# 禁用用户 site-packages，杜绝 HOME 混入
export PYTHONNOUSERSITE=1

# 运行时目录（已由部署建立则跳过）
mkdir -p "$CODESCOUT_DATA_ROOT"/{envs,cache,models,datasets,workspaces,sandboxes,runs,logs,tmp,tools,secrets} 2>/dev/null || true

# 主环境入口未就绪时给出显式提示（不静默回退系统 Python）
codescout_env_ready() {
  [ -x "$CODESCOUT_PYTHON" ] || { echo "ERROR: CODESCOUT_PYTHON ($CODESCOUT_PYTHON) 不存在。先运行 scripts/setup_conda_verl.sh。" >&2; return 1; }
}
