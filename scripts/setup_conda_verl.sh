#!/bin/bash
# 建立 CodeScout 隔离 conda + verl v0.9.1 环境（kml-1005 允许根内）。
# 用法：bash scripts/setup_conda_verl.sh [--skip-conda] [--skip-deps]
# 依赖：scripts/project_env.sh；服务器 PATH 中有 uv/git/curl。
# 组合来源：verl v0.9.1 官方 pyproject/uv.lock（torch 2.11.0 cu130 / vllm 0.24.0 /
#   transformers 5.9.0 / flash-attn 2.8.3），见 docs/reproduction/skyrl-to-verl.md §5。
set -euo pipefail

VERL_TAG="v0.9.1"
VERL_REPO="https://github.com/volcengine/verl.git"
# verl 声明 requires-python >=3.10,<3.13，但 v0.9.1 官方 uv.lock 的 GPU 后端
# wheel（torch/vllm/flash-attn 等）marker 均为 python_full_version >= '3.12'——
# 3.11 会让整组被跳过。故取 3.12（2026-10-07 实测决策，入账 cs4b-env-a01）。
PYTHON_VERSION="${CODESCOUT_PYTHON_VERSION:-3.12}"
MINIFORGE_VERSION="26.7.2-0"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"
MINIFORGE_SHA_URL="${MINIFORGE_URL}.sha256"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/project_env.sh"

SKIP_CONDA=0
SKIP_DEPS=0
for arg in "$@"; do
  case "$arg" in
    --skip-conda) SKIP_CONDA=1 ;;
    --skip-deps) SKIP_DEPS=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

CACHE="$CODESCOUT_DATA_ROOT/cache"
SRC_DIR="$CODESCOUT_DATA_ROOT/tools/src"
PREFIX="$CODESCOUT_DATA_ROOT/envs/verl-vllm"
mkdir -p "$CACHE" "$SRC_DIR" "$CODESCOUT_DATA_ROOT/envs"

step() { echo; echo "=== [setup_conda_verl] $* ==="; }

# ---------------------------------------------------------------- 1. conda 发行版
if [ "$SKIP_CONDA" -eq 0 ] && [ ! -x "$CONDA_DIST/bin/conda" ]; then
  step "下载并校验 Miniforge3 ${MINIFORGE_VERSION}"
  INSTALLER="$CACHE/miniforge/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"
  mkdir -p "$(dirname "$INSTALLER")"
  if [ ! -f "$INSTALLER" ]; then
    curl -fL --retry 3 -C - -o "$INSTALLER" "$MINIFORGE_URL"
  fi
  EXPECTED=$(curl -fsSL "$MINIFORGE_SHA_URL" | awk '{print $1}')
  ACTUAL=$(sha256sum "$INSTALLER" | awk '{print $1}')
  if [ "$EXPECTED" != "$ACTUAL" ]; then
    echo "installer sha256 mismatch: expected=$EXPECTED actual=$ACTUAL" >&2
    exit 1
  fi
  step "安装 conda 发行版到 $CONDA_DIST"
  bash "$INSTALLER" -p "$CONDA_DIST" -b -f
fi

# 项目级 conda/pip 配置（关闭用户注册；不写 HOME）
if [ ! -f "$CONDARC" ]; then
  cat > "$CONDARC" <<EOF
# CodeScout 项目级 conda 配置 — 不注册用户环境，包缓存落允许根
envs_dirs:
  - $CODESCOUT_DATA_ROOT/envs
pkgs_dirs:
  - $CODESCOUT_DATA_ROOT/cache/conda-pkgs
channel_priority: strict
EOF
fi
if [ ! -f "$CODESCOUT_DATA_ROOT/envs/.pip.conf" ]; then
  cat > "$CODESCOUT_DATA_ROOT/envs/.pip.conf" <<EOF
# CodeScout 项目级 pip 配置 — 缓存落允许根，禁止用户 site 安装
[global]
cache-dir = $CODESCOUT_DATA_ROOT/cache/pip
[install]
no-user = true
EOF
fi

CONDA_BIN="$CONDA_DIST/bin/conda"
step "conda: $("$CONDA_BIN" --version 2>/dev/null || echo MISSING)"

# ---------------------------------------------------------------- 2. verl-vllm prefix
if [ ! -x "$PREFIX/bin/python" ]; then
  step "创建 conda prefix: $PREFIX (python=${PYTHON_VERSION})"
  "$CONDA_BIN" create -y -p "$PREFIX" python="${PYTHON_VERSION}" pip
else
  step "prefix 已存在: $PREFIX ($("$PREFIX/bin/python" --version 2>&1))"
fi

# ---------------------------------------------------------------- 3. verl 源码 @ tag
VERL_SRC="$SRC_DIR/verl-${VERL_TAG}"
if [ ! -d "$VERL_SRC" ]; then
  step "克隆 verl ${VERL_TAG}"
  git clone --depth 1 --branch "$VERL_TAG" "$VERL_REPO" "$VERL_SRC"
fi
VERL_COMMIT="$(git -C "$VERL_SRC" rev-parse HEAD)"
step "verl 源码: $VERL_SRC @ $VERL_COMMIT"

# ---------------------------------------------------------------- 4. 依赖安装（uv + 官方 uv.lock）
PY="$PREFIX/bin/python"
if [ "$SKIP_DEPS" -eq 0 ]; then
  step "从官方 uv.lock 导出 fsdp+vllm 锁定依赖"
  LOCK_EXPORT="$VERL_SRC/requirements-codescout-lock.txt"
  ( cd "$VERL_SRC" && uv export --frozen --no-dev --extra fsdp --extra vllm -o "$LOCK_EXPORT" )
  # 注意：导出文件首行是 `-e .`（verl 本体 editable），uv 以 cwd 解析该相对路径，
  # 因此必须在 VERL_SRC 内执行安装；既装依赖也一并 editable 安装 verl。
  step "安装锁定依赖 + verl 本体到 prefix（uv pip，cwd=$VERL_SRC）"
  ( cd "$VERL_SRC" && uv pip install --python "$PY" -r "$LOCK_EXPORT" )
  step "安装本仓库额外训练依赖（不含 OpenHands/SkyRL）"
  ( cd "$TMPDIR" && uv pip install --python "$PY" pyarrow pandas pytest )
  step "锁定组合导入断言（torch/vllm/verl/flash_attn 必须存在且版本匹配）"
  ( cd "$TMPDIR" && "$PY" - <<'PYEOF'
import importlib, sys
expected = {"torch": "2.11.0", "vllm": "0.24.0", "transformers": "5.9.0",
            "flash_attn": "2.8.3", "trl": "0.27.0"}
import importlib.metadata as md
fail = []
for name, exp in expected.items():
    try:
        got = md.version(name)
        if not got.startswith(exp):
            fail.append(f"{name}=={got} (expected {exp}*)")
        print(f"OK {name}=={got}")
    except md.PackageNotFoundError:
        fail.append(f"{name} MISSING")
import verl, torch
print(f"OK verl {verl.__version__} importable; torch cuda available: {torch.cuda.is_available()}")
if fail:
    print("LOCK MISMATCH:", "; ".join(fail), file=sys.stderr)
    sys.exit(1)
PYEOF
  )
fi

# ---------------------------------------------------------------- 5. manifest 摘要
MANIFEST_DIR="$CODESCOUT_ROOT/manifests/environment"
mkdir -p "$MANIFEST_DIR"
step "写 manifest 摘要"
"$PY" - "$PREFIX" "$VERL_SRC" "$VERL_COMMIT" "$CONDA_DIST" "$MANIFEST_DIR" <<'EOF'
import datetime, json, os, subprocess, sys
prefix, verl_src, verl_commit, conda_dist, manifest_dir = sys.argv[1:6]
out = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "prefix": prefix,
    "conda_dist": conda_dist,
    "verl_tag": "v0.9.1",
    "verl_src": verl_src,
    "verl_commit": verl_commit,
    "python": subprocess.run([prefix + "/bin/python", "--version"],
                             capture_output=True, text=True).stdout.strip(),
    "packages": {},
    "note": "完整核验由 scripts/check_environment.py 生成",
}
q = subprocess.run([prefix + "/bin/python", "-m", "pip", "list", "--format", "freeze"],
                   capture_output=True, text=True)
for line in q.stdout.splitlines():
    if "==" in line:
        name, ver = line.split("==", 1)
        out["packages"][name.strip()] = ver.strip()
path = os.path.join(manifest_dir, "install-summary.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print("wrote", path)
EOF

step "完成。验收：\"$CODESCOUT_PYTHON\" scripts/check_environment.py --output manifests/environment/check.json"
