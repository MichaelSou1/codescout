#!/usr/bin/env python3
"""CodeScout 环境核验脚本（在目标 prefix 的 Python 下运行）。

用法（服务器，source scripts/project_env.sh 后）:
    "$CODESCOUT_PYTHON" scripts/check_environment.py --output manifests/environment/check.json
    "$CODESCOUT_PYTHON" scripts/check_environment.py --with-cuda --output ...

只做只读检查：版本、路径、缓存边界、CUDA 可见性；不做训练、不加载模型权重。
"""

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_DATA_ROOT = Path(
    os.environ.get("CODESCOUT_DATA_ROOT", "/mmu_vlm_hdd/home/rhsu/playground/codescout-data")
)
EXPECTED_PREFIX = EXPECTED_DATA_ROOT / "envs" / "verl-vllm"

KEY_PACKAGES = [
    "verl", "torch", "torchvision", "torchaudio", "vllm", "transformers",
    "ray", "flash-attn", "flash-attn", "trl", "liger-kernel",
    "flash-linear-attention", "cupy-cuda13x", "tensordict", "datasets",
    "pandas", "pyarrow", "wandb", "numpy",
]


def collect_package_versions():
    versions = {}
    for name in dict.fromkeys(KEY_PACKAGES):  # dedupe, keep order
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def check_path_boundaries():
    issues = []
    # 1) 本解释器必须在项目 prefix 内
    py = Path(sys.executable).resolve()
    if EXPECTED_PREFIX not in py.parents:
        issues.append(f"python executable {py} not under {EXPECTED_PREFIX}")
    # 2) 用户 site-packages 不得混入 sys.path
    import site
    user_sites = site.getusersitepackages()
    for p in sys.path:
        if p and user_sites and str(Path(p)).startswith(user_sites):
            issues.append(f"user site-packages in sys.path: {p}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        issues.append("PYTHONNOUSERSITE != '1' (expected '1')")
    if site.ENABLE_USER_SITE:
        issues.append("site.ENABLE_USER_SITE is True")
    # 3) 关键缓存环境变量必须落允许根
    for var in [
        "CODESCOUT_DATA_ROOT", "HF_HOME", "TORCH_HOME", "PIP_CACHE_DIR",
        "UV_CACHE_DIR", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TMPDIR",
        "WANDB_DIR", "VLLM_CACHE_ROOT",
    ]:
        val = os.environ.get(var)
        if not val:
            issues.append(f"env {var} not set")
        elif not str(Path(val).resolve()).startswith(str(EXPECTED_DATA_ROOT)):
            issues.append(f"env {var}={val} outside allowed root {EXPECTED_DATA_ROOT}")
    return issues


def check_cuda():
    """导入 torch，报告 GPU/driver/kernel 可见性。只读，不分配显存训练。"""
    info = {"available": None}
    try:
        import torch
        info["available"] = torch.cuda.is_available()
        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count() if info["available"] else 0
        if info["available"]:
            props = torch.cuda.get_device_properties(0)
            info["device_0"] = {
                "name": props.name,
                "total_memory_gib": round(props.total_memory / 2**30, 1),
                "capability": f"{props.major}.{props.minor}",
            }
            info["compiled_kernel_archs"] = torch.cuda.get_arch_list()
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        info["driver_version"] = driver.stdout.strip().splitlines()[:1]
        try:
            flash = importlib.import_module("flash_attn")
            info["flash_attn_version"] = getattr(flash, "__version__", "unknown")
        except Exception as e:
            info["flash_attn_import_error"] = str(e)
    except Exception as e:
        info["import_error"] = str(e)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="manifests/environment/check.json")
    ap.add_argument("--with-cuda", action="store_true", help="import torch and report CUDA")
    args = ap.parse_args()

    report = {
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": collect_package_versions(),
        "path_boundary_issues": check_path_boundaries(),
        "ray_tmpdir": os.environ.get("RAY_TMPDIR"),
        "cmdline": sys.argv,
    }
    if args.with_cuda:
        report["cuda"] = check_cuda()

    ok = not report["path_boundary_issues"]
    report["verdict"] = "PASS" if ok else "FAIL"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps({k: report[k] for k in ("verdict", "python_version")}, indent=2))
    for issue in report["path_boundary_issues"]:
        print("ISSUE:", issue, file=sys.stderr)
    print("packages:")
    for k, v in report["packages"].items():
        print(f"  {k}: {v}")
    if args.with_cuda:
        print("cuda:", json.dumps(report.get("cuda", {}), indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
