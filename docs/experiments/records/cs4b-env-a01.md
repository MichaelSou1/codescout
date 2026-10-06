# cs4b-env-a01 — conda + verl v0.9.1 隔离环境搭建与验收

- run_id：`cs4b-env-a01`；experiment_id：`codescout_verl_4b_v1`
- 状态：**完成**（attempt 8，2026-10-07 04:15 CST 验收 PASS）
- 时间：2026-10-07 03:24–04:15（CST）；执行位置 kml-1005
- 阶段/协议：todo.md 步骤 1；组合见 [skyrl-to-verl.md §5](../../reproduction/skyrl-to-verl.md)

## 1. 目的/假设

在允许根建立隔离 conda 环境（verl v0.9.1 官方锁定组合），验收路径合规、无用户 site-packages、GPU 可见。判据：`check_environment.py` PASS 且锁定组合导入断言通过。

## 2. 对照/改动

无前序环境 run。唯一被测改动为环境本身；不装 SkyRL、不装 OpenHands（后者步骤 3 单独 prefix）。

## 3. 数据/模型/配置

- Miniforge3 `26.7.2-0`（installer sha256 官方 sidecar 校验通过）；conda 发行版装于 `codescout-data/tools/conda-dist`，prefix `envs/verl-vllm`。
- verl `v0.9.1`（clone tag，commit `1876b06d0a3e4e71e06230be10af14492ca8a75b`），依赖用官方 uv.lock `uv export --frozen --extra fsdp --extra vllm` 导出后 `uv pip` 安装到 prefix。
- 目标锁定组合：torch 2.11.0+cu130 / vllm 0.24.0 / transformers 5.9.0 / flash-attn 2.8.3 / trl 0.27.0 / ray ≥2.41。
- 服务器 driver 595.58.03（≥CUDA13 最低 580），8×H100 空闲。

## 4. seed/环境

代码 commit：`cfe6b34`（scripts）+ `9585b9d`（tests）；服务器 checkout 同步，工作树干净（服务器产物 `install-summary.json`/`check.json` 未提交前不入库）。conda `.condarc`/`.pip.conf` 项目级，关闭用户注册；`PYTHONNOUSERSITE=1`；缓存全落 `codescout-data/cache`。

## 5. 结果/成本（含全部失败）

| # | 结果 | 根因 | 修复 |
|---|---|---|---|
| 1 | conda create 失败 | `CONDA_ENVS_PATH` 与 `CONDA_ENVS_DIRS` 同时设置，conda 拒绝 | 只保留 `CONDA_ENVS_DIRS` |
| 2 | conda create 失败 | 项目 `.condarc` 写了 `no_plugins: true`，禁用 libmamba solver | 删除该行 |
| 3 | uv pip 解析失败 | `uv export` 输出首行 `-e .` 以 cwd 解析；cwd 在本仓库 → 把 py3.13 项目拉进解析 | 在 VERL_SRC 内执行 |
| 4 | uv pip 报 not a Python project | `-e .` 在空 TMPDIR 无 pyproject 可解析（同 #3 根因的另一半） | 同 #3 |
| 5 | 安装成功但锁定组缺失 | **uv.lock GPU 后端 wheel marker `python_full_version >= '3.12'`，3.11 下整组跳过**；torch 2.14.1 系无约束传递依赖混入 | prefix 改 Python 3.12（verl 声明 3.10–3.12 内），setup 加导入断言 |
| 6 | uv pip 报 torch 2.11.0+cu130 不可解析 | `+cu130` wheel 走 verl 私有 wheelhouse 路由，`uv export | uv pip -r` 丢路由 | 改官方 `uv sync` + `UV_PROJECT_ENVIRONMENT=$PREFIX`（实测接受 conda prefix） |
| 7 | flash-attn 下载 404 | v0.9.1 uv.lock 内 wheelhouse URL 为过期 tag（实际 release 带后缀） | sed 修补 lock URL 为 `flash-attention-v2.8.3-py3.12-torch2.11.0`（版本/哈希不动） |
| 8 | **成功** | — | 导入断言全过：torch 2.11.0+cu130 / vllm 0.24.0 / transformers 5.9.0 / flash_attn 2.8.3 / trl 0.27.0 / verl 0.9.1；`check_environment.py --with-cuda` **PASS**（Python 3.12.15，8×H100 可见，路径无越界，无用户 site-packages） |

manifest 已入 Git：`manifests/environment/install-summary.json`、`manifests/environment/check.json`（commit `9d39a1e`）。

**GPU 短测（2026-10-07 04:26 CST，单卡 GPU-9ea7da5d，准入核对后启动）**：vLLM 0.24 离线加载 Qwen3-4B-Instruct-2507@cdbee75f 成功（407.7s 含首次编译/CUDA graph 捕获 4s/0.98GiB），`gpu_memory_utilization=0.5` 峰值 41.5GiB；贪心短生成正常（"Ready."）；4 路批量采样正常；进程退出后显存 0 MiB（清理干净）。消耗：1 GPU × 约 9 分钟 ≈ 0.15 GPUh。FSDP 初始化与多卡通信留待步骤 6 的 2–4 卡 correctness smoke。

成本（attempt 1–8 累计）：约 45 分钟墙钟，下载缓存约 15GB（允许根），0 GPUh，0 API 调用。

## 6. 符合预期与否

- attempt 1–5 各为独立根因（非同因重试超限）：conda 配置冲突、插件禁用、uv `-e .` cwd 语义、锁定 marker 的 Python 版本门槛。全部入账。
- verl v0.9.1 "Python 3.11 主试" 假设被否定：声明矩阵允许 3.11，但锁定 wheel 分发 marker 不允许——**记录为矩阵事实，不回退 verl 版本**。

## 7. 结论/限度

todo.md 中 "Python3.11主试" 已按矩阵证据改为 3.12（协议层面：verl 声明范围内，不改变任何训练语义）。锁定组合是否可实际加载 Qwen3-4B、hermes parser 行为仍待验收（步骤 3/4）。

## 8. 下一步

- attempt 6 完成后：`check_environment.py --with-cuda` 验收；GPU 短测（模型加载+短生成）在独立 run 记预算。
- 之后进入步骤 2（模型/数据下载）与步骤 3（workspace/工具 smoke）。
