# cs4b-oracle-a01 — 数据准备与 workspace/工具 oracle（步骤 2–3 部分）

- run_id：`cs4b-oracle-a01`；experiment_id：`codescout_verl_4b_v1`
- 状态：进行中（数据与 workspace 20 任务 smoke 已完成并验收；终端工具 + 全链路 oracle 待步骤 4 适配验收）
- 时间：2026-10-07 03:55–04:40（CST）；执行位置 kml-1005
- 阶段/协议：todo.md 步骤 2 与步骤 3 的数据面；[protocol-v1.md §2/§3](../../reproduction/protocol-v1.md)

## 1. 目的/假设

下载固定 revision 资产、重建冻结划分、验证 20 个任务可确定性构造无 `.git` 的真实仓库快照。判据：资产 revision 与清单一致、划分计数正确、20/20 构建成功且无泄漏。

## 2. 对照/改动

无前序数据 run。对照源码语义 `src/build_dataset.py` 与 `src/utils/instance.py`（clone/checkout/apply patch）。

## 3. 数据/模型/配置

- 模型 `Qwen/Qwen3-4B-Instruct-2507`@`cdbee75f`：13 文件 7.6GiB 下载完成（`codescout-data/models/…`）。
- 数据（revision 见 [assets-v1.md](../../../manifests/data/assets-v1.md)）：SWE-smith@`b1e6864f`（39,287 行）、Verified@`ffa2f8bf`（500）、Pro@`0cd28255`（264）、Lite@`6bec0115`（300），全部 snapshot_download 到 `codescout-data/datasets/`。
- 划分重建（`scripts/prepare_verl_data.py`，语义=build_dataset.py）：39,287 → train 39,187 + validation 100（空 issue 0 条，seed42 打乱），train/dev instance_id 无重叠；私有标签 `swe_smith_labels.jsonl`（sha256 `5437915f…`）落 `codescout-data/secrets/private-labels/`（actor 不可读路径，不入 Git）。manifest 入 Git：`manifests/data/swe-smith-prepare-manifest.json`（commit `0268561`）。

## 4. seed/环境

代码 commit `271bb5c`；服务器 verl env（Python 3.12.15）执行；无 GPU 使用。一次失败：HF snapshot_download 数据集未传 `repo_type="dataset"`（默认 model 401）——修复后成功（记录为 infra 修复，非训练重试）。

## 5. 结果/成本

- **workspace smoke 20/20 成功**（`scripts/smoke_workspace.py`，seed42，20 个 SWE-smith 任务真实 GitHub 克隆 + mutation patch 应用 + 无 `.git` 快照导出）：耗时 mean 6.6s / median 3.7s / max 21.9s；总 319.7MiB / 8,445 文件；`.git` 残留 0；泄漏嫌疑 12 个文件名人工核实为仓库原生测试 fixture（test_patches.py/monkeypatch.py 等），非 gold 泄漏。
- 快照构建关键权衡：工作树复制（非 `git archive HEAD`——后者会静默丢掉未提交 mutation）；导出后 `git apply --check --reverse` 验证 mutation 在位。
- 成本：约 20 分钟墙钟 CPU；克隆缓存 `codescout-data/cache/clone-cache/`；0 GPUh。

## 6. 符合预期与否

符合：划分计数、无重叠、20/20、无 .git、无泄漏。未完成项如实列出：终端工具真实执行/finish 恰好一次/奖励全链路（含 infra vs 模型失败分类）依赖步骤 4 适配实现后的 oracle 验收。

## 7. 结论/限度

数据面通过条件满足（随机 20 任务可确定性构造、私有标签隔离、划分可重建）。Pro 公开 264 vs 论文 266 的分母差异已入账（assets-v1.md）。本记录不构成训练有效性证据。

## 8. 下一步

步骤 4 适配实现完成后的验收顺序（设计文档 §11.5）：①单任务离线 prompt 编码一致性 ②reward adapter 与冻结 scorer 对齐（≤1e-8，≥50 边界）③GPU 模型加载+短生成 ④固定张量 loss/梯度对齐。
