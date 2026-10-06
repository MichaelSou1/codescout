# cs4b-stage0-docs — 步骤0：源码核对、kml-1005 核验、协议冻结

- run_id：`cs4b-stage0-docs`；experiment_id：`codescout_verl_4b_v1`
- 状态：完成（文档交付物；无 GPU、无训练、无下载）
- 时间：2026-10-07 03:05–03:35 CST
- 阶段/协议：todo.md 步骤 0；产出协议 [protocol-v1.md](../../reproduction/protocol-v1.md) v1

## 1. 目的/假设

完成 todo.md 步骤 0：读源码与上游材料、核对三端 git 状态、核验 kml-1005 实例、建立部署记录、实验账本与冻结协议文档。判据：文档齐备且只记录实测事实，未把计划项写成已完成。

## 2. 对照/改动

无前序 run。唯一改动为新增文档；不改代码、不改训练语义。

## 3. 数据/模型/配置

源码 commit `9d05a644e02f102f91e7e03244217705246fc4a4`。核对过的源码事实（详见 protocol-v1 / skyrl-to-verl）：

- `scripts/run_async_training_4b.sh`：GRPO(`grpo_norm_by_std=false`) + GSPO(clip 3e-4/4e-4, `sequence_mean`)、lr 1e-6、无 KL、batch 8×8、`update_epochs_per_batch=1`、40960 输入窗、8192/次生成、train 温度 1.0 / eval 0.6、`max_turns=10`、fsdp2+cpu_offload、colocate_all=false、4+4 引擎、ckpt 每 10 / HF export 每 50 / 保留 5、hermes parser、thinking 关闭、`CUDA_LAUNCH_BLOCKING=1` 调试配置。
- `code_search_generator.py`：episode 聚合为单序列、`<|im_start|>assistant` mask（buffer_succeed=1/precede=1）、轮数耗尽 mask 全 0、`sanity_check_last_step`、无 TokenEvent 占位 151643。
- `file_localization.py` + `module_rewards.py`：multilevel F1 1/1/1（0–3）、空真值得 0、`parse_structured_outputs` 语义。
- `localization_finish.py`：schema 与恰好一次规则。
- `instance.py`/`build_dataset.py`：GitHub 克隆+checkout+apply patch；seed42 打乱、末 100 validation、`use_patch`→`base_commit=None`。
- `pyproject.toml`：SkyRL `81e5a97c…`、OpenHands SDK `85ecfd93…`、Python≥3.13、transformers 4.57.3、vLLM 0.11.0、flash-attn cp313 wheel（不迁移到新环境）。
- 轮数三方冲突：论文 6 / prompt 文本 4 / runtime 10 → 协议冻结 6。
- 奖励配置 `configs/reward_config_4b.yaml`：仅 `multilevel_localization_f1_reward` weight 1.0。

kml-1005 现场核验（详见 [deployment/kml-1005.md](../../deployment/kml-1005.md)）：SSH/hostname（`aiplatform-wlf3-ge65-7…`）、driver 595.58.03、8×H100 空闲（UUID 已录）、允许根写测试通过、Ceph 5P/3.4P、代理出口可达 GitHub/PyPI/HF/PyTorch、工具 uv/git/tmux 有、conda/docker/rg 无、`codescout-data` 未建。

## 4. seed/环境

无训练 seed。三端核对（2026-10-07 03:21 CST）：本机 HEAD = 本机 origin/main = 服务器 HEAD = 服务器 origin/main = `9d05a64`，两端工作树干净；本机另有未跟踪 `.zcodeignore`（工具产物，不入库）。服务器 push 通道未核验（无凭据，不复制本机 token）。无 GPU 作业、无进程残留。

## 5. 结果/成本

产出：`docs/deployment/kml-1005.md`、`docs/reproduction/protocol-v1.md`、`docs/reproduction/skyrl-to-verl.md`、`docs/experiments/{index,status}.md`、`records/_TEMPLATE.md`。成本：约 0.5 人时，0 GPUh，0 API 调用。

## 6. 符合预期与否

符合：步骤 0 清单的文档/核验项全部有实测或源码证据；未见不符合项。本地 parquet 行数与字段未核（本机无 pandas/pyarrow），留步骤 2。

## 7. 结论/限度

协议 v1 冻结为文档级；其中【待核验】项（verl 支持矩阵、SDK iteration 语义、官方评测参数、AdamW 超参、Hermes 兼容性）不凭默认值补齐，进入对应步骤后消解。本记录不构成环境就绪证据。

## 8. 下一步

步骤 1：建立 `codescout-data`、`scripts/project_env.sh`、conda+verl v0.9.1 隔离环境并验收（run_id `cs4b-env-a01`）。负责人：主代理调度；预算：0 GPUh（安装与 CPU 验收）。
