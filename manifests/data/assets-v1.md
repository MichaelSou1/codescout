# 数据与模型资产清单 v1

核验时间：2026-10-07（Asia/Shanghai，子代理只读核验 HF 公开 API/datasets-server；未下载本体）。所有 revision 为核验当日最新值，执行下载前再次核验并固定。

## 模型（RL 起点，唯一）

| 项 | 值 |
|---|---|
| repo | `Qwen/Qwen3-4B-Instruct-2507` |
| revision（下载时固定） | `cdbee75f17c01a7cc42f958dc650907174af0554`（main） |
| 大小 | 13 文件 / 8,060,917,568 B ≈ 7.51 GiB（3 个 safetensors 分片） |
| 上下文 | 模型卡 262,144（256K 原生，`max_position_embeddings=262144`，无 rope 外推） |
| 精度 | config `torch_dtype=bfloat16` |
| license | apache-2.0 |

注意：训练窗口 40960、评测窗口 132K 均在 256K 原生范围内，无需扩上下文配置。

## 训练数据（SWE-smith 定位版）

| 项 | 值 |
|---|---|
| repo | `OpenHands/SWE-smith-py-code-search` |
| revision（下载时固定） | `b1e6864f1c847455ea64c7ebdb4164da2b902bb6` |
| 行数 | train split 39,287；**无 validation split**（划分由 build_dataset.py 重建：seed42 打乱、末 100 条为 dev） |
| 字段 | `instance_id`(str)、`file_changes`(list struct：file + changes{added_modules,added_entities,edited_modules,edited_entities})、`repo`(str)、`base_commit`(null dtype——与 `--use_patch` 语义一致)、`problem_statement`(str)、`patch`(str，mutation patch) |
| 大小 | 下载 43,626,982 B；解压 128,343,865 B |
| license | **未声明**（审计缺口，记录在案） |
| 与原 builder 默认数据集关系 | `adityasoni17/SWE-smith-py-code-search` 与本数据集**字节级同一 parquet**（LFS sha256 `df7d0f71b994d6dfed5fbf14ba648900f9b12f9d1b6cba459175d8a8276d2011` 一致），为同数据重发布（2026-03-17 vs 2025-12-31），可互换 |

论文口径 train ≈39K / 128 repo 与公开 39,287 行一致（repo 数待下载数据实测核对 131 vs 128）。

## 评测数据

| 数据集 | revision | 行数（test split） | 论文口径 | 大小 |
|---|---|---|---|---|
| `OpenHands/SWE-bench_Verified-locagent` | `ffa2f8bf98d03bc317695ad31594cd080d503a76` | **500** | 500 ✓ | 758,779 B |
| `OpenHands/SWE-bench_Pro-locagent` | `0cd282558ac2175670518c27b3593b5e435035c5` | **264** | 266（**差 2，入账披露**） | 1,370,373 B |
| `OpenHands/SWE-bench_Lite-locagent` | `6bec0115b6fc5cf7e904be4bb746946b663b10c5` | **300** | 300 ✓ | 393,123 B |

三个评测集字段与训练集同构（instance_id/file_changes/repo/base_commit/problem_statement/patch），但 `base_commit` **有值**（克隆 checkout 语义，非 mutation patch 语义）。

## 待办

- [ ] 下载时逐项核对 revision 未变；变则以新 revision 重录本清单并评估影响。
- [ ] SWE-smith 39,287 行展开/克隆体积待测；repo 数（论文 128 vs 数据库实际）待测。
- [ ] train/dev/test 重叠审计（instance_id/repo/base_commit/文本 hash），训练前冻结处理规则。
- [ ] 四个数据集 license 均未声明——公开重发布定位数据，本项目仅研究用途，记录在案。
- [ ] 生成 actor-safe parquet（prepare_verl_data.py）：剥离 `file_changes`/`patch`，只留 issue/仓库标识。
