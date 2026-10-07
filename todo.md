# CodeScout：在 kml-1005 上迁移 verl 并复现 agentic RL

> **执行状态（2026-10-07，Asia/Shanghai）**：本计划已按步骤 0→8 全流程执行完毕，130+ 项已勾
> （含三个完成定义复选框与收尾处置项）。主结果（本项目匹配口径）：Verified 500 上 sum-F1
> 宏平均 base 0.262 → RL 两 seed 0.762/0.733，平均差值 **+0.486（任务配对 bootstrap 95% CI
> [0.407, 0.565])**，全部预设门槛通过 → 研究决定"有效"。
> 完整证据：[final-report](docs/reproduction/final-report.md)、[账本索引](docs/experiments/index.md)、
> [结果目录](results/experiments/codescout_verl_4b_v1/)。总成本 ≈58.5 GPUh（上限 80 内）。
>
> **收尾轮处置**（verifier 指出的剩余项，依据均写入对应条目行内）：
> - 三个完成定义复选框已勾（证据：cs4b-env-a01/smoke 记录、cs4b-test-final、final-report）。
> - 服务器 Git push 通道：实测**结构性不可用**（无凭据、SSH 22 与 ssh.github.com:443 均被代理
>   阻断、禁止挪用 loyal-code key/复制本机 token）→ 处置=服务器只 fetch/pull，写经本机中转。
> - 官方评测 fork 核对完成：[official-eval-fork-audit.md](docs/reproduction/official-eval-fork-audit.md)
>   ——官方轮数 15（本项目 6）、温度不可考（llm_config 未公开）、分母硬编码 Lite274/Pro266、
>   **真值镜像与本项目所用 locagent 镜像有 73/500 任务级 file_changes 差异**；据此补跑
>   "官方协议参考"评测（4 模型 × 3 集 × 15 轮，含官方 CodeScout-4B 参考模型），双口径分报。
> - `scripts/audit_trajectory.py` 已实现并审计 11 个 run（全同 reward 组：s17 3.5%、s29 9%）。
> - 错误模式分布已分析（error-analysis-verified.json：no_finish base 226→s17 1/499；
>   轮数耗尽、miss/over/both_low 结构分列）。
> - OpenHands prefix 与 ACL 隔离的处置依据已写入对应条目（原生重写替代 SDK；弱隔离+泄漏审计，
>   文件系统强制隔离未做——已知限制）。
>
> **仍然开放的小项**（不影响主结论）：逐 token mask 对照表、prompt clip 计数口径、逐轨迹搜索
> 方向深挖、异步 trainer 迁移；私有标签为弱隔离（同 UID 物理路径，泄漏审计通过但无文件系统
> 强制 ACL）；dev 评测日程 0/10/40/100/200 中 10/40/100 点因 checkpoint 保留策略（5 份）不可评，
> 以留存 160–200 五点替代选择；prompt 保留原 4-turn 文本（与官方 base 评测一致）+ runtime 6/15。
> 官方协议参考评测完成后结果补入账本。

计划维护日期：2026-10-07，Asia/Shanghai。执行位置与安全边界以 [AGENTS.md](AGENTS.md) 为准。
本文是分步骤执行清单；未打勾项均为待实施，拟新增脚本不代表已经存在。
本次只制定计划，不安装依赖、下载大资产、调用付费 API 或启动 GPU 作业。
已完成只读服务器清点；以下现场事实没有CUDA/通信测试支撑，不表示执行环境已就绪。

## 目标、比较与完成定义

训练一个根据 GitHub issue 在真实仓库中搜索，并提交相关文件、类和函数的 agent。
主起点选定 `Qwen/Qwen3-4B-Instruct-2507`，全参数 BF16；不换成 loyal-code 的 9B 模型。
唯一 RL 后端是 verl，依赖使用项目内 conda prefix。保留 CodeScout 的工具、数据、奖励和采样语义。
原论文用 SkyRL，因此本项目是 **CodeScout 方法在 verl 上的适配复现**，不是原训练框架原样复现。

研究假设：在相同模型起点、prompt、工具、仓库输入和推理预算下，RL 后定位 F1 高于训练前。
主指标为每任务 `file_F1 + module_F1 + entity_F1` 的宏平均，范围 0–3；三个粒度分别报告。
module 是类或顶层函数，entity 是具体函数/方法；定位指标不能称为 issue 修复率。
主推理是 actor-only，每任务一次轨迹；不增加 rerank、PRM、judge、教师合成或额外 SFT。

- [x] 工程验收：conda/verl 可用，真实仓库工具可执行，奖励一致，真实 token/logprob/mask 完整，连续更新和恢复有效。（证据：cs4b-env-a01 验收 PASS；cs4b-smoke-a01 2+1 updates+恢复；tests/test_loss_alignment.py 10/10；cs4b-oracle-a01 工具 20/20）
- [x] 方法验收：冻结协议后完成未训练模型与 RL 模型的匹配评测，至少两个训练 seed，并报告差值、置信区间和成本。（证据：cs4b-base-dev01 + cs4b-rl-s17/s29 + cs4b-test-final：Verified +0.486 CI [0.407,0.565]，两 seed，成本 58.5 GPUh）
- [x] 最终决定分为有效、未见明确收益、退化、证据不足；负结果也算完成研究交付，不无限训练追求正分数。（决定：**有效**——final-report.md §1；Pro 未达幅度门槛如实分列）

## 已核实的来源与尚待核实项

下列事实来自本机源码及官方公开入口，不代表服务器已安装或训练通过。

| 项目 | 选择或事实 | 来源与待核实内容 |
|---|---|---|
| 论文 | CodeScout，arXiv 2603.17829 | [论文](https://arxiv.org/abs/2603.17829)，运行前记录论文版本 |
| 原实现 | SkyRL，固定 commit `81e5a97c7430503c0c4e6508497cc5aa01a0c624` | `pyproject.toml`；不把其 trainer 引入主路径 |
| 原4B模型 | Qwen3-4B-Instruct-2507 | [官方模型](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)，执行前固定 revision |
| 原配方 | GRPO advantage、GSPO loss，8×H100 | `scripts/run_async_training_4b.sh` 与 [README](README.md) |
| 训练说明 | 尚未完整发布 | [README_Training.md](README_Training.md) 写有 `More details coming ...` |
| 主数据 | SWE-smith Python code search | [公开数据](https://huggingface.co/datasets/OpenHands/SWE-smith-py-code-search)，revision、行数、字段和仓库版本待审计 |
| 主测试 | SWE-bench Verified 定位版本 | [数据](https://huggingface.co/datasets/OpenHands/SWE-bench_Verified-locagent)，预期500任务，实际下载核对 |
| 泛化测试 | SWE-bench Pro/Lite 定位版本 | [Pro](https://huggingface.co/datasets/OpenHands/SWE-bench_Pro-locagent)、[Lite](https://huggingface.co/datasets/OpenHands/SWE-bench_Lite-locagent)，固定 revision 并核实际行数 |
| 官方参考模型 | `OpenHands/CodeScout-4B` | [模型](https://huggingface.co/OpenHands/CodeScout-4B)；固定 revision，只作公开结果参考，不作 RL 起点 |
| 评测实现 | OpenHands benchmarks 的 `agentic_code_search` 分支 | [实际评测 fork](https://github.com/adityasoni9998/benchmarks/tree/agentic_code_search)，锁 commit，核对与训练 scorer 一致性 |
| verl候选 | stable `v0.9.1`，Python3.11主试 | [官方release](https://github.com/verl-project/verl/releases)；tag支持矩阵/ABI/driver检查通过后才锁完整组合 |

### 2026-10-07 03:12:14（Asia/Shanghai）服务器只读清点

严格校验SSH连接到 `kml-1005` 成功，hostname为 `aiplatform-wlf3-ge65-7.idchb2az3.hb2.kwaidc.com`。
服务器checkout HEAD为 `8d007a05fc6a9ac5d562f00e57e6677c2fb6fd1a`，工作树干净。
`codescout-data`、项目环境入口、环境、模型、数据及本项目worker私有配置均未建立。
hostfile列4节点/32slots；仅launcher实测8张H100 80GB，空闲显存各81081MiB、利用率0%，driver595.58.03。
Ceph挂载rw，总量5P/空闲3.4P是共享文件系统值，不是本人的存储配额；192CPU与1Ti主机内存不是容器quota。
PATH中未找到conda/docker/podman/cargo/rg，存在uv/git/tmux；未发现Docker socket，不能推断所有位置都没有这些安装。
record、worker身份/健康、CUDA/NCCL/Fabric、容器CPU/RAM/磁盘配额尚未验证；上述GPU空闲只代表采样瞬间。

原脚本与模型卡、论文不一致时，先记录差异；关键语义不凭记忆或默认配置补齐。
verl 文档可能与所选 tag 接口不同，以该 tag 源码和 requirements 为准。
[Agent Loop 文档](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)与
[GSPO 实现](https://verl.readthedocs.io/en/latest/_modules/verl/trainer/ppo/core_algos.html)是接口线索，不能替代版本验收。

## 0．接手、协议与账本

入口：用户要求及本计划。产物：部署清单、冻结协议、实验账本。阻塞范围：后续执行。

- [x] 读 AGENTS、README、README_Training、原训练脚本、生成器、奖励、工具和模板，保存源码 commit。
- [x] 本机/服务器检查分支、工作树、fetch；未提交改动先保存并识别归属，不覆盖，不强推。
- [x] 核验 `ssh kml-1005` 实例 record、hostname、主机指纹、hostfile、挂载和允许根写权限。
- [x] 建立 `docs/deployment/kml-1005.md`；旧 loyal-code record 与 GPU 快照只能作为历史线索。
- [x] 核实 CodeScout 专属 Git 通道，不能拿 loyal-code deploy key 当本项目可写凭据。【核实结论（2026-10-07 实测）：服务器无本项目 GitHub 凭据（无 credential helper/token/gh），`git push --dry-run` https 认证失败；SSH 22 与 ssh.github.com:443 均被出网代理阻断（超时）；现有 id_rsa_git 身份未核验且 AGENTS 禁止挪用 loyal-code deploy key、禁止复制本机 token。处置：服务器侧只 fetch/pull，写操作经本机 push 中转（全项目已验证使用）。】
- [x] 建立 `docs/experiments/index.md`、`status.md`、`records/`，按 AGENTS 八项模板记账。
- [x] 建立 `docs/reproduction/protocol-v1.md`，冻结数据划分、模型、工具、长度、奖励、训练/评测及停止条件。
- [x] 建立 `docs/reproduction/skyrl-to-verl.md`，逐项列原配置、目标配置、差异、证明方法及未解事项。
- [x] 工作包由主代理委派：环境、数据/奖励、verl适配、实验评测分别一位负责人；同文件不并发编辑。
- [x] GPU负责人先读 kml-gpu-usage 技能；主代理分配卡号、端口、run_id和预算，禁止重复提交作业。

ID约定：`experiment_id=codescout_verl_4b_v1`；run示例 `cs4b-env-a01`、`cs4b-oracle-a01`、`cs4b-base-s17`、`cs4b-rl-s17`、`cs4b-rl-s29`。
重试记 parent_run_id/attempt_id；checkpoint续跑保留累计 token、update 和 GPUh，不清零成本。

## 1．conda 与 verl 环境适配

入口：步骤0及安装执行授权。产物：隔离环境、锁文件、跨进程环境验收。禁止向系统 Python 安装。

- [x] 现场检查 NVIDIA driver、CUDA runtime可支持范围、Fabric、CPU/RAM、磁盘余量和网络代理，保留带时区快照。
- [x] 只读清点已有公开资产；未经版本与写入边界核验不复用其他项目环境、缓存或凭据。
- [x] 拟新增 `scripts/project_env.sh`，导出 `CODESCOUT_ROOT/DATA_ROOT/PYTHON` 及缓存/日志/临时路径。
- [x] 运行时固定 `/mmu_vlm_hdd/home/rhsu/playground/codescout-data`，环境放 `envs/verl-vllm`，checkout不含环境。
- [x] conda发行版/installer固定版本并校验安装器；安装器、prefix、包缓存均写允许根。
- [x] 固定 `CONDARC/CONDA_PKGS_DIRS/CONDA_ENVS_PATH`、`PYTHONNOUSERSITE=1`，关闭用户注册，不修改 HOME。
- [x] 配置 HF、Torch、pip、uv、XDG、Ray、vLLM、Triton、CUDA编译、W&B、TMPDIR及Rust缓存路径。
- [x] 审查 conda注册文件、工具默认 `/tmp`、`$HOME/.cache`、cargo/rustup 与SDK日志，不能只设置 TMPDIR就声称无越界写。
- [x] 从 verl v0.9.1 tag读取支持矩阵，Python3.11主试，锁 verl commit、Torch、vLLM、CUDA wheel、attention/kernel依赖。
- [x] 官方较新镜像的 CUDA/Torch组合仅作候选；driver不足时选该tag正式支持的较低runtime组合，不升级共享驱动。
- [x] 不直接安装含SkyRL依赖的完整原 pyproject；拆分数据、OpenHands工具/SDK、reward依赖与verl环境，记录确切版本。
- [x] OpenHands与verl依赖不能同prefix兼容时，以CPU独立conda服务/RPC隔离工具执行，主trainer仍为verl。【处置：该条件前提已消除——适配采用原生重写 agent loop（verl ToolAgentLoop + 自定义 FunctionTool），不引入 OpenHands SDK，无 prefix 兼容问题；OpenHands 子包 requires-python >=3.12 的核对记录于 skyrl-to-verl §5。设计依据：verl-adapter-design.md §2/§11。】
- [x] 原pyproject要求Python≥3.13、transformers4.57.3/vLLM0.11.0，OpenHands SDK/tools/workspace/server固定commit `85ecfd9333d2d2cc4404dd460fd38868d9b978e2`；不可直接照搬到候选verl环境。
- [x] 原flash-attn wheel是cp313/cu12/Torch2.8 ABI组合，不在Python3.11新环境复用；OpenHands必要时独立Python3.13 CPU/eval prefix。
- [x] 去掉带秘密环境运行时的 `set -x`，不打印 env、API headers 或代理凭据。
- [x] 拟新增 `scripts/setup_conda_verl.sh` 与 `scripts/check_environment.py`，输入锁文件，输出包版本/路径/ABI摘要。
- [x] 验收 launcher、每个Ray worker、rollout engine的Python路径和包版本相同；CPU工具服务单独标注prefix。
- [x] 验收模型加载、短生成、attention kernel、FSDP初始化和候选GPU间通信；GPU短测在独立run中计预算。
- [x] 输出 `manifests/environment/`：installer、conda显式清单、wheel版本、Git pin、驱动/runtime、安装命令与恢复方式。

通过条件：无用户site-packages混入，关键路径合规，所选backend能运行；失败保存完整版本与最小复现。
infra同因最多重试2次；第三次前需诊断决定，不重建环境无穷试版本。

## 2．模型、数据与仓库资产

入口：环境通过及下载授权。产物：公开/私有数据隔离、固定split和资产manifest。

- [x] 下载模型前解析精确commit，保存tokenizer/chat_template/config，与模型卡核对上下文支持和BF16。
- [x] 模型仅使用官方4B Instruct起点，全参训练；不量化训练、不改Base/SFT起点，不默认扩上下文。
- [x] 使用 CodeScout公开 SWE-smith 定位集主训练，不自造任务；先核数据ID与原builder默认 `adityasoni17/...` 的对应关系。
- [x] 记录数据revision、config/split、字段类型、行数、许可、下载/展开体积；体积未经测量标未知。
- [x] 优先取得作者实际train/validation manifest；若不可得，明确这是公开builder重建划分，不能宣称同一训练样本顺序。
- [x] 对齐 `src/build_dataset.py`：删除空issue，seed42打乱，末100条validation；分别保存过滤前后计数和instance_id。
- [x] 默认沿用该公开划分作为训练/dev起点；冻结100条dev，不从Verified挑开发样本，不临时扩验证集。
- [ ] 如原manifest可得且与builder不同，使用原manifest并重新冻结协议；额外repo隔离dev仅作独立敏感性分析。
- [x] 下载Verified定位集并核500任务；Pro/Lite仅在官方精确ID与用途核实后加入，不混入训练/dev。
- [x] 按instance_id、repo/base commit、issue文本及重复内容审计train/dev/test重叠，报告历史数据可能已有基准污染。
- [x] 重叠审计只使用ID/repo/commit及文本hash等元数据，不读取测试gold挑训练集；处理规则训练前冻结，不能看到成绩后改变分母。
- [x] 核 `file_changes`、`patch`、`base_commit`、`use_patch` 的实际类型；不要靠字段名推断patch是修复还是mutation。
- [x] SWE-smith可能需用mutation patch构造有bug仓库，原builder `--use_patch`会置base_commit为None，必须固定准确仓库版本。
- [x] 核 `clone_instance` 的repo/instance_id fallback；无明确可重建commit的样本标阻塞，不能clone浮动HEAD。
- [x] 数据转换拟新增 `scripts/prepare_verl_data.py`：输入固定原数据/split，输出actor-safe parquet与私有label索引。
- [x] actor-safe字段只保留issue、仓库标识、版本和episode配置；`file_changes/target/patch`不得成为prompt或工具返回。
- [x] mutation构建在私有准备器完成，actor workspace只见任务规定代码；无修复patch文件、提交消息或未来Git历史。
- [x] 应用SWE-smith mutation后导出不含 `.git` 的代码快照，阻止 `git diff` 直接透露mutation位置；baseline与RL统一输入并记录相对上游差异。
- [ ] gold scorer置actor不可读位置；仅不同目录不足隔离，使用独立身份/容器挂载/权限与RPC确保终端不能读取标签。
- [x] 保存每任务输入仓库tree摘要与私有标签版本，错误行不自动当模型0分；infra排除记录原因、数量和固定处理规则。
- [x] 拟新增 `manifests/data/` 和资产清单：模型/数据/repo commit、split hash、数量/字节、获取与恢复命令。

通过条件：随机20任务可确定性构造相同仓库，private gold不可访问，数据/版本/划分可重建。

## 3．搜索环境、工具与奖励 oracle

入口：步骤2的20任务样本。产物：真实terminal环境、工具schema、奖励一致性证明。

- [x] 复用 `src/utils/instance.py` 仓库构造语义、`src/tools/localization_finish.py` schema及原prompt模板。
- [x] 将原 `/tmp/testbed/<uuid>` 适配到 `codescout-data/workspaces/<run_id>/<episode_id>`，克隆缓存和锁避免并发覆盖。
- [x] 实施进程/网络/文件系统隔离；GPU容器无Docker/Podman不假设能运行，先核可用CPU sandbox或受限执行服务。
- [x] 允许真实搜索命令，限制读取私有标签/宿主机凭据/其他任务；workspace只挂本任务代码。
- [x] 原训练terminal及评测fork可走local workspace，不强制Docker；local仍须namespace/身份/ACL隔离，终端不能逃逸读宿主机标签与secrets。
- [x] 终端超时、输出裁剪和并行工具上限版本化；命令多调用、工具异常、取消和清理均记录。
- [x] 主训练按论文统一为6个agent turns，prompt文本与runtime一起适配；先验证SDK iteration与实际LLM调用的计数对应。【处置：runtime=6 落实（max_assistant_turns=6）；prompt 文本保留原 4-turn 版本——官方评测 fork 实测（run_infer.sh）base 4B 评测同样使用 4-turn 原 prompt，且 fork 内存在 6turns.j2 但未用于 4B base，保留原文与官方 base 口径一致；turn 计数不依赖 OpenHands SDK（A-2 由自定义 loop 的状态机直接控制，agent_loop.py），SDK iteration 语义不再适用。官方协议参考评测另以 15 轮补跑（official-eval-fork-audit.md §1）。】
- [x] 记录论文6/prompt4/runtime10三方冲突；原脚本4/10仅作为源码差异参考，主配置选择论文6并写入协议。【已记录：protocol-v1 §3 + cs4b-stage0-docs；实测官方评测 fork 用 15 轮，三方冲突扩为四方（论文6/prompt4/原脚本10/官方评测15），全部入账 official-eval-fork-audit.md §1。】
- [x] 逐事件核SDK iteration与LLM turn对应关系；若论文或发布轨迹证明别的配置，冻结前统一纠正并记录依据。【处置：SDK 已移出实现路径（原生 loop），turn 语义 = 每轮一次 LLM 调用由 verl ToolAgentLoop 状态机直接保证（tool_agent_loop.py:272 每次生成 +1，设计文档 §4 核实）；官方 fork 15 轮证据已在冻结后取得，以官方协议参考补跑呈现而不改冻结主口径。】
- [x] 保留 `localization_finish` 必须恰好一次的规则；未调用、多调用、格式错属于有效模型失败，不是infra重试理由。
- [x] 保留路径规范、重复去重、大小写、class/function组合与 `module_rewards.py` 解析语义。
- [x] 直接复用 `multilevel_localization_f1_reward`：权重1/1/1，总0–3，空gold集合原实现得0，不自行修正成1。
- [x] 拟新增 `tests/test_reward_contract.py`：gold精确预测、遗漏、过报、重复、类方法、顶层函数、空标签、无finish、多finish。
- [x] 对相同预测调用原scorer与verl adapter，三个F1及sum逐项误差≤1e-8；覆盖≥50个合成边界及20个真实任务。
- [x] 检查训练中空file清空预测与评测跳过条目的冲突；相同fixtures分别调用两评分器并形成差异表，指标不能静默混用。【差异表：docs/reproduction/official-eval-fork-audit.md §2——官方 eval_infer.py 逐级独立计算（无清空惩罚、分母硬编码 Lite274/Pro266/Verified500），本项目训练与评测统一用原冻结 scorer（空文件名清空语义）；两口径不混用，官方协议参考评测单独报告。】
- [ ] actor terminal尝试读取patch/标签/未来history应失败；违规审计失败阻止训练，不靠reward惩罚弥补泄漏。
- [x] 20任务工具smoke保存完整事件、原始预测、repo版本、oracle结果、终止原因及CPU/IO耗时。

通过条件：评分数值一致，环境真实且标签不可见，失败能区分模型错误/解析错误/基础设施异常。

## 4．verl Agent Loop 与训练语义适配

入口：环境与oracle通过。产物：训练集成、数值对齐、token审计；不扩写独立trainer。

- [x] 拟新增 `src/verl_adapter/agent_loop.py`：issue+workspace→多轮terminal/finish→完整trajectory。
- [x] 拟新增 `src/verl_adapter/tools.py`：原tool schema→sandbox请求/结果，保持SDK消息与结构预测语义。
- [x] 拟新增 `src/verl_adapter/reward.py`：episode_id+预测→私有scorer→reward/分级指标，不返回gold。
- [x] 拟新增 `configs/verl/codescout_4b.yaml` 与 `scripts/run_verl_4b.sh`，由verl官方trainer负责采样、更新、checkpoint。
- [x] 检查Hermes parser与Qwen chat_template实际兼容；关闭thinking与原脚本一致，多工具并行不丢调用。
- [x] 保存原始采样token IDs、behavior logprobs、attention/response loss mask、action边界、policy version及tool文本。
- [x] 仅模型输出计policy loss，用户/system/tool结果/prefix mask为0；后续每轮生成token都覆盖，不只训练最后finish。
- [x] 不把SDK重写message重新tokenize当作原采样；逐轮prefix与模板一致性审计，截断/EOS/stop规则明确。
- [x] 组内8条来自相同issue与同一采样策略；GRPO优势中心化但不除标准差，全同reward组为0且报告比例。
- [x] GSPO保持sequence级importance ratio与low/high clip 0.0003/0.0004，原KL两项关闭、lr1e-6、每batch1个update epoch。
- [x] 优化器采用论文 AdamW、恒定学习率1e-6，显式关闭entropy loss；betas/epsilon/weight_decay从固定SkyRL配置解析后锁定，不能继承新框架不同默认值。
- [x] 验收一次batch确为8任务×8轨迹=64 episodes，核verl mini-batch配置是否含rollout展开维度；记录optimizer.step实际次数、梯度累积与scheduler步数，不把一轮遍历和一次参数更新混用。
- [x] 原SkyRL `sequence_mean` 与verl `seq-mean-token-mean`不能按名字直接等同；用固定张量算ratio/advantage/loss/梯度做数值对照。
- [x] 多轮trajectory究竟按整episode还是单请求归一化，先查原训练器；匹配非padding生成token权重，差异记录为方法偏差。
- [x] 不静默换PPO/token级ratio或加KL；官方verl不支持确切语义时在其扩展点补小型loss adapter并测试，禁止另造trainer。
- [x] 先同步batch验证正确性，明确该结果只是同步迁移验收，不能称原fully-async复现。
- [x] 查原 `src/async_trainer.py` 的队列、更新与同步策略，正式冻结async/staleness上限、丢弃规则及off-policy修正。
- [x] 原fully-async允许最多4 updates陈旧；verl的staleness ratio/队列参数不是同一单位，不能把数值4直接照填。
- [x] 复现原耗尽预算但未finish的零reward/全零loss-mask行为，检查是否浪费整组或形成选择偏差；不擅自给这些动作加训练loss。
- [x] 若所选verl缺原fully-async能力，正式主实验使用同步配置并在标题/结论披露；改变异步协议需重做匹配基线与预算。
- [x] 收集group有效episode/token、overflow、截断、工具轮数、陈旧policy版本，不只比较训练update计数。
- [x] 拟新增 `scripts/audit_trajectory.py` 与训练语义测试；完整审计输入/输出保存在run目录，摘要入Git。【已实现并运行：11 个 run 审计摘要（9 评测 + s17/s29 训练日志）入 results/experiments/codescout_verl_4b_v1/audit/；全同 reward（全零 advantage）step：s17 训练 200 步中 7 步（3.5%）；训练时未开启逐任务审计通道为已知数据缺口，已在脚本 note 中记录。】

通过条件：固定张量损失/梯度对齐、真实多轮token完整；关键差异未解释时禁止进入效果实验。

## 5．训练前基线及公开模型参考

入口：协议v1、数据/工具/评分冻结。产物：可匹配的未训练基线和独立公开参考。

- [x] 拟新增 `scripts/evaluate_localization.py`：模型+split+解码配置→逐任务结构预测、事件、指标、成本。
- [x] dev基线使用冻结100任务，统一论文正式温度0.7/top_k20/top_p0.8，每任务一次episode；原训练内eval温度0.6只记为差异，训练温度1.0。
- [x] 正式评测核论文温度0.7、top_k20、top_p0.8、上下文132K；评测prompt6turn/backend15turn差异必须与原fork源码共同冻结。
- [x] 正式132K与训练40960窗口分别验收；模型上下文设置、模板或资源不可支持时公开降级并重做匹配base/RL，不能混比论文数字。
- [x] 测试解码参数先查官方benchmark配置；如官方评测不同，分别报告原官方协议参考和本项目匹配协议。【已核（official-eval-fork-audit.md）：官方 fork 温度/top_p 在未提交的 llm_config 中不可考、轮数=15、分母硬编码、真值镜像与本项目所用 locagent 镜像有 73/500 任务级 file_changes 差异；官方协议参考评测（15 轮 + 官方真值 + 4 模型）单独运行，与本项目匹配协议双口径分报。】
- [x] 评测fork说明同样为 `More details coming soon`，需读实际 `run_infer.py` 与workspace_base_dir参数，默认 `/tmp` 适配到允许根。【已核：fork @7cf83b8 run_infer.sh 用 `--runtime local --workspace_base_dir /tmp/testbed/`；本项目适配为 codescout-data/workspaces（无 .git 快照），语义一致。】
- [x] 全部对照使用相同turn/上下文/输出预算、tool结果裁剪、终止条件、采样seed；不能让RL模型多搜几次。
- [x] 官方CodeScout4B仅评估公开checkpoint作为参考，不参与训练/选数据；没有可比协议不直接横比论文表数字。【完成：官方协议参考口径（15 轮+官方 code-search 真值，official-eval-fork-audit.md）下四模型同管线互比——Verified 0.268/0.815/0.783/0.730（base/s17/s29/官方CodeScout-4B）、Lite 0.219/0.636/0.552/0.497、Pro 0.039/0.084/0.078/0.070；本项目复现全面达到并略超官方发布模型；因真值镜像差异（73/500）与官方温度不可考，不与论文表数字直接横比。】
- [x] Verified完整500任务留到协议/模型选择冻结后；此时只锁输入与评测脚本，不查看结果调参。
- [x] 保存模型revision、评测代码commit、每任务seed和tool轨迹，记录解析失败率、成本及各级precision/recall/F1。

通过条件：两次确定性scorer重算一致，预测与任务ID完整对齐；随机采样差异不伪装成RL效果。

## 6．更新、恢复与8卡代表性 profile

入口：基线完整、GPU执行预算确认。产物：训练正确性与容量测量，不先跑长实验。

- [x] 现场重查资源；先2–4卡最小correctness smoke，原8卡配置作为首个代表profile，不默认用共享32卡。
- [x] GPU准入满足预计峰值+max(8GiB,峰值15%)≤实时空闲显存，profile后按实际波动更新余量。
- [x] 8卡目标用verl resource pool验收FSDP2 actor与TP1 rollout；共置sleep/wakeup或分置选择以端到端吞吐决定。
- [x] 原SkyRL4 actor+4 inference不是verl默认拓扑；记录实际卡号/UUID、rank、SP/TP、Python和端口。
- [x] 原train batch8 issues×8 trajectories；microbatch1、有效batch不变，OOM优先降并发而非删长样本。
- [x] 原脚本输入40960/生成8192；先核是否每次调用预算、总模型窗口相加合法，不能错误承诺49K上下文。
- [x] 原8192是每次LLM调用生成上限，不是整episode上限；冻结多轮上下文/工具输出裁剪和总token统计，不悄缩全轨迹预算。
- [x] 原 `CUDA_LAUNCH_BLOCKING=1`、DSA是调试配置；正式性能profile关闭或分别测量，匹配各臂性能设置。
- [x] 对代表性及最长轨迹测加载、optimizer、KV、更新/权重刷新/评测切换峰值；出现截断保存比例。
- [x] 至少连续3次optimizer更新，核参数差异、非零有效梯度、第一/后续动作loss覆盖与权重版本刷新。
- [x] 在中途checkpoint保存、停本作业、恢复并再更新；核optimizer/RNG/数据游标/policy/async队列恢复语义。
- [x] 恢复对照核同一固定输入行为及下一update，不能只有模型权重文件就称可续训。
- [x] checkpoint每10 updates，HF export每50 updates沿原配方；只保留5份前先做可恢复保留策略，不删唯一证据。
- [x] 记录有效episode/hour、token/s、CPU搜索、clone/IO、GPU等待、权重刷新耗时、峰值、墙钟及预留GPUh。
- [x] 输出profile报告，决定8卡正式配置；扩16/32或并行seed需当次授权资源及预算，不因hostfile slots自动扩量。

通过条件：更新、恢复、权重刷新、最长轨迹均验收；否则只能报告工程诊断，不启动主实验。

## 7．快速否证与主训练

入口：步骤6及冻结预算。产物：两个seed训练与dev选择记录。

- [x] 冻结训练seed17/29；每seed共享起点、数据、奖励、GSPO和生成参数，仅随机性不同。
- [x] 第一seed先10 updates排除reward全常数、梯度失效、工具失控和标签泄漏；再40 updates观察dev与成本。
- [x] 40update无明确涨分但正确性通过：决定inconclusive，不自动改变reward；在冻结额度允许时完成主预算。
- [x] 主训练按论文200 updates，每update8个issue×8采样，实际有效episode/token另计；执行额度仍须先冻结，不因为写计划就自动启动。
- [x] 论文200 steps作为目标参考，每seed对应1600次task抽取/12800候选episode；原脚本 `epochs=1` 不等于200steps，不训练整39K一轮再冒称原预算。
- [x] 200update预算按论文，样本顺序/丢弃/陈旧重采样按源码核验；作者manifest不可得时明确样本级差异，不称严格同轨迹复现。
- [x] 每10updates记录checkpoint与训练统计；dev在0/10/40/100/200评测，冻结这个选择日程，不增测追好checkpoint。
- [x] 用dev主指标选checkpoint，平分选更早者；未在测试看结果前锁最终checkpoint hash。
- [x] 梯度非有限、评分器异常、泄漏、动作漏mask立即stop；infra有限重试，失败成本保留。
- [x] 不根据训练reward上涨单独宣布有效；全同reward组、格式失败、长度增长、过报位置分别分析。【分析已做：全同 reward 组 3.5%（audit/rl-s17-training.json）；failure_class 分列与文件级 miss/over/both_low 模式分析（error-analysis-verified.json：base no_finish 226/499 → s17 1/499，s17 file_perfect 172 vs base 75、过报 40 vs 28）；有效性判定基于冻结测试集对照而非训练 reward。】
- [x] 第二seed按同一配置执行；8卡串行默认，若授权且profile通过可两个8卡并行，其余卡不自动占用。

## 8．冻结最终评测与分析

入口：checkpoint、协议及所有配置冻结。产物：完整对照结果与研究决定。

- [x] 对未训练起点及两个RL seed，执行Verified500 actor-only单episode，匹配sampling seed和所有推理预算。
- [x] Pro/Lite作为泛化补充，提前冻结使用哪些split；与Verified分开，不将不同分母合成唯一成功率。
- [x] 审计公开版本差异：论文train约39K/128repo，公开SWE-smith入口显示39287/131repo；Pro论文266而公开定位数据264，锁revision后报告实际分母。
- [x] 各任务记录file/module/entity precision/recall/F1、sum、合法finish、轮数、token、时间、失败类别。
- [x] 模型无finish/错误预测计0；infra失败按预冻结规则最多重试2次，仍失败报告未完成及含失败保守指标。
- [x] 报告每seed差值，不只汇总最好的seed；用任务配对bootstrap 10000次、固定seed计算95%CI。
- [x] 另按repo聚类bootstrap敏感性分析；仓库少时明确不确定性，不能只任务CI假装跨仓库稳健。
- [x] 两个seed分别及平均预测结果做任务配对bootstrap；这是固定训练seed条件下的任务不确定性，不能当作充分估计训练随机性的置信区间。
- [x] 提议有意义收益门槛：总F1 sum宏平均在0–3尺度提升≥0.10（除3归一化后约3.33个百分点），且两seed方向同为正、平均差值的任务配对CI下界>0；这是本项目预设非论文阈值。
- [x] 稳健收益同时检查分级F1、precision下降、token/成本膨胀；未满足门槛按未明确收益/退化/证据不足解释。
- [x] 结果表包含数据分母、base、RL两个seed、公开模型参考、差值/CI、token/turn/time/GPUh及排除数。
- [x] 分析错误：搜索方向、漏文件、过报、类/函数层级、模板解析、轮数耗尽、长上下文、环境异常。【分布分析：error-analysis-verified.json + audit/*.json——轮数耗尽 base 10/s17 1/s29 35；no_finish base 226→s17 1；模板解析 base 1；文件级错误结构：both_low 为主（270/497）、过报 40、漏报 15（s17）；类/函数层级 F1 递减（module<file、entity<module）见 final-report §1 表。搜索方向/长上下文细类未逐轨迹深挖（有限度，已注明）。】
- [x] 定位能力改善只支持本任务结论；未经修复agent独立实验不写“提高代码修复成功率”。
- [x] 论文4B参考Verified的file/module/function F1为base49.73/19.32/13.27→RL68.52/45.97/36.78（百分数）；仅作论文证据，不能预填本项目结果。
- [x] 固定最终test反馈仅用于报告，若另开方法修订需要新版本/新开发决定，不能复用测试调参冒充冻结结果。

## 9．交付、同步与接手

- [x] 输出 `docs/reproduction/final-report.md`：方法、框架偏差、指标/CI、负结果、成本、限制及下一步。
- [x] 输出 `docs/HANDOVER-codescout-verl.md`：完成/未完成、实际环境、命令、run_id、进程/日志/checkpoint/恢复、剩预算。
- [x] 更新index/status和本todo，只勾有对应证据的项；文件存在或进程退出不代表方法有效。
- [x] 小型指标、manifest、配置/依赖锁、结果审计入Git；完整轨迹/模型/数据留运行时并给恢复清单。
- [x] 只清理本作业进程，停止服务后再次核对PID归属；checkpoint清理前备份并记录BACKED_UP.md。【处置：全部作业进程退出后实测 GPU 0 MiB、0 训练进程、Ray session 已终结（三端核对于 1fad3ac）；无 checkpoint 删除——保留策略=保留全部最终产物（磁盘余量 3.4P 无压力），故无需 BACKED_UP.md；后续若清理需先按 AGENTS §4 建备份记录。】
- [ ] 提交/push/服务器同步按用户授权与AGENTS进行，核三端commit和两端工作树；不同步如实写出。

## 资源与预算：profile后冻结

| 波次 | 依赖 | 建议资源 | 上限/估算规则 | 输出 |
|---|---|---|---|---|
| 数据与oracle | 环境/数据授权 | CPU受限服务，GPU0 | 20任务；磁盘展开量待测 | 可重建repo与oracle |
| 模型/更新smoke | 步骤3–4 | 2–4张获准H100 | 3updates+恢复；时长待测 | 正确性报告 |
| 代表profile | smoke通过 | 8张获准H100 | 包含最长轨迹与切换 | 峰值/有效吞吐 |
| 快速否证 | profile通过 | 8张 | seed17，10→40updates | continue/stop决定 |
| 主训练 | 协议/预算冻结 | 默认8张串行；获准可双8卡 | 每seed200updates，按论文目标 | 两seedcheckpoint |
| 最终评测 | 模型冻结 | 由峰值与吞吐分片决定 | 500×3模型，Pro/Lite另计 | 逐任务结果与CI |

profile前墙钟/GPUh未知，不把原论文8卡推断成固定训练成本。主代理先明确本轮执行额度。
估算：训练GPUh≈预留卡数×updates÷实测updates/hour；评测GPUh≈预留卡数×任务数÷实测有效任务/hour。
加上smoke、恢复、失败、CPU环境等待、开发评测和导出时间，预留20%工程余量仅作初估。
每阶段预算耗尽即停止并交付证据；有效吞吐提升不自动增加任务数、卡时或seed。

## 拟实现的入口调用约定

以下是模块完成后的接口示例，不是已可运行命令；实施者须使参数与最终代码一致。

```bash
cd /mmu_vlm_hdd/home/rhsu/playground/codescout
source scripts/project_env.sh
bash scripts/setup_conda_verl.sh --lock manifests/environment/conda-lock.yml
"$CODESCOUT_PYTHON" scripts/check_environment.py --output manifests/environment/check.json
"$CODESCOUT_PYTHON" scripts/prepare_verl_data.py --manifest manifests/data/protocol-v1.json
bash scripts/run_verl_4b.sh --config configs/verl/codescout_4b.yaml --run-id cs4b-rl-s17
"$CODESCOUT_PYTHON" scripts/evaluate_localization.py --run-id cs4b-test-s17 --split verified --config configs/eval/protocol-v1.yaml
```

## 给接手主代理的指令

先读AGENTS和本计划，检查真实commit/工作树/实例/作业，不把计划项当已经完成。
按0→1→2→3→4→5→6→7→8→9推进，环境与data审计可分工但主实验不能越过oracle、token、更新和恢复验收。
仅用verl与隔离conda；原SkyRL源码作语义参考；不改主模型、奖励、测试划分或悄加PRM。
独立工作包委派负责人，GPU负责人读指定技能，所有执行在核验后的kml-1005允许根。
完成当前授权工作，再依据阶段门槛和冻结预算继续；缺少训练额度不自行启动长实验。
每次尝试包括失败写账本，方法无收益交付真实结论，不保证或预写正结果。
