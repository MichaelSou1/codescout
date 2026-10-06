"""verl custom reward function：调用冻结 scorer，输出 reward + 分级指标。

协议/设计文档出处
------------------
- docs/reproduction/protocol-v1.md §4（奖励冻结）：唯一奖励函数
  ``multilevel_localization_f1_reward``，权重 file/module/entity = 1/1/1，范围
  0-3；无 finish/多 finish/格式错误/sanity 失败/轮数耗尽均为**有效模型失败**，
  reward 0；infra 异常与模型失败分列。
- docs/reproduction/verl-adapter-design.md §5.1（``reward.custom_reward_function``
  注册通道与 naive reward manager 的调用签名：``compute_score(data_source=...,
  solution_str=..., ground_truth=..., extra_info=...)``，返回 dict 必含
  ``"score"``，其余键进 ``reward_extra_info``）、§5.2（私有标签数据流：
  ``ground_truth`` 即 parquet ``reward_model.ground_truth`` 列（私有标签），只经
  dataset 列到达 reward fn，**不进 prompt**）、§5.3（不用 loop 内自算 reward，
  走标准 reward manager + custom_reward_function，便于逐任务审计）。

设计要点
--------
- **不复制粘贴 scorer**：直接 import ``src.rewards.file_localization.
  file_localization.multilevel_localization_f1_reward``（冻结实现，不得修改）。
- **不走 solution_str 重解析**（B-4：naive manager 解码 response 时
  ``skip_special_tokens=True``，各轮文本粘连、标记被剥，重解析脆弱）——结构化
  预测取自 ``extra_info["codesearch_structured_locations"]``（agent loop 经
  ``tool_extra_fields`` 通道写入，设计文档 §5.1）。
- 判定逻辑复用 :mod:`src.verl_adapter.semantics`（与 agent loop 同一入口，
  保证两层判罚一致）。
- verl 的 ``get_custom_reward_fn`` 以**文件路径**动态加载本模块
  （verl/trainer/ppo/reward.py:50-86），因此这里不能做包相对导入：模块顶部先
  把仓库根插入 ``sys.path``，再用绝对包名导入（服务器从 checkout 根启动，路径
  亦天然可见；见 scripts/run_verl_4b.sh 的 PYTHONPATH 导出）。

用法（verl 配置键）::

    reward.custom_reward_function.path=<repo>/src/verl_adapter/reward.py
    reward.custom_reward_function.name=compute_score
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - 环境相关
    sys.path.insert(0, str(_REPO_ROOT))

# 冻结 scorer（不得修改 src/ 下既有语义文件；此处只 import）
from src.rewards.file_localization.file_localization import (  # noqa: E402
    multilevel_localization_f1_reward,
)
from src.rewards.file_localization.module_rewards import (  # noqa: E402
    parse_structured_outputs,
)

from src.verl_adapter import semantics  # noqa: E402

__all__ = ["compute_score"]


def _extract_file_changes(ground_truth: Any) -> Any:
    """从 ``reward_model.ground_truth`` 列提取 ``file_changes`` 私有标签。

    build_prompts.py 写入的形态为 ``{"file_changes": [...]}``。形态不符直接抛
    ``ValueError``：这是数据构建合同错误（infra），必须在 oracle 阶段暴露，
    不能静默按 0 分混入模型失败（协议 §4 分列要求）。
    """
    if isinstance(ground_truth, dict) and "file_changes" in ground_truth:
        file_changes = ground_truth["file_changes"]
        if not isinstance(file_changes, list):
            raise ValueError(
                "reward_model.ground_truth['file_changes'] must be a list; got: "
                f"{type(file_changes).__name__} (malformed labels — fix data build, "
                "do not silently score 0)"
            )
        return file_changes
    raise ValueError(
        "reward_model.ground_truth must be an object with a 'file_changes' key; got: "
        f"{type(ground_truth).__name__}"
        + (f" with keys {sorted(ground_truth.keys())}" if isinstance(ground_truth, dict) else "")
    )


def _ground_truth_sets(file_changes: Any) -> dict[str, set[str]]:
    """审计用真值集合提取（口径逐点复刻冻结 scorer 内部逻辑，仅用于 P/R 指标；
    **得分本身始终来自冻结 scorer**，本函数不参与）。"""
    gt_files: set[str] = set()
    gt_modules: set[str] = set()
    gt_entities: set[str] = set()
    changes = file_changes if isinstance(file_changes, list) else []
    for change in changes:
        if not isinstance(change, dict):
            continue
        if "file" in change:
            gt_files.add(change["file"])
        detail = change.get("changes")
        if isinstance(detail, dict):
            modules = detail.get("edited_modules") or []
            entities = detail.get("edited_entities") or []
            gt_modules.update(m for m in modules if isinstance(m, str))
            gt_entities.update(e for e in entities if isinstance(e, str))
    return {"file": gt_files, "module": gt_modules, "entity": gt_entities}


def _precision_recall_audit(
    locations: list[dict[str, Any]],
    file_changes: Any,
) -> dict[str, float]:
    """三级 precision/recall 审计指标（协议 §4"各级 precision/recall/F1 与总
    奖励分列"的账本要求；公式同 ``compute_file_f1_score`` 的 P/R 定义）。"""
    pred_files, pred_modules, pred_entities = parse_structured_outputs(locations)
    gt = _ground_truth_sets(file_changes)
    out: dict[str, float] = {}
    for level, pred in (("file", pred_files), ("module", pred_modules), ("entity", pred_entities)):
        pred_set, gt_set = set(pred), gt[level]
        tp = len(pred_set & gt_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gt_set) if gt_set else 0.0
        out[f"{level}_precision"] = float(precision)
        out[f"{level}_recall"] = float(recall)
    return out


def compute_score(
    data_source: str,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> dict:
    """verl reward 入口；返回 dict 必含 ``"score"``（naive manager 合同）。

    Args:
        data_source: parquet ``data_source`` 列（本项目常量
            ``codescout_swe_smith_py``，semantic.py；原样记录便于混源审计）。
        solution_str: 最后序列解码文本（``skip_special_tokens=True``，B-4）——
            本函数**不使用**（结构化预测走 extra_fields 通道），保留参数以匹配
            verl 调用签名。
        ground_truth: ``reward_model.ground_truth`` 列 = ``{"file_changes": ...}``
            私有标签（协议 §8：只经本通道进 reward，绝不进 prompt）。
        extra_info: dataset ``extra_info`` 列 ∪ ``tool_extra_fields``（agent loop
            写入的 ``codesearch_*``）∪ ``num_turns`` 等。

    Returns:
        dict：``score``（0-3）+ ``file_reward``/``module_reward``/``entity_reward``
        （未加权 F1，键名对齐冻结 scorer 的 reward_dict）+ ``failure_class``
        （None|"no_finish"|"multi_finish"|"parse_error"|"sanity_check"|
        "exhausted"|"infra"）+ 三级 precision/recall 审计 + 轮数/终止审计。
    """
    extra_info = dict(extra_info or {})
    result: dict[str, Any] = {
        "data_source": data_source,
        "finish_call_count": int(extra_info.get("codesearch_finish_call_count", 0) or 0),
        "trajectory_exhausted": bool(extra_info.get("codesearch_trajectory_exhausted", False)),
        "num_turns": extra_info.get("num_turns"),
    }

    locations, failure_class = semantics.resolve_prediction(extra_info)
    result["failure_class"] = failure_class

    if failure_class == "infra":
        # infra 异常（协议 §4）：reward 张量填 0 使训练不崩，但 failure_class
        # 分列上报，账本按冻结排除规则剔除，不计为模型 0 分。
        result["score"] = 0.0
        _audit_emit(extra_info, result)
        return result

    if locations is None:
        # 空预测/无 finish/多 finish/格式错/sanity 失败/轮数耗尽 → 0（原语义）。
        result["score"] = 0.0
        result["file_reward"] = 0.0
        result["module_reward"] = 0.0
        result["entity_reward"] = 0.0
        _audit_emit(extra_info, result)
        return result

    file_changes = _extract_file_changes(ground_truth)
    # 冻结 scorer：权重 1/1/1（协议 §4；configs/reward_config_4b.yaml 同源默认）。
    # final_message 在 structured 路径下被原实现忽略，传空串占位。
    reward, reward_dict = multilevel_localization_f1_reward(
        "",
        {"file_changes": file_changes},
        structured_locations=locations,
    )
    result["score"] = float(reward)
    result["file_reward"] = float(reward_dict["file_reward"])
    result["module_reward"] = float(reward_dict["module_reward"])
    result["entity_reward"] = float(reward_dict["entity_reward"])
    result.update(_precision_recall_audit(locations, file_changes))
    _audit_emit(extra_info, result)
    return result


def _audit_emit(extra_info: dict[str, Any], result: dict[str, Any]) -> None:
    """逐任务审计（协议 §4"分列"与步骤 5 逐任务指标要求）。

    环境变量 ``CODESCOUT_REWARD_AUDIT_PATH`` 设置时逐条 append JSONL
    （instance_id/三级 F1/P-R/失败分类/轮数）；不设置时零开销直通。
    评测与训练通用：审计行只含指标，绝不含 gold 内容（私有标签不出边界）。
    """
    import json as _json
    import os as _os

    path = _os.environ.get("CODESCOUT_REWARD_AUDIT_PATH")
    if not path:
        return
    row = {
        "instance_id": extra_info.get("instance_id"),
        "episode_id": extra_info.get("episode_id"),
        "data_source": result.get("data_source"),
        "score": result.get("score"),
        "file_reward": result.get("file_reward"),
        "module_reward": result.get("module_reward"),
        "entity_reward": result.get("entity_reward"),
        "failure_class": result.get("failure_class"),
        "num_turns": result.get("num_turns"),
        "finish_call_count": result.get("finish_call_count"),
        "trajectory_exhausted": result.get("trajectory_exhausted"),
    }
    row.update(
        {k: v for k, v in result.items() if k.endswith("_precision") or k.endswith("_recall")}
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(row, ensure_ascii=False) + "\n")
