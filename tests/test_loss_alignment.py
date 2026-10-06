# -*- coding: utf-8 -*-
"""SkyRL(81e5a97c) ↔ verl(v0.9.1) GSPO/GRPO 数值对齐测试（protocol-v1 §6 / skyrl-to-verl §2）。

冻结协议出处：
- docs/reproduction/protocol-v1.md §6：GSPO sequence 级 importance ratio、
  clip eps_low=0.0003 / eps_high=0.0004、loss reduction sequence_mean、
  GRPO 组内 8 条、中心化不除以标准差（全同 reward 组 advantage=0）。
- docs/reproduction/skyrl-to-verl.md §2：SkyRL `gspo_policy_loss` +
  `reduce_loss(sequence_mean)` ↔ verl `compute_policy_loss_gspo` +
  `loss_agg_mode="seq-mean-token-mean"`；GRPO `grpo_norm_by_std=false` ↔
  `norm_adv_by_std_in_grpo=False`。对齐判据 = 同全局 batch 一次 optimizer step
  的最终梯度逐张量对照（本文用固定张量的 loss/梯度逐步对照作诊断）。

参考实现（逐字复制进本文件，见下 `_SKYRL` 段；出处注释随函数标注）：
- SkyRL commit 81e5a97c，`skyrl_train/utils/ppo_utils.py`：
  `gspo_policy_loss` / `reduce_loss` / `masked_mean` /
  `compute_grpo_outcome_advantage`（仅复制本测试所需，未复制整个文件）。
- verl v0.9.1（服务器 verl env 安装；本机无）：
  `verl/trainer/ppo/core_algos.py`：`compute_policy_loss_gspo`（:1545-1621）、
  `agg_loss`（:1140-1207，seq-mean-token-mean 分支 :1194-1202）、
  `compute_grpo_outcome_advantage`（:267-331）；
  `verl/workers/config/actor.py`：`ActorConfig`（:104-231，
  clip_ratio_low/high :158-160、loss_agg_mode :164、global_batch_info :192）；
  `verl/workers/utils/losses.py` `ppo_loss()`：生产路径把
  dp_size / batch_num_tokens / global_batch_size / loss_scale_factor 写进
  `config.global_batch_info` 再 `**config.global_batch_info` 传给 agg_loss。

======================================================================
梯度聚合口径记录（D1/D2，任务第 5 项）——哪些 case 相等、哪些不等、差多少倍：

设 batch 共 R 行（含全 mask 行），非全 mask 行数 R+ ≥ 1，第 i 行有效长度
L_i = Σ_t mask_{i,t}；s_i 为该行 token-mean 损失。

- verl `agg_loss("seq-mean-token-mean")`（dp_size=1、global_batch_size=N）：
  系数 = 1/N，分子 = Σ_{非全mask行} S_i/(L_i + 1e-8)   （+1e-8 为 D1 硬编码，
  core_algos.py:1195；全 mask 行被 seq_mask 剔除）。
- SkyRL `reduce_loss(..., "sequence_mean")`：
  系数 = 1/R，分子 = Σ_{全部行} S_i/max(L_i, 1)   （全 mask 行贡献恰为 0，
  但分母 R 计入它——skyrl-to-verl §2 所述 D1）。

推论（由 test_verl_agg_loss_scaling_semantics 数值断言）：
1) N == R 时两者相等（唯一残差 = verl 每序列分母 +1e-8，量级 ≤ 1e-8·Σ|s_i|/(L_i·R)；
   本文件全部固定 case 的该偏置 < 1e-8，故 cross 断言用 ≤ 1e-8 成立）。
2) N != R 时 loss_verl(N) = loss_skyrl × R/N（分子相同）。典型：含 1 条全 mask 行的
   3 行 batch 取 N=R+=2 时差 R/R+ = 3/2 倍；N=64（生产 global_batch_size =
   ppo_mini_batch_size × rollout.n）而微批只有 R 行时差 R/64 倍——D2。
3) N 为 None（global_batch_info 留空）时 verl 回退 N = 本 micro-batch 非全 mask 行数
   （core_algos.py:1198-1201），与生产"配置值 N"不同；全 mask batch 时 0/0 → NaN。

======================================================================
运行环境分层（本文件必须在无 torch 的本机可 import 且全 skip）：

1. 无 torch：每个测试经 `_skip` 跳过并给出理由（= 整文件 skip）。
2. 有 torch、无 verl：只跑 SkyRL 参考层自洽性（测试 1-4：复制实现 vs
   独立 numpy float64 手工推导 oracle，loss/梯度/advantage 各自对账）。
3. 有 torch 且有 verl（服务器 verl env）：另跑全对齐层（测试 5-10：
   verl 实现 vs SkyRL 复制实现 vs oracle 三方对账）。

运行方式（兼容 pytest 与 __main__ 双运行器）：

    pytest tests/test_loss_alignment.py            # pytest 环境
    $CODESCOUT_PYTHON tests/test_loss_alignment.py # __main__ 直跑（本机/服务器均可）

全部张量为固定值（无随机数）、CPU、float64。
"""

from __future__ import annotations

import math
import sys
import types
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

try:
    import torch

    _TORCH_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:  # pragma: no cover - 环境探测
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = "%s: %s" % (type(_exc).__name__, _exc)

try:
    import numpy as np

    _NUMPY_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:  # pragma: no cover - 环境探测
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR = "%s: %s" % (type(_exc).__name__, _exc)


# ---------------------------------------------------------------------------
# 分层 skip helper（pytest / __main__ 双运行器；内联同款，不 import 其他测试文件）
# ---------------------------------------------------------------------------


class SkipTest(Exception):
    """分层 skip：对应依赖在本环境不可用。"""


def _skip(msg: str) -> None:
    """pytest 下走 pytest.skip；__main__ 直跑时抛本模块 SkipTest 由运行器捕获。"""
    if "pytest" in sys.modules:
        import pytest

        pytest.skip(msg)
    raise SkipTest(msg)


def _require_layer1() -> None:
    """SkyRL 参考层需要 torch（numpy 作 oracle，torch 环境必有）。"""
    if _TORCH_IMPORT_ERROR is not None:
        _skip(
            "torch 不可用（%s）——本文件全部数值对齐测试需要 torch；"
            "本机无 torch 属预期（开发约定：依赖安装在 KML 服务器执行）"
            % _TORCH_IMPORT_ERROR
        )
    if _NUMPY_IMPORT_ERROR is not None:
        _skip("numpy 不可用（%s）——oracle 推导需要 numpy" % _NUMPY_IMPORT_ERROR)


# ---------------------------------------------------------------------------
# SkyRL 冻结参考实现（逐字复制）
# 出处：SkyRL commit 81e5a97c，skyrl_train/utils/ppo_utils.py。
# 复制范围 = 本测试所需：masked_mean / reduce_loss / gspo_policy_loss /
# compute_grpo_outcome_advantage。只删去 @register_* 装饰器（需要 SkyRL 的
# Ray registry，与本文件语义无关），其余逐字保留，不改写语义。
# 原文件的 DictConfig / np.ndarray 等注解经 `from __future__ import annotations`
# 惰性化，本模块无需安装 omegaconf/loguru/ray（gspo 中 loguru 为惰性 import，
# 仅在 loss_reduction != "sequence_mean" 的告警分支触发，本测试不会命中）。
# ---------------------------------------------------------------------------


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: Optional[int] = None) -> torch.Tensor:
    # SkyRL ppo_utils.py:85-88（commit 81e5a97c），逐字复制。
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim).clamp(min=1.0)


def reduce_loss(
    loss: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    loss_reduction: Literal["token_mean", "sequence_mean", "seq_mean_token_sum_norm"],
    max_seq_len: Optional[int] = None,
) -> torch.Tensor:
    # SkyRL ppo_utils.py:811-836（commit 81e5a97c），逐字复制。
    if loss_reduction == "token_mean":
        # sum over *all* valid tokens, divide by total valid-token count
        loss = masked_mean(loss, loss_mask)
    elif loss_reduction == "sequence_mean":
        # per-sequence token-mean (dim=-1), then batch-mean
        loss = masked_mean(loss, loss_mask, dim=-1).mean()
    elif loss_reduction == "seq_mean_token_sum_norm":
        # per-sequence token-sum, normalized by the max sequence length, then batch mean
        # this is the Dr. GRPO loss reduction to avoid length bias by normalizing by a constant
        assert max_seq_len is not None, "max_seq_len must be provided for seq_mean_token_sum_norm loss reduction"
        # NOTE: max_seq_len is computed as cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length by default
        if loss_mask is not None:
            seq_losses = torch.sum(loss * loss_mask, dim=-1) / max_seq_len
        else:
            # If no mask, assume all tokens are valid
            seq_losses = torch.sum(loss, dim=-1) / max_seq_len
        loss = torch.mean(seq_losses)
    else:
        raise ValueError(f"Invalid loss reduction type: {loss_reduction}")
    return loss


def gspo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    GSPO (Group Sequence Policy Optimization) policy loss function,
    as proposed in https://arxiv.org/abs/2507.18071.

    This implements sequence-level importance sampling instead of token-level importance sampling.
    The key difference is that importance weights are computed at the sequence level and then
    applied uniformly across all tokens in the sequence. This can lead to more stable training
    dynamics by reducing the variance in clipping behavior within sequences.

    The variant of GSPO used here is GSPO-token, a generalization which allows for token-level
    advantages [equations 14 and 15 in the paper].
    """
    # GSPO must use sequence_mean reduction
    loss_reduction = config.loss_reduction
    if loss_reduction != "sequence_mean":
        # The GSPO paper uses sequence_mean reduction; there's no reason
        # why a user couldn't use token_mean reduction, but it's
        # not clear whether it would be stable or not.
        from loguru import logger as logger_  # have to do lazy import to avoid pickling error

        logger_.warning(f"With GSPO it's recommended to use 'sequence_mean' loss reduction; got {loss_reduction}")

    # Compute log ratios
    log_ratio = log_probs - old_log_probs

    # Key GSPO innovation: sequence-level importance sampling
    # Instead of using per-token ratios, compute sequence-averaged ratios
    log_importance_weights = masked_mean(log_ratio, loss_mask, dim=-1).unsqueeze(-1)

    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_probs - sg[log_probs]
    # note: we put the addition at the end to avoid precision issues,
    # per https://github.com/volcengine/verl/pull/2775#discussion_r2241500280
    log_token_importance_weights = log_probs - log_probs.detach() + log_importance_weights.detach()
    # clip to avoid overflow
    log_token_importance_weights = torch.clamp(log_token_importance_weights, max=10)
    ratio = torch.exp(log_token_importance_weights)

    # Standard PPO surrogate objective with sequence-level importance weights
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
    loss = -torch.min(surr1, surr2)

    # Compute clipping ratio for monitoring
    clip_ratio = masked_mean((-surr2 > -surr1).float(), loss_mask).mean().detach().item()

    loss = reduce_loss(loss, loss_mask, loss_reduction, config.max_seq_len)

    return loss, clip_ratio


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    grpo_norm_by_std: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).

    Expects:
        - token_level_rewards: Float[torch.Tensor, "batch_size seqlen"]
        - response_mask: Float[torch.Tensor, "batch_size seqlen"]
        - index: np.ndarray (batch_size)
        - epsilon: float
        - grpo_norm_by_std: bool

    Returns:
        - advantages: Float[torch.Tensor, "batch_size seqlen"]
        - returns: Float[torch.Tensor, "batch_size seqlen"]
    """
    # this assumes response-level rewards
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if grpo_norm_by_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


# ---------------------------------------------------------------------------
# 冻结常量（protocol-v1 §6 / skyrl-to-verl §2）
# ---------------------------------------------------------------------------

EPS_LOW = 3e-4   # SkyRL eps_clip_low = verl actor.clip_ratio_low
EPS_HIGH = 4e-4  # SkyRL eps_clip_high = verl actor.clip_ratio_high
CLIP_RATIO_C = 10.0  # 双方 GSPO 路径均不读取该值（dual-clip 常量），按任务规约同值设置
LOG_CLAMP_MAX = 10.0  # 双方对 log token-importance 的硬编码 clamp（SkyRL :639 / verl :1603）
VERL_SEQ_DENOM_EPS = 1e-8  # verl agg_loss seq-mean-token-mean 每序列分母 +1e-8（D1）
TOL = 1e-8  # 任务规定的对齐容差（float64 CPU）


def _skyrl_config(
    eps_clip_low: float = EPS_LOW,
    eps_clip_high: float = EPS_HIGH,
    loss_reduction: str = "sequence_mean",
) -> types.SimpleNamespace:
    """SkyRL DictConfig 的属性兼容替身（字段名 = gspo_policy_loss/reduce_loss 消费的键）。

    生产中传入 OmegaConf DictConfig（ppo_base_config.yaml）；
    SimpleNamespace 与其属性访问语义一致，避免本文件依赖 omegaconf。
    """
    return types.SimpleNamespace(
        loss_reduction=loss_reduction,
        eps_clip_low=eps_clip_low,
        eps_clip_high=eps_clip_high,
        max_seq_len=None,  # sequence_mean 分支不读它
    )


# ---------------------------------------------------------------------------
# 固定张量 case 表（无随机数；float64）
# ---------------------------------------------------------------------------


def _mkcase(name: str, note: str, old: List[List[float]], delta: List[List[float]], adv: List[List[float]], mask: List[List[float]]) -> Dict[str, Any]:
    old_a = np.asarray(old, dtype=np.float64)
    delta_a = np.asarray(delta, dtype=np.float64)
    adv_a = np.asarray(adv, dtype=np.float64)
    mask_a = np.asarray(mask, dtype=np.float64)
    assert old_a.shape == delta_a.shape == adv_a.shape == mask_a.shape
    return {
        "name": name,
        "note": note,
        "old": old_a,
        "logp": old_a + delta_a,
        "adv": adv_a,
        "mask": mask_a,
    }


def _fixed_cases() -> List[Dict[str, Any]]:
    """≥8 组固定 GSPO loss case（覆盖：不同长度 / 全零 mask 行 / 单 token / 多行
    batch / advantage 正负混合 / ratio 低于、高于 clip 区间 / log-ratio clamp=10 /
    全 batch 全 mask / 新旧策略相同）。mask 之外的 delta/adv 故意放"垃圾值"，
    验证 mask 语义（不泄漏到 ratio/loss/梯度）。"""
    cases: List[Dict[str, Any]] = []

    # 1. 全 mask、2 行，ratio 在 clip 区间内部（interior）。
    cases.append(
        _mkcase(
            "uniform_interior_2x6",
            "r≈exp(±1.2e-5)，无 clip；adv 正负混合",
            old=[
                [-0.10, -0.20, -0.15, -0.05, -0.25, -0.30],
                [-0.12, -0.18, -0.22, -0.08, -0.28, -0.11],
            ],
            delta=[
                [2e-5, 0.5e-5, 3e-5, -1e-5, 2e-5, 0.0],
                [-1e-5, -3e-5, 1e-5, -4e-5, -2e-5, 2e-5],
            ],
            adv=[
                [1.5, -0.5, 2.0, -1.0, 0.25, -2.0],
                [1.0, 1.25, -1.5, 0.5, -0.25, 2.0],
            ],
            mask=[
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
        )
    )

    # 2. 变长序列：有效长度 5/3/1（pad 位 mask=0）。
    cases.append(
        _mkcase(
            "varying_lengths_531",
            "有效长度 5/3/1；masked 位放垃圾 delta/adv",
            old=[
                [-0.10 * (t + 1) for t in range(5)],
                [-0.10 * (t + 1) + 0.03 for t in range(5)],
                [-0.10 * (t + 1) + 0.06 for t in range(5)],
            ],
            delta=[
                [1e-5, 1e-5, 1e-5, 1e-5, 2e-5],
                [2e-5, -1e-5, 3e-5, 7e-5, -9e-5],
                [-2e-5, 8e-5, -6e-5, 4e-5, 1e-5],
            ],
            adv=[
                [1.0, -1.0, 0.5, -0.5, 2.0],
                [-2.0, 1.5, 0.25, 5.0, 5.0],
                [1.5, 5.0, 5.0, 5.0, 5.0],
            ],
            mask=[
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0],
            ],
        )
    )

    # 3. D1 关键 case：中间一行全 mask（mask 全零），垃圾 delta/adv 不许泄漏。
    cases.append(
        _mkcase(
            "all_mask_row_3rows",
            "row1 全 mask：SkyRL 计入 .mean() 分母（贡献 0），verl 从分子剔除、"
            "分母用配置 global_batch_size —— D1",
            old=[
                [-0.10 * (t + 1) for t in range(4)],
                [-0.10 * (t + 1) - 0.5 for t in range(4)],
                [-0.10 * (t + 1) + 0.2 for t in range(4)],
            ],
            delta=[
                [1e-5, 2e-5, -1e-5, 2e-5],
                [5e-5, -8e-5, 6e-5, -4e-5],  # 全 mask 行：不得影响 ratio
                [3e-5, -1e-5, 9e-5, -7e-5],
            ],
            adv=[
                [1.0, -1.0, 0.5, 2.0],
                [9.0, -9.0, 9.0, -9.0],  # 全 mask 行：不得影响 loss/梯度
                [-0.5, 1.5, 5.0, 5.0],
            ],
            mask=[
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
            ],
        )
    )

    # 4. 单 token 行（L=1）+ 一条 3-token 行。
    cases.append(
        _mkcase(
            "single_token_rows",
            "L=1 的行：+1e-8 分母的相对影响最大处",
            old=[
                [-0.10, -0.20, -0.30],
                [-0.13, -0.23, -0.33],
            ],
            delta=[
                [3e-5, 7e-5, -9e-5],
                [1e-5, -1e-5, 2e-5],
            ],
            adv=[
                [1.5, 0.0, 0.0],
                [0.75, -1.25, 2.0],
            ],
            mask=[
                [1, 0, 0],
                [1, 1, 1],
            ],
        )
    )

    # 5. 多行 batch（4×8），mask/adv 混合。
    cases.append(
        _mkcase(
            "multi_row_4x8_mixed",
            "4 行 8 列，交错 mask，adv 正负混合",
            old=[
                [-0.10 * (t + 1) + 0.03 * r for t in range(8)]
                for r in range(4)
            ],
            delta=[
                [2e-5, -1e-5, 3e-5, 1e-5, -2e-5, 0.5e-5, 2.5e-5, -1.5e-5],
                [1e-5, -2e-5, 2e-5, 4e-5, 0.0, 0.0, 0.0, 0.0],
                [1e-5, 6e-5, 2e-5, 6e-5, -1e-5, 6e-5, 3e-5, 6e-5],
                [2e-5, -1e-5, 5e-5, -5e-5, 3e-5, -3e-5, 4e-5, -4e-5],
            ],
            adv=[
                [1.0, -2.0, 0.5, 2.0, -1.5, 0.25, -0.75, 1.25],
                [-1.0, 1.5, 2.0, -0.5, 0.0, 0.0, 0.0, 0.0],
                [0.75, 5.0, -1.25, 5.0, 2.0, 5.0, -0.5, 5.0],
                [1.5, -1.25, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            ],
            mask=[
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
            ],
        )
    )

    # 6. ratio 低于 clip 下界（eps_low=3e-4 生效）。
    cases.append(
        _mkcase(
            "ratio_below_clip",
            "row0 m=-0.05→r≈0.9512<0.9997（A>0 保留梯度，A<0 被 clip 零梯度）；"
            "row1 r≈0.99986 仍在区间内",
            old=[
                [-0.10 * (t + 1) for t in range(4)],
                [-0.10 * (t + 1) - 0.3 for t in range(4)],
            ],
            delta=[
                [-0.05, -0.04, -0.06, -0.05],
                [-1e-4, -2e-4, -5e-5, -2e-4],
            ],
            adv=[
                [1.0, -1.0, 2.0, 0.5],
                [1.5, -0.5, 1.0, -2.0],
            ],
            mask=[
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        )
    )

    # 7. ratio 高于 clip 上界（eps_high=4e-4 生效）。
    cases.append(
        _mkcase(
            "ratio_above_clip",
            "row0 m=+0.03→r≈1.0305>1.0004（A<0 保留梯度，A>0 被 clip 零梯度）；"
            "row1 r≈1.0001 仍在区间内",
            old=[
                [-0.10 * (t + 1) for t in range(4)],
                [-0.10 * (t + 1) - 0.3 for t in range(4)],
            ],
            delta=[
                [0.03, 0.03, 0.03, 0.03],
                [1e-4, 2e-4, 1e-4, 0.0],
            ],
            adv=[
                [1.0, -1.0, 2.0, -0.5],
                [0.5, 1.0, -1.5, 2.0],
            ],
            mask=[
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        )
    )

    # 8. 序列均值 log-ratio=12 > 10：双方硬编码 clamp(max=10) 生效（r=e^10）。
    #    adv 取小量级：r=e^10 时 loss 值天然 ~e^10·|A|，+1e-8 分母的 D1 偏置
    #    会放大到 >1e-8；取小 adv 让偏置 <1e-8，同时保留 clamp/clip 分支覆盖。
    cases.append(
        _mkcase(
            "log_ratio_clamp_10",
            "m=12→clamp 10→r=e^10；梯度应全 0（clamp 截断梯度路径）；"
            "A>0 token 落在 clipped 分支（值 -A·(1+eps_high)），A<0 token 未 clip",
            old=[[-0.10 * (t + 1) for t in range(3)]],
            delta=[[12.0, 12.0, 12.0]],
            adv=[[2e-4, -1e-4, 1e-4]],
            mask=[[1, 1, 1]],
        )
    )

    # 9. 全 batch 全 mask（极端 D1：SkyRL 0/R；verl 0/N；fallback N=0 → NaN）。
    cases.append(
        _mkcase(
            "all_mask_batch",
            "全部行 mask=0：双方 loss/梯度恒 0；verl global_batch_info 留空时 "
            "global_batch_size 回退 0 → 0/0 = NaN（源码推导，服务器实测断言）",
            old=[[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]],
            delta=[[1e-5, -2e-5, 3e-5], [4e-5, -5e-5, 6e-5]],
            adv=[[1.0, -1.0, 2.0], [0.5, 2.0, -3.0]],
            mask=[[0, 0, 0], [0, 0, 0]],
        )
    )

    # 10. 新旧策略完全相同（ratio 恒 1，恰在 clip 区间内部）。
    cases.append(
        _mkcase(
            "identical_policies",
            "logp==old_logp → m=0, r=1.0（区间内部，无 clip）",
            old=[
                [-0.10 * (t + 1) for t in range(4)],
                [-0.10 * (t + 1) - 0.25 for t in range(4)],
            ],
            delta=[
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            adv=[
                [1.0, -0.5, 2.0, 0.0],
                [-1.5, 0.25, 1.0, -2.0],
            ],
            mask=[
                [1, 1, 1, 1],
                [1, 1, 0, 0],
            ],
        )
    )

    return cases


# ---------------------------------------------------------------------------
# GRPO advantage 固定 case（组内 8 条、reward 集中在末 token 等）
# ---------------------------------------------------------------------------


def _grpo_cases() -> List[Dict[str, Any]]:
    """returns: {name, rewards(B,T), mask(B,T), index(B,), note}"""
    out: List[Dict[str, Any]] = []

    def mk(name: str, note: str, lengths: List[int], scores: List[float], index: List[int], t_len: int = 8) -> Dict[str, Any]:
        b = len(lengths)
        rewards = np.zeros((b, t_len), dtype=np.float64)
        mask = np.zeros((b, t_len), dtype=np.float64)
        for i, ln in enumerate(lengths):
            mask[i, :ln] = 1.0
            rewards[i, ln - 1] = scores[i]  # reward 集中在每条末个有效 token
        return {
            "name": name,
            "note": note,
            "rewards": rewards,
            "mask": mask,
            "index": np.asarray(index, dtype=np.int64),
            "scores": np.asarray(scores, dtype=np.float64),
        }

    # A. protocol-v1 §6：组内 8 条同 issue，中心化不除 std；全同 reward 组另有。
    out.append(
        mk(
            "group8_centered_no_std",
            "1 组 × 8 条；adv = score - 组均值（不除 std）",
            lengths=[6, 5, 4, 3, 2, 8, 4, 1],
            scores=[3.0, 0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 2.0],
            index=[0] * 8,
        )
    )
    # B. 全同 reward 组：advantage 全 0（protocol-v1 §6 明确要求）。
    out.append(
        mk(
            "all_same_reward_group",
            "4 条同 reward → advantage 恰为 0（两侧都必须精确为 0）",
            lengths=[5, 3, 6, 2],
            scores=[2.0, 2.0, 2.0, 2.0],
            index=[0] * 4,
        )
    )
    # C. 单条组：两侧语义都是 adv = 原始 score（baseline 0），并非 0（与 RLOO 不同）。
    out.append(
        mk(
            "single_response_group",
            "单条组：adv = score - 0 = score（两侧同语义，注意不为 0）",
            lengths=[4],
            scores=[1.75],
            index=[0],
        )
    )
    # D. 两个不同大小组（5+3）混合。
    out.append(
        mk(
            "two_groups_5_3",
            "组0×5（均值 1.0）+ 组1×3（均值 1.5）",
            lengths=[7, 4, 2, 5, 1, 6, 3, 2],
            scores=[2.0, 2.0, 1.0, 0.0, 0.0, 3.0, 1.5, 0.0],
            index=[0, 0, 0, 0, 0, 1, 1, 1],
        )
    )
    return out


# ---------------------------------------------------------------------------
# 独立 numpy float64 oracle（GSPO-token 公式手工推导，arXiv:2507.18071 eq.14/15）
# ---------------------------------------------------------------------------


def _gspo_oracle(case: Dict[str, Any], eps_low: float, eps_high: float) -> Dict[str, Any]:
    """从公式直接推导 GSPO 各中间量与 loss/梯度（不依赖任何框架实现）。

    返回：
      m               (B,)  序列级 log-ratio 均值 = masked_mean(log_ratio, mask, -1)
      s               (B,)  序列级 importance ratio = exp(min(m, 10))
      pg              (B,T) 每 token pg_losses = max(-A*s, -A*clip(s, 1-elow, 1+ehigh))
      seq_loss_skyrl  (B,)  每序列 token-mean，分母 max(L,1)（SkyRL masked_mean 口径）
      loss_skyrl      scalar = mean(seq_loss_skyrl)（含全 mask 行 → D1）
      grad_skyrl      (B,T) d loss_skyrl / d logp
      seq_loss_verl   (B,)  分母 L + 1e-8（verl agg_loss 口径，D1）
      rows / nonempty / lengths (B,)
    梯度闭式：d pg/d logp_{i,t} = -A_{i,t}·s_i（当分支"激活"），否则 0。
    激活 = (s_i 在 clip 区间内) 或 (A>0 且 s_i<下界) 或 (A<0 且 s_i>上界)；
    再乘上 min(m,10) 未触发 clamp（m<=10）与 mask>0 两个条件。
    """
    lp = case["logp"]
    old = case["old"]
    adv = case["adv"]
    mask = case["mask"]
    rows = lp.shape[0]
    lengths = mask.sum(axis=1)  # (B,)
    log_ratio = lp - old
    # SkyRL masked_mean(dim=-1)：分母 clamp(min=1)
    m = (log_ratio * mask).sum(axis=1) / np.maximum(lengths, 1.0)
    s = np.exp(np.minimum(m, LOG_CLAMP_MAX))  # (B,)
    rc = np.clip(s, 1.0 - eps_low, 1.0 + eps_high)  # (B,)
    sb = s[:, None]
    rcb = rc[:, None]
    pg = np.maximum(-adv * sb, -adv * rcb)  # (B,T)
    in_mask = mask > 0
    interior = (s >= 1.0 - eps_low) & (s <= 1.0 + eps_high)
    active = (
        interior[:, None]
        | ((adv > 0) & (sb < (1.0 - eps_low)))
        | ((adv < 0) & (sb > (1.0 + eps_high)))
    ) & (m <= LOG_CLAMP_MAX)[:, None] & in_mask
    dpg = np.where(active, -adv * sb, 0.0)  # d pg / d logp（逐 token）

    seq_loss_skyrl = (pg * mask).sum(axis=1) / np.maximum(lengths, 1.0)
    loss_skyrl = seq_loss_skyrl.sum() / rows  # .mean()：含全 mask 行（贡献 0）
    grad_skyrl = dpg / (np.maximum(lengths, 1.0) * rows)[:, None]

    seq_loss_verl = (pg * mask).sum(axis=1) / (lengths + VERL_SEQ_DENOM_EPS)
    return {
        "m": m,
        "s": s,
        "pg": pg,
        "dpg": dpg,
        "active": active,
        "lengths": lengths,
        "rows": rows,
        "nonempty": int((lengths > 0).sum()),
        "seq_loss_skyrl": seq_loss_skyrl,
        "loss_skyrl": loss_skyrl,
        "grad_skyrl": grad_skyrl,
        "seq_loss_verl": seq_loss_verl,
    }


def _loss_verl_from_oracle(orc: Dict[str, Any], global_batch_size: Optional[int]) -> float:
    """verl agg_loss("seq-mean-token-mean", dp_size=1, global_batch_size=N) 的期望值。

    N=None 时回退 N = 本 micro-batch 非全 mask 行数（core_algos.py:1198-1201）。
    """
    lengths = orc["lengths"]
    nonzero = orc["seq_loss_verl"][lengths > 0].sum()
    g = orc["nonempty"] if global_batch_size is None else global_batch_size
    return nonzero / g


def _grad_verl_from_oracle(orc: Dict[str, Any], global_batch_size: int) -> np.ndarray:
    """verl 口径梯度：dpg /((L+1e-8)·N)，全 mask 行为 0。"""
    lengths = orc["lengths"]
    denom = (lengths + VERL_SEQ_DENOM_EPS) * global_batch_size
    safe = np.where(lengths > 0, denom, 1.0)
    grad = orc["dpg"] / safe[:, None]
    return np.where((lengths > 0)[:, None], grad, 0.0)


def _grpo_oracle(case: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """GRPO outcome advantage 手工推导（中心化不除 std）：
    adv_i = score_i - 组均值（单条组均值取 0），再乘 response_mask 广播。"""
    rewards = case["rewards"]
    mask = case["mask"]
    index = case["index"]
    scores = rewards.sum(axis=1)
    adv = np.zeros_like(scores)
    for g in np.unique(index):
        sel = index == g
        mean = 0.0 if sel.sum() == 1 else float(scores[sel].mean())
        adv[sel] = scores[sel] - mean
    advantages = adv[:, None] * mask
    return advantages, scores


# ---------------------------------------------------------------------------
# torch 张量工具
# ---------------------------------------------------------------------------


def _t(case: Dict[str, Any]) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """(logp, old_logp, adv, mask) float64 CPU 张量（logp 不带 grad）。"""
    return (
        torch.tensor(case["logp"], dtype=torch.float64),
        torch.tensor(case["old"], dtype=torch.float64),
        torch.tensor(case["adv"], dtype=torch.float64),
        torch.tensor(case["mask"], dtype=torch.float64),
    )


def _skyrl_gspo(
    case: Dict[str, Any],
    eps_low: float = EPS_LOW,
    eps_high: float = EPS_HIGH,
    adv_override: Optional[np.ndarray] = None,
    requires_grad: bool = False,
) -> Tuple["torch.Tensor", float, "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """跑 SkyRL 参考实现；返回 (loss, clip_ratio, logp_leaf, oldp, adv, mask)。"""
    lp = torch.tensor(case["logp"], dtype=torch.float64, requires_grad=requires_grad)
    oldp = torch.tensor(case["old"], dtype=torch.float64)
    adv = (
        torch.tensor(adv_override, dtype=torch.float64)
        if adv_override is not None
        else torch.tensor(case["adv"], dtype=torch.float64)
    )
    mask = torch.tensor(case["mask"], dtype=torch.float64)
    cfg = _skyrl_config(eps_clip_low=eps_low, eps_clip_high=eps_high)
    loss, clip_ratio = gspo_policy_loss(
        log_probs=lp, old_log_probs=oldp, advantages=adv, config=cfg, loss_mask=mask
    )
    return loss, clip_ratio, lp, oldp, adv, mask


# ---------------------------------------------------------------------------
# verl 侧加载与调用（仅在有 torch 的环境尝试；失败给出清晰 skip 理由）
# ---------------------------------------------------------------------------

_VERL_CACHE: Optional[Dict[str, Any]] = None


def _load_verl() -> Dict[str, Any]:
    """加载 verl v0.9.1 目标实现；不可用时 skip 全对齐层（参考层仍有效）。

    实测（源码核对，v0.9.1 tag）：
    - from verl.trainer.ppo.core_algos import
        compute_policy_loss_gspo / agg_loss / compute_grpo_outcome_advantage
    - ActorConfig 首选 verl.workers.config.actor（定义处），
      备选 verl.workers.config（包 __init__ 再导出）。
    """
    global _VERL_CACHE
    if _VERL_CACHE is not None:
        return _VERL_CACHE
    try:
        from verl.trainer.ppo.core_algos import (
            agg_loss,
            compute_grpo_outcome_advantage,
            compute_policy_loss_gspo,
        )
    except Exception as exc:
        _skip(
            "verl 不可用（%s: %s）——只跑 SkyRL 参考层自洽性；"
            "全对齐层须在服务器 verl env 运行" % (type(exc).__name__, exc)
        )
        raise AssertionError("unreachable")  # pragma: no cover

    actor_config_cls = None
    import_errors: List[str] = []
    import importlib

    for mod_path in ("verl.workers.config.actor", "verl.workers.config"):
        try:
            mod = importlib.import_module(mod_path)
            actor_config_cls = getattr(mod, "ActorConfig", None)
            if actor_config_cls is not None:
                import_errors = []
                break
        except Exception as exc:
            import_errors.append("%s: %s: %s" % (mod_path, type(exc).__name__, exc))
    if actor_config_cls is None:
        _skip(
            "verl.ActorConfig 导入失败（%s）——verl 版本可能与 v0.9.1 路径不同，"
            "请核对 verl/workers/config/actor.py" % "; ".join(import_errors)
        )
    _VERL_CACHE = {
        "compute_policy_loss_gspo": compute_policy_loss_gspo,
        "agg_loss": agg_loss,
        "compute_grpo_outcome_advantage": compute_grpo_outcome_advantage,
        "ActorConfig": actor_config_cls,
    }
    return _VERL_CACHE


def _verl_actor_config(
    verl: Dict[str, Any],
    global_batch_size: Optional[int],
    dp_size: int = 1,
    clip_ratio_low: Optional[float] = EPS_LOW,
    clip_ratio_high: Optional[float] = EPS_HIGH,
    empty_gbi: bool = False,
) -> Any:
    """构造 verl ActorConfig（v0.9.1）。

    必填字段（v0.9.1 __post_init__ 断言）：strategy、rollout_n；
    非 dynamic_bsz 时还须给 ppo_micro_batch_size(_per_gpu) 之一。
    clip_ratio_low/high 传 None 可触发回退 config.clip_ratio（gspo 内
    core_algos.py:1582-1583）。global_batch_info 的生产注入方式见
    verl/workers/utils/losses.py ppo_loss()（dp_size / batch_num_tokens /
    global_batch_size / loss_scale_factor 四键）；此处按同结构构造。
    """
    gbi: Dict[str, Any] = (
        {}
        if empty_gbi
        else {
            "dp_size": dp_size,
            "batch_num_tokens": None,
            "global_batch_size": global_batch_size,
            "loss_scale_factor": None,
        }
    )
    return verl["ActorConfig"](
        strategy="fsdp",
        rollout_n=8,  # protocol-v1 §6：n_samples_per_prompt=8
        ppo_micro_batch_size_per_gpu=1,
        clip_ratio=0.2,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
        clip_ratio_c=CLIP_RATIO_C,  # GSPO 不读取（dual-clip 常量）；按任务规约同值
        loss_agg_mode="seq-mean-token-mean",
        global_batch_info=gbi,
    )


def _verl_gspo(
    verl: Dict[str, Any],
    case: Dict[str, Any],
    global_batch_size: Optional[int],
    adv_override: Optional[np.ndarray] = None,
    requires_grad: bool = False,
    clip_ratio_low: Optional[float] = EPS_LOW,
    clip_ratio_high: Optional[float] = EPS_HIGH,
    empty_gbi: bool = False,
    dp_size: int = 1,
) -> Tuple["torch.Tensor", Dict[str, Any], "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """跑 verl compute_policy_loss_gspo；返回 (loss, metrics, lp_leaf, oldp, adv, mask)。"""
    lp = torch.tensor(case["logp"], dtype=torch.float64, requires_grad=requires_grad)
    oldp = torch.tensor(case["old"], dtype=torch.float64)
    adv = (
        torch.tensor(adv_override, dtype=torch.float64)
        if adv_override is not None
        else torch.tensor(case["adv"], dtype=torch.float64)
    )
    # 生产路径 losses.py 把 response_mask 转 bool 后传入；float64 mask 在
    # gspo/agg_loss 的求和/乘法下数值等价，测试直接用 float64。
    mask = torch.tensor(case["mask"], dtype=torch.float64)
    cfg = _verl_actor_config(
        verl,
        global_batch_size=global_batch_size,
        dp_size=dp_size,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
        empty_gbi=empty_gbi,
    )
    loss, metrics = verl["compute_policy_loss_gspo"](
        old_log_prob=oldp,
        log_prob=lp,
        advantages=adv,
        response_mask=mask,
        loss_agg_mode="seq-mean-token-mean",
        config=cfg,
    )
    return loss, metrics, lp, oldp, adv, mask


# ===========================================================================
# 测试 1：SkyRL 参考层 —— masked_mean 语义（含全零 mask 行的分母 clamp）
# ===========================================================================


def test_masked_mean_reference_semantics() -> None:
    _require_layer1()
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        dtype=torch.float64,
    )
    m = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    # dim=-1：逐行 (x*mask).sum / max(rowsum,1)
    got = masked_mean(x, m, dim=-1).numpy()
    want = np.array([(1 + 2 + 3 + 4) / 4.0, (5.0 + 7.0) / 2.0, 0.0])  # 全零行→0/1=0
    assert np.max(np.abs(got - want)) <= 1e-12, (got, want)
    # dim=None：全局 masked mean
    got_all = masked_mean(x, m).item()
    want_all = (1 + 2 + 3 + 4 + 5 + 7) / 6.0
    assert abs(got_all - want_all) <= 1e-12, (got_all, want_all)
    # mask=None：普通 mean
    assert abs(masked_mean(x, None).item() - x.mean().item()) <= 1e-12
    # 全零 mask 张量：分母 clamp(min=1.0) → 0，不产生 NaN/inf
    zero = masked_mean(x, torch.zeros_like(m), dim=-1)
    assert torch.all(zero == 0.0), zero


# ===========================================================================
# 测试 2：SkyRL 参考层自洽 —— gspo loss vs 手工推导 oracle（全部固定 case）
# ===========================================================================


def test_gspo_reference_loss_self_consistency() -> None:
    _require_layer1()
    for case in _fixed_cases():
        orc = _gspo_oracle(case, EPS_LOW, EPS_HIGH)
        loss, clip_ratio, _, _, _, _ = _skyrl_gspo(case)
        diff = abs(loss.item() - orc["loss_skyrl"])
        assert diff <= TOL, (
            "case %s: |loss_skyrl - oracle| = %.3e" % (case["name"], diff)
        )
        # ratio 中间量：序列 log-ratio 均值 = masked_mean(log_ratio, mask, -1)
        lp, oldp, adv, mask = _t(case)
        m_t = masked_mean(lp - oldp, mask, dim=-1).numpy()
        assert np.max(np.abs(m_t - orc["m"])) <= 1e-12, (case["name"], m_t, orc["m"])
        # 监控值 clip_ratio ∈ [0,1]
        assert 0.0 <= clip_ratio <= 1.0, (case["name"], clip_ratio)


def _frozen_gspo_skyrl_loss(
    logp: "torch.Tensor",
    logp0: "torch.Tensor",
    adv: "torch.Tensor",
    mask: "torch.Tensor",
    m0: "torch.Tensor",
    eps_low: float,
    eps_high: float,
) -> "torch.Tensor":
    """stop-gradient 语义下 autograd 实际微分的函数（独立最小复刻，供有限差分互验）。

    GSPO 的图是 logp - logp.detach() + m.detach()：在某评估点 logp0，autograd
    微分的数学函数等价于（detach 常数冻结于评估点）：

        r_{i,t'} = exp(min(logp_{i,t'} - logp0_{i,t'} + m0_i, 10))

    注意不能对整条 SkyRL loss 直接做数值差分：重跑前向会重算序列均值 m，而
    autograd 把 m detach 掉，两者是不同的函数（GSPO 序列级 IS 的 stop-gradient
    设计本身）。本函数只在冻结 m0 下做差分，同时校验复制实现的 autograd 梯度与
    闭式 oracle。
    """
    liw = logp - logp0 + m0.unsqueeze(-1)
    r = torch.exp(torch.clamp(liw, max=LOG_CLAMP_MAX))
    pg = torch.maximum(-adv * r, -adv * torch.clamp(r, 1 - eps_low, 1 + eps_high))
    seq = (pg * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    return seq.mean()


# ===========================================================================
# 测试 3：SkyRL 参考层自洽 —— autograd 梯度 vs 闭式 oracle + 有限差分抽查
# ===========================================================================


def test_gspo_reference_gradient_self_consistency() -> None:
    _require_layer1()
    for case in _fixed_cases():
        orc = _gspo_oracle(case, EPS_LOW, EPS_HIGH)
        loss, _, lp, _, _, mask = _skyrl_gspo(case, requires_grad=True)
        loss.backward()
        grad = lp.grad.numpy()
        diff = np.max(np.abs(grad - orc["grad_skyrl"]))
        assert diff <= TOL, (
            "case %s: max|grad_skyrl - oracle| = %.3e" % (case["name"], diff)
        )
        # masked token 梯度恒 0（含全 mask 行）
        assert np.all(grad[case["mask"] == 0] == 0.0), case["name"]
        # 全 mask 行整行梯度恒 0
        for i in range(case["mask"].shape[0]):
            if case["mask"][i].sum() == 0:
                assert np.all(grad[i] == 0.0), (case["name"], i)

    # 有限差分互验（stop-gradient 语义，见 _frozen_gspo_skyrl_loss 的说明）：
    # 用冻结 m 的等价函数同时校验复制实现的 autograd 梯度与闭式 oracle。
    for case in _fixed_cases():
        orc = _gspo_oracle(case, EPS_LOW, EPS_HIGH)
        in_mask = np.argwhere(case["mask"] > 0)
        if len(in_mask) == 0:
            continue
        lp0 = torch.tensor(case["logp"], dtype=torch.float64)
        old0 = torch.tensor(case["old"], dtype=torch.float64)
        adv = torch.tensor(case["adv"], dtype=torch.float64)
        mask = torch.tensor(case["mask"], dtype=torch.float64)
        m0 = masked_mean(lp0 - old0, mask, dim=-1).detach()
        loss, _, lp_leaf, _, _, _ = _skyrl_gspo(case, requires_grad=True)
        loss.backward()
        real_grad = lp_leaf.grad.numpy()
        for pick in sorted({0, min(2, len(in_mask) - 1), len(in_mask) - 1}):
            i, t = int(in_mask[pick][0]), int(in_mask[pick][1])
            h = 1e-6

            def frozen_at(delta: float) -> float:
                lp_p = lp0.clone()
                lp_p[i, t] = lp_p[i, t] + delta
                with torch.no_grad():
                    return _frozen_gspo_skyrl_loss(
                        lp_p, lp0, adv, mask, m0, EPS_LOW, EPS_HIGH
                    ).item()

            fd = (frozen_at(h) - frozen_at(-h)) / (2 * h)
            g_closed = orc["grad_skyrl"][i, t]
            assert abs(fd - g_closed) <= 5e-7, (
                "case %s pos(%d,%d): FD=%.3e closed-form=%.3e"
                % (case["name"], i, t, fd, g_closed)
            )
            assert abs(fd - real_grad[i, t]) <= 5e-7, (
                "case %s pos(%d,%d): FD=%.3e autograd=%.3e"
                % (case["name"], i, t, fd, real_grad[i, t])
            )


# ===========================================================================
# 测试 4：SkyRL 参考层自洽 —— GRPO outcome advantage vs oracle
# ===========================================================================


def test_grpo_reference_advantage_self_consistency() -> None:
    _require_layer1()
    for case in _grpo_cases():
        adv_np, _scores = _grpo_oracle(case)
        adv_t, ret_t = compute_grpo_outcome_advantage(
            token_level_rewards=torch.tensor(case["rewards"], dtype=torch.float64),
            response_mask=torch.tensor(case["mask"], dtype=torch.float64),
            index=case["index"],
            grpo_norm_by_std=False,
        )
        diff = np.max(np.abs(adv_t.numpy() - adv_np))
        assert diff <= 1e-12, "case %s: |adv - oracle| = %.3e" % (case["name"], diff)
        # returns == advantages（两侧实现均以 scores 兼任 returns）
        assert np.max(np.abs(ret_t.numpy() - adv_t.numpy())) == 0.0, case["name"]
        if case["name"] == "all_same_reward_group":
            assert np.all(adv_t.numpy() == 0.0), "全同 reward 组必须精确为 0"
        if case["name"] == "single_response_group":
            expected = np.broadcast_to(case["scores"][:, None], case["mask"].shape) * case["mask"]
            assert np.max(np.abs(adv_t.numpy() - expected)) <= 1e-12, (
                "单条组 adv 应等于原始 score（baseline 0），语义与 RLOO 置 0 不同"
            )


# ===========================================================================
# 测试 5（全对齐层）：序列级 importance ratio 对齐（任务第 1 项）
# ===========================================================================


def test_verl_sequence_ratio_alignment() -> None:
    _require_layer1()
    verl = _load_verl()
    probe_adv_one = None  # adv 全 1 探针在循环内按 case 形状构造
    for case in _fixed_cases():
        lengths = case["mask"].sum(axis=1)
        if not np.any(lengths > 0):
            continue  # 全 mask batch 无 ratio 可探
        orc = _gspo_oracle(case, EPS_LOW, EPS_HIGH)
        rows_with_tokens = [i for i in range(case["mask"].shape[0]) if lengths[i] > 0]
        for i in rows_with_tokens[:2]:  # 每 case 抽至多 2 行做逐行 ratio 探针
            one = {
                "name": "%s[row%d]" % (case["name"], i),
                "note": case["note"],
                "old": case["old"][i : i + 1],
                "logp": case["logp"][i : i + 1],
                "adv": case["adv"][i : i + 1],
                "mask": case["mask"][i : i + 1],
            }
            orc1 = _gspo_oracle(one, EPS_LOW, EPS_HIGH)
            r = orc1["s"][0]
            probe_adv_one = np.where(one["mask"] > 0, 1.0, 0.0)  # 全 1 于有效 token
            # eps 放大到不触发 clip：probe loss = -r（SkyRL）/ -r·L/(L+1e-8)（verl）
            probe_eps = 1e9
            loss_s, _, _, _, _, _ = _skyrl_gspo(
                one, eps_low=probe_eps, eps_high=probe_eps, adv_override=probe_adv_one
            )
            loss_v, _, _, _, _, _ = _verl_gspo(
                verl,
                one,
                global_batch_size=1,
                adv_override=probe_adv_one,
                clip_ratio_low=probe_eps,
                clip_ratio_high=probe_eps,
            )
            want_s = -r
            want_v = -r * float(one["mask"].sum()) / (float(one["mask"].sum()) + VERL_SEQ_DENOM_EPS)
            assert abs(loss_s.item() - want_s) <= TOL, (
                "SkyRL ratio 探针 %s: got %.12f want %.12f" % (one["name"], loss_s.item(), want_s)
            )
            assert abs(loss_v.item() - want_v) <= TOL, (
                "verl ratio 探针 %s: got %.12f want %.12f（+1e-8 每序列分母，D1）"
                % (one["name"], loss_v.item(), want_v)
            )
            # 任务第 1 项显式口径：序列 log-ratio 均值 = masked_mean(log_ratio, mask, -1)
            lp1 = torch.tensor(one["logp"], dtype=torch.float64)
            old1 = torch.tensor(one["old"], dtype=torch.float64)
            mask1 = torch.tensor(one["mask"], dtype=torch.float64)
            assert abs(masked_mean(lp1 - old1, mask1, dim=-1).item() - orc1["m"][0]) <= 1e-12, one["name"]
            assert abs(math.log(r) - min(orc1["m"][0], LOG_CLAMP_MAX)) <= 1e-12, one["name"]


# ===========================================================================
# 测试 6（全对齐层）：GSPO loss 对齐（任务第 2 项，含全 mask 行的 D1 手工推导）
# ===========================================================================


def test_verl_gspo_loss_alignment() -> None:
    _require_layer1()
    verl = _load_verl()
    for case in _fixed_cases():
        orc = _gspo_oracle(case, EPS_LOW, EPS_HIGH)
        rows = orc["rows"]
        loss_s, _, _, _, _, _ = _skyrl_gspo(case)
        loss_v, _, _, _, _, _ = _verl_gspo(verl, case, global_batch_size=rows)

        # 两侧各自对手工推导的期望值断言（全 mask 行 case 也不许含糊通过）：
        want_s = orc["loss_skyrl"]
        want_v = _loss_verl_from_oracle(orc, rows)
        assert abs(loss_s.item() - want_s) <= TOL, (
            "case %s: |loss_skyrl - 手工期望| = %.3e" % (case["name"], abs(loss_s.item() - want_s))
        )
        assert abs(loss_v.item() - want_v) <= TOL, (
            "case %s: |loss_verl - 手工期望| = %.3e" % (case["name"], abs(loss_v.item() - want_v))
        )

        # cross：N == R 时二者相等；唯一残差 = verl 每序列分母 +1e-8（D1）。
        # 偏置上界 = Σ_{L>0} |seq_loss_s_i| · 1e-8/(L_i+1e-8) / R，逐 case 推导如下：
        bias = float(
            (np.abs(orc["seq_loss_skyrl"][orc["lengths"] > 0]) * VERL_SEQ_DENOM_EPS
             / (orc["lengths"][orc["lengths"] > 0] + VERL_SEQ_DENOM_EPS)).sum() / rows
        )
        cross = abs(loss_v.item() - loss_s.item())
        assert cross <= max(TOL, bias + 1e-12), (
            "case %s: cross=%.3e bias(+1e-8 分母)=%.3e——超出可解释口径差异"
            % (case["name"], cross, bias)
        )
        # 固定 case 的 bias < 1e-8，任务容差 1e-8 必须直接成立：
        assert cross <= TOL, "case %s: |loss_verl - loss_skyrl| = %.3e > 1e-8" % (case["name"], cross)

        # --- D1/D2 定量：global_batch_size != 行数时 loss_verl = loss_skyrl × R/N ---
        if case["name"] == "all_mask_row_3rows":
            # 手工推导（D1）：L = [4, 0, 2]，全 mask 行贡献 0 但计入 SkyRL 分母 R=3。
            #   seq_loss_s0 = -(1.0-1.0+0.5+2.0)/4 · e^{1e-5} = -0.625·e^{1e-5}
            #   seq_loss_s2 = -(-0.5+1.5)/2 · e^{1e-5}  = -0.5·e^{1e-5}
            #   loss_skyrl  = (-0.625 - 0.5)·e^{1e-5} / 3 = -0.375·e^{1e-5}
            e1 = math.exp(1e-5)
            hand_s = -0.375 * e1
            assert abs(loss_s.item() - hand_s) <= TOL, (
                "D1 手工推导 loss_skyrl: got %.12f want %.12f" % (loss_s.item(), hand_s)
            )
            # verl N=3（=R，含全 mask 行）：与 SkyRL 相等（差 +1e-8 分母）。
            # verl N=2（=非全 mask 行数）：分子相同、分母 2 → 恰为 3/2 倍（D2）。
            loss_v2, _, _, _, _, _ = _verl_gspo(verl, case, global_batch_size=2)
            want_v2 = _loss_verl_from_oracle(orc, 2)
            assert abs(loss_v2.item() - want_v2) <= TOL, (case["name"], loss_v2.item(), want_v2)
            assert abs(loss_v2.item() * 2.0 - loss_v.item() * 3.0) <= 1e-8, (
                "N=2 与 N=3 分子必须相同（全 mask 行在两种口径下都不进分子）"
            )
            ratio = loss_v2.item() / loss_v.item()
            assert abs(ratio - 3.0 / 2.0) <= 1e-8, "N=2 时应为 N=3 的 3/2 倍，got %r" % ratio
        if case["name"] == "all_mask_batch":
            # 全 batch 全 mask：SkyRL 0/R；verl 0/N；均为精确 0。
            assert loss_s.item() == 0.0 and loss_v.item() == 0.0, case["name"]


# ===========================================================================
# 测试 7（全对齐层）：GSPO 梯度对齐（任务第 3 项，含全 mask 行手工推导）
# ===========================================================================


def test_verl_gspo_gradient_alignment() -> None:
    _require_layer1()
    verl = _load_verl()
    for case in _fixed_cases():
        orc = _gspo_oracle(case, EPS_LOW, EPS_HIGH)
        rows = orc["rows"]
        loss_s, _, lp_s, _, _, mask = _skyrl_gspo(case, requires_grad=True)
        loss_s.backward()
        loss_v, _, lp_v, _, _, _ = _verl_gspo(verl, case, global_batch_size=rows, requires_grad=True)
        loss_v.backward()
        gs = lp_s.grad.numpy()
        gv = lp_v.grad.numpy()

        # 两侧各自对闭式推导的梯度断言：
        assert np.max(np.abs(gs - orc["grad_skyrl"])) <= TOL, case["name"]
        assert np.max(np.abs(gv - _grad_verl_from_oracle(orc, rows))) <= TOL, case["name"]

        # cross 逐元素 ≤ 1e-8（N == R 时唯一差异是 +1e-8 每序列分母）：
        #   g_v - g_s = g_s · 1e-8/(L_i+1e-8)（逐元素），固定 case 的 |g_s|<1
        #   ⇒ |g_v - g_s| ≤ 1e-8 直接成立；下面的 bound 断言把该推导写成代码。
        diff = np.abs(gv - gs)
        bound = TOL + np.abs(gs) * VERL_SEQ_DENOM_EPS / (orc["lengths"] + VERL_SEQ_DENOM_EPS)[:, None]
        assert np.all(diff <= bound), (
            "case %s: max cross grad diff = %.3e" % (case["name"], float(diff.max()))
        )
        assert float(diff.max()) <= TOL, (
            "case %s: |grad_verl - grad_skyrl| = %.3e > 1e-8" % (case["name"], float(diff.max()))
        )

        # masked token / 全 mask 行：两侧梯度都必须精确为 0。
        assert np.all(gs[case["mask"] == 0] == 0.0), case["name"]
        assert np.all(gv[case["mask"] == 0] == 0.0), case["name"]

        if case["name"] == "all_mask_row_3rows":
            # 手工推导（D1）：row1 全 mask → 整行梯度 0；
            # row0（L=4，r=e^{1e-5}，interior）：g_{0,t} = -A_t·e^{1e-5}/(4·3)
            # row2（L=2，r=e^{1e-5}，interior）：g_{2,t} = -A_t·e^{1e-5}/(2·3)
            e1 = math.exp(1e-5)
            assert np.all(gs[1] == 0.0) and np.all(gv[1] == 0.0), "全 mask 行梯度恒 0"
            for t in range(4):
                want = -case["adv"][0][t] * e1 / 12.0
                assert abs(gs[0][t] - want) <= TOL, (t, gs[0][t], want)
            for t in range(2):
                want = -case["adv"][2][t] * e1 / 6.0
                assert abs(gs[2][t] - want) <= TOL, (t, gs[2][t], want)
        if case["name"] == "log_ratio_clamp_10":
            # m=12>10 触发 clamp(max=10)：整行梯度为 0（clamp 截断梯度路径）
            assert np.all(gs == 0.0) and np.all(gv == 0.0), "clamp 分支梯度恒 0"
        if case["name"] == "ratio_below_clip":
            # row0 r<0.9997：A>0 的 token 保留梯度（未 clip），A<0 的 token 被 clip → 0
            adv0 = case["adv"][0]
            assert gs[0][1] == 0.0 and gv[0][1] == 0.0, "A<0 且 r<下界 → clip 分支梯度 0"
            assert abs(gs[0][0] + adv0[0] * orc["s"][0] / (4 * rows)) <= TOL, "A>0 且 r<下界 → 未 clip"
        if case["name"] == "ratio_above_clip":
            # row0 r>1.0004：A<0 的 token 保留梯度，A>0 的 token 被 clip → 0
            adv0 = case["adv"][0]
            assert gs[0][0] == 0.0 and gv[0][0] == 0.0, "A>0 且 r>上界 → clip 分支梯度 0"
            assert abs(gs[0][1] + adv0[1] * orc["s"][0] / (4 * rows)) <= TOL, "A<0 且 r>上界 → 未 clip"


# ===========================================================================
# 测试 8（全对齐层）：GRPO advantage 对齐（任务第 4 项）
# ===========================================================================


def test_verl_grpo_advantage_alignment() -> None:
    _require_layer1()
    verl = _load_verl()
    for case in _grpo_cases():
        adv_np, _scores = _grpo_oracle(case)
        rewards = torch.tensor(case["rewards"], dtype=torch.float64)
        mask = torch.tensor(case["mask"], dtype=torch.float64)
        adv_s, ret_s = compute_grpo_outcome_advantage(
            token_level_rewards=rewards.clone(),
            response_mask=mask.clone(),
            index=case["index"],
            grpo_norm_by_std=False,  # SkyRL 参数名
        )
        adv_v, ret_v = verl["compute_grpo_outcome_advantage"](
            token_level_rewards=rewards.clone(),
            response_mask=mask.clone(),
            index=case["index"],
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=False,  # verl 参数名（v0.9.1）
            config=None,
        )
        d_sv = np.max(np.abs(adv_s.numpy() - adv_v.numpy()))
        assert d_sv <= 1e-12, "case %s: |adv_skyrl - adv_verl| = %.3e" % (case["name"], d_sv)
        assert np.max(np.abs(adv_s.numpy() - adv_np)) <= 1e-12, (
            "case %s: |adv_skyrl - oracle| = %.3e" % (case["name"], np.max(np.abs(adv_s.numpy() - adv_np)))
        )
        assert np.max(np.abs(adv_v.numpy() - adv_np)) <= 1e-12, case["name"]
        # returns == advantages（两侧均以中心化 score 兼任 returns）
        assert np.max(np.abs(ret_s.numpy() - adv_s.numpy())) == 0.0, case["name"]
        assert np.max(np.abs(ret_v.numpy() - adv_v.numpy())) == 0.0, case["name"]
        if case["name"] == "all_same_reward_group":
            assert np.all(adv_s.numpy() == 0.0) and np.all(adv_v.numpy() == 0.0), (
                "全同 reward 组：两侧 advantage 都必须精确为 0"
            )
        if case["name"] == "single_response_group":
            expected = np.broadcast_to(case["scores"][:, None], case["mask"].shape) * case["mask"]
            assert np.max(np.abs(adv_v.numpy() - expected)) <= 1e-12, (
                "单条组：verl 同样取 adv=score（baseline 0），非 0"
            )

    # 反向对照：两侧默认 flag（除 std）也应互相对齐（参数名不同、语义相同）。
    case = _grpo_cases()[0]
    adv_np_scores = case["rewards"].sum(axis=1)
    groupsel = case["index"] == 0
    mean = adv_np_scores[groupsel].mean()
    std = adv_np_scores[groupsel].std(ddof=1)  # torch.std 默认无偏
    want = ((adv_np_scores - mean) / (std + 1e-6))[:, None] * case["mask"]
    adv_s, _ = compute_grpo_outcome_advantage(
        token_level_rewards=torch.tensor(case["rewards"], dtype=torch.float64),
        response_mask=torch.tensor(case["mask"], dtype=torch.float64),
        index=case["index"],
        grpo_norm_by_std=True,
    )
    adv_v, _ = verl["compute_grpo_outcome_advantage"](
        token_level_rewards=torch.tensor(case["rewards"], dtype=torch.float64),
        response_mask=torch.tensor(case["mask"], dtype=torch.float64),
        index=case["index"],
        norm_adv_by_std_in_grpo=True,
    )
    assert np.max(np.abs(adv_s.numpy() - want)) <= 1e-9, "SkyRL 除 std 口径"
    assert np.max(np.abs(adv_v.numpy() - want)) <= 1e-9, "verl 除 std 口径"


# ===========================================================================
# 测试 9（全对齐层）：梯度聚合口径（任务第 5 项的注释断言）
# ===========================================================================


def test_verl_agg_loss_scaling_semantics() -> None:
    """verl agg_loss 缩放系数 vs SkyRL reduce_loss(".mean()") 系数关系（D1/D2）。

    口径记录（亦见模块 docstring）：
    - verl(dp_size=1, global_batch_size=N)：loss = Σ_{非全mask行} S_i/(L_i+1e-8) / N
    - SkyRL sequence_mean：            loss = Σ_{全部行}     S_i/max(L_i,1)  / R
    相等条件：N == R（此时唯一差异 = +1e-8 每序列分母）。
    N != R 时：loss_verl(N) = loss_verl(R) × R/N = loss_skyrl × R/N（分子相同）。
    """
    _require_layer1()
    verl = _load_verl()
    agg_loss = verl["agg_loss"]
    cases = [c for c in _fixed_cases() if c["name"] in ("multi_row_4x8_mixed", "all_mask_row_3rows")]
    for case in cases:
        orc = _gspo_oracle(case, EPS_LOW, EPS_HIGH)
        rows = orc["rows"]
        nonempty = orc["nonempty"]
        # pg 张量直接喂 agg_loss，隔离聚合口径本身（不经 ratio 计算）
        pg = torch.tensor(orc["pg"], dtype=torch.float64)
        mask = torch.tensor(case["mask"], dtype=torch.float64)

        for g in [rows, nonempty, rows + 3, 64]:
            got = agg_loss(pg, mask, "seq-mean-token-mean", dp_size=1, global_batch_size=g).item()
            want = _loss_verl_from_oracle(orc, g)
            assert abs(got - want) <= TOL, (case["name"], g, got, want)
            # 分子不变性：loss_verl(N)·N == loss_verl(R)·R（与 N 无关）
        numer = lambda g: agg_loss(pg, mask, "seq-mean-token-mean", dp_size=1, global_batch_size=g).item() * g
        assert abs(numer(rows) - numer(64)) <= 1e-8 * rows, case["name"]
        # 与 SkyRL 的系数关系：N==R 相等；N=64 时 loss_verl = loss_skyrl × R/64
        loss_s, _, _, _, _, _ = _skyrl_gspo(case)
        got_rows = agg_loss(pg, mask, "seq-mean-token-mean", dp_size=1, global_batch_size=rows).item()
        bias = float(
            (np.abs(orc["seq_loss_skyrl"][orc["lengths"] > 0]) * VERL_SEQ_DENOM_EPS
             / (orc["lengths"][orc["lengths"] > 0] + VERL_SEQ_DENOM_EPS)).sum() / rows
        )
        assert abs(got_rows - loss_s.item()) <= max(TOL, bias + 1e-12), (
            "N==R 时 verl agg_loss 输入相同 pg 张量应与 SkyRL sequence_mean 相等"
        )
        got_64 = agg_loss(pg, mask, "seq-mean-token-mean", dp_size=1, global_batch_size=64).item()
        want_64 = loss_s.item() * rows / 64.0
        assert abs(got_64 - want_64) <= max(TOL, abs(want_64) * 1e-6 + bias), (
            "N=64 时 loss_verl = loss_skyrl × R/N（D2：%d 行微批按生产 global_batch_size=64 归一）" % rows
        )
        # dp_size 线性缩放（多 DP rank 归约口径；本测试 dp_size=1 为主路径）
        got_dp2 = agg_loss(pg, mask, "seq-mean-token-mean", dp_size=2, global_batch_size=rows).item()
        assert abs(got_dp2 - 2.0 * got_rows) <= 1e-8, "agg_loss 随 dp_size 线性"

    # fallback：global_batch_size=None（global_batch_info 留空）→ 回退非全 mask 行数。
    case = [c for c in _fixed_cases() if c["name"] == "all_mask_row_3rows"][0]
    orc = _gspo_oracle(case, EPS_LOW, EPS_HIGH)
    pg = torch.tensor(orc["pg"], dtype=torch.float64)
    mask = torch.tensor(case["mask"], dtype=torch.float64)
    got = agg_loss(pg, mask, "seq-mean-token-mean", dp_size=1, global_batch_size=None).item()
    want = _loss_verl_from_oracle(orc, None)
    assert abs(got - want) <= TOL, (
        "global_batch_size=None 回退 N=非全 mask 行数（%d），与生产配置值口径不同"
        % orc["nonempty"]
    )
    # 全 mask batch + fallback：N=0 → 0/0 = NaN（源码推导 core_algos.py:1198-1201；
    # 此断言在服务器 verl 环境实测该行为）。
    case9 = [c for c in _fixed_cases() if c["name"] == "all_mask_batch"][0]
    orc9 = _gspo_oracle(case9, EPS_LOW, EPS_HIGH)
    pg9 = torch.tensor(orc9["pg"], dtype=torch.float64)
    mask9 = torch.tensor(case9["mask"], dtype=torch.float64)
    got9 = agg_loss(pg9, mask9, "seq-mean-token-mean", dp_size=1, global_batch_size=None)
    assert torch.isnan(got9).all(), (
        "全 mask batch + global_batch_size=None 应产生 NaN（回退分母 0）；"
        "生产路径不会触发（ppo_loss 总是注入配置值），此为源码行为记录"
    )


# ===========================================================================
# 测试 10（全对齐层）：clip_ratio 回退与 global_batch_info 调用方式
# ===========================================================================


def test_verl_clip_ratio_fallback_and_gbi_modes() -> None:
    _require_layer1()
    verl = _load_verl()
    # 用 ratio_below_clip：其 row0 r≈0.9512 对 3e-4 下界触发 clip、对回退值 0.2
    # 不触发 → 两种配置的 loss 必然不同，可验证回退路径真的生效。
    case = [c for c in _fixed_cases() if c["name"] == "ratio_below_clip"][0]
    orc = _gspo_oracle(case, EPS_LOW, EPS_HIGH)
    rows = orc["rows"]

    # clip_ratio_low/high=None → 回退 config.clip_ratio（core_algos.py:1582-1583）
    loss_fb, _, _, _, _, _ = _verl_gspo(
        verl, case, global_batch_size=rows, clip_ratio_low=None, clip_ratio_high=None
    )
    # 期望：以 clip_ratio=0.2 为界的手工推导
    orc_fb = _gspo_oracle(case, 0.2, 0.2)
    want = _loss_verl_from_oracle(orc_fb, rows)
    assert abs(loss_fb.item() - want) <= TOL, (
        "clip_ratio_low/high 缺失时回退 clip_ratio：got %.12f want %.12f"
        % (loss_fb.item(), want)
    )
    # 显式 3e-4/4e-4 与回退 0.2 的输出应不同（确认 clip 真的生效）
    loss_exp, _, _, _, _, _ = _verl_gspo(verl, case, global_batch_size=rows)
    assert abs(loss_exp.item() - loss_fb.item()) > 1e-12, "两种 clip 配置的 loss 不应恰好相同"

    # global_batch_info 留空（{}）→ agg_loss 收到 global_batch_size=None → 回退
    # 非全 mask 行数（与测试 9 的 fallback 断言一致；直接从 compute_policy_loss_gspo 走一遍）
    loss_empty, _, _, _, _, _ = _verl_gspo(verl, case, global_batch_size=None, empty_gbi=True)
    want_empty = _loss_verl_from_oracle(orc, None)
    assert abs(loss_empty.item() - want_empty) <= TOL, (
        "global_batch_info={} 回退 N=非全 mask 行数：got %.12f want %.12f"
        % (loss_empty.item(), want_empty)
    )
    # 生产注入方式记录：ppo_loss() 在调用前把 4 个键写进 config.global_batch_info，
    # 再由 compute_policy_loss_gspo 以 **config.global_batch_info 传给 agg_loss；
    # 本测试的 _verl_actor_config 与该结构一致（dp_size=1, batch_num_tokens=None,
    # global_batch_size=N, loss_scale_factor=None——后两项在 seq-mean-token-mean 下不被消费）。


# ---------------------------------------------------------------------------
# __main__ 运行器（pytest 不走这里）
# ---------------------------------------------------------------------------


def _main() -> int:
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    passed = skipped = failed = 0
    print("=" * 72)
    print("tests/test_loss_alignment.py — SkyRL(81e5a97c) ↔ verl(v0.9.1) 数值对齐")
    if _TORCH_IMPORT_ERROR is not None:
        print("torch: 不可用（%s）→ 全部测试 skip" % _TORCH_IMPORT_ERROR)
    else:
        print("torch: 可用（%s）" % getattr(torch, "__version__", "?"))
        print("verl : %s" % ("缓存未建立（首个全对齐测试触发加载）" if _VERL_CACHE is None else "已加载"))
    print("=" * 72)
    for name, fn in tests:
        try:
            fn()
        except SkipTest as exc:
            skipped += 1
            print("SKIP  %-52s %s" % (name, exc))
        except AssertionError as exc:
            failed += 1
            print("FAIL  %-52s %s" % (name, exc))
        except Exception as exc:  # pragma: no cover - 意外错误也计失败
            failed += 1
            print("ERROR %-52s %s: %s" % (name, type(exc).__name__, exc))
        else:
            passed += 1
            print("PASS  %s" % name)
    print("-" * 72)
    print("汇总: %d passed, %d skipped, %d failed" % (passed, skipped, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
