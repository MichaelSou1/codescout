# -*- coding: utf-8 -*-
"""CodeScout 奖励合同测试：逐字锁定原 scorer 行为（冻结基准）。

用途
----
为冻结协议下的原奖励实现写"合同测试"（contract test），锁定其行为作为后续
verl adapter 对齐（误差 <= 1e-8）的基准。本文件只断言原实现的**实际行为**，
不做任何"修正"：原实现中与直觉不符的行为（空真值集合得 0、空文件名预测使三级
全空、class-only 预测不产生 entity 等）一律按原样锁定并在断言旁注明。

冻结协议引用
------------
- docs/reproduction/protocol-v1.md §4「奖励（冻结）」：
  * 唯一奖励函数 multilevel_localization_f1_reward，权重 file/module/entity = 1/1/1，范围 0-3；
  * 空真值集合得 0（原实现行为，不"修正"）；空文件名预测使三级全空得 0；
    去重后按 set 计算 precision/recall/F1；
  * module = "file:Class" 或 "file:func"（顶层函数），entity = "file:Class.method"
    或 "file:func"；解析语义逐字复用 src/rewards/file_localization/module_rewards.py。
- todo.md 步骤 3 要求的合同测试覆盖：gold 精确预测、遗漏、过报、重复、类方法、
  顶层函数、空标签、无 finish（多 finish 语义在生成器层，见各测试内注释）。

被锁定代码（不得修改 src/ 下任何文件）
------------------------------------
- src/rewards/file_localization/file_localization.py
  multilevel_localization_f1_reward(final_message, instance, structured_locations=None,
                                    file_level_weight=1.0, module_level_weight=1.0,
                                    entity_level_weight=1.0) -> (reward, reward_dict)
  compute_file_f1_score(predicted, true, beta=1.0)
- src/rewards/file_localization/module_rewards.py
  parse_structured_outputs(structured_locations) -> (files, modules, entities)
- src/rewards/__init__.py（注册器 REWARD_REGISTRY / get_reward_function，纯标准库）

运行方式
--------
本机（macOS）默认 python3 为 3.9.6 且无 pytest；而冻结实现使用
``list[dict] | None`` 注解（PEP 604，需 Python >= 3.10；项目 pyproject 声明
requires-python >= 3.13），故 3.9 下无法 import scorer（这是源码事实，不是本
测试的问题）。本机已验证可用的解释器为 uv 管理的 Python 3.12.14：

    # 首选（若该解释器装有 pytest）：
    ~/.local/share/uv/python/cpython-3.12.14-macos-aarch64-none/bin/python3 \\
        -m pytest tests/test_reward_contract.py -v

    # 无 pytest 时的 __main__ fallback（本文件自带运行器）：
    ~/.local/share/uv/python/cpython-3.12.14-macos-aarch64-none/bin/python3 \\
        tests/test_reward_contract.py

两种方式运行同一批 test_* 函数。数值断言精度：abs diff <= 1e-9（比协议要求的
verl adapter 对齐容差 1e-8 更紧一档）。
"""

import sys
from pathlib import Path

# 保证无论从仓库根还是 tests/ 目录启动（pytest 或 __main__），src 包都可导入。
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from src.rewards import REWARD_REGISTRY, get_reward_function  # noqa: E402
    from src.rewards.file_localization.file_localization import (  # noqa: E402
        compute_file_f1_score,
        multilevel_localization_f1_reward,
    )
    from src.rewards.file_localization.module_rewards import (  # noqa: E402
        parse_structured_outputs,
    )
except TypeError as _exc:  # pragma: no cover - 仅在不兼容解释器上触发
    raise SystemExit(
        "无法在当前解释器（Python %s）上导入冻结 scorer：%r。\n"
        "src/rewards/file_localization/file_localization.py 使用 'list[dict] | None' "
        "注解（PEP 604），需要 Python >= 3.10（项目 pyproject 声明 >=3.13）。\n"
        "本机可用解释器示例："
        "~/.local/share/uv/python/cpython-3.12.14-macos-aarch64-none/bin/python3"
        % (sys.version.split()[0], _exc)
    ) from None

# 数值断言容差：比 verl adapter 对齐要求（1e-8）更紧。
TOL = 1e-9

# reward_dict 的精确键集（原实现中 prediction/ground_truth 两个键被注释掉，不输出）。
_REWARD_DICT_KEYS = {
    "multilevel_localization_f1_reward",
    "file_reward",
    "module_reward",
    "entity_reward",
}


# ---------------------------------------------------------------------------
# 辅助构造与断言
# ---------------------------------------------------------------------------

def _loc(file, class_name=None, function_name=None):
    """构造一条 structured_location（生成器层 localization_finish 的结构化产物）。"""
    return {"file": file, "class_name": class_name, "function_name": function_name}


def _instance(file_changes=None, **extra):
    """构造 SWE-smith 风格 instance；file_changes=None 表示完全不提供该键。"""
    inst = dict(extra)
    if file_changes is not None:
        inst["file_changes"] = file_changes
    return inst


def _assert_close(actual, expected, label):
    """精确断言：abs diff <= 1e-9（手算期望值）。"""
    diff = abs(actual - expected)
    assert diff <= TOL, (
        "%s: got %r, expected %r (abs diff %.3e > %.0e)"
        % (label, actual, expected, diff, TOL)
    )


def _assert_reward_dict(rd, *, total, file_f1, module_f1, entity_f1):
    """断言 reward_dict 的键集与四级数值（total 键 == 加权总分，其余为未加权 F1）。"""
    assert set(rd.keys()) == _REWARD_DICT_KEYS, (
        "reward_dict 键集变化: %r != %r" % (sorted(rd.keys()), sorted(_REWARD_DICT_KEYS))
    )
    _assert_close(rd["multilevel_localization_f1_reward"], total, "dict total")
    _assert_close(rd["file_reward"], file_f1, "dict file_reward")
    _assert_close(rd["module_reward"], module_f1, "dict module_reward")
    _assert_close(rd["entity_reward"], entity_f1, "dict entity_reward")


def _score(instance, structured_locations, weights=()):
    """调用被测 scorer（final_message 在 structured 路径下被原实现忽略，传占位串），
    返回 (reward, reward_dict)；weights 为 (fw, mw, ew) 或省略。"""
    kwargs = dict(weights) if weights else {}
    return multilevel_localization_f1_reward(
        "unused-final-message", instance, structured_locations=structured_locations, **kwargs
    )


# ---------------------------------------------------------------------------
# 注册器 sanity（附）：import src.rewards 触发 _auto_load_rewards，冻结奖励必须可用
# ---------------------------------------------------------------------------

def test_registry_contains_frozen_rewards():
    """两个冻结奖励函数已注册，按名字可取回，未知名字按原实现抛 ValueError。"""
    assert get_reward_function("multilevel_localization_f1_reward") is (
        multilevel_localization_f1_reward
    )
    assert "multilevel_localization_f1_reward" in REWARD_REGISTRY
    assert "file_localization_f1_reward" in REWARD_REGISTRY
    try:
        get_reward_function("no_such_reward_function")
    except ValueError as e:
        assert "not found in registry" in str(e)
    else:
        raise AssertionError("未知奖励名应抛 ValueError（原实现语义）")


# ---------------------------------------------------------------------------
# compute_file_f1_score 单元合同（P/R/F1 公式与空集合语义，锁定公式本身）
# ---------------------------------------------------------------------------

def test_compute_file_f1_score_semantics():
    """直接锁定 F1 公式：2PR/(P+R)，以及原实现的全部空集合边界（不修正）。

    原实现要点（src/rewards/file_localization/file_localization.py L7-14）：
    - true 为空 -> 恒 0.0（**即使 pred 也为空也得 0，不是 1**）；
    - pred 为空 -> precision 记 0.0；
    - precision + recall == 0 -> 0.0（不相交集合得 0）。
    注意：tests/test_single_file_localization.py 内嵌的本地 f1_reward_function
    对 both-empty 返回 1.0，与冻结 scorer 不同；本合同以冻结 scorer 为准。
    """
    # 精确匹配：P = R = 1 -> F1 = 1
    _assert_close(compute_file_f1_score({"a.py"}, {"a.py"}), 1.0, "exact F1")
    # 遗漏（pred ⊂ gold）：P = 1, R = 0.5 -> F1 = 2*1*0.5/(1+0.5) = 2/3
    _assert_close(compute_file_f1_score({"a.py"}, {"a.py", "b.py"}), 2 / 3, "subset F1")
    # 过报（pred ⊃ gold）：P = 0.5, R = 1 -> F1 = 2/3
    _assert_close(compute_file_f1_score({"a.py", "b.py"}, {"a.py"}), 2 / 3, "superset F1")
    # 1/3 命中：P = R = 1/3 -> F1 = 1/3
    _assert_close(
        compute_file_f1_score({"a.py", "b.py", "c.py"}, {"a.py", "d.py", "e.py"}),
        1 / 3,
        "one-third F1",
    )
    # 完全不相交：P = R = 0 -> 0.0
    _assert_close(compute_file_f1_score({"x.py"}, {"a.py"}), 0.0, "disjoint F1")
    # gold 为空、pred 非空 -> 0.0（原实现：空真值得 0，不修正）
    _assert_close(compute_file_f1_score({"a.py"}, set()), 0.0, "empty-gold F1")
    # gold 与 pred 都为空 -> 0.0（原实现，**不是** 1.0）
    _assert_close(compute_file_f1_score(set(), set()), 0.0, "both-empty F1")
    # gold 非空、pred 为空 -> precision 记 0 -> F1 = 0
    _assert_close(compute_file_f1_score(set(), {"a.py"}), 0.0, "empty-pred F1")
    # pred 重复（列表语义下）与去重后得分一致：set() 去重后计算
    _assert_close(
        compute_file_f1_score(["a.py", "a.py", "b.py"], ["a.py", "b.py", "b.py"]),
        1.0,
        "duplicates deduped by set()",
    )


# ---------------------------------------------------------------------------
# 需求 1：gold 精确预测 -> 每级 F1 = 1，总 reward = 3
# ---------------------------------------------------------------------------

def test_exact_gold_prediction_scores_three():
    file_changes = [
        {
            "file": "src/a.py",
            "changes": {
                "edited_modules": ["src/a.py:Foo", "src/a.py:run"],
                "edited_entities": ["src/a.py:Foo.bar", "src/a.py:run"],
            },
        }
    ]
    structured = [
        _loc("src/a.py", "Foo", "bar"),   # module src/a.py:Foo；entity src/a.py:Foo.bar
        _loc("src/a.py", None, "run"),    # 顶层函数：module 与 entity 均为 src/a.py:run
    ]
    reward, rd = _score(_instance(file_changes), structured)
    _assert_close(reward, 3.0, "reward")
    _assert_reward_dict(rd, total=3.0, file_f1=1.0, module_f1=1.0, entity_f1=1.0)

    # parse 侧同时锁定集合内容（锁定 P、R 的输入）
    files, modules, entities = parse_structured_outputs(structured)
    assert set(files) == {"src/a.py"}
    assert set(modules) == {"src/a.py:Foo", "src/a.py:run"}
    assert set(entities) == {"src/a.py:Foo.bar", "src/a.py:run"}


# ---------------------------------------------------------------------------
# 需求 2：部分遗漏（predicted ⊂ gold）-> 各级 precision = 1、recall < 1，具体数值
# ---------------------------------------------------------------------------

def test_partial_miss_precision_one_recall_less():
    # gold：2 个文件、2 个 module、3 个 entity
    file_changes = [
        {
            "file": "src/a.py",
            "changes": {
                "edited_modules": ["src/a.py:Foo"],
                "edited_entities": ["src/a.py:Foo.bar", "src/a.py:Foo.baz"],
            },
        },
        {
            "file": "src/b.py",
            "changes": {
                "edited_modules": ["src/b.py:Util"],
                "edited_entities": ["src/b.py:Util.run"],
            },
        },
    ]
    # 只预测 a.py 的 Foo.bar 一处
    structured = [_loc("src/a.py", "Foo", "bar")]
    files, modules, entities = parse_structured_outputs(structured)
    assert set(files) == {"src/a.py"}
    assert set(modules) == {"src/a.py:Foo"}
    assert set(entities) == {"src/a.py:Foo.bar"}

    reward, rd = _score(_instance(file_changes), structured)
    # file 级：tp=1, P=1/1=1, R=1/2 -> F1 = 2*1*0.5/1.5 = 2/3
    # module 级：tp=1, P=1, R=1/2 -> F1 = 2/3
    # entity 级：tp=1, P=1, R=1/3 -> F1 = 2*(1/3)/(1+1/3) = 1/2
    # 总分 = 2/3 + 2/3 + 1/2 = 11/6
    _assert_close(reward, 11 / 6, "reward")
    _assert_reward_dict(rd, total=11 / 6, file_f1=2 / 3, module_f1=2 / 3, entity_f1=0.5)


# ---------------------------------------------------------------------------
# 需求 3：过报（predicted ⊃ gold）-> recall = 1、precision < 1
# ---------------------------------------------------------------------------

def test_overprediction_recall_one_precision_less():
    file_changes = [
        {
            "file": "src/a.py",
            "changes": {
                "edited_modules": ["src/a.py:Foo"],
                "edited_entities": ["src/a.py:Foo.bar"],
            },
        }
    ]
    # 预测 3 个文件/模块/实体，只有 1 个正确
    structured = [
        _loc("src/a.py", "Foo", "bar"),    # 命中
        _loc("src/b.py", "Util", "run"),   # 过报
        _loc("src/c.py", None, "top"),    # 过报
    ]
    files, modules, entities = parse_structured_outputs(structured)
    assert set(files) == {"src/a.py", "src/b.py", "src/c.py"}
    assert set(modules) == {"src/a.py:Foo", "src/b.py:Util", "src/c.py:top"}
    assert set(entities) == {"src/a.py:Foo.bar", "src/b.py:Util.run", "src/c.py:top"}

    reward, rd = _score(_instance(file_changes), structured)
    # file 级：tp=1, P=1/3, R=1 -> F1 = 2*(1/3)/(1/3+1) = 1/2；module/entity 同理
    # 总分 = 1/2 * 3 = 1.5
    _assert_close(reward, 1.5, "reward")
    _assert_reward_dict(rd, total=1.5, file_f1=0.5, module_f1=0.5, entity_f1=0.5)


# ---------------------------------------------------------------------------
# 需求 4：重复预测 -> 与去重后完全相同（不惩罚也不奖励）
# ---------------------------------------------------------------------------

def test_duplicate_predictions_not_penalized_nor_rewarded():
    file_changes = [
        {
            "file": "src/a.py",
            "changes": {
                "edited_modules": ["src/a.py:Foo"],
                "edited_entities": ["src/a.py:Foo.bar"],
            },
        }
    ]
    unique = [_loc("src/a.py", "Foo", "bar")]
    duplicated = [_loc("src/a.py", "Foo", "bar")] * 3 + [_loc("src/a.py", "Foo", "bar")]
    duplicated += [_loc("src/a.py", "Foo", "bar")]

    # parse 层：list(set(...)) 去重（module_rewards.py L186-188；
    # “发现重复则清空”的严格检查是被注释掉的，不生效）
    files, modules, entities = parse_structured_outputs(duplicated)
    assert sorted(files) == ["src/a.py"]
    assert sorted(modules) == ["src/a.py:Foo"]
    assert sorted(entities) == ["src/a.py:Foo.bar"]

    r_uniq, rd_uniq = _score(_instance(file_changes), unique)
    r_dup, rd_dup = _score(_instance(file_changes), duplicated)
    _assert_close(r_dup, r_uniq, "duplicate vs unique reward")
    assert r_dup == r_uniq, "去重后应逐位相等（同一 float）"
    assert rd_dup == rd_uniq

    # gold 侧同样按 set 去重：重复的 file_changes 条目不重复计真值
    dup_gold = _instance(file_changes + file_changes)
    r_dupgold, _ = _score(dup_gold, unique)
    _assert_close(r_dupgold, r_uniq, "duplicate gold == deduped gold")


# ---------------------------------------------------------------------------
# 需求 5：类方法预测（file:Class.method）-> module/entity 命中规则
# ---------------------------------------------------------------------------

def test_class_method_prediction_rules():
    # parse 层规则：
    #   class+function -> module "file:Class"，entity "file:Class.method"
    #   仅 class       -> module "file:Class"，entity 不产生（原实现；docstring
    #                      示例把 class-only 列进 entity，与代码不符，以代码为准）
    #   仅 file        -> 只进 files
    files, modules, entities = parse_structured_outputs(
        [_loc("src/a.py", "Foo", "bar")]
    )
    assert set(files) == {"src/a.py"}
    assert set(modules) == {"src/a.py:Foo"}
    assert set(entities) == {"src/a.py:Foo.bar"}

    files, modules, entities = parse_structured_outputs(
        [_loc("src/a.py", "Foo", None)]
    )
    assert set(files) == {"src/a.py"}
    assert set(modules) == {"src/a.py:Foo"}
    assert entities == []  # class-only 无 entity（原实现）

    files, modules, entities = parse_structured_outputs(
        [{"file": "src/a.py"}]  # 缺 class_name/function_name 键 -> .get 默认 None
    )
    assert set(files) == {"src/a.py"}
    assert modules == []
    assert entities == []

    # 真值语义：class_name 为空串（falsy）按"无 class"处理 -> 落到 function 分支
    files, modules, entities = parse_structured_outputs(
        [_loc("src/a.py", "", "run")]
    )
    assert set(modules) == {"src/a.py:run"}
    assert set(entities) == {"src/a.py:run"}

    # reward 层：class-only 预测命中 file+module 两级，entity 级为 0
    file_changes = [
        {
            "file": "src/a.py",
            "changes": {
                "edited_modules": ["src/a.py:Foo"],
                "edited_entities": ["src/a.py:Foo.__init__"],
            },
        }
    ]
    reward, rd = _score(_instance(file_changes), [_loc("src/a.py", "Foo", None)])
    # file: P=R=1 -> 1；module: P=R=1 -> 1；entity: pred 空、gt 非空 -> 0
    _assert_close(reward, 2.0, "class-only reward")
    _assert_reward_dict(rd, total=2.0, file_f1=1.0, module_f1=1.0, entity_f1=0.0)


# ---------------------------------------------------------------------------
# 需求 6：顶层函数预测（file:func）-> module 与 entity 都是 "file:func"
# ---------------------------------------------------------------------------

def test_top_level_function_module_equals_entity():
    files, modules, entities = parse_structured_outputs(
        [_loc("src/a.py", None, "run")]
    )
    assert set(files) == {"src/a.py"}
    # 原实现：顶层函数的 module 与 entity 是**同一个字符串** "file:func"
    assert set(modules) == {"src/a.py:run"}
    assert set(entities) == {"src/a.py:run"}

    file_changes = [
        {
            "file": "src/a.py",
            "changes": {
                "edited_modules": ["src/a.py:run"],
                "edited_entities": ["src/a.py:run"],
            },
        }
    ]
    reward, rd = _score(_instance(file_changes), [_loc("src/a.py", None, "run")])
    _assert_close(reward, 3.0, "top-level exact reward")
    _assert_reward_dict(rd, total=3.0, file_f1=1.0, module_f1=1.0, entity_f1=1.0)


# ---------------------------------------------------------------------------
# 需求 7：空真值 -> 对应级 F1 = 0；全部空 -> reward = 0
# ---------------------------------------------------------------------------

def test_empty_ground_truth_zero_f1():
    pred = [_loc("src/a.py", "Foo", "bar")]

    # (a) instance 完全没有 "file_changes" 键 -> 三级真值全空 -> reward = 0
    reward, rd = _score(_instance(), pred)
    _assert_close(reward, 0, "missing file_changes key reward")
    _assert_reward_dict(rd, total=0, file_f1=0, module_f1=0, entity_f1=0)

    # (b) file_changes 为空列表 -> 同上（即使有非空预测）
    reward, rd = _score(_instance([]), pred)
    _assert_close(reward, 0, "empty file_changes list reward")
    _assert_reward_dict(rd, total=0, file_f1=0, module_f1=0, entity_f1=0)

    # (c) 条目只有 "file" 无 "changes" -> 只有 file 级有真值：
    #     file F1 = 1；module/entity 真值为空 -> 各级 F1 = 0；总分 1
    reward, rd = _score(_instance([{"file": "src/a.py"}]), pred)
    _assert_close(reward, 1.0, "file-only gold reward")
    _assert_reward_dict(rd, total=1.0, file_f1=1.0, module_f1=0.0, entity_f1=0.0)

    # (d) "changes" 值为 None -> 原实现按空列表处理（None -> []），同 (c)
    reward, rd = _score(
        _instance(
            [
                {
                    "file": "src/a.py",
                    "changes": {"edited_modules": None, "edited_entities": None},
                }
            ]
        ),
        pred,
    )
    _assert_close(reward, 1.0, "None changes values reward")
    _assert_reward_dict(rd, total=1.0, file_f1=1.0, module_f1=0.0, entity_f1=0.0)

    # (e) "changes" 内缺少 edited_modules/edited_entities 键 -> .get 默认 []，同 (c)
    reward, rd = _score(
        _instance([{"file": "src/a.py", "changes": {}}]),
        pred,
    )
    _assert_close(reward, 1.0, "empty changes dict reward")

    # (f) 全部空真值 + 空预测 -> 仍然 0（原实现 both-empty 得 0，不修正为 1）
    reward, rd = _score(_instance([]), [])
    _assert_close(reward, 0, "empty gold + empty prediction reward")
    _assert_reward_dict(rd, total=0, file_f1=0, module_f1=0, entity_f1=0)


def test_change_without_file_key_counts_modules_only():
    """原实现的非对称真值解析：条目含 "changes" 但无 "file" 键时，
    module/entity 真值照常累计，file 真值不累计（"file" not in change 分支）。
    按原样锁定，不视为 bug 修正。"""
    file_changes = [
        {
            "changes": {
                "edited_modules": ["src/a.py:Foo"],
                "edited_entities": ["src/a.py:Foo.bar"],
            }
        }
    ]
    reward, rd = _score(_instance(file_changes), [_loc("src/a.py", "Foo", "bar")])
    # file 级真值空 -> F1 = 0；module/entity 级精确命中 -> F1 = 1；总分 2
    _assert_close(reward, 2.0, "no-file-key gold reward")
    _assert_reward_dict(rd, total=2.0, file_f1=0.0, module_f1=1.0, entity_f1=1.0)


# ---------------------------------------------------------------------------
# 需求 8 + 10：无 finish（structured_locations=None）-> reward = 0
# ---------------------------------------------------------------------------

def test_no_finish_structured_locations_none_zero():
    """无 finish / 无结构化输出：structured_locations=None -> 恒 0。

    合同边界说明（需求 10）：**多 finish、格式错误、轮数耗尽** 的判定发生在
    生成器/生成层（protocol-v1.md §4：localization_finish 必须恰好一次，多次/格式
    错误按"有效模型失败"处理并影响 loss mask），生成器决定传给 scorer 的是
    structured_locations、None 还是干脆不调用。因此 scorer 合同测试只覆盖
    structured_locations=None 这一路径；多 finish 语义由生成器层测试另行锁定，
    本文件不重复实现。"""
    file_changes = [
        {
            "file": "src/a.py",
            "changes": {
                "edited_modules": ["src/a.py:Foo"],
                "edited_entities": ["src/a.py:Foo.bar"],
            },
        }
    ]
    # None 路径完全忽略 final_message，且不受权重影响（早于加权直接返回）
    for final_message in ("unused", None, "<file-list>['src/a.py']</file-list>", ""):
        reward, rd = multilevel_localization_f1_reward(
            final_message,
            _instance(file_changes),
            structured_locations=None,
            file_level_weight=2.0,
            module_level_weight=2.0,
            entity_level_weight=2.0,
        )
        assert reward == 0, "structured_locations=None 必须返回 0（原实现返回 int 0）"
        _assert_reward_dict(rd, total=0, file_f1=0, module_f1=0, entity_f1=0)

    # 即使 gold 非空、且 final_message 里"看起来"有正确答案，None 路径仍为 0：
    # 原实现不解析 final_message（get_simple_results_from_raw_outputs 分支
    # 只在 structured_locations is not None 的 else 里可达，实际不可达）。
    reward, rd = multilevel_localization_f1_reward(
        "src/a.py\nclass: Foo\nfunction: bar",
        _instance(file_changes),
        structured_locations=None,
    )
    assert reward == 0
    _assert_reward_dict(rd, total=0, file_f1=0, module_f1=0, entity_f1=0)


def test_empty_location_list_finish_zero():
    """有效 finish 但提交了零个定位（structured_locations=[]）：
    parse 得到三级全空预测，gold 非空时 precision=recall=0 -> 各级 F1=0。"""
    file_changes = [
        {
            "file": "src/a.py",
            "changes": {
                "edited_modules": ["src/a.py:Foo"],
                "edited_entities": ["src/a.py:Foo.bar"],
            },
        }
    ]
    files, modules, entities = parse_structured_outputs([])
    assert (files, modules, entities) == ([], [], [])

    reward, rd = _score(_instance(file_changes), [])
    _assert_close(reward, 0, "empty location list reward")
    _assert_reward_dict(rd, total=0, file_f1=0, module_f1=0, entity_f1=0)


# ---------------------------------------------------------------------------
# 需求 9：file 为 None/空白字符串 -> parse 全空 -> 三级全 0
# ---------------------------------------------------------------------------

def test_none_or_blank_file_zeroes_all_levels():
    file_changes = [
        {
            "file": "src/a.py",
            "changes": {
                "edited_modules": ["src/a.py:Foo"],
                "edited_entities": ["src/a.py:Foo.bar"],
            },
        }
    ]

    # 单独一个 file=None
    assert parse_structured_outputs([_loc(None, "Foo", "bar")]) == ([], [], [])
    # 单独一个空白文件名（含空格）
    assert parse_structured_outputs([_loc("   ", "Foo", "bar")]) == ([], [], [])
    assert parse_structured_outputs([_loc("", None, None)]) == ([], [], [])
    # 混入一个 file=None 的预测会**整体**清空（早 break，前面已收集的有效项一并丢弃）
    assert parse_structured_outputs(
        [_loc("src/a.py", "Foo", "bar"), _loc(None, "Foo", "bar")]
    ) == ([], [], [])
    # 有效项在坏项之后同样全空（break 位置无关）
    assert parse_structured_outputs(
        [_loc(None, "Foo", "bar"), _loc("src/a.py", "Foo", "bar")]
    ) == ([], [], [])

    # reward 层：含一个 file=None 的预测使三级全部为 0（即便其余预测精确命中 gold）
    reward, rd = _score(
        _instance(file_changes),
        [_loc("src/a.py", "Foo", "bar"), _loc(None, "Foo", "bar")],
    )
    _assert_close(reward, 0, "one None-file location nukes reward")
    _assert_reward_dict(rd, total=0, file_f1=0, module_f1=0, entity_f1=0)

    # 附带锁定：非空白但带首尾空格的文件名**不做 strip**，按原字符串参与匹配
    files, modules, entities = parse_structured_outputs(
        [_loc(" src/a.py ", None, "run")]
    )
    assert files == [" src/a.py "]
    assert modules == [" src/a.py :run"]
    assert entities == [" src/a.py :run"]


# ---------------------------------------------------------------------------
# 需求 11：boundary —— 混合 gold（多文件/多 class/多 entity）、
#          跨文件同名函数、__init__ 方法
# ---------------------------------------------------------------------------

def test_boundary_mixed_gold_multi_file_multi_class():
    # gold：两个文件、6 个 module、6 个 entity
    # 注意 "src/a.py:shared" 与 "src/b.py:shared" 是两个不同 module/entity（跨文件同名），
    # "src/a.py:Foo.__init__" 保留 __init__ 后缀（strip 逻辑被注释掉，原样保留）。
    file_changes = [
        {
            "file": "src/a.py",
            "changes": {
                "edited_modules": [
                    "src/a.py:Foo",
                    "src/a.py:Bar",
                    "src/a.py:top",
                    "src/a.py:shared",
                ],
                "edited_entities": [
                    "src/a.py:Foo.__init__",
                    "src/a.py:Bar.run",
                    "src/a.py:top",
                    "src/a.py:shared",
                ],
            },
        },
        {
            "file": "src/b.py",
            "changes": {
                "edited_modules": ["src/b.py:Foo", "src/b.py:shared"],
                "edited_entities": ["src/b.py:Foo.helper", "src/b.py:shared"],
            },
        },
    ]
    gt_files = {"src/a.py", "src/b.py"}                       # 2
    gt_modules = {
        "src/a.py:Foo", "src/a.py:Bar", "src/a.py:top", "src/a.py:shared",
        "src/b.py:Foo", "src/b.py:shared",
    }                                                          # 6
    gt_entities = {
        "src/a.py:Foo.__init__", "src/a.py:Bar.run", "src/a.py:top", "src/a.py:shared",
        "src/b.py:Foo.helper", "src/b.py:shared",
    }                                                          # 6

    # 预测 5 条：__init__ 方法、类方法、顶层函数、跨文件同名 shared 只报了 b.py 的
    structured = [
        _loc("src/a.py", "Foo", "__init__"),
        _loc("src/a.py", "Bar", "run"),
        _loc("src/a.py", None, "top"),
        _loc("src/b.py", "Foo", "helper"),
        _loc("src/b.py", None, "shared"),
    ]
    files, modules, entities = parse_structured_outputs(structured)
    assert set(files) == gt_files
    assert set(modules) == gt_modules - {"src/a.py:shared"}
    assert set(entities) == gt_entities - {"src/a.py:shared"}

    reward, rd = _score(_instance(file_changes), structured)
    # file 级：2/2 全中 -> F1 = 1
    # module 级：tp=5, P=1, R=5/6 -> F1 = 10/11
    # entity 级：tp=5, P=1, R=5/6 -> F1 = 10/11
    #   （b.py:shared 不会为 a.py:shared 记功：跨文件同名互不抵扣）
    # 总分 = 1 + 10/11 + 10/11 = 31/11
    _assert_close(reward, 31 / 11, "boundary reward")
    _assert_reward_dict(
        rd, total=31 / 11, file_f1=1.0, module_f1=10 / 11, entity_f1=10 / 11
    )

    # 补上 src/a.py 的 shared 后，module/entity 两级都精确 -> 总分 3。
    # 注意：_loc(a, None, shared) 同时为 module 与 entity 各补一条 "src/a.py:shared"
    # （顶层函数的 module 与 entity 同串），所以两级一起补齐。
    structured_full = structured + [_loc("src/a.py", None, "shared")]
    reward, rd = _score(_instance(file_changes), structured_full)
    _assert_close(reward, 3.0, "boundary complete reward")
    _assert_reward_dict(rd, total=3.0, file_f1=1.0, module_f1=1.0, entity_f1=1.0)

    # __init__ 精确匹配的两侧证据：预测 {a.py, Foo, __init__} 的 entity 字符串
    # 必须逐字等于 "src/a.py:Foo.__init__" 才命中（无后缀剥离）。
    f2, m2, e2 = parse_structured_outputs([_loc("src/a.py", "Foo", "__init__")])
    assert e2 == ["src/a.py:Foo.__init__"]
    assert m2 == ["src/a.py:Foo"]


# ---------------------------------------------------------------------------
# 需求 12：权重非 1 时 reward = 加权和（sanity；锁定"不做归一化"）
# ---------------------------------------------------------------------------

def test_nonunit_weights_weighted_sum_no_normalization():
    file_changes = [
        {
            "file": "src/a.py",
            "changes": {
                "edited_modules": ["src/a.py:Foo", "src/a.py:run"],
                "edited_entities": ["src/a.py:Foo.bar", "src/a.py:run"],
            },
        }
    ]
    exact = [_loc("src/a.py", "Foo", "bar"), _loc("src/a.py", None, "run")]

    # 精确命中（各级 F1=1）：reward = 2*1 + 0.5*1 + 3*1 = 5.5。
    # 注意原实现中被注释掉的权重归一化（weight_total /= ...）**不生效**，
    # 即权重不归一化、总和不强制为 3，按原样锁定。
    reward, rd = _score(
        _instance(file_changes),
        exact,
        weights={"file_level_weight": 2.0, "module_level_weight": 0.5, "entity_level_weight": 3.0},
    )
    _assert_close(reward, 5.5, "weighted exact reward")
    # 分项键保持未加权 F1（加权只作用于总分）
    _assert_reward_dict(rd, total=5.5, file_f1=1.0, module_f1=1.0, entity_f1=1.0)

    # 遗漏场景（同需求 2 的数值：file 2/3、module 2/3、entity 1/2）配权重 (1, 2, 0)：
    # reward = 1*(2/3) + 2*(2/3) + 0*(1/2) = 2.0
    partial_changes = [
        {
            "file": "src/a.py",
            "changes": {
                "edited_modules": ["src/a.py:Foo"],
                "edited_entities": ["src/a.py:Foo.bar", "src/a.py:Foo.baz"],
            },
        },
        {
            "file": "src/b.py",
            "changes": {
                "edited_modules": ["src/b.py:Util"],
                "edited_entities": ["src/b.py:Util.run"],
            },
        },
    ]
    reward, rd = _score(
        _instance(partial_changes),
        [_loc("src/a.py", "Foo", "bar")],
        weights={"file_level_weight": 1.0, "module_level_weight": 2.0, "entity_level_weight": 0.0},
    )
    _assert_close(reward, 2.0, "weighted partial reward")
    _assert_reward_dict(rd, total=2.0, file_f1=2 / 3, module_f1=2 / 3, entity_f1=0.5)

    # 权重全 0 -> reward = 0（分项 F1 照常输出）
    reward, rd = _score(
        _instance(file_changes),
        exact,
        weights={"file_level_weight": 0.0, "module_level_weight": 0.0, "entity_level_weight": 0.0},
    )
    _assert_close(reward, 0.0, "zero-weight reward")
    _assert_reward_dict(rd, total=0.0, file_f1=1.0, module_f1=1.0, entity_f1=1.0)


# ---------------------------------------------------------------------------
# __main__ fallback：本机无 pytest 时的直跑运行器（与 pytest 执行同一批函数）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    _failed = 0
    for _name, _fn in _tests:
        try:
            _fn()
        except AssertionError as _e:
            _failed += 1
            print("FAIL  %s: %s" % (_name, _e))
        except Exception as _e:  # noqa: BLE001 - 直跑运行器需完整暴露非断言异常
            _failed += 1
            print("ERROR %s: %s: %s" % (_name, type(_e).__name__, _e))
        else:
            print("PASS  %s" % _name)
    print("\n%d/%d tests passed" % (len(_tests) - _failed, len(_tests)))
    if _failed:
        print("FAILED tests: %d" % _failed)
    sys.exit(1 if _failed else 0)
