# tests/test_permission_filter.py
"""权限过滤测试 —— 用真实测试数据验证过滤逻辑。

★★ 这个测试要补的是什么洞
--------------------------
实测索引元数据分布：

    secret_level: {0: 3953, 1: 66}    ← 等级 > 2 的：0 条
    valid_until:  全部为 0（永不过期）

所以 ``retrieval_backend.py`` 里那套过滤逻辑虽然**代码是对的**，
但**当前语料里没有任何一条数据能触发它** ——
它处于「空转」状态，从没在真实数据上被验证过。

本测试用一组带明确权限与时效标记的 fixture 数据，
把 5 个密级 × 部门 × 有效期的组合全部走一遍。

★ 为什么这个测试重要
--------------------
过滤机制「代码写了但没数据驱动」是极隐蔽的问题：
功能测试全过、单元测试全过，
但直到某天接入了真实涉密文档，才发现过滤规则有漏洞。

那时候已经在生产环境了，而且是在**越权访问**的场景下暴露的。
"""
import time

import pytest

from agent1.retrieval_backend import _passes_filter
from agent1.testdata.permission_fixtures import (
    CLEARANCE_CONFIDENTIAL,
    CLEARANCE_INTERNAL,
    CLEARANCE_PUBLIC,
    CLEARANCE_SECRET,
    CLEARANCE_TOP_SECRET,
    EXPECTED_VISIBILITY,
    build_permission_fixtures,
)

FIXTURES = build_permission_fixtures()
# chunk_id 在 fixture 顶层（不在 metadata 里 —— 真实索引中它是 metadata 字段，
# 但 fixture 把「标识」与「权限属性」分开更清晰）
BY_CHUNK_ID = {f["chunk_id"]: f for f in FIXTURES}


# ==================== fixture 自身正确性 ====================
def test_fixtures_cover_all_clearance_levels():
    """fixture 必须覆盖 0~4 全部密级，否则矩阵测不全。"""
    levels = {f["metadata"]["secret_level"] for f in FIXTURES}
    assert levels == {0, 1, 2, 3, 4}, f"密级覆盖不全：{levels}"


def test_fixtures_include_expired_and_valid():
    """必须同时有已过期和仍在有效期的文档。"""
    now = int(time.time())
    vals = [f["metadata"]["valid_until"] for f in FIXTURES]
    assert any(v != 0 and v < now for v in vals), "缺少已过期样本"
    assert any(v != 0 and v > now for v in vals), "缺少有效期内的样本"
    assert any(v == 0 for v in vals), "缺少永不过期样本"


def test_fixtures_include_dept_specific():
    """必须有部门专属样本，否则 dept 过滤测不到。"""
    depts = {f["metadata"]["dept"] for f in FIXTURES}
    assert "消防支队" in depts
    assert "通用" in depts


# ==================== 可见性矩阵 ====================
@pytest.mark.parametrize("clearance", [0, 1, 2, 3, 4])
def test_visibility_matrix(clearance):
    """★ 核心：按密级逐条核对可见性矩阵。"""
    dept = "消防支队"
    mismatches = []
    for chunk_id, expected in EXPECTED_VISIBILITY.items():
        fixture = BY_CHUNK_ID[chunk_id]
        actual = _passes_filter(fixture["metadata"], dept, clearance)
        if actual != expected[clearance]:
            mismatches.append(
                f"{chunk_id} (secret={fixture['metadata']['secret_level']}, "
                f"valid_until={fixture['metadata']['valid_until']}, "
                f"dept={fixture['metadata']['dept']}): "
                f"期望 {expected[clearance]}，实际 {actual}")
    assert not mismatches, (
        f"clearance={clearance} 时可见性不符：\n  " + "\n  ".join(mismatches))


def test_public_always_visible():
    """公开文档对任何权限都可见。"""
    for clearance in range(5):
        assert _passes_filter(
            BY_CHUNK_ID["fix_public_001"]["metadata"], "消防支队", clearance)


def test_higher_clearance_sees_more():
    """密级越高，能看到的文档越多（单调性）。"""
    dept = "消防支队"
    counts = []
    for clearance in range(5):
        n = sum(1 for f in FIXTURES
                if _passes_filter(f["metadata"], dept, clearance))
        counts.append(n)
    assert counts == sorted(counts), f"可见数量应单调不减：{counts}"
    assert counts[0] < counts[4], f"绝密用户应比公开用户看到更多：{counts}"


def test_top_secret_only_visible_to_top_secret():
    """绝密文档只对绝密权限可见。"""
    md = BY_CHUNK_ID["fix_topsecret_001"]["metadata"]
    for clearance in (0, 1, 2, 3):
        assert _passes_filter(md, "消防支队", clearance) is False
    assert _passes_filter(md, "消防支队", 4) is True


def test_expired_never_visible():
    """★ 已废止条款任何权限都看不到 —— 这条直接对应执法风险。

    一份被废止的处罚规定如果还能被检索出来，
    模型可能把它写进处罚决定书。
    """
    md = BY_CHUNK_ID["fix_expired_001"]["metadata"]
    for clearance in range(5):
        assert _passes_filter(md, "消防支队", clearance) is False


def test_soon_expired_still_visible():
    """还没失效的条款应该能看到 —— 过期过滤不能过严。"""
    md = BY_CHUNK_ID["fix_soon_expire_001"]["metadata"]
    for clearance in range(5):
        assert _passes_filter(md, "消防支队", clearance) is True


# ==================== 部门隔离 ====================
def test_dept_isolation():
    """支队专属文档，其他部门看不到。"""
    md = BY_CHUNK_ID["fix_dept_001"]["metadata"]
    assert _passes_filter(md, "消防支队", 4) is True
    assert _passes_filter(md, "市局机关", 4) is False


def test_general_docs_visible_to_all_depts():
    """通用文档对所有部门可见。"""
    md = BY_CHUNK_ID["fix_public_001"]["metadata"]
    for dept in ("消防支队", "市局机关", "其他部门"):
        assert _passes_filter(md, dept, 4) is True


# ==================== 边界与健壮性 ====================
def test_missing_metadata_defaults_to_visible_public():
    """元数据缺失时按「公开、通用」处理 —— 不能因缺字段而崩。"""
    assert _passes_filter({}, "消防支队", 0) is True


def test_non_numeric_secret_level_raises():
    """★ 已知隐患：非数字密级会抛 ValueError。

    记录这个行为是为了让它**可见**——
    如果将来要加固（改成 try/except 或校验），
    这个测试会提醒你同步更新。
    """
    with pytest.raises(ValueError):
        _passes_filter({"secret_level": "机密"}, "消防支队", 4)


def test_filter_is_pure():
    """过滤不应修改传入的 metadata。"""
    md = dict(BY_CHUNK_ID["fix_secret_001"]["metadata"])
    before = dict(md)
    _passes_filter(md, "消防支队", 4)
    assert md == before
