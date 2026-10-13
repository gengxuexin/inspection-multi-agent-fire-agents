# tests/test_budget.py
"""common/budget.py 单元测试：Token/成本预算追踪与超限终止。

运行（项目根目录）：
    python -m pytest tests/test_budget.py -v
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import pytest
except ImportError as exc:  # pragma: no cover
    raise ImportError("缺少 pytest，请先安装：pip install pytest") from exc

try:
    from common.budget import BudgetExceededError, BudgetTracker
except ImportError:  # pragma: no cover
    pytest.skip("common.budget 不可用", allow_module_level=True)


# ---------------- 基础累计 ----------------


def test_new_tracker_is_empty():
    bt = BudgetTracker()
    assert bt.check() is True
    assert bt.exceeded() is False
    assert bt.usage()["used_tokens"] == 0
    assert bt.usage()["used_cost"] == 0.0


def test_consume_accumulates_tokens():
    bt = BudgetTracker(max_tokens=1000)
    bt.consume(tokens=100)
    bt.consume(tokens=250)
    assert bt.usage()["used_tokens"] == 350
    assert bt.exceeded() is False


def test_consume_accumulates_cost():
    bt = BudgetTracker(max_cost=10.0)
    bt.consume(cost=0.5)
    bt.consume(cost=1.25)
    assert bt.usage()["used_cost"] == 1.75


# ---------------- 超限终止 ----------------


def test_token_budget_exceeded_raises():
    """超过 max_tokens 应抛 BudgetExceededError。"""
    bt = BudgetTracker(max_tokens=100)
    bt.consume(tokens=60)
    with pytest.raises(BudgetExceededError):
        bt.consume(tokens=50)


def test_cost_budget_exceeded_raises():
    bt = BudgetTracker(max_cost=1.0)
    bt.consume(cost=0.6)
    with pytest.raises(BudgetExceededError):
        bt.consume(cost=0.5)


def test_boundary_hits_limit_counts_as_exceeded():
    """等于上限即视为超限（>= 判定），consume 当场抛错终止。"""
    bt = BudgetTracker(max_tokens=100)
    bt.consume(tokens=99)
    assert bt.exceeded() is False
    with pytest.raises(BudgetExceededError):
        bt.consume(tokens=1)  # 累计到恰好 100
    assert bt.exceeded() is True
    assert bt.usage()["used_tokens"] == 100


def test_state_recorded_even_when_raising():
    """超限抛错时用量应已累加，便于上层记录实际消耗。"""
    bt = BudgetTracker(max_tokens=10)
    with pytest.raises(BudgetExceededError):
        bt.consume(tokens=25)
    assert bt.usage()["used_tokens"] == 25


def test_unlimited_budget_never_exceeded():
    """max_tokens/max_cost 均为 None 表示不限制。"""
    bt = BudgetTracker()
    bt.consume(tokens=10**9, cost=10**9)
    assert bt.exceeded() is False


def test_error_message_contains_usage_snapshot():
    """异常信息应带上当前用量，便于定位。"""
    bt = BudgetTracker(max_tokens=50, max_cost=5.0)
    with pytest.raises(BudgetExceededError) as exc:
        bt.consume(tokens=60, cost=0.1)
    msg = str(exc.value)
    assert "60" in msg and "50" in msg, f"异常信息缺少用量快照: {msg}"


def test_usage_snapshot_shape():
    bt = BudgetTracker(max_tokens=100, max_cost=2.0)
    bt.consume(tokens=30, cost=0.123456)
    u = bt.usage()
    assert set(u) == {"used_tokens", "used_cost", "max_tokens", "max_cost"}
    assert u["used_cost"] == 0.1235, "cost 应保留 4 位小数"


# ---------------- 合并 ----------------


def test_merge_accumulates_across_trackers():
    """多节点并行场景下各节点预算需合并汇总。"""
    a = BudgetTracker(max_tokens=1000)
    b = BudgetTracker(max_tokens=1000)
    a.consume(tokens=100, cost=0.1)
    b.consume(tokens=250, cost=0.2)

    a.merge(b)
    assert a.usage()["used_tokens"] == 350
    assert a.usage()["used_cost"] == 0.3


def test_merge_respects_target_limit():
    """合并后若超过目标上限应立即判定超限。"""
    a = BudgetTracker(max_tokens=300)
    b = BudgetTracker()
    a.consume(tokens=200)
    b.consume(tokens=200)
    a.merge(b)
    assert a.exceeded() is True
