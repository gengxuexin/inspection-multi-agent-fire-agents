# tests/test_resilience.py
"""common/resilience.py 单元测试：三态熔断器 + 保护调用 + 降级回退。

重点验证三态机状态流转（closed → open → half_open → closed）
与失败计数语义（重试全部失败后才计一次）。

运行（项目根目录）：
    python -m pytest tests/test_resilience.py -v
"""
import asyncio
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
    from common.resilience import (
        CircuitBreaker,
        CircuitOpenError,
        get_breaker,
        protected_call,
    )
except ImportError:  # pragma: no cover
    pytest.skip("common.resilience 不可用", allow_module_level=True)


@pytest.fixture(autouse=True)
def _clear_breakers():
    """每个用例前后清空全局熔断器注册表，避免用例间串扰。"""
    import common.resilience as res
    res._breakers.clear()
    yield
    res._breakers.clear()


# ---------------- CircuitBreaker 三态机 ----------------


def test_initial_state_is_closed():
    cb = CircuitBreaker(name="t1")
    assert cb.state == "closed"
    assert cb.allow() is True


def test_failures_below_threshold_stay_closed():
    cb = CircuitBreaker(name="t2", failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "closed", "失败次数未达阈值不应熔断"
    assert cb.allow() is True


def test_reaching_threshold_opens_circuit():
    cb = CircuitBreaker(name="t3", failure_threshold=3)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == "open"
    assert cb.allow() is False, "OPEN 状态必须拒绝调用"


def test_success_resets_failure_count():
    cb = CircuitBreaker(name="t4", failure_threshold=3)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert cb._failures == 0, "成功应清零失败计数"
    cb.record_failure()
    assert cb.state == "closed"


def test_open_transitions_to_half_open_after_timeout():
    """OPEN 状态等待 reset_timeout 后应自动转 half_open 并放行试探。"""
    cb = CircuitBreaker(name="t5", failure_threshold=1, reset_timeout=0.05)
    cb.record_failure()
    assert cb.state == "open"

    asyncio.run(asyncio.sleep(0.08))  # 超时后

    assert cb.state == "half_open"
    assert cb.allow() is True, "half_open 应放行试探调用"


def test_half_open_success_closes_circuit():
    cb = CircuitBreaker(name="t6", failure_threshold=1, reset_timeout=0.05)
    cb.record_failure()
    asyncio.run(asyncio.sleep(0.08))
    assert cb.state == "half_open"

    cb.record_success()
    assert cb.state == "closed"


def test_half_open_failure_reopens_circuit():
    """half_open 试探失败应立即回到 open，并刷新冷却计时。"""
    cb = CircuitBreaker(name="t7", failure_threshold=1, reset_timeout=0.05)
    cb.record_failure()
    asyncio.run(asyncio.sleep(0.08))
    assert cb.state == "half_open"

    cb.record_failure()
    assert cb.state == "open", "half_open 试探失败必须重新熔断"


def test_to_dict_reports_snapshot():
    cb = CircuitBreaker(name="t8", failure_threshold=2)
    cb.record_failure()
    d = cb.to_dict()
    assert d["name"] == "t8"
    assert d["state"] == "closed"
    assert d["failures"] == 1
    assert d["threshold"] == 2


# ---------------- get_breaker 单例 ----------------


def test_get_breaker_is_singleton_per_name():
    a = get_breaker("svc")
    b = get_breaker("svc")
    assert a is b, "同名熔断器必须复用同一实例"


def test_get_breaker_distinct_names():
    assert get_breaker("x") is not get_breaker("y")


# ---------------- protected_call ----------------


def test_protected_call_success_returns_value():
    async def run():
        return await protected_call(lambda: "ok", name="p1", retries=0)

    assert asyncio.run(run()) == "ok"


def test_protected_call_supports_async_callable():
    async def work():
        return "async-ok"

    async def run():
        return await protected_call(work, name="p2", retries=0)

    assert asyncio.run(run()) == "async-ok"


def test_protected_call_passes_args_and_kwargs():
    def add(a, b, c=0):
        return a + b + c

    async def run():
        return await protected_call(add, 1, 2, c=3, name="p3", retries=0)

    assert asyncio.run(run()) == 6


def test_protected_call_raises_after_retries_exhausted():
    def boom():
        raise ValueError("down")

    async def run():
        with pytest.raises(ValueError, match="down"):
            await protected_call(boom, name="p4", retries=1, base_delay=0)

    asyncio.run(run())


def test_protected_call_retries_before_giving_up():
    """验证重试确实发生：第 2 次调用成功则整体成功。"""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("第一次失败")
        return "recovered"

    async def run():
        return await protected_call(flaky, name="p5", retries=2, base_delay=0)

    assert asyncio.run(run()) == "recovered"
    assert calls["n"] == 2, "应恰好重试一次后成功"


def test_failure_counted_once_after_all_retries():
    """关键语义：一次 protected_call 的全部重试失败只计 1 次失败。

    这与"每次尝试都计数"的设计不同，测试将其固化防止回归。
    """
    def boom():
        raise RuntimeError("always")

    cb = CircuitBreaker(name="p6", failure_threshold=5)

    async def run():
        with pytest.raises(RuntimeError):
            await protected_call(boom, breaker=cb, retries=3, base_delay=0)

    asyncio.run(run())
    assert cb._failures == 1, f"重试耗尽后应只计 1 次失败，实际 {cb._failures}"


def test_fallback_used_when_circuit_open():
    """熔断打开时应直接走 fallback，不调用主函数。"""
    cb = CircuitBreaker(name="p7", failure_threshold=1)
    cb.record_failure()  # 触发 open
    called = {"main": False, "fb": False}

    def main():
        called["main"] = True

    def fallback():
        called["fb"] = True
        return "degraded"

    async def run():
        return await protected_call(main, breaker=cb, fallback=fallback, retries=0)

    assert asyncio.run(run()) == "degraded"
    assert called["fb"] is True
    assert called["main"] is False, "熔断时不应调用主函数"


def test_fallback_used_when_call_fails():
    def boom():
        raise RuntimeError("x")

    async def run():
        return await protected_call(boom, name="p8", fallback=lambda: "fb", retries=0)

    assert asyncio.run(run()) == "fb"


def test_no_fallback_and_circuit_open_raises():
    cb = CircuitBreaker(name="p9", failure_threshold=1)
    cb.record_failure()

    async def run():
        with pytest.raises(CircuitOpenError, match="OPEN"):
            await protected_call(lambda: 1, breaker=cb, retries=0)

    asyncio.run(run())


def test_protected_call_closes_circuit_on_success():
    """成功调用应把熔断器拉回 closed 并清零计数。"""
    cb = CircuitBreaker(name="p10", failure_threshold=2)
    cb.record_failure()

    async def run():
        return await protected_call(lambda: "ok", breaker=cb, retries=0)

    asyncio.run(run())
    assert cb.state == "closed"
    assert cb._failures == 0
