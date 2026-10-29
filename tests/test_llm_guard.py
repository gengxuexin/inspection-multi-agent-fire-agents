# tests/test_llm_guard.py
"""熔断/重试/降级 的单元测试。

★ 这组测试要证明什么
-------------------
1. 可重试异常（429/5xx/超时）会重试，且重试后成功 → 熔断器不计入失败
2. 不可重试异常（401/403/400）**不重试** —— 重试只会把密钥失效拖成超时
3. 连续失败达阈值 → 熔断器转 OPEN
4. 熔断 OPEN 时直接走 fallback，不发起真实调用
5. 全部重试失败 → 走 fallback
6. 没有 fallback → 抛 LLMUnavailableError（不是原始异常）
7. 线程安全：并发失败不丢计数

★ 为什么必须有这组测试
-------------------
「熔断器建好了但业务零接入」这个问题之所以能长期存在，
就是因为**没有任何测试验证业务链路真的走熔断**。
只有单元测试覆盖 protected_call，无法发现「没人调用它」。
所以这里测的是 `guarded_call` 这个**会被业务调用的入口**。
"""
import threading
import time

import pytest

from common.llm_guard import (
    LLMUnavailableError,
    SyncCircuitBreaker,
    get_llm_breaker,
    guarded_call,
    is_retryable,
    reset_breakers,
)


class _StatusError(Exception):
    """模拟 openai 的 APIStatusError（带 status_code）。"""

    def __init__(self, status_code: int, msg: str = ""):
        super().__init__(msg or f"HTTP {status_code}")
        self.status_code = status_code


class _ConnError(Exception):
    """模拟网络层异常。"""


# ==================== 可重试判定 ====================
@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retryable_status_codes(status):
    assert is_retryable(_StatusError(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_non_retryable_status_codes(status):
    """鉴权/参数类错误重试无意义 —— 只会把失败拖长。"""
    assert is_retryable(_StatusError(status)) is False


def test_retryable_by_message():
    assert is_retryable(_ConnError("Connection reset by peer")) is True
    assert is_retryable(_ConnError("Read timed out")) is True
    assert is_retryable(Exception("rate limit exceeded")) is True


def test_non_retryable_by_message():
    assert is_retryable(Exception("invalid_api_key")) is False
    assert is_retryable(Exception("Incorrect API key provided")) is False


# ==================== 重试 ====================
def test_retries_then_succeeds(monkeypatch):
    """第一次失败、第二次成功 → 返回成功结果，熔断器不计入失败。"""
    reset_breakers()
    monkeypatch.setattr("common.llm_guard._retry_config", lambda: (2, 0.0, 0.0))
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _StatusError(503, "服务暂时不可用")
        return "ok"

    result = guarded_call(flaky, breaker_name="t_retry", purpose="测试")
    assert result == "ok"
    assert calls["n"] == 3, "应该重试到第 3 次才成功"
    assert get_llm_breaker("t_retry").state == "closed"


def test_non_retryable_fails_immediately(monkeypatch):
    """401 立即失败，不重试 —— 否则密钥失效会拖成 30 秒超时。"""
    reset_breakers()
    monkeypatch.setattr("common.llm_guard._retry_config", lambda: (3, 0.0, 0.0))
    calls = {"n": 0}

    def unauthorized():
        calls["n"] += 1
        raise _StatusError(401, "invalid api key")

    with pytest.raises(LLMUnavailableError):
        guarded_call(unauthorized, breaker_name="t_401", purpose="测试")
    assert calls["n"] == 1, f"401 不该重试，实际调了 {calls['n']} 次"


# ==================== 熔断 ====================
def test_circuit_opens_after_threshold(monkeypatch):
    """连续失败达阈值 → OPEN。"""
    reset_breakers()
    monkeypatch.setattr("common.llm_guard._retry_config", lambda: (0, 0.0, 0.0))
    cb = get_llm_breaker("t_open")
    cb.failure_threshold = 3
    cb.reset_timeout = 60.0

    def always_fail():
        raise _StatusError(500)

    for _ in range(3):
        with pytest.raises(LLMUnavailableError):
            guarded_call(always_fail, breaker_name="t_open", purpose="测试")

    assert cb.state == "open", f"应转为 OPEN，实际 {cb.state}"
    assert cb.allow() is False


def test_open_circuit_skips_real_call(monkeypatch):
    """熔断 OPEN 时直接降级，不发起真实调用 —— 省下游配额。"""
    reset_breakers()
    monkeypatch.setattr("common.llm_guard._retry_config", lambda: (0, 0.0, 0.0))
    cb = get_llm_breaker("t_skip")
    cb.failure_threshold = 1
    cb.reset_timeout = 60.0
    real_calls = {"n": 0}

    def always_fail():
        real_calls["n"] += 1
        raise _StatusError(500)

    with pytest.raises(LLMUnavailableError):
        guarded_call(always_fail, breaker_name="t_skip", purpose="测试")
    assert real_calls["n"] == 1

    # 第二次：熔断已 OPEN，不该再调真实函数
    result = guarded_call(always_fail, breaker_name="t_skip",
                          purpose="测试", fallback=lambda *a, **k: "降级结果")
    assert result == "降级结果"
    assert real_calls["n"] == 1, "熔断期间不该再打真实调用"


def test_half_open_recovers(monkeypatch):
    """熔断超时后转 half_open，试探成功 → 回到 closed。"""
    reset_breakers()
    monkeypatch.setattr("common.llm_guard._retry_config", lambda: (0, 0.0, 0.0))
    cb = get_llm_breaker("t_half")
    cb.failure_threshold = 1
    cb.reset_timeout = 0.05

    state = {"fail": True}

    def maybe_fail():
        if state["fail"]:
            raise _StatusError(500)
        return "恢复成功"

    with pytest.raises(LLMUnavailableError):
        guarded_call(maybe_fail, breaker_name="t_half", purpose="测试")
    assert cb.state == "open"

    time.sleep(0.1)          # 等 reset_timeout
    state["fail"] = False
    assert cb.allow() is True, "超时后应允许试探"
    assert guarded_call(maybe_fail, breaker_name="t_half", purpose="测试") == "恢复成功"
    assert cb.state == "closed"


# ==================== 降级 ====================
def test_fallback_used_after_all_retries(monkeypatch):
    """全部重试失败 → 走 fallback，而不是抛原始异常。"""
    reset_breakers()
    monkeypatch.setattr("common.llm_guard._retry_config", lambda: (1, 0.0, 0.0))
    cb = get_llm_breaker("t_fb")
    cb.failure_threshold = 10

    def always_fail():
        raise _StatusError(500)

    result = guarded_call(always_fail, breaker_name="t_fb", purpose="测试",
                          fallback=lambda *a, **k: {"answer": "纯检索结果"})
    assert result == {"answer": "纯检索结果"}


def test_no_fallback_raises_llm_unavailable():
    """没有 fallback → 抛 LLMUnavailableError（调用方据此返 503）。"""
    reset_breakers()
    cb = get_llm_breaker("t_nofb")
    cb.failure_threshold = 10

    def always_fail():
        raise _StatusError(500)

    with pytest.raises(LLMUnavailableError) as ei:
        guarded_call(always_fail, breaker_name="t_nofb", purpose="测试")
    assert "LLM 服务不可用" in str(ei.value)
    assert ei.value.breaker_state in ("closed", "open")


def test_fallback_itself_failing_raises_unavailable():
    """降级也失败 → 仍抛 LLMUnavailableError，不能把原始异常漏出去。"""
    reset_breakers()
    cb = get_llm_breaker("t_fbfail")
    cb.failure_threshold = 10

    def always_fail():
        raise _StatusError(500)

    def bad_fallback(*a, **k):
        raise RuntimeError("降级也挂了")

    with pytest.raises(LLMUnavailableError):
        guarded_call(always_fail, breaker_name="t_fbfail",
                     purpose="测试", fallback=bad_fallback)


# ==================== 线程安全 ====================
def test_breaker_thread_safe():
    """并发失败不丢计数 —— LLM 调用跑在 FastAPI threadpool 里。"""
    cb = SyncCircuitBreaker("t_thread", failure_threshold=1000,
                             reset_timeout=60.0)
    n = 50

    def worker():
        for _ in range(n):
            cb.record_failure("x")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert cb._failures == n * 10, f"计数丢失：{cb._failures} != {n * 10}"


# ==================== 快照 ====================
def test_breaker_snapshot_shape():
    reset_breakers()
    cb = get_llm_breaker("t_snap")
    snap = cb.snapshot()
    for k in ("name", "state", "failures", "threshold"):
        assert k in snap
    assert snap["state"] in ("closed", "open", "half_open")
