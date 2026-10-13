# common/resilience.py
"""熔断器与保护调用：三态机 closed→open→half_open + 指数退避重试 + 降级回退。

用法：
    from common.resilience import protected_call, get_breaker

    result = await protected_call(handle_vision, q, fp, name="vision",
                                  fallback=vision_fallback)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable


class CircuitOpenError(Exception):
    """熔断器处于 OPEN 状态时调用被拒绝。"""


@dataclass
class CircuitBreaker:
    """三态熔断器。

    - closed：正常放行，失败累计达 threshold 后转 open
    - open：拒绝所有调用，等待 reset_timeout 后自动转 half_open
    - half_open：放行试探调用；成功→closed，失败→open
    """
    name: str
    failure_threshold: int = 5
    reset_timeout: float = 30.0
    _failures: int = 0
    _state: str = "closed"
    _opened_at: float = 0.0

    def _refresh(self) -> None:
        if self._state == "open" and time.monotonic() - self._opened_at >= self.reset_timeout:
            self._state = "half_open"

    def allow(self) -> bool:
        self._refresh()
        return self._state != "open"

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"

    def record_failure(self) -> None:
        self._refresh()
        if self._state == "half_open":
            self._state = "open"
            self._opened_at = time.monotonic()
        elif self._state == "closed":
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._state = "open"
                self._opened_at = time.monotonic()

    @property
    def state(self) -> str:
        self._refresh()
        return self._state

    def to_dict(self) -> dict:
        return {"name": self.name, "state": self.state,
                "failures": self._failures, "threshold": self.failure_threshold}


# 全局注册器
_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str, **kwargs: Any) -> CircuitBreaker:
    """获取或创建具名熔断器（单例）。"""
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name=name, **kwargs)
    return _breakers[name]


async def _call(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """统一同步/异步调用。"""
    result = fn(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result


async def protected_call(
    fn: Callable,
    *args: Any,
    breaker: CircuitBreaker | None = None,
    name: str = "default",
    fallback: Callable | None = None,
    retries: int = 2,
    base_delay: float = 0.5,
    **kwargs: Any,
) -> Any:
    """带熔断 + 指数退避重试 + 降级回退的异步保护调用。

    - breaker 或 name 二选一；name 时自动 get_breaker(name)
    - 熔断 OPEN 时：有 fallback 调 fallback，否则抛 CircuitOpenError
    - 失败时指数退避重试（base_delay * 2^attempt），最多 retries 次
    - 最终失败：有 fallback 调 fallback，否则抛最后一次异常
    """
    cb = breaker or get_breaker(name)

    if not cb.allow():
        if fallback is not None:
            return await _call(fallback, *args, **kwargs)
        raise CircuitOpenError(f"熔断器 [{name}] 处于 OPEN 状态，调用被拒绝")

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = await _call(fn, *args, **kwargs)
            cb.record_success()
            return result
        except Exception as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(base_delay * (2 ** attempt))
    # 全部重试失败
    cb.record_failure()
    if fallback is not None:
        try:
            return await _call(fallback, *args, **kwargs)
        except Exception:
            pass
    raise last_exc  # type: ignore[misc]
