# common/llm_guard.py
"""LLM 调用保护层：熔断 + 指数退避重试 + 降级。

★ 为什么需要这一层
------------------
`common/resilience.py` 里已经有完整的 `CircuitBreaker`（三态机）与
`protected_call`（异步版），但全库检索下来，**除了它自己和它的单元测试，
没有任何业务代码引用过** —— 熔断器建好了却没接线。

而 LLM 调用是同步阻塞的（`ds_client.chat.completions.create`），
`protected_call` 是 async，两者接不上。所以本模块提供**同步版**保护，
并把「可重试错误判定」「降级策略」两件事集中在这里，
让调用方只需要改一行调用方式。

★★★ 落地场景（本项目最关键的一条）★★★
消防执法场景中，如果云端 LLM（DeepSeek）不可达 —— 网络不通、限流、
密钥失效 —— 当前实现会让整个请求 500 掉。
执法员填了一半的执法文书就丢了，这在生产环境是不可接受的。

所以降级策略不是「返回一句错误提示」就完事，而是**分三级**：

    1. 优先重试（429/5xx/超时）—— 指数退避，最多 N 次
    2. 熔断后 → 切本地模型（provider_mode=local/hybrid 时可用）
    3. 本地也没有 → 抛 LLMUnavailableError，由调用方转 503
       并**保留已检索到的法条原文**（纯检索结果仍然有价值）

第 3 条最关键：**宁可给执法员一份没有 AI 解读的法条原文，
也不要给一个编造的答案，也不要让整个请求失败。**

用法：
    from common.llm_guard import guarded_chat, LLMUnavailableError

    text = guarded_chat(
        messages=[{"role": "user", "content": prompt}],
        purpose="文书生成",
    )
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Sequence

logger = logging.getLogger("fire_agents.llm_guard")

# ---- 可重试错误判定 ----
# 依据 agent1/build_index.py 里已验证过的 _RETRYABLE 集合：
#   rate limit / 429 / 500 / 502 / 503 / 504
_RETRYABLE_MARKERS: tuple[str, ...] = (
    "rate limit", "rate_limit", "429",
    "500", "502", "503", "504",
    "timeout", "timed out", "connection",
    "temporarily unavailable", "overloaded",
    "bad gateway", "service unavailable",
)


def is_retryable(exc: BaseException) -> bool:
    """判断异常是否值得重试。

    - HTTP 状态码类（openai 的 APIStatusError）看 status
    - 网络类（timeout / connection）看异常文本
    - 明确不可重试的（鉴权失败、参数错误）直接 False —— 重试也没用，
      反而会把密钥失效拖成 30 秒超时。
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        if status in (400, 401, 403, 404):
            return False
        if status == 429 or 500 <= status < 600:
            return True
        return False

    text = f"{type(exc).__name__}: {exc}".lower()
    if any(m in text for m in ("401", "403", "invalid_api_key", "authentication")):
        return False
    return any(m in text for m in _RETRYABLE_MARKERS)


# ---- 熔断器（进程内，多线程安全）----
class SyncCircuitBreaker:
    """线程安全的三态熔断器（同步版）。

    与 common/resilience.py 的 CircuitBreaker 语义一致，
    区别是加了锁 —— LLM 调用在 FastAPI 的 threadpool 里跑，
    多线程并发下计数器会竞态。
    """

    def __init__(self, name: str, failure_threshold: int = 5,
                 reset_timeout: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._state = "closed"
        self._opened_at = 0.0
        self._lock = threading.Lock()
        self._last_error: str = ""

    def _refresh(self) -> None:
        if (self._state == "open"
                and time.monotonic() - self._opened_at >= self.reset_timeout):
            self._state = "half_open"

    def allow(self) -> bool:
        with self._lock:
            self._refresh()
            return self._state != "open"

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "closed"
            self._last_error = ""

    def record_failure(self, error: str = "") -> None:
        with self._lock:
            self._refresh()
            self._last_error = error
            if self._state == "half_open":
                self._state = "open"
                self._opened_at = time.monotonic()
            elif self._state == "closed":
                self._failures += 1
                if self._failures >= self.failure_threshold:
                    self._state = "open"
                    self._opened_at = time.monotonic()
                    logger.warning(
                        "[llm_guard] 熔断器 %s 转为 OPEN（连续失败 %d 次）",
                        self.name, self._failures)

    @property
    def state(self) -> str:
        with self._lock:
            self._refresh()
            return self._state

    @property
    def last_error(self) -> str:
        return self._last_error

    def snapshot(self) -> dict:
        with self._lock:
            self._refresh()
            return {"name": self.name, "state": self._state,
                    "failures": self._failures,
                    "threshold": self.failure_threshold,
                    "last_error": self._last_error}


_breakers: dict[str, SyncCircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_llm_breaker(name: str = "llm") -> SyncCircuitBreaker:
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = SyncCircuitBreaker(name=name)
        return _breakers[name]


def breaker_snapshot() -> list[dict]:
    """给 /health 或监控端点用。"""
    with _breakers_lock:
        return [b.snapshot() for b in _breakers.values()]


class LLMUnavailableError(RuntimeError):
    """LLM 完全不可用，且没有可用的降级路径。

    调用方应当把它转成 503（服务暂不可用）而不是 500，
    并且**不要丢弃已有结果** —— 纯检索结果仍然有价值。
    """

    def __init__(self, message: str, *, breaker_state: str = "",
                 last_error: str = ""):
        super().__init__(message)
        self.breaker_state = breaker_state
        self.last_error = last_error


# ---- 配置读取（延迟导入避免循环依赖）----
def _cfg() -> Any:
    from common.config import get_settings
    return get_settings()


def _retry_config() -> tuple[int, float, float]:
    """(retries, base_delay, max_delay)"""
    try:
        s = _cfg()
        return (int(getattr(s, "llm_max_retries", 2)),
                float(getattr(s, "llm_retry_base_delay", 0.5)),
                float(getattr(s, "llm_retry_max_delay", 8.0)))
    except Exception:
        return (2, 0.5, 8.0)


def _timeout_config() -> float:
    try:
        s = _cfg()
        return float(getattr(s, "llm_request_timeout", 30.0))
    except Exception:
        return 30.0


# ---- 核心：带保护的同步调用 ----
def guarded_call(
    fn: Callable[..., Any],
    *args: Any,
    breaker_name: str = "llm",
    purpose: str = "",
    fallback: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """熔断 + 退避重试 + 降级 的同步保护调用。

    参数
    ----
    fn            实际要执行的函数（通常是一个 bound method）
    breaker_name  熔断器名，不同上游（如 llm / embedding）分开计数
    purpose       用途描述，仅用于日志
    fallback      降级函数，签名与 fn 一致；熔断或全部重试失败时调用

    行为
    ----
    - 熔断器 OPEN → 直接走 fallback（不浪费一次真实调用）
    - 可重试异常 → 指数退避重试（base_delay * 2^n，上限 max_delay）
    - 不可重试异常（401/403/400）→ 立即失败，不重试
    - 全部失败 → 走 fallback；没有 fallback 则抛 LLMUnavailableError
    """
    cb = get_llm_breaker(breaker_name)
    retries, base_delay, max_delay = _retry_config()

    if not cb.allow():
        logger.warning(
            "[llm_guard] 熔断器 %s 处于 OPEN，跳过调用直接降级（用途=%s）",
            breaker_name, purpose)
        if fallback is not None:
            return fallback(*args, **kwargs)
        raise LLMUnavailableError(
            f"LLM 服务暂不可用（熔断器 {breaker_name} 处于 OPEN 状态）",
            breaker_state=cb.state, last_error=cb.last_error)

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = fn(*args, **kwargs)
            cb.record_success()
            return result
        except Exception as e:  # noqa: BLE001 — 这里要兜住一切上游异常
            last_exc = e
            retryable = is_retryable(e)
            if attempt < retries and retryable:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    "[llm_guard] %s 第 %d 次调用失败（%s），%.1fs 后重试",
                    purpose or breaker_name, attempt + 1, e, delay)
                time.sleep(delay)
                continue
            break  # 不可重试，或重试次数用尽

    cb.record_failure(f"{type(last_exc).__name__}: {last_exc}")
    logger.error("[llm_guard] %s 最终失败：%s", purpose or breaker_name, last_exc)

    if fallback is not None:
        try:
            return fallback(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            logger.error("[llm_guard] 降级路径也失败了：%s", e)
    raise LLMUnavailableError(
        f"LLM 服务不可用：{last_exc}",
        breaker_state=cb.state,
        last_error=f"{type(last_exc).__name__}: {last_exc}",
    ) from last_exc


def _resolve_client() -> tuple[Any, str]:
    """返回 (client, model)。

    provider_mode=local/hybrid 时优先用本地 Ollama。
    """
    from agent1.config import get_deepseek, get_llm_model, get_local_llm

    try:
        s = _cfg()
        mode = str(getattr(s, "provider_mode", "cloud")).lower()
    except Exception:
        mode = "cloud"

    if mode in ("local", "hybrid"):
        local = get_local_llm()
        if local is not None:
            return local, getattr(local, "model_name", "") or "local"

    return get_deepseek(), get_llm_model()


def guarded_chat(
    messages: Sequence[dict],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    json_mode: bool = False,
    purpose: str = "",
    breaker_name: str = "llm",
    fallback: Callable[..., Any] | None = None,
    timeout: float | None = None,
) -> str:
    """带保护的 LLM 文本生成，返回纯文本内容。

    这是所有 LLM 调用点应该改用的入口。
    """
    client, default_model = _resolve_client()
    use_model = model or default_model
    tmo = timeout if timeout is not None else _timeout_config()

    def _do(msgs, **kw):
        kwargs = {"model": use_model, "messages": msgs,
                  "temperature": temperature, "timeout": tmo, **kw}
        if json_mode:
            try:
                resp = client.chat.completions.create(
                    response_format={"type": "json_object"}, **kwargs)
            except Exception:
                # 模型不支持 json_object —— 回退普通调用（不是失败，是降级）
                resp = client.chat.completions.create(**kwargs)
        else:
            resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    return guarded_call(
        _do, list(messages),
        breaker_name=breaker_name, purpose=purpose, fallback=fallback)


def guarded_chat_json(
    messages: Sequence[dict],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    purpose: str = "",
    breaker_name: str = "llm",
    fallback: Callable[..., Any] | None = None,
    timeout: float | None = None,
) -> dict:
    """带保护的 LLM JSON 生成，返回已解析的 dict。"""
    from agent1.config import parse_json_safe

    text = guarded_chat(
        messages, model=model, temperature=temperature, json_mode=True,
        purpose=purpose, breaker_name=breaker_name, fallback=fallback,
        timeout=timeout)
    return parse_json_safe(text)


def reset_breakers() -> None:
    """测试用：重置所有熔断器。"""
    with _breakers_lock:
        for b in _breakers.values():
            b.record_success()
