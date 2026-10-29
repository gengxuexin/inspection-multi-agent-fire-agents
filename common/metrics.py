# common/metrics.py
"""Prometheus 指标封装：让四个服务暴露 /metrics 供 Prometheus 抓取。

## 为什么需要这一层

现有的 `common/tracing.py` 管的是 **trace**（单次请求的调用链），
但排查线上问题时，trace 回答的是"这一次请求发生了什么"，
而真正高频的问题是：

- 哪个接口的 P99 延迟在涨？
- 熔断器打开过多少次？
- RAG 检索的召回率有没有下滑？
- 每天烧掉多少 token / 多少钱？

这些是**聚合指标**，属于 metrics 的职责。trace 每次请求产生几百个 span，
不适合用来算统计量；反过来，用日志手工统计更是不可行。

## 设计原则（与 tracing.py 保持一致）

1. **全部可选依赖**：`prometheus_client` 没装时自动降级为 no-op，
   不抛错、不影响服务启动。这一点很关键 ——
   监控组件缺失不该让业务服务起不来。
2. **重复注册不报错**：本文件被 import 两次（服务入口 + 测试）时
   `Counter` 会抛 `Duplicated timeseries`。这里用模块级缓存兜住。
3. **低基数标签**：标签值不能是 question / file_path 这类高基数字段，
   否则 Prometheus 内存会爆。业务标识放日志里，不放标签里。

## 指标清单

| 指标 | 类型 | 用途 |
|------|------|------|
| `fire_http_requests_total` | Counter | 请求总数，按服务/方法/端点/状态 |
| `fire_http_request_duration_seconds` | Histogram | 请求延迟分布 |
| `fire_rag_retrieve_duration_seconds` | Histogram | 检索延迟（与生成区分） |
| `fire_rag_retrieved_chunks` | Histogram | 每次检索返回的片段数 |
| `fire_llm_calls_total` | Counter | LLM 调用数，按服务/用途/成败 |
| `fire_llm_tokens_total` | Counter | token 消耗，按模型/用途 |
| `fire_llm_duration_seconds` | Histogram | LLM 调用延迟 |
| `fire_circuit_breaker_state` | Gauge | 熔断器状态 0=closed 1=open 2=half |
| `fire_circuit_breaker_transitions_total` | Counter | 熔断器状态切换次数 |
| `fire_budget_exhausted_total` | Counter | 预算耗尽次数 |
| `fire_mcp_tool_calls_total` | Counter | MCP 工具调用数 |
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger("fire_agents.metrics")

# ---- 可用性探测：没装 prometheus_client 就整体降级 ----
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    _PROM_AVAILABLE = True
except ImportError:
    Counter = Gauge = Histogram = None      # type: ignore
    _PROM_AVAILABLE = False

# 需要在 Registry 里手工登记、避免重复创建的类型
_METRICS: dict[str, Any] = {}

# 标签里出现的所有服务名（固定枚举，保证低基数）
SERVICES = ("gateway", "agent1", "agent2", "agent3")


def _get(key: str, factory: Callable[..., Any], *args, **kwargs) -> Any:
    """取指标实例，重复调用返回同一个。

    prometheus_client 的 `Counter(...)` 第二次用同名会抛异常。
    在多进程/重复 import 的场景下（pytest 反复导入、服务起两次），
    这个异常会让服务起不来 —— 所以必须缓存。
    """
    if not _PROM_AVAILABLE:
        return None
    if key not in _METRICS:
        _METRICS[key] = factory(*args, **kwargs)
    return _METRICS[key]


# ==================== 指标定义 ====================
def http_requests_total():
    return _get("http_requests_total", Counter,
                "fire_http_requests_total",
                "HTTP 请求总数",
                ["service", "method", "endpoint", "status"])


def http_request_duration_seconds():
    return _get("http_dur", Histogram,
                "fire_http_request_duration_seconds",
                "HTTP 请求延迟（秒）",
                ["service", "method", "endpoint"],
                # 桶选的是"用户能感知的延迟档位"：
                # 0.1=无感，0.5=略慢，1/2/5=明显卡顿，10=不可接受。
                # 刻意不设 60s 的桶 —— 超长尾的根因通常是下游挂了，
                # 会被单独的超时计数覆盖，不需要拉一个巨大的桶。
                buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0))


def rag_retrieve_duration_seconds():
    return _get("rag_ret_dur", Histogram,
                "fire_rag_retrieve_duration_seconds",
                "RAG 检索延迟（秒）",
                ["service", "backend"],
                buckets=(0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0))


def rag_retrieved_chunks():
    return _get("rag_chunks", Histogram,
                "fire_rag_retrieved_chunks",
                "单次检索返回的片段数",
                ["service"],
                buckets=(0, 1, 3, 5, 8, 10, 15, 20, 30, 50))


def llm_calls_total():
    return _get("llm_calls", Counter,
                "fire_llm_calls_total",
                "LLM 调用总数",
                ["service", "purpose", "outcome"])


def llm_tokens_total():
    return _get("llm_tokens", Counter,
                "fire_llm_tokens_total",
                "LLM token 消耗",
                ["model", "purpose", "kind"])     # kind: prompt/completion


def llm_duration_seconds():
    return _get("llm_dur", Histogram,
                "fire_llm_duration_seconds",
                "LLM 调用延迟（秒）",
                ["service", "purpose"],
                buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0))


def circuit_breaker_state():
    return _get("cb_state", Gauge,
                "fire_circuit_breaker_state",
                "熔断器状态（0=closed 1=open 2=half_open）",
                ["service", "target"])


def circuit_breaker_transitions_total():
    return _get("cb_trans", Counter,
                "fire_circuit_breaker_transitions_total",
                "熔断器状态切换次数",
                ["service", "target", "from_state", "to_state"])


def budget_exhausted_total():
    return _get("budget", Counter,
                "fire_budget_exhausted_total",
                "预算耗尽次数",
                ["service"])


def mcp_tool_calls_total():
    return _get("mcp_calls", Counter,
                "fire_mcp_tool_calls_total",
                "MCP 工具调用数",
                ["tool", "outcome"])


# ==================== 便捷接口 ====================
def observe_http(service: str, method: str, endpoint: str,
                 status: int, duration: float) -> None:
    """记一次 HTTP 请求。异常时静默 —— 监控不该拖垮业务。"""
    if not _PROM_AVAILABLE:
        return
    try:
        http_requests_total().labels(
            service=service, method=method,
            endpoint=endpoint, status=str(status)).inc()
        http_request_duration_seconds().labels(
            service=service, method=method, endpoint=endpoint).observe(duration)
    except Exception as e:                       # noqa: BLE001
        logger.debug("metrics observe_http 失败：%s", e)


def observe_rag_retrieve(service: str, backend: str,
                         duration: float, chunks: int) -> None:
    """记一次 RAG 检索。"""
    if not _PROM_AVAILABLE:
        return
    try:
        rag_retrieve_duration_seconds().labels(
            service=service, backend=backend).observe(duration)
        rag_retrieved_chunks().labels(service=service).observe(chunks)
    except Exception as e:                       # noqa: BLE001
        logger.debug("metrics observe_rag_retrieve 失败：%s", e)


def observe_llm(service: str, purpose: str, model: str,
                duration: float, ok: bool = True,
                prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    """记一次 LLM 调用。"""
    if not _PROM_AVAILABLE:
        return
    try:
        llm_calls_total().labels(
            service=service, purpose=purpose,
            outcome="ok" if ok else "error").inc()
        llm_duration_seconds().labels(
            service=service, purpose=purpose).observe(duration)
        if prompt_tokens:
            llm_tokens_total().labels(
                model=model, purpose=purpose, kind="prompt").inc(prompt_tokens)
        if completion_tokens:
            llm_tokens_total().labels(
                model=model, purpose=purpose,
                kind="completion").inc(completion_tokens)
    except Exception as e:                       # noqa: BLE001
        logger.debug("metrics observe_llm 失败：%s", e)


def set_circuit_breaker(service: str, target: str, state: str) -> None:
    """更新熔断器状态（gauge 语义：设值而非累加）。"""
    if not _PROM_AVAILABLE:
        return
    mapping = {"closed": 0, "open": 1, "half_open": 2}
    if state not in mapping:
        return
    try:
        circuit_breaker_state().labels(
            service=service, target=target).set(mapping[state])
    except Exception as e:                       # noqa: BLE001
        logger.debug("metrics set_circuit_breaker 失败：%s", e)


def record_cb_transition(service: str, target: str,
                         from_state: str, to_state: str) -> None:
    """记一次熔断器状态切换。"""
    if not _PROM_AVAILABLE:
        return
    try:
        circuit_breaker_transitions_total().labels(
            service=service, target=target,
            from_state=from_state, to_state=to_state).inc()
    except Exception as e:                       # noqa: BLE001
        logger.debug("metrics record_cb_transition 失败：%s", e)


@contextmanager
def timed(service: str, purpose: str):
    """计时上下文管理器。

    用法：
        with metrics.timed("agent1", "rag_ask"):
            answer = rag_ask(q)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        observe_llm(service, purpose, model="", duration=time.perf_counter() - t0)


def instrument_endpoint(service: str, endpoint: Optional[str] = None) -> Callable:
    """给 FastAPI 端点加埋点的装饰器。

    刻意不用 `endpoint=None` 自动取函数名 —— 那会把
    `POST /api/rag/ask` 和 `POST /api/rag/retrieve` 都记成同一个名字，
    指标维度直接失去意义。宁可要求显式传。
    """
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        async def awrapper(*args, **kwargs):
            t0 = time.perf_counter()
            status = 200
            try:
                return await fn(*args, **kwargs)
            except Exception:
                status = 500
                raise
            finally:
                observe_http(service, fn.__name__,
                             endpoint or fn.__name__,
                             status, time.perf_counter() - t0)
        return awrapper
    return deco


# ==================== 审计可靠性 ====================
# 审计写失败是「执法留痕丢失」，属于需要立即告警的事件，
# 所以除了计数器还要有一个 gauge 反映当前积压的死信数。
def audit_write_failures_total():
    return _get("audit_write_fail", Counter,
                "fire_audit_write_failures_total",
                "审计日志写入失败次数（业务不中断，失败记录进死信队列）",
                ["service"])


def audit_write_dead_letters():
    return _get("audit_dead", Gauge,
                "fire_audit_write_dead_letters",
                "审计写入失败积压的死信条数（>0 需人工补录）",
                ["service"])


def render_latest() -> tuple[bytes, str]:
    """渲染 /metrics 响应体。

    :return: (内容, Content-Type)
    """
    if not _PROM_AVAILABLE:
        return (b"# prometheus_client not installed\n", "text/plain; charset=utf-8")
    try:
        return (generate_latest(REGISTRY), CONTENT_TYPE_LATEST)
    except Exception as e:                       # noqa: BLE001
        return (f"# metrics render error: {e}\n".encode(), "text/plain; charset=utf-8")


def available() -> bool:
    return _PROM_AVAILABLE