# common/tracing.py
"""可观测性封装：OpenTelemetry Tracing + Langfuse LLM 调用日志。

logger = logging.getLogger("fire_agents.tracing")

设计原则：
- 全部为可选依赖：opentelemetry / langfuse / fastapi 任一未安装或未配置时优雅降级为 no-op；
- setup_tracing() 幂等（模块级 flag 防重复初始化），可在各服务入口重复调用；
- 业务侧用法：
    from common.tracing import setup_tracing, trace_span, get_langfuse_logger
    setup_tracing(app)                    # FastAPI 入口处调用一次
    @trace_span("gateway.dispatch")       # 同步/异步函数均可
    async def dispatch(...): ...
    get_langfuse_logger().log_llm_call("classify", q, answer, usage={...})
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional
import logging

try:
    from common.config import get_settings
except ImportError:  # pragma: no cover - 以包内相对方式导入
    from .config import get_settings

# ---- OpenTelemetry 可用性探测（未安装则降级为 no-op，不抛错） ----
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import Status, StatusCode
    _OTEL_AVAILABLE = True
except ImportError:
    otel_trace = None
    Status = None
    StatusCode = None
    _OTEL_AVAILABLE = False

_TRACER_NAME = "fire-agents"

# setup_tracing 幂等标记
_setup_done = False
_setup_ok = False


def _get_tracer():
    """惰性取 tracer：provider 尚未设置时 otel 返回 Proxy/NoOp tracer，安全无害。"""
    return otel_trace.get_tracer(_TRACER_NAME)


def _make_exporter(endpoint: str):
    """按 endpoint 形态选择 OTLP exporter：HTTP 路径型用 http，否则优先 gRPC。"""
    ep = (endpoint or "").strip().lower()
    if "/v1/traces" in ep or "/api/public/otel" in ep:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        return OTLPSpanExporter(endpoint=endpoint)
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        return OTLPSpanExporter(endpoint=endpoint)
    except ImportError:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        return OTLPSpanExporter(endpoint=endpoint)


def setup_tracing(app=None, service_name: Optional[str] = None) -> bool:
    """初始化 OpenTelemetry Tracing（幂等）。

    返回 True 表示已接入；返回 False 表示未配置 otel endpoint 或初始化失败，
    业务代码无需关心差异（trace_span 会自动降级为无开销直通）。
    """
    global _setup_done, _setup_ok
    if _setup_done:
        return _setup_ok
    _setup_done = True

    try:
        settings = get_settings()
    except Exception as exc:  # 配置层异常不阻断业务启动
        logger.error(f"[tracing] 读取配置失败，跳过 tracing 初始化: {exc}")
        _setup_ok = False
        return False

    endpoint = (getattr(settings, "otel_exporter_otlp_endpoint", "") or "").strip()
    if not endpoint:
        _setup_ok = False  # 未配置 OTLP endpoint → no-op
        return False

    svc = service_name or getattr(settings, "otel_service_name", "") or "fire-agents"

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        exporter = _make_exporter(endpoint)
        provider = TracerProvider(resource=Resource.create({"service.name": svc}))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # FastAPI 自动埋点（app 未传或 fastapi 未安装则跳过）
        if app is not None:
            try:
                from fastapi import FastAPI
                from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
                if isinstance(app, FastAPI):
                    FastAPIInstrumentor.instrument_app(app)
            except Exception as exc:
                logger.error(f"[tracing] FastAPI 自动埋点失败(忽略): {exc}")

        # httpx 出站请求埋点（gateway 转发各 Agent 均走 httpx）
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            HTTPXClientInstrumentor.instrument()
        except Exception as exc:
            logger.error(f"[tracing] httpx(同步) 自动埋点失败(忽略): {exc}")
        try:
            from opentelemetry.instrumentation.httpx import AsyncHTTPXClientInstrumentor
            AsyncHTTPXClientInstrumentor.instrument()
        except Exception as exc:
            logger.error(f"[tracing] httpx(异步) 自动埋点失败(忽略): {exc}")

        _setup_ok = True
        return True
    except Exception as exc:
        logger.error(f"[tracing] OpenTelemetry 初始化失败，已降级为无追踪模式: {exc}")
        _setup_ok = False
        return False


def trace_span(name: str) -> Callable:
    """span 装饰器：同时支持 sync / async 函数。

    - otel 可用：包一层 span，异常时 record_exception + set_status(ERROR) 后原样抛出；
      span 上附带属性 fire.span=<name> 便于按业务维度检索。
    - otel 不可用：原函数直接返回，零装饰开销。
    """
    def decorator(func: Callable) -> Callable:
        if not _OTEL_AVAILABLE:
            return func

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with _get_tracer().start_as_current_span(name) as span:
                    span.set_attribute("fire.span", name)
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                        raise
            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with _get_tracer().start_as_current_span(name) as span:
                span.set_attribute("fire.span", name)
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
        return sync_wrapper

    return decorator


class LangfuseLogger:
    """Langfuse LLM 调用日志；三个密钥配置不全时自动禁用为 no-op。"""

    def __init__(self) -> None:
        self._client: Any = None
        self.enabled = False
        try:
            s = get_settings()
        except Exception as exc:
            logger.error(f"[tracing] 读取配置失败，Langfuse 未启用: {exc}")
            return
        public_key = getattr(s, "langfuse_public_key", "") or ""
        secret_key = getattr(s, "langfuse_secret_key", "") or ""
        host = getattr(s, "langfuse_host", "") or ""
        if not (public_key and secret_key and host):
            return  # 未完整配置 → 保持 no-op
        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=public_key, secret_key=secret_key, host=host
            )
            self.enabled = True
        except Exception as exc:
            logger.error(f"[tracing] Langfuse 初始化失败(静默降级): {exc}")
            self._client = None

    def log_llm_call(
        self,
        name: str,
        input_text: str,
        output_text: str,
        usage: Optional[dict] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """记录一次 LLM 调用（trace + generation），全程静默失败不影响主流程。"""
        if not self.enabled or self._client is None:
            return
        try:
            meta = dict(metadata or {})
            if latency_ms is not None:
                meta.setdefault("latency_ms", latency_ms)
            trace = self._client.trace(name=name, metadata=meta)
            gen_kwargs: dict = {
                "name": name,
                "input": input_text,
                "output": output_text,
                "metadata": meta,
            }
            if usage is not None:
                gen_kwargs["usage"] = usage
            trace.generation(**gen_kwargs)
        except Exception:
            pass  # 可观测日志失败不抛出

    def flush(self) -> None:
        """同步刷盘（如进程退出前调用）。"""
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception:
            pass


_langfuse_logger: Optional[LangfuseLogger] = None


def get_langfuse_logger() -> LangfuseLogger:
    """模块级单例。"""
    global _langfuse_logger
    if _langfuse_logger is None:
        _langfuse_logger = LangfuseLogger()
    return _langfuse_logger
