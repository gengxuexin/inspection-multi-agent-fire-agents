# common/metrics_middleware.py
"""把 /metrics 端点挂到 FastAPI 应用上。

为什么单独一个文件而不是放进 metrics.py：
  metrics.py 是纯指标定义，不该依赖 FastAPI；
  而挂载逻辑要处理"路由是否已存在""是否会与业务路由冲突"这些问题，
  属于web 层关注点。分开后 metrics.py 可以被非 HTTP 场景
  （CLI 脚本、后台任务、MCP Server）直接复用。
"""
from __future__ import annotations

import logging
import re

from fastapi import Response

from common import metrics

logger = logging.getLogger("fire_agents.metrics")

_PATH = "/metrics"


# 路径里的高基数片段：uuid、纯数字 id、hash、任务号等
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.I)
_HEXID_RE = re.compile(r"\b[0-9a-f]{16,}\b", re.I)      # hash / token

# 纯数字 id 段。
# ★ 第一版写成 `/(?<=[a-z_])[0-9]{2,}(?=/|$)`，用了「前面必须是字母」的断言，
#   结果 `/tasks/12345` 匹配不上 —— 因为数字前面是斜杠不是字母。
#   这个 bug 极隐蔽：单测里若只测 `/api/x/12345` 就会通过，
#   而真实路由 `/tasks/{task_id}` 恰好漏掉。
#   改成不依赖前视断言，直接匹配「斜杠 + 数字段」，
#   靠"出现在路径段里"这个语义条件约束即可。
_NUMID_RE = re.compile(r"/\d{2,}(?=/|$)")

# 业务 id：形如 CR-20261016-48aeda / RV-U / review-abc123。
# 这类不是纯数字也不是 uuid，很容易被当成普通路径段留下来。
# 但它们同样是**每个请求都不同**的高基数值 —— agent1 的
# /api/doc/change-reports/{report_id} 就是这类。
_BIZID_RE = re.compile(
    r"/(?:CR|RV|REV|REVIEW)[-_][A-Za-z0-9_-]+", re.I)


def normalize_path(path: str) -> str:
    """把路径里的高基数字段压成占位符。

    ★ 这是 Prometheus 指标设计里最容易踩、后果又最严重的坑：

      agent1 有 `GET /api/doc/change-reports/{report_id}` 这类路由。
      如果把实际路径直接当标签值，
      每个 review_id 都会生成一条新的时间序列 ——
      一千次请求就是一千条 series，Prometheus 的内存和时间
      都会被拖垮，而且没有任何报错，只表现为"查询变慢"。

      Prometheus 官方建议一条指标的 series 数控制在 1000 以内。
      所以标签值必须是**有限枚举**，不能是任何用户输入。

    压法：
      uuid全文-> {uuid}
      长 hex（hash/token） -> {hash}
      业务 id（CR-xxx / RV-xxx） -> {id}
      纯数字段 -> {id}
    """
    p = _UUID_RE.sub("{uuid}", path)
    p = _HEXID_RE.sub("{hash}", p)
    p = _BIZID_RE.sub("/{id}", p)
    p = _NUMID_RE.sub("/{id}", p)
    return p or "/"


def install_metrics_endpoint(app, service: str) -> bool:
    """给 FastAPI app 挂 /metrics。

    :return: 是否挂载成功
    """
    try:
        # 幂等：重复调用（比如网关与子服务都装了同一份）不产生第二个路由
        for r in app.routes:
            if getattr(r, "path", "") == _PATH:
                logger.info("/metrics 已存在，跳过")
                return True
    except Exception as e:                       # noqa: BLE001
        logger.debug("检查已有路由失败：%s", e)

    @app.get(_PATH, include_in_schema=False)
    async def metrics_endpoint():
        """Prometheus 抓取端点。

        刻意不放进 OpenAPI schema —— 它不是业务 API，
        出现在 Swagger 里会干扰使用者的理解。
        """
        body, content_type = metrics.render_latest()
        return Response(content=body, media_type=content_type)

    logger.info("/metrics 已挂载（service=%s, 客户端可用=%s）",
                service, metrics.available())
    return True


def install_http_middleware(app, service: str) -> None:
    """给所有 HTTP 请求自动打埋点（耗时 + 状态码）。

    比逐个端点加装饰器更可靠：
      - 装饰器要人工加，漏一个端点就少一段数据，而漏了不会有任何报错
      - 中间件是全局的，新增端点自动纳入统计

    端点路径做了归一化：把实际的 /api/rag/ask/<uuid> 压成
    /api/rag/ask/{id}，否则每个请求 id 都是一个新时间序列，
    几分钟就能把 Prometheus 撑爆（高基数问题）。
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request

    class _MetricsMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if request.url.path == _PATH:
                # 不给 /metrics 自身打点，否则每次抓取都产生一条样本，
                # 让"请求延迟"指标被自己的噪声污染。
                return await call_next(request)

            import time
            t0 = time.perf_counter()
            status = 500
            try:
                response = await call_next(request)
                status = response.status_code
                return response
            finally:
                metrics.observe_http(
                    service=service,
                    method=request.method,
                    endpoint=normalize_path(request.url.path),
                    status=status,
                    duration=time.perf_counter() - t0,
                )

    app.add_middleware(_MetricsMiddleware)
    logger.info("HTTP 埋点中间件已启用（service=%s）", service)