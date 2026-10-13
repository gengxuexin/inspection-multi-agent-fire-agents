# server.py (DEPRECATED)
"""
[已废弃] 消防智能执法系统旧入口。

所有请求透明反代到 gateway:8000（统一鉴权 + 审计 + 级联路由入口）。
请直接使用 gateway.py 作为统一入口：
    uvicorn gateway:app --host 0.0.0.0 --port 8000

本文件仅为向后兼容保留，不推荐独立启动。
"""
from __future__ import annotations

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI(
    title="消防智能执法系统 (DEPRECATED)",
    description="⚠️ 此入口已废弃，请使用 gateway:8000",
    version="0.1.0-deprecated",
)

GATEWAY_URL = "http://127.0.0.1:8000"

# 不转发的 hop-by-hop 头
_EXCLUDED_HEADERS = frozenset({
    "content-encoding", "transfer-encoding", "connection",
    "keep-alive", "host",
})


@app.get("/")
async def root():
    return JSONResponse(
        status_code=410,
        content={
            "status": "deprecated",
            "message": "此入口已废弃，请使用 gateway:8000",
            "gateway_url": GATEWAY_URL,
        },
    )


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy_to_gateway(request: Request, path: str):
    """透明反代到 gateway:8000。"""
    url = f"{GATEWAY_URL}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _EXCLUDED_HEADERS
    }

    try:
        async with httpx.AsyncClient(trust_env=False, timeout=300) as client:
            resp = await client.request(
                request.method, url,
                headers=headers, content=body,
            )
    except httpx.ConnectError:
        return JSONResponse(
            status_code=503,
            content={
                "error": "gateway 不可达",
                "detail": f"无法连接 {GATEWAY_URL}，请先启动 gateway",
            },
        )

    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in _EXCLUDED_HEADERS
    }
    resp_headers["X-Deprecated"] = "true"

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=resp.headers.get("content-type"),
    )


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  ⚠️  server.py 已废弃！")
    print("  请改用: uvicorn gateway:app --port 8000")
    print("  本文件仅在 9000 端口提供兼容反代")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=9000)
