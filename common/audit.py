# common/audit.py
"""审计日志与敏感信息脱敏。

logger = logging.getLogger("fire_agents.audit")

- redact：手机号 / 身份证 / 密码键值对 / sk-密钥 脱敏 + 超长截断
- install_audit_middleware：请求/响应摘要审计（JSONL，best-effort）
- audit_event：显式业务事件记录（如 review approve/reject、task submit）
"""
import json
import os
import re
import time
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
from starlette.types import ASGIApp

from common.config import get_settings
import logging

# ---- 脱敏规则 ----
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ID_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_SECRET_KV_RE = re.compile(
    r"(?i)(password|passwd|pwd|secret|token|api[_-]?key)\s*[=:：]\s*\S+"
)
_SK_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")

# 请求体捕获上限（字节）
_MAX_BODY_BYTES = 51200

_jsonl_write_warned = False


def redact(text: str, max_len: int = 600) -> str:
    """对文本脱敏并截断：None/空 -> ""；仅当实际截断时追加 "...(截断)"。"""
    if text is None:
        return ""
    text = str(text)
    if not text:
        return ""
    # 1) 18 位身份证：保留前 6 后 4（先于手机号，避免长数字串被部分命中）
    text = _ID_RE.sub(lambda m: m.group(0)[:6] + "********" + m.group(0)[-4:], text)
    # 2) 手机号：保留前 3 后 4
    text = _PHONE_RE.sub(lambda m: m.group(0)[:3] + "****" + m.group(0)[-4:], text)
    # 3) 密码类键值对：键名 + =***
    text = _SECRET_KV_RE.sub(lambda m: m.group(1) + "=***", text)
    # 4) sk- 开头密钥
    text = _SK_KEY_RE.sub("sk-***", text)
    # 5) 超长截断
    if max_len and len(text) > max_len:
        text = text[:max_len] + "...(截断)"
    return text


def get_client_ip(request: Request) -> str:
    """尽力获取客户端真实 IP（代理头优先）。"""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    xri = request.headers.get("x-real-ip", "")
    if xri:
        return xri.strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _default_log_path() -> str:
    return os.path.join(get_settings().log_dir, "audit.jsonl")


def _write_jsonl(path: str, record: dict) -> None:
    """best-effort 追加一行 JSONL；目录自动创建；失败吞掉异常仅提示一次。"""
    global _jsonl_write_warned
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        if not _jsonl_write_warned:
            _jsonl_write_warned = True
            logger.error(f"[audit] 警告：审计日志写入失败（仅提示一次）：{e}")


def _user_sub(user: Any) -> str:
    """从 UserCtx / dict / str 中提取 sub。"""
    if user is None:
        return "anonymous"
    if isinstance(user, str):
        return user or "anonymous"
    sub = getattr(user, "sub", None)
    if sub is None and isinstance(user, dict):
        sub = user.get("sub")
    return str(sub) if sub else "anonymous"


def _sub_from_state(request: Request) -> str:
    """从 request.state.user 读取 sub（鉴权依赖写入），取不到用 anonymous。"""
    try:
        state = request.scope.get("state") or {}
        return _user_sub(state.get("user"))
    except Exception:  # noqa: BLE001
        return "anonymous"


class AuditMiddleware(BaseHTTPMiddleware):
    """审计中间件：记录请求/响应摘要（脱敏后）到 JSONL 文件。"""

    def __init__(self, app: ASGIApp, log_path: str):
        super().__init__(app)
        self.log_path = log_path

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        method = request.method
        path = request.url.path
        client = get_client_ip(request)
        input_summary = ""
        output_summary = ""
        status_code = 500
        try:
            input_summary = await self._capture_request_body(request)
            response = await call_next(request)
            status_code = response.status_code
            response, output_summary = await self._capture_response_body(response)
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            record = {
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "sub": _sub_from_state(request),
                "method": method,
                "path": path,
                "status": status_code,
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                "client": client,
                "input_summary": redact(input_summary),
                "output_summary": redact(output_summary),
            }
            _write_jsonl(self.log_path, record)

    async def _capture_request_body(self, request: Request) -> str:
        """仅捕获小型 JSON 请求体；multipart 记 "file" 占位；其他类型不读取。"""
        try:
            ct = request.headers.get("content-type", "")
            if "multipart/form-data" in ct:
                return "file"
            if "application/json" not in ct:
                return ""
            cl = request.headers.get("content-length", "")
            if not cl.isdigit() or int(cl) >= _MAX_BODY_BYTES:
                return ""
            body = await request.body()

            # 重建 receive，保证下游路由仍能读到请求体
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            request._receive = receive  # noqa: SLF001
            return body.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""

    async def _capture_response_body(self, response):
        """仅缓冲 JSON 响应体用于审计，其余响应直接透传。"""
        ct = response.headers.get("content-type", "")
        if not ct.startswith("application/json") or not hasattr(response, "body_iterator"):
            return response, ""
        chunks = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunks.append(chunk)
            else:
                chunks.append(str(chunk).encode("utf-8"))
        buffered = b"".join(chunks)
        # 重建响应：保留除 content-length 外的原始头（长度不变，CL 由新响应重算）
        new_resp = StarletteResponse(content=buffered, status_code=response.status_code)
        new_resp.raw_headers.extend(
            (k, v) for (k, v) in response.raw_headers if k.lower() != b"content-length"
        )
        return new_resp, buffered.decode("utf-8", errors="replace")


def install_audit_middleware(app: FastAPI, log_path: str | None = None) -> None:
    """为 FastAPI 应用挂载审计中间件，默认写 <log_dir>/audit.jsonl。"""
    path = log_path or _default_log_path()
    app.add_middleware(AuditMiddleware, log_path=path)


def audit_event(action: str, user: Any = None, log_path: str | None = None, **fields: Any) -> None:
    """显式业务事件审计（如 review approve/reject、task submit），写同一 JSONL 文件。"""
    record = {
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "sub": _user_sub(user),
        "action": action,
    }
    for key, value in fields.items():
        record[key] = redact(str(value))
    _write_jsonl(log_path or _default_log_path(), record)
