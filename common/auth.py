# common/auth.py
"""JWT 鉴权（本地 HS256 / OIDC RS256-ready）与 Agent 内部调用 Token 校验。

用法：
    from common.auth import get_current_user, require_internal_token, UserCtx

    @app.get("/secure")
    async def secure(user: UserCtx = Depends(get_current_user)):
        ...
"""
import logging
import time

import jwt
from fastapi import HTTPException, Request
from pydantic import BaseModel

from common.config import get_settings

logger = logging.getLogger("fire_agents.auth")

# 免鉴权路径（健康检查 / 文档 / 根路径）
EXEMPT_PATHS = frozenset({"/", "/agents/health", "/health", "/docs", "/openapi.json"})

# clearance 等级含义：0=公开 1=内部 2=秘密 3=机密 4=绝密
CLEARANCE_LEVELS = {0: "公开", 1: "内部", 2: "秘密", 3: "机密", 4: "绝密"}

_auth_disabled_warned = False


class UserCtx(BaseModel):
    """当前用户上下文（由 JWT claims 构造）。"""

    sub: str
    dept: str = "消防支队"
    role: str = "officer"
    clearance: int = 2


def create_token(
    sub: str,
    dept: str = "消防支队",
    role: str = "officer",
    clearance: int = 2,
    expires_min: int | None = None,
) -> str:
    """签发本地 JWT（HS256，算法取 settings.jwt_algorithm）。"""
    settings = get_settings()
    if not settings.jwt_secret:
        raise RuntimeError("JWT_SECRET 未配置")
    now = int(time.time())
    ttl_min = settings.jwt_expire_min if expires_min is None else expires_min
    payload = {
        "sub": sub,
        "dept": dept,
        "role": role,
        "clearance": clearance,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": now + int(ttl_min) * 60,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_token(token: str) -> dict:
    """校验 JWT 并返回 claims。

    - 配置了 jwt_oidc_jwks_url：走 OIDC JWKS（RS256），校验 iss/aud；
    - 否则：本地密钥按 settings.jwt_algorithm 校验。
    失败统一抛 ValueError("token 无效或已过期")；JWT_SECRET 未配置抛 RuntimeError。
    """
    settings = get_settings()
    try:
        if settings.jwt_oidc_jwks_url:
            jwks_client = jwt.PyJWKClient(settings.jwt_oidc_jwks_url)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                issuer=settings.jwt_issuer,
                audience=settings.jwt_audience,
            )
        else:
            if not settings.jwt_secret:
                raise RuntimeError("JWT_SECRET 未配置")
            payload = jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm],
                issuer=settings.jwt_issuer,
                audience=settings.jwt_audience,
            )
        return dict(payload)
    except jwt.InvalidTokenError as e:
        raise ValueError("token 无效或已过期") from e
    except RuntimeError:
        raise
    except Exception as e:
        raise ValueError("token 无效或已过期") from e


async def get_current_user(request: Request) -> UserCtx:
    """FastAPI 依赖：解析 Bearer JWT 构造 UserCtx，并写入 request.state.user 供审计用。"""
    global _auth_disabled_warned
    settings = get_settings()

    # 1) 豁免路径直接放行
    if request.url.path in EXEMPT_PATHS:
        ctx = UserCtx(sub="anonymous")
        request.state.user = ctx
        return ctx

    # 2) 本地开发跳过鉴权（仅提示一次）
    if settings.auth_disabled:
        if not _auth_disabled_warned:
            _auth_disabled_warned = True
            logger.warning("[auth] AUTH_DISABLED 已开启，跳过 JWT 校验（仅限本地开发/测试！）")
        ctx = UserCtx(sub="dev")
        request.state.user = ctx
        return ctx

    # 3) 解析 Authorization: Bearer <token>
    auth_header = request.headers.get("Authorization", "")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少 Authorization 头或格式非 Bearer token")
    token = auth_header[len("Bearer "):].strip()
    if not token:
        raise HTTPException(status_code=401, detail="token 为空")

    try:
        claims = verify_token(token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=401, detail="token 校验失败") from e

    try:
        clearance = int(claims.get("clearance", 2))
    except (TypeError, ValueError):
        clearance = 2

    ctx = UserCtx(
        sub=str(claims.get("sub") or "anonymous"),
        dept=str(claims.get("dept") or "消防支队"),
        role=str(claims.get("role") or "officer"),
        clearance=clearance,
    )
    request.state.user = ctx
    return ctx


async def require_internal_token(request: Request) -> None:
    """FastAPI 依赖：Agent 服务间内部调用校验（X-Internal-Token）。

    - 根路径（健康检查）放行；
    - settings.agent_shared_token 为空时视为未启用，放行；
    - 否则请求头 X-Internal-Token 必须与配置一致，不匹配返回 401。
    """
    if request.url.path == "/":
        return
    settings = get_settings()
    if not settings.agent_shared_token:
        return
    provided = request.headers.get("X-Internal-Token", "")
    if provided != settings.agent_shared_token:
        raise HTTPException(
            status_code=401,
            detail="内部调用校验失败：X-Internal-Token 缺失或不匹配",
        )
