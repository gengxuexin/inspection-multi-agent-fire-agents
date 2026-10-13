# common/config.py
"""全局配置：统一读取项目根 .env（pydantic-settings）。

所有生产化改造的新配置项集中在此，环境变量名与字段名大小写不敏感。
用法：
    from common.config import get_settings
    settings = get_settings()
"""
import os
from functools import lru_cache

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - 环境未装 pydantic-settings 时给出明确报错
    raise ImportError(
        "缺少 pydantic-settings，请先安装：pip install -r requirements.txt"
    )

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(_ROOT, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Agent 地址 ----
    agent1_url: str = "http://127.0.0.1:8001"
    agent2_url: str = "http://127.0.0.1:8002"
    agent3_url: str = "http://127.0.0.1:8003"

    # ---- 鉴权（P1）----
    jwt_secret: str = ""
    jwt_issuer: str = "fire-agents"
    jwt_audience: str = "fire-agents"
    jwt_expire_min: int = 480
    jwt_algorithm: str = "HS256"
    jwt_oidc_jwks_url: str = ""          # 配置后走 OIDC RS256 校验
    agent_shared_token: str = ""          # Agent 间内部 Token，空=关闭校验
    auth_disabled: bool = False           # AUTH_DISABLED=1 本地开发跳过鉴权

    # ---- Redis / 任务状态（P8）----
    redis_url: str = ""

    # ---- Milvus（P9）----
    milvus_uri: str = ""
    milvus_collection: str = "fire_laws"

    # ---- 可观测（P10）----
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""
    otel_exporter_otlp_endpoint: str = ""
    otel_service_name: str = "fire-agents"

    # ---- 预算（P6）----
    budget_max_tokens: int = 200000
    budget_max_cost: float = 10.0

    # ---- 编排与路由（P2/P7）----
    orchestrator_mode: str = "legacy"     # legacy | langgraph
    classifier_conf_threshold: float = 0.6
    route_max_hops: int = 3

    # ---- 部署（P11）----
    registry: str = ""
    base_image: str = "python:3.10-slim"

    # ---- 日志目录 ----
    log_dir: str = os.path.join(_ROOT, "logs")

    # ---- LLM（级联路由 L3 兜底用，复用 DeepSeek 配置）----
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def project_root() -> str:
    return _ROOT
