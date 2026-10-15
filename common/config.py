# common/config.py
"""全局配置：统一读取项目根 .env（pydantic-settings）。

所有生产化改造的新配置项集中在此，环境变量名与字段名大小写不敏感。
用法：
    from common.config import get_settings
    settings = get_settings()
"""
import os
import subprocess
import sys
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
    orchestrator_mode: str = "langgraph"  # langgraph（生产默认） | legacy
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

    # ---- Agent1：Embedding（索引依赖，建库后不可变）----
    # 本地 Ollama 向量化 3841 块需 3 小时以上（3.04s/chunk），已改为云端 API。
    # 选型：阿里云 text-embedding-v4
    #   - 维度 1024，与既有 FAISS 索引（bge-large-zh-v1.5）**完全一致**，
    #     换API 无需改动索引维度，重建即可
    #   - 中文语义检索专项优化，匹配法律条文场景
    #   - 前 1000 万 token 免费（3841 chunk ×约 400 字 ≈ 150 万字，一次性重建足够）
    #   - 提供 OpenAI 兼容端点，现有 OpenAI SDK 代码零改动
    embedding_provider: str = "dashscope"     # dashscope（云端）| ollama（本地）
    embedding_api_key: str = ""                # 云端 key；留空则复用 DASHSCOPE_API_KEY
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024                  # ⚠️ 换模型必须核对，维度变了索引要重建
    embedding_batch_size: int = 10# 云端单批文本数（DashScope 单请求上限 10）
    embedding_timeout: int = 120# 单次请求超时（秒）

    # ---- 本地 Ollama（embedding_provider=ollama 时回退使用）----
    ollama_base_url: str = ""            # 空则按平台自动探测
    ollama_embedding_model: str = "quentinz/bge-large-zh-v1.5"

    # ---- Agent2：讯飞（在线听写 + 录音文件转写）----
    xfyun_app_id: str = ""
    xfyun_api_key: str = ""
    xfyun_api_secret: str = ""
    lfasr_secret_key: str = ""
    lfasr_hot_words: str = ""

    # ---- Agent2：离线 ASR ----
    whisper_model_path: str = ""         # 空则使用 faster-whisper 默认缓存
    whisper_download_root: str = ""      # 空则 agent2/models/whisper
    vosk_model_path: str = ""

    # ---- Agent2：本地 LLM / TTS ----
    ollama_url: str = "http://localhost:11434/api/chat"
    ollama_model: str = "qwen2.5:7b"
    tts_voice_edge: str = "zh-CN-XiaoxiaoNeural"
    tts_emotion: str = "正常"

    # ---- Agent2/3：阿里云 CosyVoice / 百炼 ----
    dashscope_api_key: str = ""
    dashscope_workspace_id: str = ""
    dashscope_region: str = "beijing"
    cosyvoice_model: str = "cosyvoice-v2"
    cloned_voice_id: str = ""

    # ---- Agent3：百度 OCR ----
    baidu_api_key: str = ""
    baidu_secret_key: str = ""

    # ---- Agent3：模型 ----
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vlm_model: str = "qwen3-vl-flash"    # 视觉理解
    judge_model: str = "qwen-plus"       # 综合判断（纯文本推理）
    yolo_weights: str = ""               # 空则用 ultralytics 默认模型

    # ---- 模型缓存根目录（各 Agent 运行时产物）----
    model_cache_dir: str = ""            # 空则 <project_root>/models


@lru_cache
def get_settings() -> Settings:
    return Settings()


def project_root() -> str:
    return _ROOT


# ==================== 派生路径与平台探测 ====================


def agent_dir(name: str) -> str:
    """返回 agent1/agent2/agent3 的绝对路径。"""
    return os.path.join(_ROOT, name)


def models_root() -> str:
    """模型缓存根目录。"""
    s = get_settings()
    return s.model_cache_dir or os.path.join(_ROOT, "models")


def ollama_base_url() -> str:
    """Ollama 地址：显式配置 > 环境变量 > 平台默认。

    Windows/macOS 直接用 localhost；Linux 尝试从默认路由推断宿主 IP。
    """
    s = get_settings()
    if s.ollama_base_url:
        return s.ollama_base_url
    env = os.getenv("OLLAMA_BASE_URL")
    if env:
        return env
    if os.name == "nt" or sys.platform == "darwin":
        return "http://localhost:11434/v1"
    try:
        out = subprocess.check_output(
            "ip route show default | awk '{print $3}'",
            shell=True, timeout=2,
        ).decode().strip()
        if out:
            return f"http://{out}:11434/v1"
    except Exception:
        pass
    return "http://localhost:11434/v1"


def whisper_device() -> tuple[str, str]:
    """探测 Whisper 运行设备与计算精度，有 CUDA 用 GPU，否则 CPU。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


def embedding_settings() -> dict:
    """解析后的 Embedding 配置，供 agent1 统一消费。

    优先级：EMBEDDING_API_KEY > DASHSCOPE_API_KEY（同一个阿里云账号，
    避免让用户为embedding 再配一份 key）。

    :return: {provider, api_key, base_url, model, dim, batch_size, timeout}
    :raises RuntimeError: 云端模式下两者都未配置时
    """
    s = get_settings()
    provider = (s.embedding_provider or "dashscope").strip().lower()

    if provider == "ollama":
        return {
            "provider": "ollama",
            "api_key": "ollama",
            "base_url": ollama_base_url(),
            "model": s.ollama_embedding_model or "quentinz/bge-large-zh-v1.5",
            "dim": 1024,
            "batch_size": 8,
            "timeout": int(s.embedding_timeout or 120),
        }

    key = (s.embedding_api_key or s.dashscope_api_key or "").strip()
    if not key:
        raise RuntimeError(
            "云端 Embedding 未配置密钥。请在项目根 .env 里设置其一：\n"
            "  EMBEDDING_API_KEY=sk-你的阿里云百炼密钥   （推荐，独立管理）\n"
            "  DASHSCOPE_API_KEY=sk-...              （与 Agent3 共用同一把）\n"
            "若想退回本地 Ollama，把 EMBEDDING_PROVIDER 设为 ollama。"
        )
    return {
        "provider": "dashscope",
        "api_key": key,
        "base_url": s.embedding_base_url
        or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": s.embedding_model or "text-embedding-v4",
        "dim": int(s.embedding_dim or 1024),
        "batch_size": max(1, min(int(s.embedding_batch_size or 10), 10)),
        "timeout": int(s.embedding_timeout or 120),
    }
