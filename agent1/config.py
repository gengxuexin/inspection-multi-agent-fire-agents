# agent1/config.py
"""Agent1 配置（消防 RAG + 文书生成）。

配置来源已统一到 common.config.get_settings()（项目根 .env，单一配置源）。
本模块只保留 Agent1 特有的 LLM / Embedding 客户端与工具函数。

- LLM：DeepSeek 官方 API（云端，走系统代理）
- Embedding：云端 API（阿里云 text-embedding-v4，OpenAI 兼容）
            本地 Ollama 仍可通过 EMBEDDING_PROVIDER=ollama 启用
"""
import json
import logging
import os
import re
import sys

from openai import OpenAI

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.config import embedding_settings, get_settings, ollama_base_url

logger = logging.getLogger("fire_agents.agent1")

# ==================== 读取统一配置 ====================
_s = get_settings()

OLLAMA_BASE = ollama_base_url()

# ★ Embedding 配置在首次调用时懒解析（缺 key 时不该import 就崩）
DEFAULT_EMB_MODEL = _s.embedding_model

DEEPSEEK_API_KEY = _s.deepseek_api_key
DEEPSEEK_BASE_URL = _s.deepseek_base_url
DEEPSEEK_MODEL = _s.deepseek_model        # 或 deepseek-reasoner

# ★ provider_mode：cloud（默认，纯云端）| local（全本地 Ollama）| hybrid（云端优先，失败降本地）
#
# 落地背景：消防执法/政务场景通常是内网或涉密环境，云端 API 根本不可达。
# 当前版本（2026-10）默认仍走 cloud，因为开发调试方便；
# 但代码必须保证切到 local 时不需要改任何业务逻辑。
PROVIDER_MODE = (_s.provider_mode or "cloud").lower()


# ==================== 本地 Ollama LLM 客户端 ====================
_local_llm = None
_local_llm_checked = False


def get_local_llm():
    """获取本地 Ollama 的 OpenAI 兼容客户端；不可用时返回 None。

    ★ 与历史兼容逻辑的区别
    ----------------------
    下方 `set_deepseek` 有一段兼容代码：历史调用如果传了 ollama 相关入参，
    会被**强制改道回 DeepSeek**。那是为了防止老代码误连本地。
    但它同时意味着「本地模型路径被完全堵死」——
    provider_mode=local 时根本走不到 Ollama。

    所以这里单独提供一个真正的本地 client，
    由 `common/llm_guard.py` 根据 provider_mode 决定用哪个。

    不抛异常，只返回 None —— 调用方（llm_guard）会据此决定是否降级。
    """
    global _local_llm, _local_llm_checked
    if _local_llm_checked:
        return _local_llm
    _local_llm_checked = True

    try:
        client = OpenAI(
            api_key="ollama",              # Ollama 不校验 key，占位即可
            base_url=OLLAMA_BASE,
            timeout=float(getattr(_s, "llm_request_timeout", 30.0)),
        )
        # 探活：列模型，确认服务在、模型在
        models = client.models.list()
        available = {getattr(m, "id", "") for m in (models.data or [])}
        want = _s.ollama_model or "qwen2.5:7b"
        if want not in available:
            logger.warning(
                "[agent1] 本地 Ollama 已连通但缺少模型 %s（现有: %s）",
                want, ", ".join(sorted(available)) or "无")
            _local_llm = None
            return None
        logger.info("[agent1] 本地 Ollama 就绪：%s @ %s", want, OLLAMA_BASE)
        _local_llm = client
    except Exception as e:  # noqa: BLE001 — 探活失败就是「不可用」，不该崩
        logger.info("[agent1] 本地 Ollama 不可用（%s），将走云端或降级路径", e)
        _local_llm = None
    return _local_llm


def reset_local_llm_probe() -> None:
    """测试用：重置本地探活缓存。"""
    global _local_llm, _local_llm_checked
    _local_llm = None
    _local_llm_checked = False


# ==================== DeepSeek 官方 API 配置（LLM 用）====================
_ds_client = None
_ds_config: dict = {}
_emb_config: dict = {}


def set_deepseek(api_key=None, base_url=None, model=None):
    """初始化 DeepSeek 客户端（幂等，可重复调用以切换配置）。"""
    global _ds_client, _ds_config
    api_key = api_key or DEEPSEEK_API_KEY
    base_url = base_url or DEEPSEEK_BASE_URL
    model = model or DEEPSEEK_MODEL

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未设置。请在项目根 .env 里配置：\n"
            "  DEEPSEEK_API_KEY=sk-你的密钥"
        )

    # ★ DeepSeek 走外网，不能设 trust_env=False（那样会绕开系统代理）
    _ds_client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)
    _ds_config = {"api_key": api_key, "base_url": base_url, "model": model}


def get_deepseek():
    if _ds_client is None:
        set_deepseek()
    return _ds_client


def get_deepseek_config():
    if not _ds_config:
        set_deepseek()
    return _ds_config


def get_llm_model():
    return get_deepseek_config()["model"]


# ==================== Embedding 客户端 ====================
def set_embedding(api_key=None, base_url=None, model=None):
    """初始化 Embedding 配置（幂等）。默认走common.config 的解析结果。"""
    global _emb_config
    cfg = embedding_settings()
    if api_key or base_url or model:
        logger.warning(
            "[config] set_embedding 显式入参覆盖已生效。"
            "建议改用 .env 的 EMBEDDING_* 配置，避免多处设置不一致。"
        )
    _emb_config = {
        "api_key": api_key or cfg["api_key"],
        "base_url": base_url or cfg["base_url"],
        "model": model or cfg["model"],
        "provider": cfg["provider"],
        "dim": cfg["dim"],
        "batch_size": cfg["batch_size"],
        "timeout": cfg["timeout"],
    }


def get_embedding_config():
    if not _emb_config:
        set_embedding()
    return _emb_config


def get_embedding_dim() -> int:
    """向量维度。⚠️ 换模型时必须核对，维度与既有 FAISS 索引不一致会检索错乱。"""
    return int(get_embedding_config().get("dim") or 1024)


# ==================== JSON 容错解析 ====================
def parse_json_safe(text):
    """从 LLM 输出中稳健提取 JSON 对象。

    处理三类脏输出：thinking 标签混入、```json 代码块围栏、前后缀噪声。
    """
    if not text:
        return {}
    text = re.sub(r" thinking.*? response", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    pass
    return {}


# ==================== LLM 调用（走 DeepSeek 官方 API）====================
def chat_json(messages, temperature=0.1, model=None, timeout=600):
    """要求模型返回 JSON 对象；模型不支持 json_object 时自动回退普通调用。"""
    cfg = get_deepseek_config()
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=timeout)
    kwargs = {
        "model": model or cfg["model"],
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
    }
    try:
        resp = client.chat.completions.create(
            **kwargs, response_format={"type": "json_object"}
        )
    except Exception:
        logger.debug("[chat_json] json_object 模式不被支持，回退普通调用")
        resp = client.chat.completions.create(**kwargs)
    return parse_json_safe(resp.choices[0].message.content)


def chat_text(messages, temperature=0.3, model=None, timeout=600):
    cfg = get_deepseek_config()
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=timeout)
    resp = client.chat.completions.create(
        model=model or cfg["model"],
        messages=messages,
        temperature=temperature,
        timeout=timeout,
    )
    return resp.choices[0].message.content.strip()

# ==================== ds_client 兼容层 ====================
# 多数文件写死 model="deepseek-chat"，这里拦截并统一到实际配置的模型。


class _PatchedCompletions:
    def __init__(self, real, cfg):
        self._real = real
        self._cfg = cfg

    def create(self, **kwargs):
        # 只替换那些明显是"占位符"的模型名
        m = kwargs.get("model", "")
        if m in ("deepseek-chat", "deepseek-reasoner", "gpt-3.5-turbo", "gpt-4"):
            kwargs["model"] = self._cfg["model"]
        return self._real.create(**kwargs)


class _PatchedChat:
    def __init__(self, real, cfg):
        self._real = real
        self._cfg = cfg
        self.completions = _PatchedCompletions(real.completions, cfg)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _LazyDSClient:
    """延迟初始化代理：首次访问属性时才真正建连，避免导入期失败。"""

    def __getattr__(self, name):
        real = get_deepseek()
        cfg = get_deepseek_config()
        if name == "chat":
            return _PatchedChat(real.chat, cfg)
        return getattr(real, name)


ds_client = _LazyDSClient()


# ==================== 兼容旧调用签名 ====================
# 历史代码存在 set_deepseek("ollama", "...", "qwen2.5:7b") 这类调用，
# 统一拦截指向本地 Ollama 的入参，强制走 DeepSeek。
_original_set_deepseek = set_deepseek


def set_deepseek(api_key=None, base_url=None, model=None):   # noqa: F811
    if api_key and "ollama" in str(api_key).lower():
        return _original_set_deepseek()
    if base_url and "11434" in str(base_url):
        return _original_set_deepseek()
    return _original_set_deepseek(api_key, base_url, model)
