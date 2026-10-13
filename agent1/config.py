# config.py
"""
Agent1 配置
- LLM：走 DeepSeek 官方 API（快，云端）
- Embedding：走本地 Ollama（索引依赖，不能变）
"""
import os
import json
import re
import subprocess
import httpx
from openai import OpenAI

# ==================== 加载 .env ====================
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        with open(_env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


_ds_client = None
_ds_config = {}
_emb_config = {}


# ==================== Ollama 地址检测（Embedding 用）====================
def _detect_ollama_base():
    env = os.getenv("OLLAMA_BASE_URL")
    if env:
        return env
    if os.name == "nt":
        return "http://localhost:11434/v1"
    try:
        ip = subprocess.check_output(
            "ip route show default | awk '{print $3}'", shell=True
        ).decode().strip()
        return f"http://{ip}:11434/v1"
    except Exception:
        return "http://172.30.32.1:11434/v1"


OLLAMA_BASE = _detect_ollama_base()
DEFAULT_EMB_MODEL = "quentinz/bge-large-zh-v1.5"   # ★ 必须和建库时一致


# ==================== DeepSeek 官方 API 配置（LLM 用）====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")   # 或 deepseek-reasoner


# ==================== LLM 客户端（DeepSeek）====================
def set_deepseek(api_key=None, base_url=None, model=None):
    global _ds_client, _ds_config
    api_key = api_key or DEEPSEEK_API_KEY
    base_url = base_url or DEEPSEEK_BASE_URL
    model = model or DEEPSEEK_MODEL

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未设置。请在 agent1/.env 里配置：\n"
            "  DEEPSEEK_API_KEY=sk-你的密钥"
        )

    # ★ DeepSeek 走外网，不能用 trust_env=False（那样会绕开系统代理）
    _ds_client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=300,
    )
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


# ==================== Embedding 客户端（Ollama）====================
def set_embedding(api_key="ollama", base_url=None, model=None):
    global _emb_config
    base_url = base_url or OLLAMA_BASE
    model = model or DEFAULT_EMB_MODEL
    _emb_config = {"api_key": "ollama", "base_url": base_url, "model": model}


def get_embedding_config():
    if not _emb_config:
        set_embedding()
    return _emb_config


# ==================== JSON 容错解析 ====================
def parse_json_safe(text):
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
        # 某些模型不支持 json_object 模式，回退到普通调用
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
# 很多旧文件里写死了 model="deepseek-chat"，这里拦截并统一
# 注意：DeepSeek 官方模型的默认名就是 deepseek-chat，一般不用替换

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
    def __getattr__(self, name):
        real = get_deepseek()
        cfg = get_deepseek_config()
        if name == "chat":
            return _PatchedChat(real.chat, cfg)
        return getattr(real, name)


ds_client = _LazyDSClient()


# ==================== 兼容旧函数名 ====================
# 有些文件可能还调用 set_deepseek("ollama", "...", "qwen2.5:7b")
# 这里做个保护：如果传的是 "ollama" 或本地地址，就忽略，统一走 DeepSeek
_original_set_deepseek = set_deepseek

def set_deepseek(api_key=None, base_url=None, model=None):   # noqa: F811
    # 忽略任何指向 ollama 的调用
    if api_key and "ollama" in str(api_key).lower():
        return _original_set_deepseek()
    if base_url and "11434" in str(base_url):
        return _original_set_deepseek()
    return _original_set_deepseek(api_key, base_url, model)