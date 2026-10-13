# agent2/config.py
"""Agent2 配置（语音：ASR + TTS + 语音问答）。

配置来源统一为 common.config.get_settings()（项目根 .env，单一配置源）。
本模块只负责路径生成与设备探测。
"""
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.config import get_settings, whisper_device

logger = logging.getLogger("fire_agents.agent2")

_s = get_settings()

# ================== 路径 ==================
BASE_DIR = os.path.join(_ROOT, "agent2")
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 各模块输出子目录
OUT_XFYUN_ONLINE = os.path.join(OUTPUT_DIR, "xfyun_online")
OUT_XFYUN_LFASR = os.path.join(OUTPUT_DIR, "xfyun_lfasr")
OUT_WHISPER = os.path.join(OUTPUT_DIR, "whisper_llm")
OUT_VOSK = os.path.join(OUTPUT_DIR, "vosk")
OUT_TTS = os.path.join(OUTPUT_DIR, "tts_demo")
OUT_VOICE_BOT = os.path.join(OUTPUT_DIR, "voice_bot")

for _d in (INPUT_DIR, OUTPUT_DIR, MODEL_DIR,
           OUT_XFYUN_ONLINE, OUT_XFYUN_LFASR, OUT_WHISPER,
           OUT_VOSK, OUT_TTS, OUT_VOICE_BOT):
    os.makedirs(_d, exist_ok=True)


# ================== 讯飞（在线听写 WebSocket）==================
XFYUN_APP_ID = _s.xfyun_app_id
XFYUN_API_KEY = _s.xfyun_api_key
XFYUN_API_SECRET = _s.xfyun_api_secret

# ================== 讯飞（录音文件转写 LFASR）==================
LFASR_APP_ID = _s.xfyun_app_id
LFASR_SECRET_KEY = _s.lfasr_secret_key
LFASR_HOT_WORDS = _s.lfasr_hot_words

# ================== Whisper（离线 ASR）==================
# 路径留空则使用 faster-whisper 的默认 HuggingFace 缓存
WHISPER_MODEL_SIZE = _s.whisper_model_path or "small"
WHISPER_DOWNLOAD_ROOT = _s.whisper_download_root or os.path.join(MODEL_DIR, "whisper")

WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = whisper_device()

# ================== Vosk（离线 ASR）==================
# 路径留空则由 vosk 自身按 VOSK_MODEL_PATH 环境变量或默认位置解析
VOSK_MODEL_PATH = _s.vosk_model_path

# ================== Ollama（本地 LLM）==================
OLLAMA_URL = _s.ollama_url
OLLAMA_MODEL = _s.ollama_model

# ================== TTS ==================
TTS_VOICE_EDGE = _s.tts_voice_edge
TTS_EMOTION = _s.tts_emotion

# 阿里云 CosyVoice
DASHSCOPE_API_KEY = _s.dashscope_api_key
WORKSPACE_ID = _s.dashscope_workspace_id
REGION = _s.dashscope_region
COSYVOICE_MODEL = _s.cosyvoice_model
CLONED_VOICE_ID = _s.cloned_voice_id


# ================== 启动检查 ==================
def check_env(verbose: bool = False) -> bool:
    """检查关键环境变量是否配置，返回是否全部就绪。"""
    missing = []
    if not XFYUN_APP_ID:
        missing.append("XFYUN_APP_ID")
    if not DASHSCOPE_API_KEY:
        missing.append("DASHSCOPE_API_KEY")
    if missing:
        logger.warning("以下环境变量未设置（相关功能不可用）：%s", ", ".join(missing))
    if verbose:
        logger.info("Whisper 设备：%s (%s)", WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
    return len(missing) == 0


if __name__ == "__main__":
    check_env(verbose=True)
