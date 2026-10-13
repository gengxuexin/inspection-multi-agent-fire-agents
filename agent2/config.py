# -*- coding: utf-8 -*-
"""全局配置：所有密钥从环境变量读取，路径按项目结构生成"""
import os
import sys
import logging
logger = logging.getLogger("fire_agents.agent2")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass
# ================== 路径 ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

for d in [INPUT_DIR, OUTPUT_DIR, MODEL_DIR,
          OUT_XFYUN_ONLINE, OUT_XFYUN_LFASR, OUT_WHISPER,
          OUT_VOSK, OUT_TTS, OUT_VOICE_BOT]:
    os.makedirs(d, exist_ok=True)


# ================== 讯飞（在线听写 WebSocket）==================
XFYUN_APP_ID = os.getenv("XFYUN_APP_ID", "")
XFYUN_API_KEY = os.getenv("XFYUN_API_KEY", "")
XFYUN_API_SECRET = os.getenv("XFYUN_API_SECRET", "")

# ================== 讯飞（录音文件转写 LFASR）==================
LFASR_APP_ID = os.getenv("LFASR_APP_ID", XFYUN_APP_ID)
LFASR_SECRET_KEY = os.getenv("LFASR_SECRET_KEY", "")
LFASR_HOT_WORDS = os.getenv("LFASR_HOT_WORDS", "")

# ================== Whisper（离线 ASR）==================
WHISPER_MODEL_SIZE = os.getenv(
    "WHISPER_MODEL_PATH",
    "/home/geng/fire_agents/agent2/models/models/pengzhendong--faster-whisper-small/snapshots/master"
)
WHISPER_DOWNLOAD_ROOT = os.path.join(MODEL_DIR, "whisper")

# 自动检测 GPU：有 CUDA 就用，没有就 CPU
def _detect_whisper_device():
    """自动检测 Whisper 运行设备，优先使用 CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"

WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = _detect_whisper_device()

# ================== Vosk（离线 ASR）==================
VOSK_MODEL_PATH = os.getenv(
    "VOSK_MODEL_PATH",
    r"D:\Docker\vosk\vosk-model-cn-0.22"
)

# ================== Ollama（本地 LLM）==================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# ================== TTS ==================
TTS_VOICE_EDGE = os.getenv("TTS_VOICE_EDGE", "zh-CN-XiaoxiaoNeural")
TTS_EMOTION = os.getenv("TTS_EMOTION", "正常")

# 阿里云 CosyVoice
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
WORKSPACE_ID = os.getenv("DASHSCOPE_WORKSPACE_ID", "")
REGION = os.getenv("DASHSCOPE_REGION", "beijing")
COSYVOICE_MODEL = os.getenv("COSYVOICE_MODEL", "cosyvoice-v2")
CLONED_VOICE_ID = os.getenv("CLONED_VOICE_ID", "")

# ================== 启动检查 ==================
def check_env(verbose=False):
    """检查关键环境变量是否配置"""
    missing = []
    if not XFYUN_APP_ID:
        missing.append("XFYUN_APP_ID")
    if not DASHSCOPE_API_KEY:
        missing.append("DASHSCOPE_API_KEY")
    if missing:
        logger.info(f"⚠️ 以下环境变量未设置（相关功能将不可用）：{', '.join(missing)}")
    if verbose:
        logger.info(f"✅ Whisper 设备：{WHISPER_DEVICE} ({WHISPER_COMPUTE_TYPE})")
    return len(missing) == 0


if __name__ == "__main__":
    check_env(verbose=True)