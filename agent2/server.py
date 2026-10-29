# ~/fire_agents/agent2/server.py
import os
import sys
import time
import shutil
import tempfile
import traceback

# ★ import 路径要包含项目根，而不只是 agent2 自身目录。
#
# 原来只 append 了 `os.path.dirname(__file__)`（即 /app/agent2），
# 于是`from common.tracing import ...` 这类跨包 import 在容器里会失败：
#     ModuleNotFoundError: No module named 'common'
# （common 包在 /app/common，而 WORKDIR 是 /app/agent2）
#
# 本地开发时看不出来，因为 CWD 是项目根，Python 会自动把 CWD 放进 sys.path。
# 只有按容器的形态跑才暴露 —— 所以 preflight 里加了
# check_container_imports.py专门查这一项。
_SVC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SVC_DIR)
for _p in (_PROJECT_ROOT, _SVC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Agent2 语音服务")

# ============ Whisper 懒加载（首次调用才加载，避免启动慢） ============
_whisper = None

def get_whisper():
    global _whisper
    if _whisper is None:
        from asr.whisper_engine import WhisperEngine
        _whisper = WhisperEngine()
    return _whisper


# ============ 请求模型 ============
class TTSReq(BaseModel):
    text: str
    voice: str = "zh-CN-XiaoxiaoNeural"
    emotion: str = "正常"

class QAReq(BaseModel):
    audio_path: str
    voice: str = "zh-CN-XiaoxiaoNeural"
    emotion: str = "正常"


# ============ 接口 ============
@app.get("/")
def root():
    return {"service": "Agent2 语音服务", "status": "ok"}


@app.get("/health")
def health():
    """存活探针：只回答「进程还在不在」。

    刻意不加载 Whisper 模型、不探 ffmpeg、不连 TTS 服务。
    agent2 的模型有几百 MB，真在探针里加载会让每次健康检查都变成一次
    重量级操作；而且模型加载失败属于「就绪失败」，该由 /ready 报，不是重启能解决的。
    """
    return {"service": "Agent2 语音服务", "status": "ok"}


@app.get("/ready")
def ready():
    """就绪探针：检查语音链路真正需要的外部条件。

    这里只做廉价的静态检查（ffmpeg 是否在 PATH），不做模型加载 ——
    探针必须轻量，否则每次探测都会给系统带来额外负担。
    """
    import shutil as _shutil
    ffmpeg_ok = _shutil.which("ffmpeg") is not None
    return {"service": "Agent2 语音服务",
            "status": "ok" if ffmpeg_ok else "degraded",
            "checks": {"ffmpeg": "ok" if ffmpeg_ok else "missing"}}


@app.post("/api/voice/transcribe")
async def transcribe(file: UploadFile = File(...), engine: str = "whisper"):
    """上传音频/视频，返回识别文本"""
    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_path = os.path.join(tmp_dir, file.filename)
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 视频 → 先抽音频
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"):
            from utils.audio_extract import extract_audio_from_video
            wav_path = os.path.join(tmp_dir, "extracted.wav")
            audio_path = extract_audio_from_video(tmp_path, wav_path)
            if not audio_path:
                return {"status": "error", "message": "音频提取失败"}
        else:
            audio_path = tmp_path

        # 识别
        if engine == "whisper":
            text = get_whisper().transcribe(audio_path)
        elif engine == "vosk":
            from asr.vosk_engine import VoskEngine
            text = VoskEngine().transcribe(audio_path)
        else:
            return {"status": "error", "message": f"不支持的引擎：{engine}"}

        return {"status": "success", "text": text}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/voice/tts")
def tts(req: TTSReq):
    """文字转语音，返回 mp3 文件路径"""
    try:
        from tts.edge_engine import synthesize
        from config import OUT_TTS
        out = os.path.join(OUT_TTS, f"tts_{int(time.time())}.mp3")
        synthesize(req.text, out, voice=req.voice, emotion=req.emotion)
        return {"status": "success", "audio_path": out}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/api/voice/qa")
def voice_qa(req: QAReq):
    """语音问答：ASR → LLM → TTS，给定本地音频路径"""
    try:
        if not os.path.exists(req.audio_path):
            return {"status": "error", "message": f"音频不存在：{req.audio_path}"}

        # 1) ASR
        user_text = get_whisper().transcribe(req.audio_path)
        if not user_text:
            return {"status": "error", "message": "未识别到语音内容"}

        # 2) LLM
        from llm.ollama_client import ask_fire_bot
        answer_text = ask_fire_bot(user_text, [])

        # 3) TTS
        from tts.edge_engine import synthesize
        from config import OUT_VOICE_BOT
        out = os.path.join(OUT_VOICE_BOT, f"qa_{int(time.time())}.mp3")
        synthesize(answer_text, out, voice=req.voice, emotion=req.emotion)

        return {
            "status": "success",
            "user_text": user_text,
            "answer_text": answer_text,
            "audio_path": out,
        }
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)