# -*- coding: utf-8 -*-
"""Vosk 离线语音识别引擎（并发安全版）"""
import os
import json
import wave
from vosk import Model, KaldiRecognizer, SetLogLevel
from config import VOSK_MODEL_PATH
import logging


logger = logging.getLogger("fire_agents.agent2")

class VoskEngine:
    """Vosk 识别引擎（模型单例，识别器每次新建 → 线程安全）"""

    _instance = None
    _model = None
    _lock = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if VoskEngine._model is None:
            import threading
            VoskEngine._lock = threading.Lock()
            SetLogLevel(0)
            if not os.path.exists(VOSK_MODEL_PATH):
                raise FileNotFoundError(
                    f"❌ Vosk 模型路径不存在：{VOSK_MODEL_PATH}"
                )
            logger.info(f"⏳ 正在加载 Vosk 模型：{VOSK_MODEL_PATH}")
            VoskEngine._model = Model(VOSK_MODEL_PATH)
            logger.info("✅ Vosk 模型加载完成")

    def transcribe(self, audio_path):
        """识别音频，返回文本（每次新建 KaldiRecognizer，线程安全）"""
        if not audio_path or not os.path.exists(audio_path):
            logger.info(f"❌ 找不到音频：{audio_path}")
            return ""

        logger.info(f"🎙️ [Vosk] 识别：{audio_path}")
        wf = wave.open(audio_path, "rb")

        if (wf.getnchannels() != 1
                or wf.getsampwidth() != 2
                or wf.getcomptype() != "NONE"):
            logger.info("⚠️ 音频格式需为 16kHz/16bit/单声道 WAV")
            wf.close()
            return ""

        # ★ 每次调用都创建新识别器，避免并发问题
        rec = KaldiRecognizer(VoskEngine._model, wf.getframerate())
        rec.SetWords(True)

        full_text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result["text"]:
                    full_text += result["text"]

        final_result = json.loads(rec.FinalResult())
        if final_result["text"]:
            full_text += final_result["text"]

        wf.close()
        logger.info(f"📝 [Vosk] 识别结果：{full_text}")
        return full_text.strip()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        logger.info("用法：python -m asr.vosk_engine <音频路径>")
        sys.exit(1)
    engine = VoskEngine()
    logger.info(engine.transcribe(sys.argv[1]))