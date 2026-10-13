# -*- coding: utf-8 -*-
"""Whisper 离线语音识别引擎（自动使用 CUDA 加速）"""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from faster_whisper import WhisperModel
from config import (
    WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    WHISPER_DOWNLOAD_ROOT,
)


class WhisperEngine:
    """Whisper 识别引擎（单例加载，可复用）"""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if WhisperEngine._model is None:
            print(f"⏳ 正在加载 Whisper [{WHISPER_MODEL_SIZE}] "
                  f"设备={WHISPER_DEVICE} 精度={WHISPER_COMPUTE_TYPE}...")
            try:
                WhisperEngine._model = WhisperModel(
                WHISPER_MODEL_SIZE,  # 现在是本地路径
                device="cpu",
                compute_type="int8",
            )
                print("✅ Whisper 模型加载完成")
            except Exception as e:
                # GPU 加载失败 → 降级到 CPU
                if WHISPER_DEVICE == "cuda":
                    print(f"⚠️ CUDA 加载失败（{e}），降级到 CPU")
                    WhisperEngine._model = WhisperModel(
                        WHISPER_MODEL_SIZE,
                        device="cpu",
                        compute_type="int8",
                        download_root=WHISPER_DOWNLOAD_ROOT,
                    )
                else:
                    raise

    def transcribe(self, audio_path):
        """识别音频，返回连贯的文本"""
        if not audio_path or not os.path.exists(audio_path):
            print(f"❌ 找不到音频：{audio_path}")
            return ""

        print(f"🎙️ [Whisper] 识别：{audio_path}")
        initial_prompt = (
            "这是一段消防相关的语音，可能包含火情、地址、人员被困、"
            "值班、持证上岗、灭火器、消火栓等消防术语。"
        )

        segments, info = WhisperEngine._model.transcribe(
            audio_path,
            beam_size=5,
            language="zh",
            initial_prompt=initial_prompt,
            vad_filter=True,
            condition_on_previous_text=True,   # ★ 提升长音频连贯性
        )

        # 拼接时用句号分隔，避免粘连
        parts = []
        for seg in segments:
            t = seg.text.strip()
            if t:
                parts.append(t)
        text = "，".join(parts) if parts else ""
        # 连续逗号合并
        import re
        text = re.sub(r"[，,]{2,}", "，", text)

        print(f"📝 [Whisper] 识别结果：{text[:200]}"
              f"{'...' if len(text) > 200 else ''}")
        return text.strip()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法：python -m asr.whisper_engine <音频路径>")
        sys.exit(1)
    engine = WhisperEngine()
    print(engine.transcribe(sys.argv[1]))