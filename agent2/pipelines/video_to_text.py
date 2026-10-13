# -*- coding: utf-8 -*-
"""
logger = logging.getLogger("fire_agents.agent2")

2.1-2.3 视频 → 音频 → 识别 → 语义梳理 → 关键词提取
支持三种 ASR 引擎：xfyun_lfasr / whisper / vosk
"""
import os
import json
from utils.audio_extract import extract_audio_from_video
from utils.fire_keywords import FireKeywordExtractor
from llm.ollama_client import refine_text_with_llm
import logging


def process_video(video_path, engine="whisper", output_dir=None, refine=True):
    """
    处理单个视频
    engine: "xfyun_lfasr" | "whisper" | "vosk"
    refine: 是否用大模型语义梳理
    """
    video_path = os.path.abspath(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(video_path), "output")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    audio_path = os.path.join(output_dir, f"{base_name}.wav")
    text_path = os.path.join(output_dir, f"{base_name}_识别文本.txt")
    refined_path = os.path.join(output_dir, f"{base_name}_梳理文本.txt")
    keywords_path = os.path.join(output_dir, f"{base_name}_消防关键信息.json")

    logger.info(f"\n{'=' * 70}\n📹 处理视频：{video_path}\n引擎：{engine}\n{'=' * 70}")

    # 1. 提取音频
    wav_path = extract_audio_from_video(video_path, audio_path)
    if not wav_path:
        return None

    # 2. 识别
    if engine == "xfyun_lfasr":
        from asr.xfyun_lfasr import LFASRClient
        client = LFASRClient()
        result = client.recognize(wav_path)
        raw_text = result["text"] if result else ""
    elif engine == "whisper":
        from asr.whisper_engine import WhisperEngine
        engine_inst = WhisperEngine()
        raw_text = engine_inst.transcribe(wav_path)
    elif engine == "vosk":
        from asr.vosk_engine import VoskEngine
        engine_inst = VoskEngine()
        raw_text = engine_inst.transcribe(wav_path)
    else:
        raise ValueError(f"未知引擎：{engine}")

    if not raw_text:
        logger.info("⚠️ 识别结果为空")
        return None

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(raw_text)
    logger.info(f"💾 识别文本：{text_path}")

    # 3. 语义梳理
    final_text = refine_text_with_llm(raw_text) if refine else raw_text
    with open(refined_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    logger.info(f"💾 梳理文本：{refined_path}")

    # 4. 关键词提取
    extractor = FireKeywordExtractor()
    keywords = extractor.extract(final_text)
    with open(keywords_path, "w", encoding="utf-8") as f:
        json.dump(keywords, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 关键信息：{keywords_path}")

    extractor.print_summary(keywords)
    return keywords