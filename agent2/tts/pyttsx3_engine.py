# -*- coding: utf-8 -*-
"""pyttsx3 语音合成（完全离线，系统引擎）"""
import os
import pyttsx3
from config import OUT_TTS
import logging


logger = logging.getLogger("fire_agents.agent2")

def text_to_speech_pyttsx3(text, output_path=None, rate=150, volume=0.9):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)

    voices = engine.getProperty('voices')
    for voice in voices:
        name_l = voice.name.lower()
        id_l = voice.id.lower()
        if 'chinese' in name_l or 'zh' in id_l or 'huihui' in id_l:
            engine.setProperty('voice', voice.id)
            break

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        logger.info(f"✅ 语音已保存至：{output_path}")
    else:
        engine.say(text)
        engine.runAndWait()
    engine.stop()


if __name__ == "__main__":
    os.makedirs(OUT_TTS, exist_ok=True)
    text_to_speech_pyttsx3(
        "消防值班室，请立即启动应急预案",
        output_path=os.path.join(OUT_TTS, "pyttsx3_output.wav"),
    )