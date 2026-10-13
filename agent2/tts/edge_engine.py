# -*- coding: utf-8 -*-
"""edge-tts 语音合成封装（支持同步/异步、Gradio 兼容）"""
import asyncio
import os
import edge_tts
from config import OUT_TTS
import logging


logger = logging.getLogger("fire_agents.agent2")

RECOMMENDED_VOICES = {
    "晓晓（女声-温柔）":   "zh-CN-XiaoxiaoNeural",
    "晓伊（女声-活泼）":   "zh-CN-XiaoyiNeural",
    "云希（男声-沉稳）":   "zh-CN-YunxiNeural",
    "云健（男声-新闻）":   "zh-CN-YunjianNeural",
    "云扬（男声-大气）":   "zh-CN-YunyangNeural",
    "辽宁小北（东北女声）": "zh-CN-liaoning-XiaobeiNeural",
    "陕西小妮（陕西方言）": "zh-CN-shaanxi-XiaoniNeural",
}

EMOTION_PRESETS = {
    "正常":   {"rate": "+0%",  "pitch": "+0Hz",  "volume": "+0%"},
    "温和":   {"rate": "-10%", "pitch": "-2Hz",  "volume": "+0%"},
    "严肃":   {"rate": "-5%",  "pitch": "-5Hz",  "volume": "+10%"},
    "急促":   {"rate": "+30%", "pitch": "+5Hz",  "volume": "+20%"},
    "温柔":   {"rate": "-15%", "pitch": "+0Hz",  "volume": "-10%"},
    "兴奋":   {"rate": "+20%", "pitch": "+10Hz", "volume": "+20%"},
    "悲伤":   {"rate": "-20%", "pitch": "-10Hz", "volume": "-20%"},
}


async def _list_voices_async(locale_prefix="zh"):
    voices = await edge_tts.list_voices()
    return [v for v in voices if v["Locale"].lower().startswith(locale_prefix)]


def list_chinese_voices():
    voices = asyncio.run(_list_voices_async("zh"))
    logger.info(f"📢 找到 {len(voices)} 个中文音色：\n")
    for v in voices:
        logger.info(f"  {v['ShortName']:45s} | {v['Gender']:6s} | {v['Locale']}")
    return voices


async def synthesize_async(text, output_path,
                           voice="zh-CN-XiaoxiaoNeural",
                           rate="+0%", pitch="+0Hz", volume="+0%"):
    """异步合成（Gradio 里用这个）"""
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    communicate = edge_tts.Communicate(
        text=text, voice=voice, rate=rate, pitch=pitch, volume=volume,
    )
    await communicate.save(output_path)
    logger.info(f"✅ 语音已保存：{output_path}")
    return output_path


def synthesize(text, output_path, voice="zh-CN-XiaoxiaoNeural",
               emotion="正常", rate=None, pitch=None, volume=None):
    """同步合成（命令行脚本里用这个）"""
    preset = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS["正常"])
    rate = rate if rate is not None else preset["rate"]
    pitch = pitch if pitch is not None else preset["pitch"]
    volume = volume if volume is not None else preset["volume"]
    logger.info("合成：voice=%s, emotion=%s, rate=%s, pitch=%s, volume=%s",
                 voice, emotion, rate, pitch, volume)
    asyncio.run(synthesize_async(
        text, output_path, voice, rate, pitch, volume
    ))
    return output_path


class VoiceProfile:
    """音色档案：为不同说话人保存音色和参数配置"""

    def __init__(self):
        self.profiles = {
            "值班员": {"voice": "zh-CN-YunxiNeural", "emotion": "严肃",
                       "rate": "+0%", "pitch": "-3Hz", "volume": "+10%"},
            "接警员": {"voice": "zh-CN-XiaoxiaoNeural", "emotion": "温和",
                       "rate": "+0%", "pitch": "+0Hz", "volume": "+0%"},
            "报警人": {"voice": "zh-CN-YunjianNeural", "emotion": "急促",
                       "rate": "+20%", "pitch": "+5Hz", "volume": "+15%"},
        }

    def add_profile(self, name, voice, emotion="正常",
                    rate=None, pitch=None, volume=None):
        preset = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS["正常"])
        self.profiles[name] = {
            "voice": voice, "emotion": emotion,
            "rate":   rate   if rate   is not None else preset["rate"],
            "pitch":  pitch  if pitch  is not None else preset["pitch"],
            "volume": volume if volume is not None else preset["volume"],
        }
        logger.info(f"✅ 已添加音色档案：{name}")

    def speak(self, speaker_name, text, output_path):
        if speaker_name not in self.profiles:
            logger.info(f"⚠️ 未找到档案 {speaker_name}，使用默认音色")
            return synthesize(text, output_path)
        p = self.profiles[speaker_name]
        return synthesize(text, output_path, voice=p["voice"],
                          rate=p["rate"], pitch=p["pitch"], volume=p["volume"])


if __name__ == "__main__":
    os.makedirs(OUT_TTS, exist_ok=True)
    list_chinese_voices()

    test_text = "你好，我是消防语音助手，请注意值班室必须二十四小时在岗。"
    synthesize(test_text, os.path.join(OUT_TTS, "1_晓晓女声.mp3"))
    synthesize(test_text, os.path.join(OUT_TTS, "2_云希男声.mp3"),
               voice="zh-CN-YunxiNeural")

    fire_text = "着火了！三楼着火，有两个人被困，请立即派消防车！"
    synthesize(fire_text, os.path.join(OUT_TTS, "4_紧急急促.mp3"),
               emotion="急促")
    synthesize(fire_text, os.path.join(OUT_TTS, "5_悲伤.mp3"),
               emotion="悲伤")

    vp = VoiceProfile()
    vp.speak("接警员", "你好，这里是119，请问您的位置在哪里？",
             os.path.join(OUT_TTS, "7_接警员.mp3"))
    vp.speak("报警人", "我们这里三楼着火了，有两个人被困！",
             os.path.join(OUT_TTS, "8_报警人.mp3"))

    logger.info(f"\n🎉 测试音频已生成到：{OUT_TTS}")