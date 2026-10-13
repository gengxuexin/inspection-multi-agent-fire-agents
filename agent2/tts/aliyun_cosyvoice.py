# -*- coding: utf-8 -*-
"""阿里云 CosyVoice：音色提取 / 音色克隆 / 语音合成 / 情感润色"""
import os
import requests
import dashscope
from dashscope.audio.tts_v2 import (
    SpeechSynthesizer, VoiceEnrollmentService, AudioFormat,
)

from config import (
    DASHSCOPE_API_KEY, WORKSPACE_ID, REGION,
    COSYVOICE_MODEL, CLONED_VOICE_ID, OUT_TTS,
)
import logging


logger = logging.getLogger("fire_agents.agent2")
if not DASHSCOPE_API_KEY:
    raise RuntimeError(
        "未设置 DASHSCOPE_API_KEY 环境变量，CosyVoice 不可用"
    )

# 端点配置
if REGION == "beijing":
    dashscope.base_websocket_api_url = (
        f"wss://{WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference"
    )
    HTTP_BASE = f"https://{WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/api/v1"
elif REGION == "singapore":
    dashscope.base_websocket_api_url = (
        f"wss://{WORKSPACE_ID}.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/inference"
    )
    HTTP_BASE = f"https://{WORKSPACE_ID}.ap-southeast-1.maas.aliyuncs.com/api/v1"
else:
    raise ValueError("REGION 只能是 'beijing' 或 'singapore'")

dashscope.api_key = DASHSCOPE_API_KEY


BUILTIN_VOICES = {
    "cosyvoice-v2": [
        {"name": "龙小淳", "voice": "longxiaochun_v2",  "gender": "女", "desc": "温暖亲和"},
        {"name": "龙小夏", "voice": "longxiaoxia_v2",   "gender": "女", "desc": "沉稳女声"},
        {"name": "龙小诚", "voice": "longxiaocheng_v2", "gender": "男", "desc": "磁性男声"},
        {"name": "龙小白", "voice": "longxiaobai_v2",   "gender": "男", "desc": "成熟男声"},
        {"name": "龙老铁", "voice": "longlaotie_v2",    "gender": "男", "desc": "东北直率男"},
        {"name": "龙婉",   "voice": "longwan_v2",       "gender": "女", "desc": "温柔女声"},
        {"name": "龙橙",   "voice": "longcheng_v2",     "gender": "女", "desc": "活泼女声"},
        {"name": "龙可",   "voice": "longke_v2",        "gender": "女", "desc": "懵懂乖乖女"},
    ],
    "cosyvoice-v3-flash": [
        {"name": "龙安洋", "voice": "longanyang",    "gender": "男",   "desc": "阳光大男孩"},
        {"name": "龙安欢", "voice": "longanhuan",    "gender": "女",   "desc": "欢脱元气女"},
        {"name": "龙呼呼", "voice": "longhuhu_v3",   "gender": "女童", "desc": "天真烂漫"},
        {"name": "龙泡泡", "voice": "longpaopao_v3", "gender": "男童", "desc": "泡泡音"},
    ],
}


def list_builtin_voices(model=None):
    if model and model in BUILTIN_VOICES:
        voices = BUILTIN_VOICES[model]
    else:
        voices = [v for vs in BUILTIN_VOICES.values() for v in vs]
    logger.info(f"📢 共 {len(voices)} 个预置音色：")
    for v in voices:
        logger.info("   %-8s | %-25s | %-4s | %s",
                     v["name"], v["voice"], v["gender"], v["desc"])
    return voices


def list_cloned_voices(prefix=""):
    url = f"{HTTP_BASE}/services/audio/tts/customization"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "voice-enrollment",
        "input": {"action": "list_voice", "prefix": prefix,
                  "page_index": 0, "page_size": 10},
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        result = resp.json()
        voices = result.get("output", {}).get("voices", [])
        logger.info(f"📢 已克隆音色 {len(voices)} 个：")
        for v in voices:
            logger.info(f"   {v.get('voice_id')} | 状态: {v.get('status')}")
        return voices
    except Exception as e:
        logger.error(f"❌ 查询异常：{e}")
        return []


def clone_voice(audio_url, prefix, target_model=None):
    """提交声音复刻任务"""
    target_model = target_model or COSYVOICE_MODEL
    service = VoiceEnrollmentService(api_key=DASHSCOPE_API_KEY)
    try:
        voice = service.create_voice(
            target_model=target_model, prefix=prefix, url=audio_url,
        )
        voice_id = voice.get_voice_id()
        logger.info(f"✅ 音色克隆成功！voice_id = {voice_id}")
        return voice_id
    except Exception as e:
        logger.error(f"❌ 音色克隆失败：{e}")
        return None


def text_to_speech(text, output_path, voice=None, model=None):
    model = model or COSYVOICE_MODEL
    voice = voice or (CLONED_VOICE_ID or "longxiaochun_v2")
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    logger.info(f"🎙️ [CosyVoice] 合成：model={model}, voice={voice}")
    synthesizer = SpeechSynthesizer(
        model=model, voice=voice,
        format=AudioFormat.MP3_22050HZ_MONO_256KBPS,
    )
    try:
        audio_data = synthesizer.call(text)
        with open(output_path, "wb") as f:
            f.write(audio_data)
        logger.info(f"✅ 语音已保存：{output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ 合成失败：{e}")
        return None


SUPPORTED_EMOTIONS = {
    "neutral": "平静自然", "happy": "高兴愉快", "sad": "悲伤",
    "angry": "愤怒", "fearful": "恐惧", "surprised": "惊讶",
    "disgusted": "厌恶",
}
SCENE_PRESETS = {
    "闲聊互动": "闲聊互动", "新闻播报": "新闻播报", "广告促销": "广告促销",
    "比赛解说": "比赛解说", "语音导航": "语音导航",
    "儿童内容": "一些儿童内容解说", "脱口秀": "脱口秀表演",
}


def text_to_speech_with_emotion(text, output_path, voice=None, model=None,
                                 emotion="neutral", scene=None, role=None,
                                 identity=None, dialect=None):
    model = model or COSYVOICE_MODEL
    voice = voice or (CLONED_VOICE_ID or "longxiaochun_v2")
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if dialect:
        instruction = f"请用{dialect}表达。"
    elif scene:
        instruction = f"你正在进行{scene}，你说话的情感是{emotion}。"
    elif role:
        instruction = f"你现在说话的角色是{role}，你说话的情感是{emotion}。"
    elif identity:
        instruction = f"你正在以一个{identity}的身份说话，你说话的情感是{emotion}。"
    else:
        instruction = f"你说话的情感是{emotion}。"

    logger.info("[CosyVoice 情感] voice=%s, emotion=%s, instruction=%s",
                 voice, emotion, instruction)

    synthesizer = SpeechSynthesizer(
        model=model, voice=voice,
        format=AudioFormat.MP3_22050HZ_MONO_256KBPS,
    )
    try:
        audio_data = synthesizer.call(text)
        with open(output_path, "wb") as f:
            f.write(audio_data)
        logger.info(f"✅ 语音已保存：{output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ 合成失败：{e}")
        return None


if __name__ == "__main__":
    os.makedirs(OUT_TTS, exist_ok=True)

    logger.info("【2.4.1.1】音色提取")
    list_builtin_voices("cosyvoice-v2")

    logger.info("\n【2.4.1.3】语音合成")
    test_text = "你好，我是消防语音助手，请注意值班室必须二十四小时在岗。"
    text_to_speech(test_text, os.path.join(OUT_TTS, "cosy_龙小淳.mp3"),
                   voice="longxiaochun_v2")
    text_to_speech(test_text, os.path.join(OUT_TTS, "cosy_龙小诚.mp3"),
                   voice="longxiaocheng_v2")

    logger.info("\n【2.4.1.4】情感润色")
    fire_text = "着火了！三楼着火，有两个人被困，请立即派消防车！"
    for emo in ["neutral", "angry", "fearful"]:
        text_to_speech_with_emotion(
            fire_text, os.path.join(OUT_TTS, f"cosy_情感_{emo}.mp3"),
            emotion=emo,
        )

    logger.info(f"\n🎉 输出目录：{OUT_TTS}")