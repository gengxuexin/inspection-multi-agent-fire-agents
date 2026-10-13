# -*- coding: utf-8 -*-
"""
logger = logging.getLogger("fire_agents.agent2")

消防语音问答机器人
ASR: asr/whisper_engine.py
LLM: llm/ollama_client.py
TTS: tts/edge_engine.py (主) + tts/pyttsx3_engine.py (降级)
UI:  Gradio
"""
import os
import time
import gradio as gr

from config import OUT_VOICE_BOT, TTS_VOICE_EDGE, TTS_EMOTION
from asr.whisper_engine import WhisperEngine
from llm.ollama_client import ask_fire_bot
from tts.edge_engine import synthesize, EMOTION_PRESETS
import logging


# ============ 单例加载 Whisper ============
logger.info("⏳ 初始化 Whisper 引擎...")
whisper = WhisperEngine()
logger.info("✅ Whisper 引擎就绪")


MAX_HISTORY = 6   # 最多保留 6 条消息（3 轮对话）


# ============ ASR 封装 ============
def speech_to_text(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        return ""
    return whisper.transcribe(audio_path)


# ============ TTS 封装（带重试 + 降级）============
def text_to_speech(text, output_path, voice=None, emotion=None, retry=2):
    voice = voice or TTS_VOICE_EDGE
    emotion = emotion or TTS_EMOTION

    # 主引擎：edge-tts
    for attempt in range(retry + 1):
        try:
            return synthesize(text, output_path, voice=voice, emotion=emotion)
        except Exception as e:
            logger.error(f"⚠️ [edge-tts] 第 {attempt+1} 次失败：{e}")
            if attempt < retry:
                time.sleep(1)

    # 降级：pyttsx3
    logger.error("⚠️ edge-tts 全部失败，降级到 pyttsx3")
    try:
        from tts.pyttsx3_engine import text_to_speech_pyttsx3
        wav_path = os.path.splitext(output_path)[0] + ".wav"
        text_to_speech_pyttsx3(text, wav_path)
        return wav_path
    except Exception as e:
        logger.error(f"❌ [pyttsx3] 也失败：{e}")
        return None


# ============ 主流程 ============
def voice_qa_pipeline(audio_path, chat_history, voice_choice, emotion_choice):
    if chat_history is None:
        chat_history = []

    if not audio_path:
        return "请先录音或上传音频", chat_history, None

    # 1. ASR
    t0 = time.time()
    user_text = speech_to_text(audio_path)
    t_asr = time.time() - t0

    if not user_text:
        chat_history.append(
            {"role": "assistant", "content": "⚠️ 没有识别到语音内容，请重试"}
        )
        return "⚠️ 没有识别到语音内容，请重试", chat_history, None

    # 2. LLM
    t0 = time.time()
    answer_text = ask_fire_bot(user_text, chat_history)
    t_llm = time.time() - t0

    # 3. TTS
    t0 = time.time()
    output_audio = os.path.join(
        OUT_VOICE_BOT, f"answer_{int(time.time())}.mp3"
    )
    tts_result = text_to_speech(answer_text, output_audio,
                                voice=voice_choice, emotion=emotion_choice)
    t_tts = time.time() - t0

    logger.info(f"⏱️ ASR {t_asr:.1f}s | LLM {t_llm:.1f}s | TTS {t_tts:.1f}s")

    # 4. 追加对话
    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": answer_text})

    # ★ 限制历史长度
    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[-MAX_HISTORY:]

    return answer_text, chat_history, tts_result


# ============ UI ============
def create_ui():
    with gr.Blocks(title="🚒 消防语音问答机器人") as demo:
        gr.Markdown("# 🚒 消防语音问答机器人")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="🎤 语音输入（录音 或 上传音频）",
                )
                with gr.Accordion("⚙️ 语音合成配置", open=False):
                    voice_choice = gr.Dropdown(
                        choices=[
                            "zh-CN-XiaoxiaoNeural",
                            "zh-CN-XiaoyiNeural",
                            "zh-CN-YunxiNeural",
                            "zh-CN-YunjianNeural",
                            "zh-CN-YunyangNeural",
                            "zh-CN-liaoning-XiaobeiNeural",
                            "zh-CN-shaanxi-XiaoniNeural",
                        ],
                        value=TTS_VOICE_EDGE,
                        label="音色选择",
                    )
                    emotion_choice = gr.Dropdown(
                        choices=list(EMOTION_PRESETS.keys()),
                        value=TTS_EMOTION,
                        label="情感/语气",
                    )
                submit_btn = gr.Button("🚀 开始问答",
                                       variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ 清空对话")

            with gr.Column(scale=2):
                answer_output = gr.Textbox(label="🤖 机器人回答", lines=4,
                                           interactive=False)
                audio_output = gr.Audio(label="🔊 语音播报",
                                        type="filepath", autoplay=True)
                chatbot = gr.Chatbot(label="💬 对话历史", height=350)

        submit_btn.click(
            fn=voice_qa_pipeline,
            inputs=[audio_input, chatbot, voice_choice, emotion_choice],
            outputs=[answer_output, chatbot, audio_output],
        )
        clear_btn.click(
            fn=lambda: ([], None, ""),
            outputs=[chatbot, audio_output, answer_output],
        )

        gr.Markdown("### 💡 示例问题")
        gr.Markdown(
            "- 消防车必须多少小时在岗在位？\n"
            "- 灭火器多久需要检查一次？\n"
            "- 发现火灾应该先做什么？\n"
            "- 值班室为什么必须持证上岗？\n"
            "- 仓库堆放货物有什么消防要求？"
        )
    return demo


if __name__ == "__main__":
    import requests
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3)
        logger.info("✅ Ollama 服务正常")
    except Exception:
        logger.info("⚠️ 无法连接 Ollama，请先启动 Ollama")

    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )