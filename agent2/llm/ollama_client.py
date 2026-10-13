# -*- coding: utf-8 -*-
"""Ollama 本地大模型封装：语义梳理 + 消防问答"""
import requests
import os
import subprocess

logger = logging.getLogger("fire_agents.agent2")

def _detect_ollama_url():
    env = os.getenv("OLLAMA_URL")
    if env:
        return env
    if os.name == "nt":
        return "http://localhost:11434/api/chat"
    try:
        ip = subprocess.check_output(
            "ip route show default | awk '{print $3}'", shell=True
        ).decode().strip()
        return f"http://{ip}:11434/api/chat"
    except Exception:
        return "http://172.30.32.1:11434/api/chat"

from config import OLLAMA_MODEL
import logging
OLLAMA_URL = _detect_ollama_url()


def refine_text_with_llm(raw_text):
    """语义梳理：纠正错别字、加标点"""
    if not raw_text.strip():
        return raw_text

    logger.info("\n🧠 大模型语义梳理中...")
    system_prompt = (
        "你是一个消防领域的语音识别纠错助手。接下来我会给你一段由语音识别生成的消防录音文本。"
        "任务是：\n"
        "1. 纠正明显的错别字（特别是消防专业术语）。\n"
        "2. 补充标点符号，使语句通顺。\n"
        "3. 去除无意义的重复词。\n"
        "4. 只做纠正和润色，不要增加原文没有的信息，不要总结或提炼。\n"
        "5. 直接输出润色后的文本。"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"原始文本：\n{raw_text}"},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        if resp.status_code != 200:
            logger.info(f"⚠️ HTTP {resp.status_code}，返回原文")
            return raw_text
        result = resp.json()
        return result.get("message", {}).get("content", "").strip() or raw_text
    except Exception as e:
        logger.error(f"⚠️ 大模型调用失败（用原文）：{e}")
        return raw_text


def ask_fire_bot(user_input, chat_history):
    """消防问答机器人：基于历史对话生成回答"""
    system_prompt = (
        "你是一个专业的消防知识问答助手，服务于消防值班室、消防检查、火灾报警等场景。"
        "回答要求：\n"
        "1. 准确专业，引用《消防法》、GB 标准、消防安全常识。\n"
        "2. 简洁明了，控制在 100 字以内，方便语音播报。\n"
        "3. 口语化，像人对话一样自然。\n"
        "4. 用户描述火情或紧急情况时，优先给出'立即行动'的指导（拨打119、疏散、启动预案）。\n"
        "5. 直接输出回答，不要加格式标记。"
    )
    messages = [{"role": "system", "content": system_prompt}]
    for msg in (chat_history or [])[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.6, "num_ctx": 2048, "num_predict": 200},
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        if resp.status_code != 200:
            return f"抱歉，大模型服务异常（HTTP {resp.status_code}）"
        result = resp.json()
        return result.get("message", {}).get("content", "").strip() \
            or "（模型返回为空）"
    except requests.exceptions.Timeout:
        return "抱歉，大模型处理超时。"
    except Exception as e:
        return f"抱歉，调用失败：{e}"