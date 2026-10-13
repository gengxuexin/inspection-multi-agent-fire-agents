# -*- coding: utf-8 -*-
"""
logger = logging.getLogger("fire_agents.agent3")

3.2.2 视频内容直接分析
将视频切分为3~5秒小段，逐段上传Qwen3-VL进行识别，保存完整结果
"""
import os
import base64
import time
import json
from datetime import datetime
from openai import OpenAI

from split_video import split_video
from config import BASE_URL, VLM_MODEL, require_dashscope
import logging

# ========== 1. 配置 ==========
MODEL = VLM_MODEL

VIDEO_PATH = r"D:\AI\ollama_deepseek\Agent4-图片和视频处理器\input\4.mp4"
SEGMENT_DIR = r"D:\AI\ollama_deepseek\Agent4-图片和视频处理器\output\4\video_segments"
OUTPUT_DIR = r"D:\AI\ollama_deepseek\Agent4-图片和视频处理器\output\4"
SEGMENT_SECONDS = 4

SEGMENT_JSON = os.path.join(OUTPUT_DIR, "video_segment_results.json")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "video_summary.json")

ANALYSIS_PROMPT = (
    "这是一段消防检查视频的片段。请分析：\n"
    "1. 画面中出现了哪些消防设施（灭火器/消防栓/消防水带/消防柜/安全出口标志等）？\n"
    "2. 消防通道是否被占用？\n"
    "3. 是否存在明显的消防安全隐患？\n"
    "注意：只描述你能看到的内容，看不到的不要推测。\n"
    "请用简洁的中文回答。"
)


# ========== 2. 视频转Base64 ==========
def video_to_base64(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ========== 3. 单段视频分析 ==========
def analyze_segment(client, segment_path, prompt=ANALYSIS_PROMPT):
    """上传单个视频片段给 Qwen3-VL 进行分析"""
    b64_video = video_to_base64(segment_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{b64_video}"},
                    "max_pixels": 640 * 32 * 32,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=512,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"[分析失败] {e}"


# ========== 4. 主流程 ==========
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Step 1: 切分视频 ----
    logger.info("=" * 50)
    logger.info("Step 1: 切分视频")
    logger.info("=" * 50)
    segments = split_video(VIDEO_PATH, SEGMENT_DIR, SEGMENT_SECONDS)

    # ---- Step 2: 逐段分析 ----
    logger.info("=" * 50)
    logger.info("Step 2: 逐段上传 Qwen3-VL 分析")
    logger.info("=" * 50)

    client = OpenAI(api_key=require_dashscope(), base_url=BASE_URL)
    results = []

    for i, seg in enumerate(segments):
        logger.info(f"\n正在分析第 {i + 1}/{len(segments)} 段：{os.path.basename(seg)}")
        answer = analyze_segment(client, seg)
        results.append({
            "segment": os.path.basename(seg),
            "index": i,
            "answer": answer,
        })
        logger.info(answer)
        logger.info("-" * 60)
        time.sleep(0.5)

    # ---- Step 3: 保存逐段结果 ----
    with open(SEGMENT_JSON, "w", encoding="utf-8") as fp:
        json.dump({
            "model": MODEL,
            "segment_seconds": SEGMENT_SECONDS,
            "total_segments": len(segments),
            "analyzed_at": datetime.now().isoformat(),
            "results": results,
        }, fp, ensure_ascii=False, indent=2)
    logger.info(f"\n✅ 逐段结果已保存：{SEGMENT_JSON}")

    # ---- Step 4: 整体综合判断 ----
    logger.info("\n" + "=" * 50)
    logger.info("Step 4: 整体综合判断")
    logger.info("=" * 50)

    summary_prompt = (
        "以下是消防检查视频各片段的AI分析结果，请综合判断：\n"
        "1. 该场所的消防设施配备情况如何？\n"
        "2. 是否存在消防安全隐患？如有，请列出。\n"
        "3. 总体评价（合格/基本合格/不合格）。\n\n"
    )
    for r in results:
        summary_prompt += f"【{r['segment']}】{r['answer']}\n"

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=1500,
    )
    summary_text = completion.choices[0].message.content
    logger.info(summary_text)

    # ---- Step 5: 保存综合判断 ----
    with open(SUMMARY_JSON, "w", encoding="utf-8") as fp:
        json.dump({
            "model": MODEL,
            "analyzed_at": datetime.now().isoformat(),
            "total_segments": len(segments),
            "summary": summary_text,
        }, fp, ensure_ascii=False, indent=2)
    logger.info(f"\n✅ 综合判断已保存：{SUMMARY_JSON}")


if __name__ == "__main__":
    main()