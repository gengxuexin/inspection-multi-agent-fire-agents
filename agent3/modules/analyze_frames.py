# -*- coding: utf-8 -*-
"""
logger = logging.getLogger("fire_agents.agent3")

3.2.3 抽帧分析（完整输出所有结果）
"""
import os
import base64
import json
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import BASE_URL, VLM_MODEL, require_dashscope
import logging

# ========== 1. 配置 ==========
MODEL = VLM_MODEL

FRAMES_DIR = r"D:\AI\ollama_deepseek\Agent4-图片和视频处理器\output\4\frames"
OUTPUT_DIR = r"D:\AI\ollama_deepseek\Agent4-图片和视频处理器\output\4"
RESULT_JSON = os.path.join(OUTPUT_DIR, "frame_analysis_results.json")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "frame_summary.json")
REPORT_MD = os.path.join(OUTPUT_DIR, "frame_analysis_report.md")

FRAME_PROMPT = (
    "这是消防检查视频的一帧画面。请识别：\n"
    "1. 画面中有哪些消防设施？（灭火器、消防栓、消防水带、消防柜、"
    "火灾报警控制器、消防联动控制柜、安全出口标志等）\n"
    "2. 消防通道是否被占用？\n"
    "3. 是否存在明显的消防安全隐患？\n"
    "注意：只描述你能看到的内容，看不到的不要推测。"
    "未出现的设施说'本帧未观察到'。\n"
    "回答控制在150字以内。"
)


# ========== 2. 工具函数 ==========
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_frame(client, frame_path, prompt=FRAME_PROMPT):
    """上传单张图片给 Qwen3-VL 分析"""
    b64 = image_to_base64(frame_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        resp = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=400)
        return resp.choices[0].message.content
    except Exception as e:
        return f"[分析失败] {e}"


# ========== 3. 主流程 ==========
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frames = sorted(Path(FRAMES_DIR).glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError(f"没找到图片，请先运行 extract_frames.py：{FRAMES_DIR}")

    logger.info(f"共 {len(frames)} 张图片待分析\n")
    client = OpenAI(api_key=require_dashscope(), base_url=BASE_URL)

    # ---- 并发逐帧分析 ----
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_frame, client, str(f)): f for f in frames}
        done = 0
        for future in as_completed(futures):
            f = futures[future]
            answer = future.result()
            results[f.name] = answer
            done += 1
            logger.info(f"\n[{done}/{len(frames)}] {f.name}")
            logger.info(answer)
            logger.info("-" * 60)

    # ---- 保存逐帧结果（按文件名排序）----
    sorted_results = {k: results[k] for k in sorted(results.keys())}
    with open(RESULT_JSON, "w", encoding="utf-8") as fp:
        json.dump(sorted_results, fp, ensure_ascii=False, indent=2)
    logger.info(f"\n✅ 逐帧结果已保存：{RESULT_JSON}")

    # ---- 综合判断 ----
    logger.info("\n" + "=" * 60)
    logger.info("综合判断")
    logger.info("=" * 60)

    summary_input = "\n\n".join(
        f"【{k}】{sorted_results[k]}" for k in sorted_results.keys()
    )

    summary_prompt = (
        "以下是消防检查视频各帧的AI识别结果，请综合判断：\n"
        "1. 该场所的消防设施配备情况如何？\n"
        "2. 是否存在消防安全隐患？如有，请列出。\n"
        "3. 总体评价（合格/基本合格/不合格）。\n\n"
        + summary_input
    )

    summary_resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=2000,
    )
    summary_text = summary_resp.choices[0].message.content

    logger.info(summary_text)

    # ---- 保存综合判断 ----
    summary_data = {
        "model": MODEL,
        "analyzed_at": datetime.now().isoformat(),
        "total_frames": len(frames),
        "summary": summary_text,
    }
    with open(SUMMARY_JSON, "w", encoding="utf-8") as fp:
        json.dump(summary_data, fp, ensure_ascii=False, indent=2)
    logger.info(f"\n✅ 综合判断已保存：{SUMMARY_JSON}")

    # ---- 生成 Markdown 报告 ----
    md_lines = [
        f"# 消防视频抽帧分析报告\n",
        f"- **分析时间**：{summary_data['analyzed_at']}",
        f"- **使用模型**：{MODEL}",
        f"- **抽帧总数**：{len(frames)} 张（1 秒 1 帧）",
        "\n---\n",
        "## 一、综合判断\n",
        summary_text,
        "\n---\n",
        "## 二、逐帧识别明细\n",
    ]
    for name, answer in sorted_results.items():
        md_lines.append(f"### {name}\n")
        md_lines.append(answer)
        md_lines.append("\n")

    with open(REPORT_MD, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md_lines))
    logger.info(f"✅ Markdown 报告已生成：{REPORT_MD}")


if __name__ == "__main__":
    main()