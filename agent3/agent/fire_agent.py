# -*- coding: utf-8 -*-
"""
Agent4 - 图片和视频处理器 统一入口
整合 3.1.1 / 3.2.1 / 3.2.2 / 3.2.3 / 3.2.5
"""
import os
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# 已有模块
from baiduOCR import get_access_token, general_ocr, extract_hotel_name

# 新增模块
from yolo_detector import YoloDetector
from judgment_agent import JudgmentAgent

# 视频处理模块
from split_video import split_video
from extract_frames import extract_frames
from video_analyzer import analyze_segment
from analyze_frames import analyze_frame

from config import (
    BAIDU_API_KEY, BAIDU_SECRET_KEY,
    require_dashscope,
    BASE_URL, VLM_MODEL,
    INPUT_DIR, OUTPUT_DIR,
)
import logging


logger = logging.getLogger("fire_agents.agent3.agent")

class FireInspectionAgent:
    """消防检查 Agent"""

    def __init__(self, use_yolo=True):
        self.client = OpenAI(api_key=require_dashscope(), base_url=BASE_URL)
        self.judge_agent = JudgmentAgent()
        self.yolo = None
        if use_yolo:
            try:
                self.yolo = YoloDetector()
            except FileNotFoundError as e:
                logger.warning(f"⚠️ YOLO 未加载（{e}），将跳过 3.2.1 检测")

    # ========== 3.1.1 图片文字识别 ==========
    def analyze_image_text(self, image_path):
        """在线 OCR 识别图片文字"""
        if not BAIDU_API_KEY:
            return {"error": "未设置 BAIDU_API_KEY / BAIDU_SECRET_KEY 环境变量"}

        token = get_access_token(BAIDU_API_KEY, BAIDU_SECRET_KEY)
        texts = general_ocr(token, image_path)
        hotel_name = extract_hotel_name(texts)
        return {
            "ocr_texts": texts,
            "hotel_name": hotel_name,
        }

    # ========== 图片分析 ==========
    def analyze_image(self, image_path, do_yolo=True, do_ocr=True):
        """对单张图片做综合分析"""
        image_path = Path(image_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"分析图片：{image_path.name}")
        logger.info("=" * 60)

        result = {"source": str(image_path), "type": "image"}

        # 3.1.1 OCR
        if do_ocr:
            logger.info("\n▶ 3.1.1 在线 OCR...")
            try:
                result["ocr"] = self.analyze_image_text(str(image_path))
                logger.info(f"  识别 {len(result['ocr'].get('ocr_texts', []))} 条文字")
                logger.info(f"  酒店名称：{result['ocr'].get('hotel_name', '')}")
            except Exception as e:
                result["ocr"] = {"error": str(e)}
                logger.error(f"  ✗ OCR 失败：{e}")

        # 3.2.1 YOLO
        if do_yolo and self.yolo:
            logger.info("\n▶ 3.2.1 YOLO 检测...")
            try:
                result["yolo"] = self.yolo.detect(str(image_path))
                logger.info(f"  检测到 {result['yolo']['total_objects']} 个目标")
                logger.info(f"  分类：{result['yolo']['counts']}")
            except Exception as e:
                result["yolo"] = {"error": str(e)}
                logger.error(f"  ✗ YOLO 失败：{e}")

        # 3.2.5 综合判断（传入图片路径，让 VLM 融合判断）
        logger.info("\n▶ 3.2.5 综合判断（VLM 融合版）...")
        judgment = self.judge_agent.judge(
            yolo_result=result.get("yolo"),
            source_info=f"单张图片：{image_path.name}",
            image_path=str(image_path),    # ★ 新增：传原图路径
        )
        result["judgment"] = judgment
        self.judge_agent.print_report(judgment)

        # 保存
        out = Path(OUTPUT_DIR) / f"image_result_{image_path.stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"\n✅ 结果已保存：{out}")

        return result

    # ========== 视频分析 ==========
    def analyze_video(self, video_path, segment_seconds=4, fps=1,
                      do_yolo=False):
        """对视频做综合分析（3.2.2 + 3.2.3 + 3.2.5）"""
        video_path = Path(video_path)
        stem = video_path.stem
        out_dir = Path(OUTPUT_DIR) / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"分析视频：{video_path.name}")
        logger.info("=" * 60)

        result = {"source": str(video_path), "type": "video"}

        # ---- 3.2.2 视频切分 + 分析 ----
        logger.info("\n▶ 3.2.2 视频切分与分析...")
        seg_dir = out_dir / "video_segments"
        segments = split_video(str(video_path), str(seg_dir), segment_seconds)

        seg_results = []
        for i, seg in enumerate(segments):
            logger.info(f"  [{i+1}/{len(segments)}] {Path(seg).name}")
            ans = analyze_segment(self.client, seg)
            seg_results.append({"segment": Path(seg).name, "answer": ans})

        # 3.2.2 综合
        seg_summary_prompt = (
            "以下是消防检查视频各片段的AI分析结果，请综合判断：\n"
            "1. 消防设施配备情况\n2. 消防通道是否被占用\n"
            "3. 是否存在消防安全隐患\n\n"
        )
        for r in seg_results:
            seg_summary_prompt += f"【{r['segment']}】{r['answer']}\n"

        seg_summary = self.client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": seg_summary_prompt}],
            max_tokens=1500,
        ).choices[0].message.content

        result["video_segments"] = {
            "total": len(segments),
            "details": seg_results,
            "summary": seg_summary,
        }
        logger.info(f"  ✅ 3.2.2 完成，共 {len(segments)} 段")

        # ---- 3.2.3 抽帧 + 分析 ----
        logger.info("\n▶ 3.2.3 抽帧与分析...")
        frame_dir = out_dir / "frames"
        frames = extract_frames(str(video_path), str(frame_dir), fps=fps)

        frame_results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(analyze_frame, self.client, str(f)): f
                       for f in frames}
            for future in as_completed(futures):
                f = futures[future]
                frame_results[f.name] = future.result()
        frame_results = {k: frame_results[k] for k in sorted(frame_results.keys())}

        # 3.2.3 综合
        frame_summary_prompt = (
            "以下是消防检查视频各帧的AI识别结果，请综合判断：\n"
            "1. 消防设施配备情况\n2. 消防通道是否被占用\n"
            "3. 是否存在消防安全隐患\n\n"
        )
        for k, v in frame_results.items():
            frame_summary_prompt += f"【{k}】{v}\n"

        frame_summary = self.client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": frame_summary_prompt}],
            max_tokens=2000,
        ).choices[0].message.content

        result["frames"] = {
            "total": len(frames),
            "details": frame_results,
            "summary": frame_summary,
        }
        logger.info(f"  ✅ 3.2.3 完成，共 {len(frames)} 帧")

        # ---- 3.2.1 YOLO（可选）----
        yolo_result = None
        if do_yolo and self.yolo:
            logger.info("\n▶ 3.2.1 YOLO 对抽帧图检测...")
            yolo_counts = {}
            for f in frames[:20]:
                r = self.yolo.detect(str(f))
                for k, v in r["counts"].items():
                    yolo_counts[k] = yolo_counts.get(k, 0) + v
            yolo_result = {"counts": yolo_counts,
                           "total_objects": sum(yolo_counts.values()),
                           "frames_analyzed": min(20, len(frames))}
            result["yolo"] = yolo_result
            logger.info(f"  ✅ YOLO 完成，检测：{yolo_counts}")

        # ---- 3.2.5 综合判断 ----
        logger.info("\n▶ 3.2.5 综合判断...")
        judgment = self.judge_agent.judge(
            yolo_result=yolo_result,
            video_result=result["video_segments"],
            frame_result=result["frames"],
            source_info=f"视频：{video_path.name}",
        )
        result["judgment"] = judgment
        self.judge_agent.print_report(judgment)

        # 保存完整报告
        out_file = out_dir / "final_report.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"\n✅ 完整报告已保存：{out_file}")

        return result


if __name__ == "__main__":
    agent = FireInspectionAgent(use_yolo=False)
    # agent.analyze_video(r"D:\AI\ollama_deepseek\Agent4-图片和视频处理器\input\4.mp4")