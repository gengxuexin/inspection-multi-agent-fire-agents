# -*- coding: utf-8 -*-
"""
logger = logging.getLogger("fire_agents.agent3")

视频切分工具：将长视频按固定时长切成小段（Windows 兼容版）
"""
import subprocess
from pathlib import Path
import logging


def split_video(video_path, output_dir, segment_seconds=4):
    """将视频按 segment_seconds 秒切分为多个小段"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 获取视频总时长 ----
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(
        probe_cmd, capture_output=True, text=True,
        encoding='utf-8', errors='ignore',        # ★ 修复 GBK 崩溃
    )
    total_duration = float(result.stdout.strip())
    expected = int(total_duration / segment_seconds) + 1
    logger.info(f"视频总时长：{total_duration:.1f} 秒")
    logger.info(f"每段 {segment_seconds} 秒，预计切分 {expected} 段")

    # ---- 2. ffmpeg 切分（重新编码，精确切分）----
    output_pattern = str(output_dir / "segment_%03d.mp4")
    split_cmd = [
        "ffmpeg", "-i", video_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        "-force_key_frames", f"expr:gte(t,n_forced*{segment_seconds})",
        "-map", "0",
        "-segment_time", str(segment_seconds),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_pattern,
        "-y"
    ]
    result = subprocess.run(
        split_cmd, capture_output=True, text=True,
        encoding='utf-8', errors='ignore',        # ★ 修复 GBK 崩溃
    )

    # 收集切分结果
    segments = sorted(output_dir.glob("segment_*.mp4"))
    logger.info(f"切分完成，共 {len(segments)} 段\n")
    return [str(s) for s in segments]


if __name__ == "__main__":
    video = r"D:\AI\ollama_deepseek\Agent4-图片和视频处理器\input\4.mp4"
    segs = split_video(video, r"D:\AI\ollama_deepseek\Agent4-图片和视频处理器\output\4\video_segments", segment_seconds=4)
    for s in segs:
        logger.info(s)