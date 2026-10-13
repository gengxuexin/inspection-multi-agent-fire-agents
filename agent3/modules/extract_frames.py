# -*- coding: utf-8 -*-
"""
logger = logging.getLogger("fire_agents.agent3")

视频抽帧工具：将视频按固定帧率抽成图片
"""
import subprocess
from pathlib import Path
import logging


def extract_frames(video_path, output_dir, fps=1, max_width=1280):
    """
    将视频抽帧为图片。

    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        fps: 抽帧频率，1 表示每秒 1 帧
        max_width: 图片最大宽度（等比缩放），防止图片过大浪费Token

    Returns:
        list[Path]: 抽出的图片路径列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 清空旧帧，避免重复累积
    for old in output_dir.glob("frame_*.jpg"):
        old.unlink()

    # ffmpeg 抽帧命令
    output_pattern = str(output_dir / "frame_%04d.jpg")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps},scale='min({max_width},iw)':-2",   # 抽帧 + 等比缩放
        "-q:v", "2",                                           # JPEG质量（1~31，越小越好）
        output_pattern,
        "-y"
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="ignore"
    )

    if result.returncode != 0:
        logger.info("ffmpeg 报错：")
        logger.info(result.stderr[-2000:])
        raise RuntimeError("抽帧失败")

    frames = sorted(output_dir.glob("frame_*.jpg"))
    logger.info(f"✅ 抽帧完成：{len(frames)} 张图片 → {output_dir}")
    return frames


if __name__ == "__main__":
    video = r"D:\AI\ollama_deepseek\Agent4-图片和视频处理器\input\4.mp4"
    out_dir = r"D:\AI\ollama_deepseek\Agent4-图片和视频处理器\output\4\frames"

    frames = extract_frames(video, out_dir, fps=1, max_width=1280)
    for f in frames[:5]:
        logger.info(f)
    if len(frames) > 5:
        logger.info(f"... 还有 {len(frames)-5} 张")