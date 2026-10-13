# -*- coding: utf-8 -*-
"""从视频提取音频（16kHz/16bit/单声道）"""
import os
import sys
import subprocess


def extract_audio_from_video(video_path, output_path):
    """从视频中提取音频，输出 16kHz/16bit/单声道 WAV"""
    if not os.path.exists(video_path):
        print(f"❌ 找不到视频：{video_path}")
        return None

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        output_path,
    ]

    # Windows 下隐藏子进程黑框
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            creationflags=creation_flags,
        )
        print(f"✅ 音频提取完成：{output_path}")
        return output_path
    except FileNotFoundError:
        print("❌ 找不到 ffmpeg，请确认已安装并加入 PATH")
        return None
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
        print(f"❌ ffmpeg 失败：{err[-500:]}")
        return None