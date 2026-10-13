# -*- coding: utf-8 -*-
"""
Agent3 语音处理器主入口
用法：
    python main.py video2text <video> [whisper|xfyun_lfasr|vosk]
    python main.py batch [engine]
    python main.py voicebot
    python main.py tts_demo
    python main.py list_voices
    python main.py check_env
"""
import sys
import os
from config import INPUT_DIR, OUT_WHISPER, OUT_XFYUN_LFASR, OUT_VOSK


def cmd_video2text(video_path, engine="whisper"):
    from pipelines.video_to_text import process_video
    out_map = {
        "whisper": OUT_WHISPER,
        "xfyun_lfasr": OUT_XFYUN_LFASR,
        "vosk": OUT_VOSK,
    }
    return process_video(
        os.path.abspath(video_path),
        engine=engine,
        output_dir=out_map[engine],
    )


def cmd_batch(engine="whisper"):
    videos = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))
    ]
    if not videos:
        print(f"⚠️ {INPUT_DIR} 下没有视频文件")
        return
    for v in videos:
        cmd_video2text(os.path.join(INPUT_DIR, v), engine)


def cmd_voicebot():
    from pipelines.voice_qa_bot import create_ui
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


def cmd_tts_demo():
    from tts.edge_engine import synthesize, list_chinese_voices
    from config import OUT_TTS
    list_chinese_voices()
    synthesize("你好，我是消防语音助手。",
               os.path.join(OUT_TTS, "demo.mp3"))


def cmd_list_voices():
    from tts.edge_engine import list_chinese_voices
    list_chinese_voices()


def cmd_check_env():
    from config import check_env
    check_env(verbose=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd == "video2text":
        if len(sys.argv) < 3:
            print("用法：python main.py video2text <video路径> [engine]")
            sys.exit(1)
        engine = sys.argv[3] if len(sys.argv) > 3 else "whisper"
        cmd_video2text(sys.argv[2], engine)
    elif cmd == "batch":
        cmd_batch(sys.argv[2] if len(sys.argv) > 2 else "whisper")
    elif cmd == "voicebot":
        cmd_voicebot()
    elif cmd == "tts_demo":
        cmd_tts_demo()
    elif cmd == "list_voices":
        cmd_list_voices()
    elif cmd == "check_env":
        cmd_check_env()
    else:
        print(f"未知命令：{cmd}")
        print(__doc__)