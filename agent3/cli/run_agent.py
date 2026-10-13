# -*- coding: utf-8 -*-
"""
Agent4 命令行入口
用法：
  python run_agent.py --image "path/to/img.jpg"
  python run_agent.py --video "path/to/video.mp4"
  python run_agent.py --image xx.jpg --no-yolo
"""
import argparse
from pathlib import Path
from fire_agent import FireInspectionAgent


def main():
    parser = argparse.ArgumentParser(description="Agent4 消防检查")
    parser.add_argument("--image", type=str, help="图片路径")
    parser.add_argument("--video", type=str, help="视频路径")
    parser.add_argument("--no-yolo", action="store_true", help="禁用 YOLO")
    parser.add_argument("--no-ocr", action="store_true", help="禁用 OCR")
    parser.add_argument("--segment", type=int, default=4, help="视频切段秒数")
    parser.add_argument("--fps", type=int, default=1, help="抽帧频率")
    parser.add_argument("--yolo-video", action="store_true",
                        help="视频也跑 YOLO（慢）")

    args = parser.parse_args()

    if not args.image and not args.video:
        parser.print_help()
        return

    agent = FireInspectionAgent(use_yolo=not args.no_yolo)

    if args.image:
        agent.analyze_image(
            args.image,
            do_yolo=not args.no_yolo,
            do_ocr=not args.no_ocr,
        )

    if args.video:
        agent.analyze_video(
            args.video,
            segment_seconds=args.segment,
            fps=args.fps,
            do_yolo=args.yolo_video,
        )


if __name__ == "__main__":
    main()