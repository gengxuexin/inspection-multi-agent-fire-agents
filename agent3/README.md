# Agent4 - 图片和视频处理器

## 环境要求
- Python 3.11
- 依赖：`pip install ultralytics openai requests pillow pyyaml`

## 环境变量
DASHSCOPE_API_KEY=sk-xxx # 阿里百炼 Qwen3-VL
BAIDU_API_KEY=xxx # 百度OCR
BAIDU_SECRET_KEY=xxx


## 快速开始
```bash
# 分析图片
python cli/run_agent.py --image input/酒店外观_4.jpg

# 分析视频
python cli/run_agent.py --video input/4.mp4

# 视频 + YOLO
python cli/run_agent.py --video input/4.mp4 --yolo-video

模块说明
     模块	         对应章节	   功能
baiduOCR.py	3.1.1	在线 OCR      识别图片文字
yolo_detector.py	 3.2.1	    YOLO 识别消防设施
video_analyzer.py	 3.2.2	   视频切段 + VLM 分析
analyze_frames.py	 3.2.3	    抽帧 + VLM 分析
judgment_agent.py	 3.2.5	   综合判断 + 告警生成
fire_agent.py	      整体	    统一 Agent 入口

输出说明
每次分析结果保存到 output/<视频名>/final_report.json，包含：

video_segments：分段分析

frames：逐帧分析

judgment：结构化告警