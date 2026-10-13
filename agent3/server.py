# ~/fire_agents/agent3/server.py
import os
import sys
import shutil
import tempfile
import traceback

# ★ 第一步：先把三个子目录加入 import 路径
_ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("agent", "modules", "cli"):
    p = os.path.join(_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

# ★ 第二步：现在路径就绪了，才能 import
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn


app = FastAPI(title="Agent3 视觉服务")

# 延迟加载（避免启动就加载 YOLO/OCR）
_agent = None

def get_agent(use_yolo=True):
    global _agent
    if _agent is None:
        from fire_agent import FireInspectionAgent   # ← 到这里再 import
        _agent = FireInspectionAgent(use_yolo=use_yolo)
    return _agent


@app.get("/")
def root():
    return {"service": "Agent3 视觉服务", "status": "ok"}


@app.post("/api/vision/analyze_image")
async def analyze_image(file: UploadFile = File(...),
                        do_yolo: bool = Form(True),
                        do_ocr: bool = Form(True)):
    """上传图片，返回 YOLO + OCR + 综合判断"""
    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_path = os.path.join(tmp_dir, file.filename)
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        agent = get_agent(use_yolo=do_yolo)
        result = agent.analyze_image(tmp_path, do_yolo=do_yolo, do_ocr=do_ocr)
        return {"status": "success", "result": result}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/vision/analyze_video")
async def analyze_video(file: UploadFile = File(...),
                        segment: int = Form(4),
                        fps: int = Form(1),
                        do_yolo: bool = Form(False)):
    """上传视频，返回分段分析 + 抽帧分析 + 综合判断"""
    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_path = os.path.join(tmp_dir, file.filename)
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        agent = get_agent(use_yolo=do_yolo)
        result = agent.analyze_video(
            tmp_path, segment_seconds=segment, fps=fps, do_yolo=do_yolo
        )
        return {"status": "success", "result": result}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)