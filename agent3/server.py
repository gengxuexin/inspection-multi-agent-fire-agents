# ~/fire_agents/agent3/server.py
import os
import sys
import shutil
import tempfile
import traceback

# ★ 第一步：先把 import 路径配好
#
# 这里原来只加了 agent/ modules/ cli/ 三个子目录，漏了**项目根**。
# 后果：`from common.tracing import ...` 这类跨包import 会失败 ——
# 实测挂metrics 时暴露出来：
#     ModuleNotFoundError: No module named 'common'
# 而 agent1/gateway 能正常 import，因为它们的工作目录就是项目根，
# Python 会自动把 CWD 加进 sys.path；agent3 的 WORKDIR 是 /app/agent3，
# 所以不会自动带上 /app。
#
# 修法：显式把项目根加进 sys.path，与子目录一起加。
_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_ROOT)
for sub in (os.path.join(_PROJECT_ROOT, ""),
            os.path.join(_ROOT, "agent"),
            os.path.join(_ROOT, "modules"),
            os.path.join(_ROOT, "cli")):
    if sub and sub not in sys.path:
        sys.path.insert(0, sub)

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


@app.get("/health")
def health():
    """存活探针：不加载 YOLO 权重、不做推理。

    ultralytics 的 YOLO() 会读权重文件并初始化网络，耗时以秒计。
    放进健康检查意味着每 30 秒就重载一次模型 —— 既拖慢探针，
    又可能在内存上翻车。而模型加载失败是「就绪问题」，重启不解决。
    """
    return {"service": "Agent3 视觉服务", "status": "ok"}


@app.get("/ready")
def ready():
    """就绪探针：检查 YOLO 权重文件是否就位。

    权重是运行时挂载/下载的，缺失时服务能起来但一推理就报错。
    这个检查能让问题在流量进来之前就暴露出来。
    """
    import shutil as _shutil
    try:
        from config import YOLO_WEIGHTS
    except Exception:
        YOLO_WEIGHTS = ""
    weights_ok = bool(YOLO_WEIGHTS) and os.path.isfile(str(YOLO_WEIGHTS))
    ffmpeg_ok = _shutil.which("ffmpeg") is not None
    return {"service": "Agent3 视觉服务",
            "status": "ok" if (weights_ok and ffmpeg_ok) else "degraded",
            "checks": {"yolo_weights": "ok" if weights_ok else "missing",
                       "ffmpeg": "ok" if ffmpeg_ok else "missing"}}


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