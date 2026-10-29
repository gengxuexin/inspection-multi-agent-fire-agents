ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

LABEL service="agent3" \
      maintainer="fire-agents"

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    WEB_CONCURRENCY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

# ---- 系统依赖 ----
# ffmpeg：拆帧/抽音轨。libgl1+libglib2.0：opencv 运行时（cv2 import 时就 dlopen，
# 缺了会在 import cv2 那一刻直接报ImportError，跟 opencv 装没装无关）。
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        libgl1 libglib2.0-0t64 \
        fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 依赖 ----
COPY requirements.txt requirements-agent3.txt ./

# ★ torch 必须先装 CPU 版，且必须在 requirements-agent3.txt 之前。
#
#   实测：默认 `pip install torch` 会连带装整套 CUDA 运行时，
#   agent3 镜像到 11.3 GB，其中 nvidia CUDA 占 3.2 GB。
#   本项目是 CPU 推理，这 3.2 GB 是纯浪费：
#   拉取慢、占磁盘、容器冷启动还要解压几 GB 的 .so 库。
#
#   顺序很关键：先装 CPU 版 torch，ultralytics 之后装时
#   检测到 torch 已满足依赖就不会再去拉 CUDA 版。
#   （反过来先装 ultralytics，它会把 CUDA 版 torch 一起拖进来。）
#
#   ★ 用 --extra-index-url 而不是 --index-url：
#     --index-url 会**完全替换**索引源，于是 torch 的构建依赖
#     flit_core 在 PyTorch 索引里找不到，直接报
#       No matching distribution found for flit_core<4,>=3.11
#     --extra-index-url 是"额外加一个"，PyPI 仍然可用。
RUN pip install --no-cache-dir torch \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-agent3.txt

# 装完立刻清掉 torch 的测试资产（几百 MB，用不到）
RUN python -c "import torch, pathlib; p = pathlib.Path(torch.__file__).parent; \
    [__import__('shutil').rmtree(d, ignore_errors=True) for d in (p/'test', p/'include')]" \
    && rm -rf /root/.cache/pip /tmp/*

# ---- 共享包 ----
COPY common/ common/

# ---- agent3 业务代码 ----
# ★ 必须整目录 COPY：modules/yolo_detector.py、agent/fire_agent.py 都在子目录里，
#   COPY agent3/*.py agent3/ 会让这些模块全部缺失。
COPY agent3/ agent3/

RUN find agent3 -type d -name __pycache__ -prune -exec rm -rf {} + \
    && mkdir -p logs

# ---- 非 root 运行 ----
# ★ uid/gid 必须与宿主用户一致，否则 compose 挂载目录一律 Permission denied。
#   隐蔽之处：单独 docker run 不挂目录时一切正常，镜像「验证通过」，
#   一上 compose 就崩，很容易误判成镜像问题。
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN useradd --create-home --uid ${HOST_UID} --gid ${HOST_GID} appuser 2>/dev/null \
    || (groupadd --gid ${HOST_GID} appuser \
        && useradd --create-home --uid ${HOST_UID} --gid ${HOST_GID} appuser) \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8003/health || exit 1

WORKDIR /app/agent3
CMD uvicorn server:app --host 0.0.0.0 --port 8003 --workers ${WEB_CONCURRENCY:-1}