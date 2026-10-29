ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

LABEL service="agent1" \
      maintainer="fire-agents"

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    WEB_CONCURRENCY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

# ---- 系统依赖 ----
# pytesseract 调 tesseract 二进制；poppler 是 pdf2image 的后端；
# fonts 装中文字体，否则 Pillow 渲染中文会炸成方块。
#
# ★ libglib2.0-0 在 python:3.10-slim（Debian 13 trixie）里已不存在，
#   实际提供的是带 t64 后缀的 libglib2.0-0t64。
#   写错包名的后果是构建 exit 100（apt: Unable to locate package）——
#   而"能构建成功"这件事本身也可能是侥幸：某些包名会被 apt 当虚拟包静默解析，
#   真正缺的是运行时库，症状要等到 import cv2 /渲染图片时才暴露。
#   换基础镜像前先跑 _check_syspkgs.py 确认包名。
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        tesseract-ocr tesseract-ocr-chi-sim \
        poppler-utils \
        libgl1 libglib2.0-0t64 \
        fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 依赖 ----
# 依赖分层放在代码之前：改业务代码时不会失效这一层缓存。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 共享包 ----
COPY common/ common/

# ---- agent1 业务代码 ----
# ★ 这里必须整目录 COPY，不能用 COPY agent1/*.py agent1/。
#   glob 只匹配当前层，agent1/eval/、agent1/ui/ 这些子目录会被整个漏掉 ——
#   症状很隐蔽：镜像构建成功，但运行时一旦走到 eval/ui 的代码就报
#   ModuleNotFoundError: No module named 'eval'。
COPY agent1/ agent1/

# 把索引/知识库/输出里遗留的产物清掉，只留代码。
# .dockerignore 已排除大部分，这里兜底防止漏网文件进镜像。
RUN find agent1 -type d -name __pycache__ -prune -exec rm -rf {} + \
    && rm -rf agent1/faiss_full agent1/faiss_multi agent1/faiss_single \
              agent1/knowledge_base agent1/inputs \
    && mkdir -p logs \
              agent1/output/pending agent1/output/official agent1/output/reviews

# ---- 非 root 运行 ----
# 以 root 跑服务是交付事故的常见来源：容器被突破时攻击者直接拿到宿主 root。
#
# ★ uid/gid 必须与宿主用户一致，否则 compose 挂载目录一律Permission denied。
#   compose 把宿主的 agent1/output/ 挂进 /app/agent1/output，
#   挂载点属主是宿主用户的 uid，而容器内用户是另一个 uid → 写不进去。
#
#   这个坑的隐蔽之处在于：单独 docker run（不挂 output/）一切正常，
#   镜像「验证通过」，一上 compose 就 PermissionError 崩溃。
#   很容易误判成镜像有问题，实际是 uid 不匹配。
#
#   两个方案：
#     A. uid 对齐宿主（本文件采用）—— 挂载目录天然可读写，
#        容器写出的文件在宿主上也属主正确。
#     B. 保留固定 uid + compose 里配 user: —— 换宿主要重新 chown，
#        且容器写出的文件在宿主机上属主是 10001，宿主机其他工具读不了。
#
#   HOST_UID/HOST_GID 可覆盖：HOST_UID=1000 HOST_GID=1000 docker compose build
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN useradd --create-home --uid ${HOST_UID} --gid ${HOST_GID} appuser 2>/dev/null \
    || (groupadd --gid ${HOST_GID} appuser \
        && useradd --create-home --uid ${HOST_UID} --gid ${HOST_GID} appuser) \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

# 存活探针：打 /health（只确认进程能响应 HTTP，不查外部依赖）。
# 就绪判断在 compose 里用 /ready，两者故意分开：
#   /health 失败 -> 重启容器；依赖抖动时重启毫无意义，只会把抖动放大。
#   /ready 失败 -> 摘流量，等依赖恢复。
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

WORKDIR /app/agent1
CMD uvicorn server:app --host 0.0.0.0 --port 8001 --workers ${WEB_CONCURRENCY:-1}