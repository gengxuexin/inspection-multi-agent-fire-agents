ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

LABEL service="agent2" \
      maintainer="fire-agents"

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    WEB_CONCURRENCY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

# ---- 系统依赖 ----
# ffmpeg：faster-whisper 解码音频、以及任何音视频转码都要它。缺了它 WhisperEngine
#   会在首次调用时抛异常，而不是启动时报错 —— 那种失败更难排查。
# libasound2：TTS 播放需要。fonts：中文字体。
#
# ★ 包名与基础镜像强相关，不要凭印象写：
#   python:3.10-slim 现在是 Debian 13 (trixie)，做了 64-bit time_t 迁移，
#   libasound2 / libglib2.0-0 在这个版本里已不存在，实际提供的是带 t64 后缀的
#   libasound2t64 / libglib2.0-0t64。
#   写错包名的后果是构建直接 exit 100（apt: Unable to locate package）。
#   要换基础镜像版本时，先跑 _check_syspkgs.py 确认包名。
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        libasound2t64 \
        libgl1 libglib2.0-0t64 \
        fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 依赖 ----
# 分两层装：核心依赖先行，语音重依赖单独一层。
# 这样改业务代码时两层缓存都还在，重建只需几秒。
COPY requirements.txt requirements-agent2.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-agent2.txt

# ---- 共享包 ----
COPY common/ common/

# ---- agent2 业务代码 ----
# ★ 必须整目录 COPY。server.py 里 `from asr.whisper_engine import ...`、
#   `from llm.ollama_client import ...` 都是子目录 import，
#   COPY agent2/*.py agent2/ 只搬顶层文件，asr/ llm/ tts/ pipelines/ 全都不在镜像里。
COPY agent2/ agent2/

RUN find agent2 -type d -name __pycache__ -prune -exec rm -rf {} + \
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

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8002/health || exit 1

WORKDIR /app/agent2
CMD uvicorn server:app --host 0.0.0.0 --port 8002 --workers ${WEB_CONCURRENCY:-1}