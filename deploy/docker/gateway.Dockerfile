ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

LABEL service="gateway" \
      maintainer="fire-agents"

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    WEB_CONCURRENCY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 依赖 ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 业务代码 ----
COPY common/ common/
COPY gateway.py .
COPY orchestrator_langgraph.py .
# legacy 编排器为可选回退路径，按需构建
COPY multimodal_orchestrator.py* ./
COPY data/intent_train.jsonl data/

RUN mkdir -p logs

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

EXPOSE 8000

# gateway 已有 /agents/health（会连带探测下游 agent），
# 存活探针单独用根路径 —— 用 /agents/health 的话，
# 下游 agent1 挂了会把 gateway 也判成不健康并重启，
# 结果就是「一个服务的问题引发全部服务重启」。
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/ || exit 1

CMD uvicorn gateway:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-1}