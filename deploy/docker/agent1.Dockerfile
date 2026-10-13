ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

LABEL service="agent1"

WORKDIR /app

# 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 共享包
COPY common/ common/

# agent1 业务代码
COPY agent1/*.py agent1/
COPY agent1/kb_meta.yaml agent1/

# 日志目录
RUN mkdir -p logs agent1/output/pending agent1/output/official agent1/output/reviews

EXPOSE 8001

WORKDIR /app/agent1
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
