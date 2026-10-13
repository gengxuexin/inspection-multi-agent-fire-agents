ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

LABEL service="agent2"

WORKDIR /app

# 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 共享包
COPY common/ common/

# agent2 业务代码
COPY agent2/*.py agent2/

# 日志目录
RUN mkdir -p logs

EXPOSE 8002

WORKDIR /app/agent2
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
