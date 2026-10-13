ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

LABEL service="agent3"

WORKDIR /app

# 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 共享包
COPY common/ common/

# agent3 业务代码
COPY agent3/*.py agent3/

# 日志目录
RUN mkdir -p logs

EXPOSE 8003

WORKDIR /app/agent3
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003"]
