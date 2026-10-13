ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

LABEL service="gateway"

WORKDIR /app

# 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 业务代码
COPY common/ common/
COPY gateway.py .
COPY multimodal_orchestrator.py .
COPY orchestrator_langgraph.py .
COPY data/intent_train.jsonl data/
COPY models/ models/

# 日志目录
RUN mkdir -p logs

EXPOSE 8000

CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8000"]
