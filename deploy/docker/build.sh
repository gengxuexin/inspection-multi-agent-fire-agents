#!/usr/bin/env bash
# 构建并推送 Docker 镜像到私有仓库
# 用法：REGISTRY=registry.local/fire ./build.sh
set -euo pipefail

REGISTRY="${REGISTRY:-}"
BASE_IMAGE="${BASE_IMAGE:-python:3.10-slim}"
TAG="${TAG:-latest}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [ -z "$REGISTRY" ]; then
    echo "⚠️  REGISTRY 未设置，仅本地构建（不推送）"
    PUSH=""
else
    PUSH="--push"
    echo "📦 推送到: $REGISTRY (tag=$TAG)"
fi

services=("gateway" "agent1" "agent2" "agent3")

for svc in "${services[@]}"; do
    echo ""
    echo "========== 构建 $svc =========="
    if [ -z "$REGISTRY" ]; then
        docker build \
            -f "$SCRIPT_DIR/${svc}.Dockerfile" \
            --build-arg BASE_IMAGE="$BASE_IMAGE" \
            -t "fire-$svc:$TAG" \
            "$PROJECT_ROOT"
    else
        docker build \
            -f "$SCRIPT_DIR/${svc}.Dockerfile" \
            --build-arg BASE_IMAGE="$BASE_IMAGE" \
            -t "$REGISTRY/fire-$svc:$TAG" \
            "$PROJECT_ROOT" \
            $PUSH
    fi
done

echo ""
echo "✅ 构建完成"
