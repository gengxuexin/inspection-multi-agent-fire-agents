#!/bin/bash
# 消防智能执法系统 - 一键启动/停止/状态
# 用法：
#   ./start_all.sh                    启动所有服务
#   ./start_all.sh --with-redis       启动并预检 Redis
#   ./start_all.sh --with-milvus      启动并预检 Milvus
#   ./start_all.sh stop               停止所有服务
#   ./start_all.sh status             查看状态
#   ./start_all.sh restart            重启

set -euo pipefail

# ==================== 配置 ====================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="$SCRIPT_DIR"
LOG_DIR="$BASE/logs"
mkdir -p "$LOG_DIR"

AGENT1_PORT=8001
AGENT2_PORT=8002
AGENT3_PORT=8003
GATEWAY_PORT=8000

# venv 路径（按需修改）
VENV="${VENV:-$HOME/.venv/bin/activate}"

# 组件预检开关
CHECK_REDIS=false
CHECK_MILVUS=false

# ==================== 参数解析 ====================
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --with-redis)  CHECK_REDIS=true ;;
        --with-milvus) CHECK_MILVUS=true; CHECK_REDIS=true ;;
        start|stop|status|restart) ;;
        *) ARGS+=("$arg") ;;
    esac
done
COMMAND="${1:-start}"

# ==================== 工具函数 ====================
check_port() {
    local port=$1
    lsof -i:$port -sTCP:LISTEN -t 2>/dev/null || true
}

wait_healthy() {
    local name=$1 port=$2 max=${3:-30}
    local i=0
    while [ $i -lt $max ]; do
        if [ -n "$(check_port $port)" ]; then
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done
    return 1
}

preflight_redis() {
    echo "  🔍 预检 Redis..."
    if command -v redis-cli &>/dev/null; then
        if redis-cli ping 2>/dev/null | grep -q PONG; then
            echo "  ✅ Redis 可用"
        else
            echo "  ❌ Redis 未响应（redis-cli ping 失败）"
            echo "     请启动: redis-server --daemonize yes 或 docker run -d -p 6379:6379 redis:7-alpine"
            return 1
        fi
    else
        echo "  ⚠️  redis-cli 未安装，跳过 Redis 预检"
    fi
}

preflight_milvus() {
    echo "  🔍 预检 Milvus..."
    local uri="${MILVUS_URI:-}"
    if [ -z "$uri" ]; then
        echo "  ℹ️  MILVUS_URI 未配置，将使用 FAISS 回退"
        return 0
    fi
    echo "  ℹ️  MILVUS_URI=$uri"
    # 简单 TCP 探测
    local host port
    host=$(echo "$uri" | sed -E 's|.*://([^:/]+).*|\1|')
    port=$(echo "$uri" | sed -E 's|.*:([0-9]+).*|\1|')
    port="${port:-19530}"
    if command -v nc &>/dev/null; then
        if nc -z -w2 "$host" "$port" 2>/dev/null; then
            echo "  ✅ Milvus 可达 ($host:$port)"
        else
            echo "  ⚠️  Milvus 不可达 ($host:$port)，将回退 FAISS"
        fi
    else
        echo "  ⚠️  nc 未安装，跳过 Milvus 连通性检查"
    fi
}

# ==================== 启动单服务 ====================
start_service() {
    local name=$1 dir=$2 script=$3 port=$4
    local log="$LOG_DIR/${name}.log"

    if [ -n "$(check_port $port)" ]; then
        echo "  ⚠️  $name 已在运行（端口 $port）"
        return
    fi

    cd "$dir"
    source "$VENV" 2>/dev/null || true
    nohup python "$script" > "$log" 2>&1 &
    sleep 2

    if wait_healthy "$name" "$port" 15; then
        echo "  ✅ $name (端口 $port)  日志: $log"
    else
        echo "  ❌ $name 启动失败，查看日志:"
        tail -10 "$log"
    fi
}

# ==================== 启动 ====================
do_start() {
    echo "=========================================="
    echo "  消防智能执法系统 - 启动"
    echo "=========================================="
    echo

    # 预检
    if [ "$CHECK_REDIS" = true ]; then
        preflight_redis || true
    fi
    if [ "$CHECK_MILVUS" = true ]; then
        preflight_milvus || true
    fi
    echo

    # 编排模式
    local mode="${ORCHESTRATOR_MODE:-legacy}"
    echo "  📋 编排模式: $mode"
    if [ "$mode" = "langgraph" ]; then
        echo "  ℹ️  LangGraph 编排已启用（需安装 langgraph）"
    fi
    echo

    # 启动服务
    start_service "Agent1(RAG+文书)" "$BASE/agent1" "server.py" $AGENT1_PORT
    start_service "Agent2(语音)"     "$BASE/agent2" "server.py" $AGENT2_PORT
    start_service "Agent3(视觉)"     "$BASE/agent3" "server.py" $AGENT3_PORT
    start_service "Gateway"          "$BASE"       "gateway.py" $GATEWAY_PORT

    # 健康检查
    echo
    echo "=========================================="
    echo "  健康检查"
    echo "=========================================="
    sleep 2
    for port in $GATEWAY_PORT $AGENT1_PORT $AGENT2_PORT $AGENT3_PORT; do
        name=$(case $port in
            $GATEWAY_PORT) echo "Gateway";;
            $AGENT1_PORT)   echo "Agent1(RAG+文书)";;
            $AGENT2_PORT)   echo "Agent2(语音)";;
            $AGENT3_PORT)   echo "Agent3(视觉)";;
        esac)
        resp=$(curl -s --max-time 3 "http://127.0.0.1:$port/" 2>/dev/null || true)
        if [ -n "$resp" ]; then
            echo "  ✅ $name"
        else
            echo "  ❌ $name (无响应)"
        fi
    done

    # Redis 状态
    if command -v redis-cli &>/dev/null; then
        if redis-cli ping 2>/dev/null | grep -q PONG; then
            echo "  ✅ Redis"
        else
            echo "  ⚠️  Redis (未运行，异步任务不可用)"
        fi
    fi

    echo
    echo "🎉 启动完成！"
    echo "  · Gateway: http://127.0.0.1:$GATEWAY_PORT"
    echo "  · 日志目录: $LOG_DIR"
    echo "  · 鉴权: 在 .env 中 AUTH_DISABLED=0 时需要 JWT token"
    echo
}

# ==================== 停止 ====================
do_stop() {
    echo "停止所有服务..."
    for port in $GATEWAY_PORT $AGENT1_PORT $AGENT2_PORT $AGENT3_PORT; do
        fuser -k ${port}/tcp 2>/dev/null || true
    done
    echo "✅ 已停止 Gateway + Agent1/2/3"
}

# ==================== 状态 ====================
do_status() {
    echo "=========================================="
    echo "  服务状态"
    echo "=========================================="
    for port in $AGENT1_PORT $AGENT2_PORT $AGENT3_PORT $GATEWAY_PORT; do
        name=$(case $port in
            $GATEWAY_PORT) echo "Gateway";;
            $AGENT1_PORT)   echo "Agent1(RAG+文书)";;
            $AGENT2_PORT)   echo "Agent2(语音)";;
            $AGENT3_PORT)   echo "Agent3(视觉)";;
        esac)
        pid=$(check_port $port)
        if [ -n "$pid" ]; then
            echo "  ✅ $name (端口 $port, PID $(echo $pid | head -1))"
        else
            echo "  ❌ $name (端口 $port 未启动)"
        fi
    done
    echo
    # Redis
    if command -v redis-cli &>/dev/null; then
        if redis-cli ping 2>/dev/null | grep -q PONG; then
            echo "  ✅ Redis"
        else
            echo "  ⚠️  Redis (未运行)"
        fi
    fi
    # Milvus
    local uri="${MILVUS_URI:-}"
    if [ -n "$uri" ]; then
        echo "  ℹ️  Milvus URI: $uri"
    else
        echo "  ℹ️  Milvus: 未配置（FAISS 回退）"
    fi
    # 编排模式
    echo "  ℹ️  编排模式: ${ORCHESTRATOR_MODE:-legacy}"
}

# ==================== 主入口 ====================
case "$COMMAND" in
    start)   do_start ;;
    stop)    do_stop ;;
    status)  do_status ;;
    restart) do_stop; sleep 2; do_start ;;
    *)       echo "用法: $0 {start|stop|status|restart} [--with-redis] [--with-milvus]" ;;
esac
