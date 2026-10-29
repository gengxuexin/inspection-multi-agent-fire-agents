#!/usr/bin/env bash
# 启动/重启 RAG 链路演示服务。
#
# 为什么不用 pkill -f demo_server.py：
#   那会把**当前这条命令行自己**也匹配上（命令行里就含这个字符串），
#   于是 shell 被自己的信号杀掉，exit 15，表现为"命令莫名失败"。
#   正确做法是按端口杀，或先取出 PID 再 kill。
set -uo pipefail

PORT="${1:-8800}"
PROJECT=/home/geng/fire_agent_v2/fire_agents
PY=/home/geng/.venv/bin/python
LOG=/tmp/demo_server.log

cd "$PROJECT" || exit 1

# 按端口找 PID（比 pkill -f 精确：只杀监听该端口的进程）
OLD=$(ss -lptn "sport = :$PORT" 2>/dev/null \
      | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
if [ -n "${OLD:-}" ]; then
  echo "停止旧进程 PID=$OLD"
  kill "$OLD" 2>/dev/null
  sleep 2
fi

echo "启动演示服务（端口 $PORT）"
setsid "$PY" agent1/demo/demo_server.py --port "$PORT" > "$LOG" 2>&1 &

# 等健康检查通过
for i in $(seq 1 30); do
  sleep 1
  if curl -sf --max-time 2 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    echo "✓ 服务已就绪http://127.0.0.1:$PORT"
    # 再等预热完成，让首次查询不至于 6 秒
    for j in $(seq 1 25); do
      if grep -q '预热完成' "$LOG" 2>/dev/null; then
        grep '预热完成' "$LOG" | tail -1
        break
      fi
      sleep 1
    done
    exit 0
  fi
done

echo "✗ 启动失败，日志尾部："
tail -12 "$LOG"
exit 1