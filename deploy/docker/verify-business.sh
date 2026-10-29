#!/usr/bin/env bash
# 容器内业务闭环验证：带token 调真实接口。
# 只验 /health 是不够的 —— 服务能响应不代表检索链路能用。
set -uo pipefail

PORT="${1:-18001}"
ROOT=/home/geng/fire_agent_v2/fire_agents

# ★ 键名以 .env 里实际存在的为准：AGENT_SHARED_TOKEN。
#   写死一个猜的键名，脚本会一直报"没找到 token"，看起来像鉴权坏了，
#   其实是脚本自己的问题 —— 这种假警报比没有脚本更浪费时间。
TOKEN="$(grep -iE '^(AGENT_SHARED_TOKEN|INTERNAL_TOKEN)[ \t]*=' "$ROOT/.env" \
         | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')"

if [ -z "${TOKEN:-}" ]; then
  echo "!! .env 里没找到 AGENT_SHARED_TOKEN / INTERNAL_TOKEN，无法验证鉴权接口"
  echo "   .env 中与 token 相关的键："
  grep -iE 'token' "$ROOT/.env" | sed 's/=.*/= <hidden>/'
  exit 2
fi

echo "=== /api/doc/templates ==="
curl -s --max-time 10 -H "X-Internal-Token: $TOKEN" \
  "http://localhost:${PORT}/api/doc/templates" | head -c 400
echo; echo

echo "=== /api/rag/retrieve（真实检索：验证挂载的向量库 + embedding 配置）==="
curl -s --max-time 60 -X POST \
  -H "X-Internal-Token: $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query":"消防监督检查的内容包括哪些","top_k":3}' \
  "http://localhost:${PORT}/api/rag/retrieve" | head -c 700
echo; echo

echo "=== /api/rag/ask（走完整生成链路）==="
curl -s --max-time 120 -X POST \
  -H "X-Internal-Token: $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question":"消防监督检查的主要内容包括什么？","top_k":5}' \
  "http://localhost:${PORT}/api/rag/ask" | head -c 700
echo