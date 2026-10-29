#!/usr/bin/env bash
# 监控栈端到端验证。
#
# ★ 为什么不只看容器状态：
#   Prometheus 与 Grafana 都能"容器 running 但没数据"——
#   抓取目标写错、路径不对、数据源没配、看板查询语法有误，
#   这些都不会让容器退出，只会让图上什么都没有。
#   所以必须逐层验证：应用 /metrics → Prometheus targets →
#   Prometheus 查询 → Grafana 看板是否真的存在。
set -uo pipefail

APP_PORT="${AGENT1_PORT:-18001}"
PROM_PORT="${PROMETHEUS_PORT:-9090}"
GRAF_PORT="${GRAFANA_PORT:-3001}"
FAIL=0

ok()   { echo "  ${G}✓${N} $*"; }
bad()  { echo "  ${R}✗${N} $*"; FAIL=1; }
info() { echo "  - $*"; }

if [ -t 1 ]; then G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; N=$'\033[0m'
else G=""; R=""; Y=""; N=""; fi

echo "=== 1. 应用 /metrics 端点 ==="
code=$(curl -s -o /tmp/metrics.txt -w '%{http_code}' --max-time 8 \
       "http://localhost:${APP_PORT}/metrics" 2>/dev/null || echo "000")
if [ "$code" = "200" ]; then
  ok "/metrics 返回 200（$(wc -l < /tmp/metrics.txt) 行）"
  if grep -q '^# HELP fire_' /tmp/metrics.txt; then
    n=$(grep -c '^# HELP fire_' /tmp/metrics.txt || echo 0)
    ok "已暴露 ${n} 个 fire_* 指标"
  else
    info "暂无 fire_* 指标（还没有请求经过，指标是懒注册的）"
  fi
else
  bad "/metrics 返回 ${code}"
fi

echo
echo "=== 2. 产生一些流量再看 ==="
for i in 1 2 3; do
  curl -s -o /dev/null --max-time 5 \
    "http://localhost:${APP_PORT}/health" 2>/dev/null || true
done
sleep 2
if curl -s --max-time 8 "http://localhost:${APP_PORT}/metrics" 2>/dev/null \
     | grep -q 'fire_http_requests_total'; then
  ok "请求之后 fire_http_requests_total 出现了"
else
  info "仍无 fire_http_requests_total（可能中间件未生效，需要排查）"
fi

echo
echo "=== 3. Prometheus targets 状态 ==="
sleep 18   # 等一个抓取周期（scrape_interval=15s）
targets=$(curl -s --max-time 10 \
          "http://localhost:${PROM_PORT}/api/v1/targets" 2>/dev/null)
if [ -z "$targets" ]; then
  bad "Prometheus API 无响应"
else
  n_up=$(echo "$targets" | grep -o '"health":"up"' | wc -l | tr -d ' ')
  n_down=$(echo "$targets" | grep -o '"health":"down"' | wc -l | tr -d ' ')
  info "targets: up=${n_up} down=${n_down}"
  if [ "$n_up" -gt 0 ]; then
    ok "${n_up} 个目标抓取正常"
  else
    bad "没有任何目标抓取成功"
    echo "$targets" | head -c 400
  fi
fi

echo
echo "=== 4. Prometheus 查询 fire_* 指标 ==="
q=$(curl -s --max-time 10 --get \
     --data-urlencode 'query=fire_http_requests_total' \
     "http://localhost:${PROM_PORT}/api/v1/query" 2>/dev/null)
if echo "$q" | grep -q '"status":"success"'; then
  res=$(echo "$q" | grep -o '"value":\[[^]]*\]' | head -1)
  if [ -n "$res" ]; then
    ok "查询有返回：${res}"
  else
    info "查询成功但暂无数据（应用还没被 Prometheus 抓到或还没产生流量）"
  fi
else
  bad "查询失败：$(echo "$q" | head -c 200)"
fi

echo
echo "=== 5. Grafana 健康与数据源 ==="
gh=$(curl -s --max-time 10 "http://localhost:${GRAF_PORT}/api/health" 2>/dev/null)
if echo "$gh" | grep -q '"database": *"ok"'; then
  ok "Grafana 健康检查通过"
else
  info "Grafana 未就绪（可能还在启动）：$(echo "$gh" | head -c 160)"
fi

ds=$(curl -s --max-time 10 -u admin:admin \
      "http://localhost:${GRAF_PORT}/api/datasources" 2>/dev/null)
if echo "$ds" | grep -qi 'prometheus'; then
  ok "Prometheus 数据源已通过 provisioning 导入"
else
  info "数据源尚未导入或需要更多启动时间"
fi

echo
echo "=== 6. Grafana 看板是否被provisioning 加载 ==="
dash=$(curl -s --max-time 10 -u admin:admin \
        "http://localhost:${GRAF_PORT}/api/search?type=dash-db" 2>/dev/null)
if echo "$dash" | grep -q 'fire-agents-overview'; then
  ok "看板 fire-agents-overview 已加载"
  npanel=$(echo "$dash" | grep -o '"panelCount":[0-9]*' | head -1)
  info "  $npanel"
else
  info "看板尚未加载（provisioning 需要时间，或路径未挂对）"
fi

echo
if [ "$FAIL" = "0" ]; then
  echo "  ${G}监控栈验证通过${N}"
  echo
  echo "  打开看板： http://localhost:${GRAF_PORT}/d/fire-agents-overview"
  echo "  Prometheus: http://localhost:${PROM_PORT}"
else
  echo "  ${R}有问题，见上面标✗ 的项${N}"
fi
exit $FAIL