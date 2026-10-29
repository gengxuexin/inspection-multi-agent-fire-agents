#!/usr/bin/env python3
"""把 Prometheus 与 Grafana 加进 docker-compose.yml 的 obs profile。

用脚本改是因为 compose 文件手改容易缩进错位，
而 YAML 的缩进错误只在 `docker compose config` 时才暴露。
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
COMPOSE = ROOT / "deploy" / "docker" / "docker-compose.yml"

BLOCK = '''
  # ---- Prometheus（指标存储与查询）----
  prometheus:
    image: prom/prometheus:v2.54.1
    profiles: ["obs"]
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      # 保留 15 天。本地演示够了；生产要按实际容量调，
      # 太小会导致看板查不到历史数据（表现为"图突然从昨天开始"）。
      - "--storage.tsdb.retention.time=15d"
      # 允许通过 9090 暴露表达式调试接口。
      # ★ 生产环境必须关掉（--web.enable-lifecycle），
      #   它允许任何人远程重载配置与清空数据。
      - "--web.enable-lifecycle"
    volumes:
      - ./../monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "${PROMETHEUS_PORT:-9090}:9090"
    # depends_on 各应用服务，保证它们先起来；
    # 但**不用** condition: service_healthy —— Prometheus 抓取失败会
    # 自动重试，等应用健康反而让监控自己变成被依赖对象。
    depends_on: [agent1, agent2, agent3, gateway]

  # ---- Grafana（看板）----
  grafana:
    image: grafana/grafana:11.2.0
    profiles: ["obs"]
    environment:
      GF_SECURITY_ADMIN_USER: ${GRAFANA_USER:-admin}
      # 生产必须换掉这个默认值。
      # ★ 用 /etc/grafana/provisioning 里的 datasources.yml 也可以带 token，
      #   但放在环境变量里更直观，也更容易在部署时被 review 到。
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
      GF_USERS_ALLOW_SIGN_UP: "false"
      GF_INSTALL_PLUGINS: ""
      GF_LOG_LEVEL: warn
    volumes:
      - ./../monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./../monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana-data:/var/lib/grafana
    ports:
      - "${GRAFANA_PORT:-3001}:3000"
    depends_on: [prometheus]
'''

src = COMPOSE.read_text(encoding="utf-8")

if "prometheus:" in src and "grafana:" in src:
    print("· compose 里已有 prometheus/grafana，跳过")
    sys.exit(0)

# 插到 langfuse-db 服务之后、volumes: 之前
marker = "\nvolumes:\n"
idx = src.find(marker)
if idx < 0:
    print("✗ 找不到 volumes: 锚点")
    sys.exit(1)

src = src[:idx] + "\n" + BLOCK + src[idx:]

# 在顶层 volumes 里加两个数据卷
src = src.replace(
    "  langfuse-db-data:\n",
    "  langfuse-db-data:\n  prometheus-data:\n  grafana-data:\n",
)

COMPOSE.write_text(src, encoding="utf-8")
print(f"✓ {COMPOSE.name} 已加入 prometheus 与 grafana")

# ---- 用 docker compose config 校验 YAML ----
r = subprocess.run(
    ["docker", "compose", "config", "--quiet"],
    cwd=str(COMPOSE.parent), capture_output=True, text=True,
)
if r.returncode == 0:
    print("✓ compose 配置校验通过")
else:
    print("✗ compose 配置有问题：")
    print((r.stderr or r.stdout)[:800])
    sys.exit(1)