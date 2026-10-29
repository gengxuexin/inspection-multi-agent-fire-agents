#!/usr/bin/env python3
"""生成 Grafana 看板 JSON。

用脚本生成而不是手写 JSON 的原因：手写一千多行 JSON
既慢又容易在某处漏一个逗号导致整份看板导入失败，
而失败时 Grafana 只会静默不显示，改起来很痛苦。

看板设计围绕「出问题时先看哪几个数」组织，
而不是把指标平铺一堆 —— 平铺的看板等于没有看板。
"""
import json
import pathlib
import sys

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)

DS = {"type": "prometheus", "uid": "fire-prom"}


def target(expr: str, legend: str, refid: str = "A") -> dict:
    return {
        "datasource": DS,
        "expr": expr,
        "legendFormat": legend,
        "refId": refid,
        "interval": "",
        "editorMode": "code",
    }


def stat(title, targets, unit="short", desc="", gridPos=None, thresholds=None):
    return {
        "type": "stat",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": gridPos or {"h": 5, "w": 4, "x": 0, "y": 0},
        "targets": [target(e, l, r) for e, l, r in targets],
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "values": False,
                              "fields": ""},
            "colorMode": "background",
            "graphMode": "area",
            "textMode": "auto",
            "justifyMode": "auto",
        },
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "thresholds": thresholds or {
                    "mode": "absolute",
                    "steps": [{"color": "text", "value": None}],
                },
                "mappings": [],
            },
            "overrides": [],
        },
        "options": {},
    }


def timeseries(title, targets, unit="short", desc="", gridPos=None, stack=False):
    return {
        "type": "timeseries",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": gridPos,
        "targets": [target(e, l, r) for e, l, r in targets],
        "options": {
            "legend": {"displayMode": "table",
                       "placement": "bottom",
                       "showLegend": True,
                       "calcs": ["mean", "max", "p95"]},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "custom": {
                    "drawStyle": "line",
                    "lineWidth": 1,
                    "fillOpacity": 8 if stack else 0,
                    "showPoints": "never",
                    "spanNulls": True,
                    "stacking": {"mode": "normal" if stack else "none",
                                 "group": "A"},
                    "axisSoftMin": 0,
                },
                "thresholds": {"mode": "absolute",
                               "steps": [{"color": "text", "value": None}]},
            },
            "overrides": [],
        },
    }


def row(title, y):
    return {"type": "row", "title": title, "collapsed": False,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
            "panels": []}


P = []          # panels
y = 0

# ==================== 第一行：服务健康总览 ====================
P.append(row("服务健康总览", y)); y += 1

P.append(stat(
    "在线服务数",
    [("sum(up{job=~\"agent1|agent2|agent3|gateway\"})", "up", "A")],
    desc="四个服务里有多少个Prometheus 能抓到 /metrics。"
         "掉到 3 以下说明有服务没起来或没暴露端点。",
    gridPos={"h": 5, "w": 4, "x": 0, "y": y},
    thresholds={"mode": "absolute",
                "steps": [{"color": "red", "value": None},
                          {"color": "orange", "value": 3},
                          {"color": "green", "value": 4}]},
))

P.append(stat(
    "请求速率 (req/s)",
    [("sum(rate(fire_http_requests_total[1m]))", "req/s", "A")],
    unit="reqps",
    desc="全部服务的请求速率。持续为 0 说明流量没进来。",
    gridPos={"h": 5, "w": 5, "x": 4, "y": y},
))

P.append(stat(
    "错误率",
    [("sum(rate(fire_http_requests_total{status=~\"5..\"}[5m])) "
      "/ clamp_min(sum(rate(fire_http_requests_total[5m])), 0.001)", "err", "A")],
    unit="percentunit",
    desc="5xx 占比。超过 1% 需要查日志；持续 5% 以上说明有服务崩了。",
    gridPos={"h": 5, "w": 5, "x": 9, "y": y},
    thresholds={"mode": "absolute",
                "steps": [{"color": "green", "value": None},
                          {"color": "orange", "value": 0.01},
                          {"color": "red", "value": 0.05}]},
))

P.append(stat(
    "P99 延迟",
    [("histogram_quantile(0.99, "
      "sum by (le) (rate(fire_http_request_duration_seconds_bucket[5m])))",
      "p99", "A")],
    unit="s",
    desc="全局 P99。注意：桶边界决定了分辨率，"
         "最细一档是 10s，超过 10s 的延迟都会落进 +Inf 桶。",
    gridPos={"h": 5, "w": 5, "x": 14, "y": y},
    thresholds={"mode": "absolute",
                "steps": [{"color": "green", "value": None},
                          {"color": "orange", "value": 2},
                          {"color": "red", "value": 5}]},
))

P.append(stat(
    "熔断器 OPEN 数",
    [("sum(fire_circuit_breaker_state == 1)", "open", "A")],
    desc="处于 OPEN 状态的熔断器数量。非 0 说明有下游在拒绝请求。",
    gridPos={"h": 5, "w": 5, "x": 19, "y": y},
    thresholds={"mode": "absolute",
                "steps": [{"color": "green", "value": None},
                          {"color": "red", "value": 1}]},
))
y += 5

# ==================== 第二行：请求与延迟 ====================
P.append(row("请求与延迟", y)); y += 1

P.append(timeseries(
    "各服务请求速率",
    [("sum by (service) (rate(fire_http_requests_total[1m]))",
      "{{service}}", "A")],
    unit="reqps",
    desc="哪个服务在扛流量。gateway 通常最高（所有外部流量都过它）。",
    gridPos={"h": 8, "w": 12, "x": 0, "y": y},
))

P.append(timeseries(
    "各服务 P50/P95/P99 延迟",
    [("histogram_quantile(0.50, sum by (le, service) "
      "(rate(fire_http_request_duration_seconds_bucket[5m])))",
      "p50 {{service}}", "A"),
     ("histogram_quantile(0.95, sum by (le, service) "
      "(rate(fire_http_request_duration_seconds_bucket[5m])))",
      "p95 {{service}}", "B"),
     ("histogram_quantile(0.99, sum by (le, service) "
      "(rate(fire_http_request_duration_seconds_bucket[5m])))",
      "p99 {{service}}", "C")],
    unit="s",
    desc="P50 与 P99 的差距大 = 存在长尾问题，通常是某个下游不稳定。",
    gridPos={"h": 8, "w": 12, "x": 12, "y": y},
))
y += 8

P.append(timeseries(
    "响应状态码分布",
    [("sum by (status) (rate(fire_http_requests_total[1m]))", "{{status}}", "A")],
    unit="reqps",
    desc="4xx 多说明是调用方参数问题；5xx 多说明是服务端问题。",
    gridPos={"h": 7, "w": 8, "x": 0, "y": y},
    stack=True,
))

P.append(timeseries(
    "Top 端点（按请求量）",
    [("topk(8, sum by (service, endpoint) "
      "(rate(fire_http_requests_total[5m])))", "{{service}} {{endpoint}}", "A")],
    unit="reqps",
    desc="定位热点接口。注意 endpoint 标签已做高基数归一化"
         "（uuid / 业务 id / 数字 id 都会压成 {id}）。",
    gridPos={"h": 7, "w": 16, "x": 8, "y": y},
))
y += 7

# ==================== 第三行：RAG 与 LLM ====================
P.append(row("RAG 检索", y)); y += 1

P.append(timeseries(
    "检索延迟分布",
    [("histogram_quantile(0.50, sum by (le, backend) "
      "(rate(fire_rag_retrieve_duration_seconds_bucket[5m])))",
      "p50 {{backend}}", "A"),
     ("histogram_quantile(0.95, sum by (le, backend) "
      "(rate(fire_rag_retrieve_duration_seconds_bucket[5m])))",
      "p95 {{backend}}", "B")],
    unit="s",
    desc="检索延迟对比 FAISS 与 Milvus。检索慢而 LLM 正常时，"
         "瓶颈在向量库或分词，不在生成。",
    gridPos={"h": 8, "w": 12, "x": 0, "y": y},
))

P.append(timeseries(
    "单次检索返回片段数",
    [("histogram_quantile(0.50, sum by (le) "
      "(rate(fire_rag_retrieved_chunks_bucket[5m])))", "p50", "A"),
     ("histogram_quantile(0.95, sum by (le) "
      "(rate(fire_rag_retrieved_chunks_bucket[5m])))", "p95", "B")],
    desc="检索返回多少片段。持续为 0 说明检索层挂了"
         "（服务活着但检索不出东西，这是只做存活探测发现不了的）。",
    gridPos={"h": 8, "w": 12, "x": 12, "y": y},
))
y += 8

P.append(row("LLM 调用与成本", y)); y += 1

P.append(timeseries(
    "LLM 调用速率（按用途）",
    [("sum by (purpose) (rate(fire_llm_calls_total[1m]))", "{{purpose}}", "A")],
    unit="reqps",
    desc="classify（意图分类）/ retrieve 判定 / generate（生成）分开看。"
         "generate 占比过高要留意成本。",
    gridPos={"h": 8, "w": 8, "x": 0, "y": y},
))

P.append(timeseries(
    "token 消耗速率",
    [("sum by (kind) (rate(fire_llm_tokens_total[1m]))", "{{kind}}", "A")],
    unit="short",
    desc="prompt 与 completion 分开。completion/prompt 比值突然升高，"
         "通常意味着生成的长度失控（比如 prompt 没约束好导致模型长篇大论）。",
    gridPos={"h": 8, "w": 8, "x": 8, "y": y},
))

P.append(timeseries(
    "LLM 延迟",
    [("histogram_quantile(0.50, sum by (le, purpose) "
      "(rate(fire_llm_duration_seconds_bucket[5m])))", "p50 {{purpose}}", "A"),
     ("histogram_quantile(0.95, sum by (le, purpose) "
      "(rate(fire_llm_duration_seconds_bucket[5m])))", "p95 {{purpose}}", "B")],
    unit="s",
    desc="生成延迟是端到端延迟的主要构成。p95 超过 20s 时，"
         "用户已经会以为系统挂了。",
    gridPos={"h": 8, "w": 8, "x": 16, "y": y},
))
y += 8

# ==================== 第四行：韧性 ====================
P.append(row("韧性与预算", y)); y += 1

P.append(timeseries(
    "熔断器状态（0=closed 1=open 2=half_open）",
    [("fire_circuit_breaker_state", "{{service}}/{{target}}", "A")],
    desc="值变成 1（OPEN）说明该下游正在拒绝所有请求。"
         "常驻 2（HALF_OPEN）说明熔断器反复试探，"
         "下游在好时坏之间摇摆 —— 这通常比一直 OPEN 更难处理。",
    gridPos={"h": 7, "w": 8, "x": 0, "y": y},
))

P.append(timeseries(
    "熔断器状态切换次数",
    [("sum by (service, target, to_state) "
      "(rate(fire_circuit_breaker_transitions_total[5m]))",
      "{{service}}→{{to_state}}", "A")],
    unit="short",
    desc="频繁切换 = 阈值设置不合理，在抖动的边界上反复跳。",
    gridPos={"h": 7, "w": 8, "x": 8, "y": y},
))

P.append(timeseries(
    "LLM 失败率",
    [("sum by (service, purpose) "
      "(rate(fire_llm_calls_total{outcome=\"error\"}[5m]))", "{{service}} {{purpose}}", "A")],
    unit="short",
    desc="按用途拆分，能直接看出是哪一类调用在失败"
         "（比如只有 classify 失败，多半是 Key 或模型名配错）。",
    gridPos={"h": 7, "w": 8, "x": 16, "y": y},
))
y += 7

P.append(timeseries(
    "预算耗尽事件",
    [("sum by (service) (rate(fire_budget_exhausted_total[5m]))", "{{service}}", "A")],
    desc="token 预算耗尽会直接拒绝请求。出现非 0 说明限额设置偏紧，"
         "或者有异常重试在放大消耗。",
    gridPos={"h": 6, "w": 12, "x": 0, "y": y},
))

P.append(timeseries(
    "MCP 工具调用",
    [("sum by (tool, outcome) (rate(fire_mcp_tool_calls_total[5m]))",
      "{{tool}} {{outcome}}", "A")],
    desc="对外暴露的 MCP 工具调用情况。error 占比高说明工具本身有问题。",
    gridPos={"h": 6, "w": 12, "x": 12, "y": y},
))
y += 6

# ==================== 看板元信息 ====================
dashboard = {
    "__inputs": [],
    "__requires": [
        {"type": "grafana", "id": "grafana", "name": "Grafana", "version": "10.0.0"},
        {"type": "prometheus", "id": "prometheus", "name": "Prometheus", "version": "1.0.0"},
        {"type": "datasource", "id": "fire-prom", "name": "Prometheus", "version": "1.0.0"},
    ],
    "annotations": {"list": [{
        "builtIn": 1,
        "datasource": {"type": "grafana", "uid": "-- Grafana --"},
        "enable": True,
        "hide": True,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard",
    }]},
    "description": (
        "消防智能执法系统 · 服务可观测总览。\n\n"
        "组织逻辑按「出问题时先看哪个数」排：\n"
        "1. 服务健康总览 —— 先确认是不是服务挂了\n"
        "2. 请求与延迟 —— 再看是不是慢\n"
        "3. RAG 检索 —— 检索不出东西时看这里\n"
        "4. LLM 调用与成本 —— 端到端延迟的主要构成\n"
        "5. 韧性与预算 —— 下游不稳定与成本失控时看这里"
    ),
    "editable": True,
    "fiscalYearStartMonth": 0,
    "graphTooltip": 1,          # 1 = 多图联动，鼠标悬停时同时高亮
    "links": [],
    "panels": P,
    "preload": False,
    "refresh": "30s",
    "schemaVersion": 39,
    "tags": ["fire-agents", "fde", "rag"],
    "templating": {"list": [
        {
            "name": "datasource",
            "type": "datasource",
            "query": "prometheus",
            "current": {},
        },
        {
            "name": "service",
            "type": "query",
            "datasource": DS,
            "query": "label_values(fire_http_requests_total, service)",
            "refresh": 1,
            "includeAll": True,
            "multi": True,
            "current": {"text": ["All"], "value": ["$__all"]},
        },
    ]},
    "time": {"from": "now-1h", "to": "now"},
    "timepicker": {},
    "timezone": "browser",
    "title": "消防智能执法 · 服务可观测总览",
    "uid": "fire-agents-overview",
    "version": 1,
    "weekStart": "",
}

# ---- 校验：Grafana 对无效 JSON 只会静默不显示，必须提前验 ----
try:
    text = json.dumps(dashboard, ensure_ascii=False, indent=2)
except (TypeError, ValueError) as e:
    print(f"✗ 看板 JSON 序列化失败：{e}")
    sys.exit(1)

path = OUT / "fire-agents-overview.json"
path.write_text(text, encoding="utf-8")

# 回读校验
back = json.loads(path.read_text(encoding="utf-8"))
n_panels = sum(1 for x in back["panels"] if x["type"] != "row")
n_rows = sum(1 for x in back["panels"] if x["type"] == "row")
print(f"✓ 已生成 {path}")
print(f"  文件大小 {path.stat().st_size / 1024:.1f} KB")
print(f"  面板 {n_panels} 个，分组 {n_rows} 行")
print(f"  uid: {back['uid']}")
print()
print("导入方式（二选一）：")
print("  A. 自动：已配置 provisioning，容器起来后自动出现在「消防智能执法」目录")
print("  B. 手动：Grafana → Dashboards → New → Import → 上传本文件")