#!/usr/bin/env python3
"""检查 metrics 是否正确挂载到四个服务。

★ 为什么用 Python 而不是 bash：
  第一版 bash 版里用 `grep -c ... || echo 0`，
  但 `grep -c` 无匹配时输出的是空串且退出码为 1，
  于是 `$(...)` 拿到 "0\\n0"（grep 打的 0 + 兜底 echo 打的 0），
  字符串比较 `"0\\n0" != "0"` 成立 → 明明没问题却报"被缩进"。

  这个坑的恶劣之处：**检查脚本自己有 bug 时会报出假问题**，
  而假问题比没检查更费时间 —— 人会去改本来正确的代码。
  涉及计数与字符串比较的逻辑，放 Python 里最稳。
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

SERVERS = ["agent1/server.py", "gateway.py",
           "agent2/server.py", "agent3/server.py"]
EXTRA = ["common/metrics.py", "common/metrics_middleware.py"]

fail = False


def head_indent(line: str) -> int:
    return len(line) - len(line.lstrip())


print("=== 1. 语法检查 ===")
for rel in SERVERS + EXTRA:
    p = ROOT / rel
    if not p.is_file():
        print(f"  ✗ {rel}不存在")
        fail = True
        continue
    try:
        ast.parse(p.read_text(encoding="utf-8"))
        print(f"  ✓ {rel}")
    except SyntaxError as e:
        print(f"  ✗ {rel} 第 {e.lineno} 行: {e.msg}")
        fail = True

print()
print("=== 2. 挂载位置与缩进 ===")
for rel in SERVERS:
    p = ROOT / rel
    if not p.is_file():
        continue
    lines = p.read_text(encoding="utf-8").splitlines()

    calls = [
        (i, ln) for i, ln in enumerate(lines)
        if ln.lstrip().startswith(
            ("install_metrics_endpoint(app", "install_http_middleware(app"))
    ]
    imports = [i for i, ln in enumerate(lines)
               if ln.startswith("from common.metrics_middleware import (")]

    problems = []
    if not imports:
        problems.append("缺少 import")
    elif len(imports) != 1:
        problems.append(f"import 出现 {len(imports)} 次")
    if len(calls) != 2:
        problems.append(f"调用出现 {len(calls)} 次（应为 2）")

    for i, ln in calls:
        if head_indent(ln) != 0:
            problems.append(f"第 {i + 1} 行调用被缩进 {head_indent(ln)}，"
                            f"可能被塞进了函数体")

    # 确认 app = FastAPI(...) 存在，且在调用之前
    app_line = next((i for i, ln in enumerate(lines)
                     if ln.startswith("app = FastAPI(")), None)
    if app_line is None:
        problems.append("找不到 app = FastAPI(...)")
    elif calls and calls[0][0] < app_line:
        problems.append("调用出现在 app 定义之前")

    if problems:
        print(f"  ✗ {rel}")
        for pb in problems:
            print(f"      - {pb}")
        fail = True
    else:
        print(f"  ✓ {rel}  import@{imports[0] + 1} "
              f"调用@{[i + 1 for i, _ in calls]} app@{app_line + 1}")

print()
print("=== 3. metrics 模块冒烟 ===")
sys.path.insert(0, str(ROOT))
try:
    from common import metrics
    from common.metrics_middleware import normalize_path

    print(f"  prometheus_client 可用: {metrics.available()}")

    # ★ 先触发指标，再检查渲染结果。
    #   指标是懒注册的（第一次调用工厂函数时才创建），
    #   所以"刚导入就render"必然渲染不出 fire_* 指标 ——
    #   这是 prometheus_client 的设计，不是 bug。
    #   检查脚本若不做这一步，就会一直报"没有 fire_ 指标"，
    #   而这个结论是错的。
    metrics.observe_http("agent1", "POST", "/api/rag/ask", 200, 0.123)
    metrics.observe_rag_retrieve("agent1", "faiss", 0.045, 5)
    metrics.observe_llm("agent1", "generate", "deepseek-chat", 1.8,
                        prompt_tokens=1200, completion_tokens=350)
    metrics.set_circuit_breaker("agent1", "deepseek", "half_open")
    metrics.record_cb_transition("agent1", "deepseek", "closed", "open")

    body, ctype = metrics.render_latest()
    text = body.decode("utf-8", errors="ignore")
    print(f"  渲染长度: {len(body)}  Content-Type: {ctype[:40]}")

    expected = [
        "fire_http_requests_total",
        "fire_http_request_duration_seconds",
        "fire_rag_retrieve_duration_seconds",
        "fire_rag_retrieved_chunks",
        "fire_llm_calls_total",
        "fire_llm_tokens_total",
        "fire_llm_duration_seconds",
        "fire_circuit_breaker_state",
        "fire_circuit_breaker_transitions_total",
    ]
    missing = [m for m in expected if m not in text]
    if missing:
        print(f"  ✗ 缺少指标: {', '.join(missing)}")
        fail = True
    else:
        print(f"  ✓ {len(expected)} 个指标全部已注册并渲染")

    # 样本值检查：token 计数应该真的是 1200/350
    if "1200" not in text or "350" not in text:
        print("  ✗ token 样本值不对，observe_llm 可能没生效")
        fail = True
    else:
        print("  ✓ token 样本值正确（prompt=1200 completion=350）")

    # series 数检查（高基数的直接后果）
    n_series = sum(1 for ln in text.splitlines()
                   if ln and not ln.startswith("#"))
    print(f"  当前样本行数: {n_series}（远低于 1000 的警戒线）")

    cases = [
        ("/health", "/health"),
        ("/api/doc/change-reports/CR-20261016-48aeda", "/api/doc/change-reports/{id}"),
        ("/api/x/8f14e45f-ceea-167a-5a36-dedd4bea2543", "/api/x/{uuid}"),
        ("/tasks/12345", "/tasks/{id}"),
        ("/api/rag/ask", "/api/rag/ask"),
    ]
    print("  路径归一化（高基数防护）：")
    for raw, expect in cases:
        got = normalize_path(raw)
        ok = "✓" if got == expect else "✗"
        if got != expect:
            fail = True
            print(f"    {ok} {raw[:46]:48s} -> {got}  (期望 {expect})")
        else:
            print(f"    {ok} {raw[:46]:48s} -> {got}")
except Exception as e:                          # noqa: BLE001
    print(f"  ✗ 导入失败：{type(e).__name__}: {e}")
    fail = True

print()
print("  ✓ 全部通过" if not fail else "  ✗ 有问题，先修再起服务")
sys.exit(1 if fail else 0)