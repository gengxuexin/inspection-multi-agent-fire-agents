"""加载 MCP Server 模块并检查工具是否可用（不启动服务）。

单独一个脚本是因为内联 python -c 的引号在 Git Bash → wsl 链路里
被吃掉太多次了，写文件最稳。
"""
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]   # deploy/tools -> 根
SERVER = ROOT / "mcp" / "change_detector_server.py"

if not SERVER.is_file():
    print(f"找不到 {SERVER}")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("srv", SERVER)
m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m)
except Exception as e:
    print(f"加载失败：{type(e).__name__}: {e}")
    sys.exit(1)

print("模块加载 OK")
print(f"  服务名     : {m.server.name}")
print(f"  报告目录: {m.REVIEWS_DIR}")
print(f"  目录存在   : {m.REVIEWS_DIR.is_dir()}")

# change_detector 的 API 是否符合预期
cd = m.cd
for fn in ("analyze_changes", "safe_filename"):
    print(f"  cd.{fn:16s}: {'有' if hasattr(cd, fn) else '★缺失'}")

# 直接调工具函数，确认能返回结构化结果
print()
print("=== 调用 list_reports() ===")
try:
    r = m.list_reports(limit=5)
    print(f"  status={r.get('status')} count={r.get('count')}")
    for it in r.get("reports", [])[:3]:
        print(f"    {it.get('report_id')} {it.get('created_at')} "
              f"改动={it.get('total_changes')} 高危={it.get('high_risk_count')}")
except Exception as e:
    print(f"  调用失败：{type(e).__name__}: {e}")

print()
print("=== 调用 compare_documents(错误路径，应返回结构化错误而非抛异常) ===")
try:
    r = m.compare_documents("/nonexistent/a.docx", "/nonexistent/b.docx")
    print(f"  {r}")
except Exception as e:
    print(f"  ★ 直接抛异常了（客户端会看到笼统的失败）：{type(e).__name__}: {e}")

print()
print("=== 异步工具列表（MCP 协议实际暴露的工具）===")
try:
    tools = m.server.list_tools()
    if hasattr(tools, "__await__"):
        import asyncio
        tools = asyncio.run(tools)
    for t in tools:
        print(f"  {t.name:20s} {t.description[:60] if t.description else ''}")
except Exception as e:
    print(f"  列取失败：{type(e).__name__}: {e}")