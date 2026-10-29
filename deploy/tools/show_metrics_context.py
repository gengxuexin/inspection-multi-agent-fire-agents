"""打印 metrics 插入点的上下文（含缩进可视化），用于人工核对。"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

for rel in ("agent1/server.py", "gateway.py",
            "agent2/server.py", "agent3/server.py"):
    p = ROOT / rel
    if not p.is_file():
        continue
    lines = p.read_text(encoding="utf-8").splitlines()
    print("=" * 68)
    print(rel)
    print("=" * 68)

    marks = [i for i, ln in enumerate(lines)
             if "metrics_middleware import" in ln
             or "install_metrics_endpoint" in ln
             or "install_http_middleware" in ln]

    if not marks:
        print("  (未找到 metrics 相关行)")
        print()
        continue

    lo = max(0, min(marks) - 4)
    hi = min(len(lines), max(marks) + 3)
    for i in range(lo, hi):
        ln = lines[i]
        lead = len(ln) - len(ln.lstrip())
        bar = "|" if "|" else ""
        flag = ">>" if i in marks else "  "
        print(f"  {flag} {i + 1:4d} [{lead:2d}] {ln}")
    print()