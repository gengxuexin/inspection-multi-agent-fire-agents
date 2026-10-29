"""精确回滚 install_metrics.py 对四个服务做的错误插入。

为什么要精确回滚而不是 git checkout：
  这四个文件在本轮之前已经有改动（新增 /health 与 /ready 端点），
  git checkout 会一并丢掉那些有效改动。

回滚方式：删掉插入的两段内容
  1) import 块
  2) install_* 调用块（含注释行）
只按内容匹配删除，不动其他任何字节。
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

FILES = ["agent1/server.py", "gateway.py",
         "agent2/server.py", "agent3/server.py"]

IMPORT_BLOCK = re.compile(
    r"from common\.metrics_middleware import \(\n"
    r"    install_http_middleware,\n"
    r"    install_metrics_endpoint,\n"
    r"\)\n"
)

CALL_BLOCK = re.compile(
    r"\n*# ---- Prometheus 指标（见 common/metrics\.py 的指标清单）----\n"
    r"install_metrics_endpoint\(app, service=\"[a-z0-9]+\"\)\n"
    r"install_http_middleware\(app, service=\"[a-z0-9]+\"\)\n"
)

for rel in FILES:
    p = ROOT / rel
    if not p.is_file():
        print(f"{rel}: 不存在")
        continue
    src = p.read_text(encoding="utf-8")
    orig = src

    src = IMPORT_BLOCK.sub("", src)
    src = CALL_BLOCK.sub("", src)
    # 清掉插入时可能留下的多余空行（连续 3 个换行压成 2 个）
    src = re.sub(r"\n{3,}(?=(if __name__|# ==|@app))", "\n\n", src)

    if src != orig:
        p.write_text(src, encoding="utf-8")
        import ast
        try:
            ast.parse(src)
            print(f"✓ {rel} 已回滚，语法正常")
        except SyntaxError as e:
            print(f"✗ {rel} 回滚后仍有语法错误：{e}")
            for ln in range(max(0, e.lineno - 3), e.lineno + 2):
                lines = src.splitlines()
                if 0 <= ln - 1 < len(lines):
                    print(f"    {ln}: {lines[ln - 1]}")
    else:
        print(f"· {rel} 未含插入内容，无需回滚")