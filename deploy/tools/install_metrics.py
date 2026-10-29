"""给四个服务挂 /metrics 与 HTTP 埋点中间件。

★ 第一版按文本锚点插入，结果四个服务全被插坏：
    - import 被插进了 `try:` 块中间 → SyntaxError
    - 调用被插进了函数体内 → IndentationError
  原因是"最后一行 import"、"setup_tracing(app) 之后"这类锚点
  在真实文件里并不可靠：try/except 块、函数内部的同名调用都会骗过它。

第二版改用 AST 定位：
  - import 插到**第一个顶层语句是 import 的位置**之后，
    并跳过 try/except 内部（那是控制流，不是模块级 import）
  - 调用插到 `app = FastAPI(...)` 这个赋值语句**之后**，
    用 AST 的 end_lineno 精确拿到语句结束行，
    这样无论 FastAPI(...) 是单行还是多行都对
这样定位与文件实际结构无关，不依赖文本猜测。
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

TARGETS = [
    ("agent1", "agent1/server.py"),
    ("gateway", "gateway.py"),
    ("agent2", "agent2/server.py"),
    ("agent3", "agent3/server.py"),
]

IMPORT_BLOCK = (
    "from common.metrics_middleware import (\n"
    "    install_http_middleware,\n"
    "    install_metrics_endpoint,\n"
    ")\n"
)


def call_block(service: str) -> str:
    return (
        f"\n# ---- Prometheus 指标（指标清单见 common/metrics.py）----\n"
        f"install_metrics_endpoint(app, service=\"{service}\")\n"
        f"install_http_middleware(app, service=\"{service}\")\n"
    )


def top_level_insert_points(tree: ast.Module):
    """找出模块级的 import 语句与 app 赋值语句位置。

    只看 tree.body（即模块级直接子节点），
    try/except 里的 import 属于 If.body，不在这里面 ——
    这正是第一版插错的地方。
    """
    last_import_end = None
    app_end = None

    for node in tree.body:
        # 模块级 import：Import / ImportFrom / 也允许 try 顶层的 import
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_end = node.end_lineno
        # app = FastAPI(...)
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets
                       if isinstance(t, ast.Name)]
            if "app" in targets and isinstance(node.value, ast.Call):
                fn = node.value.func
                name = getattr(fn, "attr", getattr(fn, "id", ""))
                if name == "FastAPI":
                    app_end = node.end_lineno

    return last_import_end, app_end


for service, rel in TARGETS:
    p = ROOT / rel
    if not p.is_file():
        print(f"[{service}] 跳过：{rel} 不存在")
        continue

    src = p.read_text(encoding="utf-8")

    if "install_metrics_endpoint" in src:
        print(f"[{service}] 已埋点，跳过")
        continue

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"[{service}] ★ 原文件语法就坏了（line {e.lineno}），跳过")
        continue

    imp_end, app_end = top_level_insert_points(tree)
    if imp_end is None:
        print(f"[{service}] ★ 找不到模块级 import 锚点")
        continue
    if app_end is None:
        print(f"[{service}] ★ 找不到 app = FastAPI(...) 语句")
        continue

    lines = src.splitlines(keepends=True)

    # 插调用（在 app 赋值之后），再插 import（在 import 之后）。
    # ★ 顺序很重要：从后往前插，前面插入不会让后面的行号失效。
    lines.insert(app_end, call_block(service))
    lines.insert(imp_end, IMPORT_BLOCK + "\n")

    new_src = "".join(lines)
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        print(f"[{service}] ★ 插入后语法错误（line {e.lineno}），已放弃")
        continue

    p.write_text(new_src, encoding="utf-8")
    print(f"[{service}] {rel} 已加 /metrics 与中间件"
          f"（import 在第 {imp_end} 行后，调用在第 {app_end} 行后）")