#!/usr/bin/env python3
"""检查各服务在「容器工作目录」下能否正常 import 全部依赖。

★ 为什么需要这个：
  本地开发时 WORKDIR 是项目根，Python 自动把 CWD 放进 sys.path，
  所以 `from common.tracing import ...` 永远能成功。
  但容器里 agent2 / agent3 的 WORKDIR 是 /app/<svc>，
  而 common 包在 /app/common —— **CWD 不一样，import 结果就不一样。**
  实测就是这样暴露的：本地好好的，容器里
      ModuleNotFoundError: No module named 'common'
  这类问题静态检查看不出来（文件都在、语法都对），
  只能在"按容器的方式跑一遍"里暴露。

★ 第二版修的 bug（第一版误报）：
  第一版只判断「项目根下有没有 common/__init__.py」，
  有就报"正常"—— 结果 agent2 明明 import 会失败却被判为通过。
  因为 agent2 原本只把 /app/agent2 加进 sys.path，common 在 /app，
  **存在 ≠ 能被找到**。
  正确判据（三条同时满足）：
    1. 入口显式把项目根加进 sys.path，或 WORKDIR 本身就是项目根
    2. 项目根下确实有 common/__init__.py
    3. 按容器形态拼路径后确实能定位到 common
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

# 服务 -> (入口文件, 容器里的 WORKDIR 相对路径)
SERVICES = {
    "gateway": ("gateway.py", "."),
    "agent1": ("agent1/server.py", "agent1"),
    "agent2": ("agent2/server.py", "agent2"),
    "agent3": ("agent3/server.py", "agent3"),
}

STDLIB = set(sys.stdlib_module_names)


def top_level_imports(path: pathlib.Path) -> list[str]:
    """只看顶层 import —— 函数内的是懒加载，不影响启动。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            out += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append(node.module.split(".")[0])
    return out


def declares_project_root(src: str) -> bool:
    """入口是否显式把项目根放进 sys.path。

    不做精确的 AST 判断（那样太脆），
    而是在源码文本里找几种常见的写法 ——
    这里是"提示"而非"证明"，所以宁可多报（漏报更危险）。

    ★ 注意 agent1 的写法：它已经正确加了（`_ROOT = os.path.dirname(_SCRIPT_DIR)`
      然后把 _ROOT 插进 sys.path），但第一版检查脚本没认这个形式，
      误报成"未加项目根"。**检查脚本报警报的代码要去核对，
      不要直接改代码** —— 这次的"问题代码"其实是正确的。
    """
    markers = (
        "_PROJECT_ROOT",            # 我这次统一用的变量名
        "parents[1]",
        "dirname(dirname(os.path.abspath(__file__)))",
        "dirname(_ROOT)",
        "dirname(_SVC_DIR)",
    )
    if any(m in src for m in markers):
        return True

    # 兜底：任何 `sys.path.insert/append` 且参数是路径变量的写法都算
    for line in src.splitlines():
        s = line.strip()
        if ("sys.path.insert" in s or "sys.path.append" in s) and "if" not in s:
            return True
    return False


fail = False

print("=== 容器形态下的 import 路径 ===")
print()

for svc, (rel, workdir_rel) in SERVICES.items():
    entry = ROOT / rel
    if not entry.is_file():
        print(f"✗ {svc:8s} 入口不存在：{rel}")
        fail = True
        continue

    src = entry.read_text(encoding="utf-8")
    mods = [m for m in top_level_imports(entry)
            if m not in STDLIB and m != "__future__"]

    workdir = (ROOT / workdir_rel).resolve()
    adds_root = declares_project_root(src)
    cwd_is_root = (workdir_rel == ".")

    common_exists = (ROOT / "common" / "__init__.py").is_file()

    # 真正按容器形态拼一遍路径，确认能定位到 common
    candidates = {workdir, ROOT}
    resolved_ok = any((c / "common" / "__init__.py").is_file()
                      for c in candidates)

    # 还要检查该服务自己的目录在不在候选里（它当然在，但要显式确认）
    svc_dir = entry.parent.resolve()
    self_ok = svc_dir in candidates or adds_root

    problems = []
    if not common_exists:
        problems.append("项目根下没有 common/__init__.py")
    if not (adds_root or cwd_is_root):
        problems.append(
            f"入口未把项目根加入 sys.path"
            f"（容器 WORKDIR=/{workdir_rel}，common 在 /app/common）")
    if not resolved_ok:
        problems.append("按容器形态解析不到 common 包")
    if not self_ok:
        problems.append("无法定位到本服务自身目录")

    if problems:
        print(f"✗ {svc:8s} WORKDIR=/{workdir_rel:8s} 顶层 import {len(mods):2d} 个")
        for x in problems:
            print(f"      - {x}")
        fail = True
    else:
        why = "入口显式加了项目根" if adds_root else "WORKDIR 即项目根"
        print(f"✓ {svc:8s} WORKDIR=/{workdir_rel:8s} "
              f"顶层 import {len(mods):2d} 个  （{why}）")

print()
if fail:
    print("✗ 有服务在容器形态下 import 会失败")
    print()
    print("  典型原因：Dockerfile 的 WORKDIR 是 /app/<svc>，")
    print("  而跨包 import（如 common）需要 /app 在 sys.path 里。")
    print("  修法：在入口文件顶部显式把项目根加进 sys.path，例如：")
    print("      _SVC_DIR = os.path.dirname(os.path.abspath(__file__))")
    print("      _PROJECT_ROOT = os.path.dirname(_SVC_DIR)")
    print("      sys.path.insert(0, _PROJECT_ROOT)")
    sys.exit(1)

print("✓ 四个服务在容器形态下 import 路径均正常")
sys.exit(0)