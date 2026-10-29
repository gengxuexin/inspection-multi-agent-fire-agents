"""依赖审计：找出 requirements.txt 里声明但代码未 import 的包，以及代码 import 但未声明的包。

关键修正点（对比旧版 _audit.sh）：
  包名归一化 pydantic_settings -> pydantic-settings、rank_bm25 -> rank-bm25，
  再做前缀匹配。否则 grep "^pydantic_settings" 永远匹配不到 "pydantic-settings>=2.2"，
  会同时误报"缺依赖"和"未使用"。
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()


def norm(name: str) -> str:
    """把 import 名归一化成 PyPI 名。"""
    n = name.lower().strip()
    n = n.split("[")[0].split("=")[0].split(">")[0].split("<")[0].split("!")[0].split(";")[0]
    n = n.replace("_", "-")
    return n


def read_reqs(path: pathlib.Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9._\-]+)", line)
        if m:
            out[norm(m.group(1))] = line
    return out


# ---------- 收集源码里的 import ----------
IMPORT_TO_PKG = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "fitz": "pymupdf",
    "jwt": "pyjwt",
    "dotenv": "python-dotenv",
    "serial": "pyserial",
    "win32com": "pywin32",
    "rank_bm25": "rank-bm25",
    "pydantic_settings": "pydantic-settings",
    "faiss": "faiss-cpu",
    "torch": "torch",
    "whisper": "faster-whisper",
    "ultralytics": "ultralytics",
    "opentelemetry": "opentelemetry-api",
    "langchain_core": "langchain-core",
    "langchain_community": "langchain-community",
    "langchain_openai": "langchain-openai",
    "langgraph": "langgraph",
}

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    "index", "data", "output", "logs", "eval_runs",
}

stdlib_hint = set(sys.stdlib_module_names)

imports: dict[str, list[str]] = {}   # top-level module -> files
for py in ROOT.rglob("*.py"):
    if any(p in SKIP_DIRS for p in py.parts):
        continue
    try:
        tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        continue
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
    for m in mods:
        if m in stdlib_hint or m == "__future__":
            continue
        imports.setdefault(m, []).append(str(py.relative_to(ROOT)))

# 展开 import 名 -> PyPI 名
used: dict[str, list[str]] = {}
for mod, files in imports.items():
    pkg = IMPORT_TO_PKG.get(mod, norm(mod))
    used.setdefault(pkg, [])
    for f in files:
        if f not in used[pkg]:
            used[pkg].append(f)

declared = read_reqs(ROOT / "requirements.txt")

# ---------- 对比 ----------
def match(pkg: str, table: dict[str, str]) -> str | None:
    if pkg in table:
        return table[pkg]
    for k in table:
        if pkg.startswith(k + "-") or k.startswith(pkg + "-"):
            return table[k]
    return None


# ---------- 过滤掉项目内部模块 ----------
# 项目里 `from common import x` / `import chunking` 这类本地模块会被当成第三方包，
# 必须先排除，否则"缺依赖"列表会被几十个自家文件名淹没，真正的问题反而看不见。
LOCAL_ROOTS = {p.name for p in ROOT.iterdir() if p.is_dir()} - SKIP_DIRS
# 凡是带 __init__.py 的目录就是一个真实的 Python 包，它的名字绝不可能是第三方依赖。
# 这比"看目录名在不在顶层"可靠：agent2/asr/、agent2/tts/ 这些嵌套包也能覆盖到。
LOCAL_PKGS = {p.parent.name for p in ROOT.rglob("__init__.py")
              if p.parent.name not in SKIP_DIRS}
LOCAL_PY = {p.stem for p in ROOT.glob("*.py")}
LOCAL_PY |= {p.stem for p in ROOT.rglob("*.py")
             if p.parent.name in LOCAL_ROOTS or p.parent.parent.name in LOCAL_ROOTS}
# ★ 关键：LOCAL_PY 也必须走norm()，否则 stem "case_pipeline" 与used 里的
#   "case-pipeline"（norm 把下划线换成连字符）比不上，本地模块会全部漏进第三方列表。
LOCAL_PY_N = {norm(x) for x in LOCAL_PY} | {norm(x) for x in LOCAL_PKGS}

third_party: dict[str, list[str]] = {
    pkg: files for pkg, files in used.items() if pkg not in LOCAL_PY_N
}

missing = sorted(p for p in third_party
                 if p not in declared and match(p, declared) is None)
unused = sorted(p for p in declared if p not in third_party and not any(
    u == p or u.startswith(p + "-") or p.startswith(u + "-") for u in third_party))

print(f"声明 {len(declared)} / 第三方 import {len(third_party)} "
      f"/ 已排除本地模块 {len(used) - len(third_party)}")
print()
print("=== [缺依赖] 代码 import 了但 requirements.txt 没声明 ===")
for p in missing:
    files = third_party[p]
    print(f"  {p:24s} <- {', '.join(files[:3])}{' ...' if len(files) > 3 else ''}")
print()
print("=== [未使用] requirements.txt 声明了但代码没import ===")
for p in unused:
    print(f"  {p:24s} <- {declared[p]}")