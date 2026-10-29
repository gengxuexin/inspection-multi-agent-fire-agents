"""追踪 agent1 的启动导入链，找出「容器启动即崩」的缺失依赖。

背景：server.py 顶层直接 import 了 pdf_converter / case_pipeline 等模块，
这些模块又在顶层 import 第三方库。任何一环缺失，容器都是启动即崩，
而且是 ImportError 这种最难看的方式 —— 服务根本起不来。
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(".").resolve()
STDLIB = set(sys.stdlib_module_names)
LOCAL_MAP = {"config", "common", "hybrid", "rag_module", "chunking", "schemas",
             "document_io", "templates", "ui", "eval"}

SKIP = {"__pycache__", ".venv", "venv", "knowledge_base", "faiss_full",
        "faiss_multi", "faiss_single", "outputs", "node_modules", ".git"}


def local_pkg_names() -> set[str]:
    """项目内所有可作为 import 目标的模块名。

    注意不能只 glob 顶层 *.py —— agent1/agent2/agent3 下的模块同样会被
    `from case_pipeline import ...` 这样直接导入，必须一起收进来，
    否则它们会被误判成第三方包，报告出一堆「缺依赖」的假警报。
    """
    names = {p.parent.name for p in ROOT.rglob("__init__.py") if p.parent.name not in SKIP}
    names |= {p.stem for p in ROOT.rglob("*.py")
              if not any(part in SKIP for part in p.parts)}
    return ({n.replace("_", "-") for n in names} | set(names)
            | {n.replace("-", "_") for n in names})


LOCAL = local_pkg_names()
ENTRY = ROOT / "agent1" / "server.py"

seen: set[str] = set()
third_party: dict[str, str] = {}   # pkg -> 首次出现的文件
local_chain: dict[str, str] = {}   # module -> importer

queue = [ENTRY]
while queue:
    f = queue.pop(0)
    key = str(f)
    if key in seen:
        continue
    seen.add(key)
    try:
        tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, FileNotFoundError):
        continue

    mods: list[str] = []
    for node in tree.body:              # 只看顶层 import，函数内的是懒加载
        if isinstance(node, ast.Import):
            mods += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.append(node.module.split(".")[0])

    for m in mods:
        if m in STDLIB or m == "__future__":
            continue
        if m in LOCAL or m.replace("_", "-") in LOCAL:
            # 本地模块：继续往下钻
            local_chain.setdefault(m, f.relative_to(ROOT).as_posix())
            for cand in (f.parent / f"{m}.py", f.parent / m / "__init__.py"):
                if cand.exists():
                    queue.append(cand)
                    break
            continue
        third_party.setdefault(m, f.relative_to(ROOT).as_posix())

# 读 requirements
#注意两件事，否则会误报「缺依赖」：
#   1) uvicorn[standard] 这类extras 方括号要剥掉，否则包名变成 "uvicorn[standard]" 匹配不上；
#   2) 导入名与 PyPI 名不一致的必须映射（import docx -> python-docx）。
IMPORT_TO_PKG = {"docx": "python-docx", "PIL": "pillow", "fitz": "pymupdf",
                 "jwt": "pyjwt", "yaml": "pyyaml", "cv2": "opencv-python"}
declared = set()
for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
    line = line.split("#")[0].strip()
    if not line or line.startswith("-"):
        continue
    name = re.split(r"[><=\[]", line)[0].strip().lower().replace("_", "-")
    declared.add(name)

print("=== agent1 启动链上的第三方 import ===")
for m, where in sorted(third_party.items()):
    print(f"  {m:22s} <- {where}")
print()
print("=== 启动链上的本地模块（会继续递归）===")
for m, where in sorted(local_chain.items()):
    print(f"  {m:22s} <- {where}")
print()
def pypi(mod: str) -> str:
    m = mod.lower().replace("_", "-")
    return IMPORT_TO_PKG.get(mod.lower(), m)


miss = [m for m in sorted(third_party) if pypi(m) not in declared]
print(f"=== 启动链缺失依赖（{len(miss)} 个）—— 这些缺失会导致容器启动即崩 ===")
for m in miss:
    print(f"  {m:22s} <- {third_party[m]}")