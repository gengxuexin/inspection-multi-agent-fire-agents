"""对 agent2 / agent3 / gateway 做与 agent1 相同的启动链依赖审计。

agent1 已经实测过启动即崩（缺 pdfplumber）。agent2/3 的 Dockerfile
改动后还没构建完，先用同样的方法静态确认一遍，避免又花十几分钟
构建出一个起不来的镜像。
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(".").resolve()
STDLIB = set(sys.stdlib_module_names)
SKIP = {"__pycache__", ".venv", "venv", "knowledge_base", "faiss_full",
        "faiss_multi", "faiss_single", "output", "outputs", "node_modules", ".git"}

IMPORT_TO_PKG = {"docx": "python-docx", "PIL": "pillow", "fitz": "pymupdf",
                 "jwt": "pyjwt", "yaml": "pyyaml", "cv2": "opencv-python",
                 "sklearn": "scikit-learn", "dotenv": "python-dotenv"}

# 各服务自己的重依赖文件
EXTRA_REQ = {
    "agent2": "requirements-agent2.txt",
    "agent3": "requirements-agent3.txt",
    "gateway": None,
    "agent1": None,
}


def norm(n: str) -> str:
    return re.split(r"[><=\[]", n.strip())[0].lower().replace("_", "-")


def read_reqs(path: pathlib.Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            out.add(norm(line))
    return out


def local_names() -> set[str]:
    names = {p.parent.name for p in ROOT.rglob("__init__.py") if p.parent.name not in SKIP}
    names |= {p.stem for p in ROOT.rglob("*.py")
              if not any(part in SKIP for part in p.parts)}
    return {norm(n) for n in names} | set(names) | {n.replace("-", "_") for n in names}


LOCAL = local_names()
base_reqs = read_reqs(ROOT / "requirements.txt")

for svc in ("gateway", "agent2", "agent3"):
    entry = ROOT / (f"{svc}.py" if svc == "gateway" else f"{svc}/server.py")
    if not entry.exists():
        print(f"[{svc}] 入口不存在: {entry}")
        continue

    declared = set(base_reqs)
    extra = EXTRA_REQ.get(svc)
    if extra:
        declared |= read_reqs(ROOT / extra)

    seen: set[str] = set()
    third: dict[str, str] = {}
    queue = [entry]

    while queue:
        f = queue.pop(0)
        if str(f) in seen:
            continue
        seen.add(str(f))
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, FileNotFoundError):
            continue
        mods: list[str] = []
        for node in tree.body:          # 只看顶层 import
            if isinstance(node, ast.Import):
                mods += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.append(node.module.split(".")[0])
        for m in mods:
            if m in STDLIB or m == "__future__":
                continue
            if m in LOCAL or norm(m) in LOCAL:
                for cand in (f.parent / f"{m}.py", f.parent / m / "__init__.py"):
                    if cand.exists():
                        queue.append(cand)
                        break
                continue
            third.setdefault(m, f.relative_to(ROOT).as_posix())

    miss = sorted(m for m in third
                  if IMPORT_TO_PKG.get(m.lower(), norm(m)) not in declared)
    print(f"[{svc}] 顶层第三方 {len(third)} 个，缺失 {len(miss)} 个")
    for m in miss:
        print(f"    缺 {m:20s} <- {third[m]}")
    if not miss:
        print("    ✓ 无缺失")
    print()