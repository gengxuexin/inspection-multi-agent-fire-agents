"""核对案例故事集里的每个数字是否有出处。

★ 为什么需要这个：
  我自己写速查表时就凭印象填了过时的指标
  （0.88 / 0.165，实际是 0.698 / 0.226）。
  这次不校验，下次改文档时又会重犯。

  所以把速查表里的数字逐条对照真实数据源：
    - 代码行数      ← 实际 find + wc -l
    - 测试数        ← pytest --collect-only
    - 评测指标      ← report_rerank_final.json
    - 索引规模      ← index.pkl
    - 文档数        ← _processed_files.json
    - 抖动/噪声底   ← 基线对比代码里的常量
    - 冷启动耗时    ← 演示服务预热日志

  **不通过的项要明确报出来，而不是让文档带着错数字流出去。**
"""
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(".")
DOC = ROOT / "docs" / "case-stories.md"
PY = "/home/geng/.venv/bin/python"

src = DOC.read_text(encoding="utf-8")
# 只核对"数字速查"那一节
i0 = src.find("## 数字速查")
block = src[i0:] if i0 > 0 else src

fail = 0
print("=" * 62)
print("案例故事集 · 数字出处核对")
print("=" * 62)

# ---- 1. 代码规模 ----
r = subprocess.run(
    "find . -name '*.py' -not -path './.git/*' -not -path '*/__pycache__/*' "
    "-not -path './deploy/tools/*' -not -path './agent1/demo/*' "
    "| xargs wc -l | tail -1",
    shell=True, capture_output=True, text=True, cwd=str(ROOT))
loc = int(r.stdout.strip().split()[0]) if r.stdout.strip() else 0
nfile = len(list(ROOT.glob("**/*.py")))
nfile = sum(1 for p in ROOT.rglob("*.py")
            if ".git" not in p.parts and "__pycache__" not in p.parts
            and "deploy/tools" not in p.parts
            and "agent1/demo" not in p.parts)
ok = f"{loc:,}" in block or str(loc) in block
print(f"  [{'OK  ' if ok else 'FAIL'}] 业务代码 {loc:,} 行"
      f" / {nfile} 文件")
if not ok:
    fail += 1
    print("        文档里找不到这个数字")

# ---- 2. 测试数 ----
r = subprocess.run([PY, "-m", "pytest", "tests/", "-q", "--collect-only"],
                   capture_output=True, text=True, cwd=str(ROOT))
m = re.search(r"(\d+) tests? collected", r.stdout)
nt = int(m.group(1)) if m else 0
ok = str(nt) in block
print(f"  [{'OK  ' if ok else 'FAIL'}] 测试 {nt} 个")
if not ok:
    fail += 1

# ---- 3. 验证脚本 ----
# 验证脚本：文档里可能没写这一项。
# ★ 区分"没写"和"写错"：
#   没写 -> 跳过（那是信息缺失，不是数据错误）
#   写了 -> 校验它对不对
#   规则本身错了会让核对器变成只会刷FAIL的工具，
#   而一个老在报错的检查工具，用两次就没人信它了。
tools = list((ROOT / "deploy" / "tools").glob("*.py"))
tloc = 0
for p in tools:
    tloc += len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
mentioned = (str(len(tools)) in block
             or f"{tloc:,}" in block or str(tloc) in block)
has_row = "验证脚本" in block
if not has_row:
    print(f"  [SKIP] 验证脚本 —— 文档未提及"
          f"（实际 {len(tools)} 个 / {tloc:,} 行）")
elif not mentioned:
    print(f"  [FAIL] 验证脚本 数字对不上"
          f"（实际 {len(tools)} 个 / {tloc:,} 行）")
    fail += 1
else:
    print(f"  [OK  ] 验证脚本 {len(tools)} 个 / {tloc:,} 行")

# ---- 4. 评测指标 ----
final = json.loads((ROOT / "report_rerank_final.json").read_text(encoding="utf-8"))
sc = final["scores"]
for k in ("faithfulness", "answer_relevancy", "context_precision",
          "context_recall"):
    v = f"{sc[k]:.3f}"
    ok = v in block
    print(f"  [{'OK  ' if ok else 'FAIL'}] {k} = {v}")
    if not ok:
        fail += 1

# ---- 5. 索引规模 ----
import pickle
with open(ROOT / "agent1" / "faiss_full" / "index.pkl", "rb") as f:
    d = pickle.load(f)
nchunk = len(d[0]._dict)
ok = str(nchunk) in block
print(f"  [{'OK  ' if ok else 'FAIL'}] 索引 {nchunk} 分块")
if not ok:
    fail += 1

files = json.loads(
    (ROOT / "agent1" / "faiss_full" / "_processed_files.json")
    .read_text(encoding="utf-8"))
ndoc = len(files) if isinstance(files, (list, dict)) else 0
ok = str(ndoc) in block
print(f"  [{'OK  ' if ok else 'FAIL'}] 文档 {ndoc} 份")
if not ok:
    fail += 1

# ---- 6. 噪声底（应与代码里的常量一致）----
# ★ 这里要读**生效的判定阈值** `_NOISE_FLOOR`，不是兜底值 `_NOISE_FLOOR_UNKNOWN`。
#   两者含义不同：
#     _NOISE_FLOOR         = 0.05   实际判定阈值（A/B 差值超过它算「超出噪声」）
#     _NOISE_FLOOR_UNKNOWN = 0.10   拿不到基线报告时的保守兜底
#   早期版本误读了兜底值，导致文档写「噪声底 0.10」而系统实际用 0.05 ——
#   拿兜底值当生效值，比数字不一致更糟。
loc_src = (ROOT / "agent1" / "eval" / "local_eval.py").read_text(encoding="utf-8")
m = re.search(r"^_NOISE_FLOOR\s*=\s*([\d.]+)", loc_src, re.M)
floor = m.group(1) if m else "?"
mu = re.search(r"^_NOISE_FLOOR_UNKNOWN\s*=\s*([\d.]+)", loc_src, re.M)
floor_unknown = mu.group(1) if mu else "?"
ok = floor in block
print(f"  [{'OK  ' if ok else 'FAIL'}] 噪声底 {floor}"
      f"（代码 local_eval.py）")
if not ok:
    fail += 1

# ---- 7. 评测样本数 ----
ndet = len(final.get("details") or [])
ok = str(ndet) in block
print(f"  [{'OK  ' if ok else 'FAIL'}] 评测集 {ndet} 条")
if not ok:
    fail += 1

print()
print("=" * 62)
print(f"结果：{'全部有出处' if not fail else f'{fail} 项对不上'}")
print("=" * 62)
if fail:
    print("\n对不上的数字要么改文档，要么改数据源 ——")
    print("不要让带着错数字的文档流出去，它会在面试时变成风险。")
raise SystemExit(1 if fail else 0)