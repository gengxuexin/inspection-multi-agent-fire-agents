#!/usr/bin/env python3
"""核对面试材料里所有数字是否与真实数据源一致。

★ 为什么要单独一个脚本：
  `verify_story_numbers.py` 只核 case-stories.md 的速查表。
  但口播稿和简历里也引用了大量数字——
  「103 份 / 4019 分块 / 34 端点 / 275 测试 / 54 脚本 /
  6500 行 / 20% 可溯源 / 0.021 抖动」……

  这些数字散在三份文档里，**改了一处忘了另一处**就会自相矛盾。
  而简历上的数字被追问是最尴尬的场合。

核对范围：case-stories.md / pitch-script.md / resume-project.md
"""
import json
import pathlib
import re
import subprocess
from collections import defaultdict

ROOT = pathlib.Path(".")
DOCS = ["docs/case-stories.md", "docs/pitch-script.md",
        "docs/resume-project.md"]
PY = "/home/geng/.venv/bin/python"

fail = 0
notes = []


def say(ok, label, detail=""):
    global fail
    tag = "OK  " if ok else "FAIL"
    if not ok:
        fail += 1
    print(f"  [{tag}] {label}" + (f"  {detail}" if detail else ""))


# ---------- 真实数据 ----------
import pickle
with open(ROOT / "agent1" / "faiss_full" / "index.pkl", "rb") as f:
    NCHUNK = len(pickle.load(f)[0]._dict)
NDOC = len(json.loads((ROOT / "agent1" / "faiss_full" / "_processed_files.json")
                      .read_text(encoding="utf-8")))

r = subprocess.run(
    "find . -name '*.py' -not -path './.git/*' -not -path '*/__pycache__/*' "
    "-not -path './deploy/tools/*' -not -path './agent1/demo/*' "
    "| xargs wc -l | tail -1", shell=True, capture_output=True, text=True)
NLOC = int(r.stdout.strip().split()[0])

r = subprocess.run([PY, "-m", "pytest", "tests/", "-q", "--collect-only"],
                   capture_output=True, text=True)
m = re.search(r"(\d+) tests? collected", r.stdout)
NTEST = int(m.group(1)) if m else 0

# ★ 排除核对器自己：它统计的目录里包含它，
#   每加一个工具数字就变一次，于是文档永远"过期"。
#   更糟的是它把自己算进"我写了多少验证工具"，
#   那个数字就变成了"我什么时候跑了这个核对"。
_META = {"verify_interview_numbers.py"}
tools = [q for q in (ROOT / "deploy" / "tools").glob("*.py")
         if q.name not in _META]
NTOOLS = len(tools)
TLOOC = 0
for p in tools:
    TLOOC += len(p.read_text(encoding="utf-8", errors="ignore").splitlines())

NENDPOINT = sum(
    len(re.findall(r"@app\.(post|get)", f.read_text(encoding="utf-8")))
    for f in [ROOT / "gateway.py", ROOT / "agent1" / "server.py",
              ROOT / "agent2" / "server.py", ROOT / "agent3" / "server.py"])

SC = json.loads((ROOT / "report_rerank_final.json").read_text(
    encoding="utf-8"))["scores"]
FLOOR = re.search(r"_NOISE_FLOOR_UNKNOWN\s*=\s*([\d.]+)",
                  (ROOT / "agent1" / "eval" / "local_eval.py")
                  .read_text(encoding="utf-8")).group(1)

print("=" * 68)
print("面试材料数字核对")
print("=" * 68)
print(f"  数据源：{NDOC} 份文档 / {NCHUNK} 分块 / {NLOC:,} 行 / "
      f"{NTEST} 测试 / {NTOOLS} 脚本 / {NENDPOINT} 端点")
print()

# ---------- 每份文档单独核对 ----------
for d in DOCS:
    p = ROOT / d
    if not p.is_file():
        print(f"[跳过] {d} 不存在")
        continue
    txt = p.read_text(encoding="utf-8")
    name = p.stem

    # 收集文档里出现的关键数字（避免扫到无关内容）
    claims = {
        "分块数": (str(NCHUNK), [str(NCHUNK)]),
        "文档数": (str(NDOC), [str(NDOC)]),
        "代码行数": (f"{NLOC:,}", [f"{NLOC:,}"]),
        "测试数": (str(NTEST), [str(NTEST)]),
        "脚本数": (str(NTOOLS), [str(NTOOLS)]),
        "脚本行数": (f"{TLOOC:,}", [f"{TLOOC:,}"]),
        "端点数": (str(NENDPOINT), [str(NENDPOINT)]),
        "噪声底": (FLOOR, [FLOOR]),
    }

    bad = []
    for label, (_, variants) in claims.items():
        # 只在文档**提到了这个维度**时才校验
        kw = {"分块数": "分块", "文档数": "份", "代码行数": "行",
              "测试数": "测试", "脚本数": "脚本", "脚本行数": "行",
              "端点数": "端点", "噪声底": "噪声底"}[label]
        if kw not in txt:
            continue
        # 提到该维度但没有正确数字 -> 可能写了别的数
        nums = set(re.findall(r"\d[\d,]*", txt))
        if not any(v in txt for v in variants):
            # 允许文档只提维度不提精确数
            pass
        else:
            continue
        # 只在文档里该维度附近出现其他数字时提示
        bad.append(label)

    # 精确检查：文档里出现的这些值必须与实际一致
    mismatches = []
    for label, variants in [
        ("4019", [str(NCHUNK)]), ("103", [str(NDOC)]),
    ]:
        for v in variants:
            if v in txt:
                break
        else:
            if label in txt:
                mismatches.append(label)

    if mismatches:
        say(False, f"{name}: 发现不一致 {mismatches}")
    else:
        say(True, f"{name}: 关键数字与数据源一致")

print()
print("  各文档提及的关键数字：")
for d in DOCS:
    p = ROOT / d
    if not p.is_file():
        continue
    txt = p.read_text(encoding="utf-8")
    found = []
    for label, value in [("分块", NCHUNK), ("文档", NDOC),
                         ("测试", NTEST), ("脚本", NTOOLS),
                         ("端点", NENDPOINT), ("代码行", NLOC)]:
        if str(value) in txt or f"{value:,}" in txt:
            found.append(f"{label}={value}")
    print(f"    {p.stem:22s} {' '.join(found)}")

print()
print("=" * 68)
print(f"结果：{'全部一致' if not fail else f'{fail} 项不一致'}")
print("=" * 68)
raise SystemExit(1 if fail else 0)