"""隔离 recall 退化根因（修正版：用报告实际返回的 contexts 定位目标）。

★ 第一版的错误：
  用 ground_truth 前 12 字去 texts 里找片段，
  但 ground_truth 是人工撰写的答案摘要，不是片段原文 —— 六条全部匹配失败，
  什么都没验证。
  教训：**定位目标要用"确实进过 top5 的那个片段"，而不是"看起来应该对的文本"。**

修正：
  用旧精排报告里的 contexts（当时 recall=1.00，说明答案在里面）
  去 texts 里反查它的索引，作为目标片段。
"""
import json
import math
import pathlib
import pickle
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent1"))
sys.path.insert(0, str(ROOT))

import jieba  # noqa: E402

with open(ROOT / "agent1" / "faiss_full" / "index.pkl", "rb") as f:
    data = pickle.load(f)
texts = [d.page_content for d in data[0]._dict.values()]
n = len(texts)

# 建立 片段 -> 索引 的映射（用前 200 字做键，避开截断差异）
by_prefix: dict[str, int] = {}
for i, t in enumerate(texts):
    by_prefix.setdefault(t[:200], i)

df = {}
for t in texts:
    for tok in {x for x in jieba.cut(t) if x.strip()}:
        df[tok] = df.get(tok, 0) + 1


def make_idf(cap):
    return {w: min(cap, max(0.15, math.log(n / (1.0 + c)))) for w, c in df.items()}


IDF_OLD = make_idf(3.0)
IDF_NEW = make_idf(8.0)

VARIANTS = {
    "V0旧 (idf≤3,线性,bool)":   (IDF_OLD, "lin"),
    "V1 只放宽到8 (线性,bool)":  (IDF_NEW, "lin"),
    "V2 只用平方 (idf≤3)":       (IDF_OLD, "sq"),
    "V3 新精排 (idf≤8,平方)":    (IDF_NEW, "sq"),
}


def cov(text, ws, tw, mode):
    if mode == "lin":
        return sum(w for t, w in ws.items() if t in text) / (tw or 1.0)
    return sum(w * w for t, w in ws.items() if t in text) / (tw or 1.0)


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


old = load("report_rerank_old.json")
a = load("report_rerank_new.json")
b = load("report_rerank_fixb.json")


def rmap(rep):
    ps = (rep.get("scores") or {}).get("_per_sample") or []
    return {i: float((x.get("context_recall") if isinstance(x, dict) else x) or 0)
            for i, x in enumerate(ps)}


ro, ra, rb = rmap(old), rmap(a), rmap(b)
od = old.get("details") or []
dropped = sorted([i for i in ro if ro[i] - ra.get(i, 0) > 0.4],
                 key=lambda i: ro[i] - ra[i], reverse=True)

print(f"退化严重样本：{len(dropped)} 条")
same_ab = all(abs(ra.get(i, 0) - rb.get(i, 0)) < 1e-9 for i in dropped)
print(f"方案 A 与 B 在这些样本上recall 完全相同：{same_ab}")
print()
print("=" * 78)
print("四个变体的目标排名（目标=旧精排 top5 里的片段）")
print("=" * 78)

stat = {k: [] for k in VARIANTS}
locatable = 0

for i in dropped:
    q = od[i].get("question", "")
    ctx = od[i].get("contexts") or []
    if not ctx:
        continue

    toks = {t for t in jieba.cut(q) if t.strip()}
    # 旧精排 contexts 是 [:500] 截断的，要用前缀匹配
    tgt = {by_prefix[c[:200]] for c in ctx if c[:200] in by_prefix}
    if not tgt:
        continue
    locatable += 1

    line = f"\n[{i}] {q[:46]}"
    line += f"  旧={ro[i]:.2f} A={ra.get(i,0):.2f} B={rb.get(i,0):.2f}"
    print(line)
    print(f"    目标片段索引 {sorted(tgt)}")

    for vname, (idf, wmode) in VARIANTS.items():
        ws = {t: idf.get(t, 0.15) for t in toks}
        tw = (sum(ws.values()) if wmode == "lin"
              else sum(w * w for w in ws.values())) or 1.0
        scored = sorted(range(n), key=lambda j: -cov(texts[j], ws, tw, wmode))
        pos = next((k + 1 for k, j in enumerate(scored) if j in tgt), None)
        stat[vname].append(pos if pos is not None else 999)
        print(f"      {vname:24s} 目标排名 = {pos}")

print()
print(f"可定位的样本：{locatable}/{len(dropped)}")
print()
print("=" * 78)
print("汇总：各变体把目标排进 top5 的次数")
print("=" * 78)
for vname in VARIANTS:
    vals = stat[vname]
    ok = sum(1 for v in vals if v <= 5)
    print(f"  {vname:24s} top5 内 {ok:2d}/{len(vals)}   "
          f"中位排名 {sorted(vals)[len(vals)//2] if vals else '-'}")

print()
print("=" * 78)
print("判读")
print("=" * 78)
print("""
V0 最好、其余都差  ->  上限放宽与平方都有害
V1 最好、V3 差    ->  平方是主因（超线性把常见词压太狠）
V2 最好、V3 差    ->  上限放宽是主因（idf 8.0 让生僻词权重过高压掉相关片段）
三者与 V0 差不多  ->  退化不在覆盖率信号，而在结构匹配/长度惩罚与覆盖率的交互
""")