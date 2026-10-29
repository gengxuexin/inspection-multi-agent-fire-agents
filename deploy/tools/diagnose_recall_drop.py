"""诊断 context_recall 下降 0.063 的原因。

A/B 实测：
    faithfulness        0.976 -> 0.971  (-0.005)
    answer_relevancy    0.719 -> 0.736  (+0.017)
    context_precision   0.209 -> 0.230  (+0.021)   ← 上升
    context_recall      0.574 -> 0.511  (-0.063)   ← 下降，且超出噪声底

precision 涨、recall 降，是典型的**排序偏移**特征：
精排更"聚焦"了少数片段，但丢掉了原本能召回的片段。

要定位到具体是哪些样本退化了，以及退化样本的共同特征。
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
old = json.loads((ROOT / "report_rerank_old.json").read_text(encoding="utf-8"))
new = json.loads((ROOT / "report_rerank_new.json").read_text(encoding="utf-8"))

od = old.get("details") or []
nd = new.get("details") or []
ops = (old.get("scores") or {}).get("_per_sample") or []
nps = (new.get("scores") or {}).get("_per_sample") or []

print(f"样本数：old {len(od)} / new {len(nd)}")
print(f"per_sample：old {len(ops)} / new {len(nps)}")
print()

# ---- 1. 逐条对比 recall 变化 ----
rows = []
for i, (o, n, op, np_) in enumerate(zip(od, nd, ops, nps)):
    ro = op.get("context_recall", 0) or 0
    rn = np_.get("context_recall", 0) or 0
    rows.append((i, o, n, ro, rn, rn - ro))

worse = [r for r in rows if r[5] < -0.001]
better = [r for r in rows if r[5] > 0.001]
same = [r for r in rows if abs(r[5]) <= 0.001]

print("=" * 70)
print("context_recall 逐条变化分布")
print("=" * 70)
print(f"  变好: {len(better)} 条")
print(f"  变差: {len(worse)} 条")
print(f"  持平: {len(same)} 条")
print()

if worse:
    print("退化的样本（按降幅排序，前 8 条）：")
    for i, o, n, ro, rn, d in sorted(worse, key=lambda x: x[5])[:8]:
        cat = o.get("category", "?")
        diff = o.get("difficulty", "?")
        print(f"\n  [{i}] {cat} / {diff}   recall {ro:.2f} -> {rn:.2f} ({d:+.2f})")
        print(f"      Q: {o.get('question','')[:60]}")
        # contexts 是否变了
        co, cn = o.get("contexts") or [], n.get("contexts") or []
        if co and cn and co[0] != cn[0]:
            print(f"      top1 变了：")
            print(f"        旧: {co[0][:70]}")
            print(f"        新: {cn[0][:70]}")
        else:
            print(f"      top1 未变")

# ---- 2. 按 category / difficulty 分解 ----
print()
print("=" * 70)
print("按分层分解 recall 变化")
print("=" * 70)


def group(key):
    agg = {}
    for i, o, n, op, np_, d in rows:
        k = o.get(key, "?")
        agg.setdefault(k, []).append((op.get("context_recall", 0) or 0,
                                      np_.get("context_recall", 0) or 0))
    for k, vals in sorted(agg.items()):
        a = sum(x for x, _ in vals) / len(vals)
        b = sum(y for _, y in vals) / len(vals)
        print(f"  {key:12s} {k:14s} n={len(vals):3d}  "
              f"{a:.3f} -> {b:.3f}  ({b - a:+.3f})")

group("category")
print()
group("difficulty")

# ---- 3. 检索结果有没有整体变少 ----
print()
print("=" * 70)
print("检索片段数与弃答情况")
print("=" * 70)
oc = [len(d.get("contexts") or []) for d in od]
nc = [len(d.get("contexts") or []) for d in nd]
print(f"  平均片段数  old {sum(oc)/len(oc):.2f}  new {sum(nc)/len(nc):.2f}")
oa = sum(1 for x in ops if x.get("_abstained"))
na = sum(1 for x in nps if x.get("_abstained"))
print(f"  弃答条数    old {oa}  new {na}")
print(f"  样本数      old {len(ops)}  new {len(nps)}")