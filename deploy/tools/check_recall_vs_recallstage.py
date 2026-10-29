"""查 recall 退化的真正机制：召回候选池 vs 覆盖率信号。

★ 至此已排除的可能：
  - 疑问词折扣（B 方案改的，分层数字逐位不变 -> 排除）
  - 覆盖率信号本身（4 变体纯覆盖率测试，6/6 全部top5 内 -> 排除）

★ 剩下的最大嫌疑：**召回阶段候选池变了**。
  hybrid 是 BM25 + 向量 + RRF 融合后取候选，再精排。
  BM25 的打分本身也用 idf —— 而我改了 _IDF_MAX，
  **BM25 的 idf 跟着一起变了**！
  这意味着我以为只改了"精排覆盖率"，实际上改的是**整个检索链路**。

  这是本轮 A/B 设计上的漏洞：
  _IDF_MAX 是 HybridRetriever 的类属性，被 BM25 与精排共用。
  改它 → BM25 召回结果变 → 精排排序变 → 两处一起动。
  A/B 只跑一轮分不清是谁的锅。

验证方式：看退化样本在"旧精排"与"新精排"两次的 top5 是否真的不同，
  以及差异片段是被 BM25 挤掉的还是被精排挤掉的。
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

old = json.loads((ROOT / "report_rerank_old.json").read_text(encoding="utf-8"))
a = json.loads((ROOT / "report_rerank_new.json").read_text(encoding="utf-8"))
b = json.loads((ROOT / "report_rerank_fixb.json").read_text(encoding="utf-8"))

od = old.get("details") or []
ad = a.get("details") or []
bd = b.get("details") or []


def rmap(rep):
    ps = (rep.get("scores") or {}).get("_per_sample") or []
    return {i: float((x.get("context_recall") if isinstance(x, dict) else x) or 0)
            for i, x in enumerate(ps)}


ro, ra, rb = rmap(old), rmap(a), rmap(b)
dropped = sorted([i for i in ro if ro[i] - ra.get(i, 0) > 0.4],
                 key=lambda i: ro[i] - ra[i], reverse=True)

print(f"退化样本：{len(dropped)} 条")
print()

print("=" * 78)
print("A 与 B 的 contexts 是否完全相同")
print("=" * 78)
same_ab = sum(1 for i in dropped
              if (ad[i].get("contexts") or []) == (bd[i].get("contexts") or []))
print(f"  {same_ab}/{len(dropped)} 条 A 与 B 的 contexts 逐字相同")
print("  → 若全部相同，说明 B 方案对检索结果毫无影响（折扣确实不是原因）")
print()

print("=" * 78)
print("旧 vs 新：top5 的重叠度（判断是召回变了还是重排变了）")
print("=" * 78)
for i in dropped[:8]:
    co = od[i].get("contexts") or []
    cn = ad[i].get("contexts") or []
    so = set(co)
    sn = set(cn)
    overlap = len(so & sn)
    print(f"\n[{i}] {od[i].get('question','')[:44]}")
    print(f"  recall {ro[i]:.2f} -> {ra[i]:.2f}")
    print(f"  top5 重叠 {overlap}/5")
    if overlap == 5:
        print("  → 集合相同、顺序变了：**纯重排问题**")
    elif overlap >= 3:
        print(f"  → 丢失 {5-overlap} 个、新增 {5-overlap} 个：部分召回变化")
    else:
        print(f"  → 集合大变：**召回阶段就变了**，不是精排的问题")

print()
print("=" * 78)
print("全部样本的重叠度统计")
print("=" * 78)
full_same = part = mostly = 0
for i in range(len(od)):
    so = set(od[i].get("contexts") or [])
    sn = set(ad[i].get("contexts") or [])
    ov = len(so & sn)
    if ov == 5:
        full_same += 1
    elif ov >= 3:
        part += 1
    else:
        mostly += 1
n = len(od)
print(f"  top5 完全相同 : {full_same:3d}/{n}  ({full_same/n*100:.0f}%)")
print(f"  部分相同(>=3): {part:3d}/{n}  ({part/n*100:.0f}%)")
print(f"  大变(<3)     : {mostly:3d}/{n}  ({mostly/n*100:.0f}%)")
print()
print("判读：")
print("  若「完全相同」占多数 -> 召回没变，是纯重排问题")
print("  若「大变」很多-> 召回阶段（BM25 的 idf 也被改了）就变了，")
print("     那么 A/B 设计的变量隔离是失败的，")
print("     必须把 _IDF_MAX 拆成「BM25 用」与「精排用」两个独立参数再测")