"""验证 B 方案的新判据是否真的消除了「跨文档问题被误伤」。

★ 为什么不直接跑评测验证：
   一轮RAGAS 要 3-5 分钟。先用确定性的方式验证机制，
   确认方向对了再花这个时间 —— 这是本轮工作的一贯做法。

要验证的：
  退化最严重的几条样本，在新判据下 cue_ratio 应该显著下降。

判据：cue_ratio = 疑问词 IDF 和 / 全部查询词 IDF 和
"""
import math
import pathlib
import pickle
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent1"))
sys.path.insert(0, str(ROOT))

import jieba  # noqa: E402
import hybrid as H  # noqa: E402

CUES = H.HybridRetriever._QUERY_CUES

# 从 A/B 诊断里挑出退化最严重的样本
CASES = [
    ("消防安全技术服务机构可以从事哪些业务？", "跨文档综合", 0.524),
    ("消防安全重点单位有哪些消防安全义务？", "跨文档综合", None),
    ("浙江省消防执法全过程记录制度要求记录哪些内容？", "跨文档综合", None),
    ("消防产品的认证目录包含哪些类别？", "陷阱题", None),
    ("消防技术文书应当使用什么式样？", "直查法条", None),
    # 对照组：这两类在 A/B 里没退化或上涨
    ("建筑消防设施包括哪些", "对照-曾退化", None),
    ("电线电缆类产品属于消防产品吗", "对照-曾改善", None),
    ("消防产品目录里有哪些火灾报警产品", "对照", None),
]

with open(ROOT / "agent1" / "faiss_full" / "index.pkl", "rb") as f:
    data = pickle.load(f)
texts = [d.page_content for d in data[0]._dict.values()]
n = len(texts)

df: dict[str, int] = {}
for t in texts:
    for tok in {x for x in jieba.cut(t) if x.strip()}:
        df[tok] = df.get(tok, 0) + 1

idf = {w: min(H.HybridRetriever._IDF_MAX,
              max(H.HybridRetriever._IDF_MIN, math.log(n / (1.0 + c))))
       for w, c in df.items()}


def cue_old(q):
    toks = {t for t in jieba.cut(q) if t.strip()}
    if not toks:
        return 0.0, 0.0
    return sum(1 for t in toks if t in CUES) / len(toks), 0.0


def cue_new(q):
    toks = {t for t in jieba.cut(q) if t.strip()}
    if not toks:
        return 0.0, 0.0
    w = {t: idf.get(t, H.HybridRetriever._IDF_MIN) for t in toks}
    tw = sum(w.values()) or 1.0
    cw = sum(v for t, v in w.items() if t in CUES)
    return min(1.0, cw / tw), cw / tw


print(f"索引 {n} 片段")
print(f"疑问词集合大小 {len(CUES)}")
print()
print("=" * 78)
print("B 方案判据验证：新旧 cue_ratio 对比")
print("=" * 78)
print(f"  {'查询':40s} {'类型':12s} {'旧':>6s} {'新':>6s} {'变化':>8s}")
print("  " + "-" * 78)

rows = []
for q, kind, _ in CASES:
    old_r, _ = cue_old(q)
    new_r, _ = cue_new(q)
    toks = [t for t in jieba.cut(q) if t.strip()]
    cue_toks = [t for t in toks if t in CUES]
    change = new_r - old_r
    rows.append((q, kind, old_r, new_r, change, cue_toks))

    mark = ""
    if change < -0.15:
        mark = "  ✓ 大幅下降"
    elif change < 0:
        mark = "  ↓ 略降"
    print(f"  {q[:38]:40s} {kind:12s} {old_r:6.3f} {new_r:6.3f} "
          f"{change:+8.3f}{mark}")

print()
print("=" * 78)
print("疑问词的 idf（决定新判据的关键）")
print("=" * 78)
seen = set()
for _, _, _, _, _, cue_toks in rows:
    for t in cue_toks:
        if t in seen:
            continue
        seen.add(t)
        cnt = sum(1 for x in texts if t in x)
        print(f"  {t:10s} df={cnt:5d}  idf={idf.get(t, 0.15):6.3f}")

print()
print("=" * 78)
print("结论")
print("=" * 78)

bad = [r for r in rows if r[1] in ("跨文档综合", "陷阱题") and r[3] > 0.3]
if bad:
    print("✗ 以下退化样本在新判据下 cue_ratio 仍然偏高，可能仍被误伤：")
    for q, kind, o, nn, c, _ in bad:
        print(f"    {q[:40]}  new={nn:.3f}")
else:
    print("✓ 跨文档综合 / 陷阱题 类问题在新判据下 cue_ratio 已显著下降")

print()
print("跨文档/陷阱题平均 cue_ratio：")
grp = [r for r in rows if r[1] in ("跨文档综合", "陷阱题")]
if grp:
    print(f"  旧判据 {sum(r[2] for r in grp) / len(grp):.3f}"
          f"  ->  新判据 {sum(r[3] for r in grp) / len(grp):.3f}")

print()
print("对照组（直查法条等）是否被过度削弱：")
ctrl = [r for r in rows if r[1].startswith("对照")]
if ctrl:
    print(f"  旧判据 {sum(r[2] for r in ctrl) / len(ctrl):.3f}"
          f"  ->  新判据 {sum(r[3] for r in ctrl) / len(ctrl):.3f}")
    print("  注意：「建筑消防设施包括哪些」这类真正虚词多的查询，")
    print("       新判据下折扣会减弱 —— 这是预期的，")
    print("       因为「包括」idf=2.57 属于常见词，折扣本就该轻。")
    print("       真正的虚词（「吗」「呢」「的」）idf≈0.15，折扣依然有效。")