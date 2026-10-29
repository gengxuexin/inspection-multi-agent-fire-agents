"""查清 C 实现仍然不如 A 的根本原因，不再靠猜。

已确认的事实：
  - A（idf≤3 + bool）在 "消防救援机构有哪些职责" 上排 79，C 排 212
  - C 做了两件事：放宽 idf 上限（3→8）、改用 idf^2、疑问句打折

需要拆开看：到底是哪一步导致退化？
单独对比 "只放宽 idf" vs "只改 idf^2" vs "只加疑问句打折"，
才能知道该保留哪一步、回退哪一步。
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
CUE_D = H.HybridRetriever._CUE_DISCOUNT

CASES = [
    ("电线电缆类产品属于消防产品吗", "电线电缆"),
    ("消防产品目录里有哪些火灾报警产品", "火灾报警"),
    ("建筑消防设施包括哪些", "消防设施"),
    ("消防救援机构有哪些职责", "消防救援机构"),
]


def build_idf(texts, cap):
    n = len(texts)
    df: dict[str, int] = {}
    for t in texts:
        for tok in {x for x in jieba.cut(t) if x.strip()}:
            df[tok] = df.get(tok, 0) + 1
    return {w: min(cap, max(H.HybridRetriever._IDF_MIN,
                             math.log(n / (1.0 + c)))) for w, c in df.items()}


def rank(texts, scores, k):
    return sorted(range(len(texts)), key=lambda i: -scores[i]).index(k) + 1


def main() -> None:
    with open(ROOT / "agent1" / "faiss_full" / "index.pkl", "rb") as f:
        data = pickle.load(f)
    texts = [d.page_content for d in data[0]._dict.values()]
    n = len(texts)
    idf3, idf8 = build_idf(texts, 3.0), build_idf(texts, 8.0)

    variants = [
        ("A 基线idf≤3+bool", lambda s, ws, cr: sum(
            w for t, w in ws.items() if t in s) / (sum(ws.values()) or 1)),
        ("B 只放宽到8+bool", lambda s, ws, cr: sum(
            w for t, w in ws.items() if t in s) / (sum(ws.values()) or 1)),
        ("C idf≤3+平方", lambda s, ws, cr: sum(
            w * w for t, w in ws.items() if t in s) / (sum(w * w for w in ws.values()) or 1)),
        ("D idf≤8+平方", lambda s, ws, cr: sum(
            w * w for t, w in ws.items() if t in s) / (sum(w * w for w in ws.values()) or 1)),
        ("E idf≤8+bool+打折", lambda s, ws, cr: (
            sum(w for t, w in ws.items() if t in s) / (sum(ws.values()) or 1))
            * (1 - CUE_D * cr if cr > 0 else 1)),
    ]

    print(f"索引 {n} 片段\n")
    hdr = f"{'查询':30s}" + "".join(f"{v[0][:13]:>14s}" for v in variants)
    print(hdr)
    print("-" * len(hdr))

    totals = {v[0]: 0 for v in variants}
    for q, target in CASES:
        tset = {t for t in jieba.cut(q) if t.strip()}
        cr = sum(1 for t in tset if t in CUES) / len(tset) if tset else 0.0
        tgt = [i for i, t in enumerate(texts) if target in t]
        if not tgt:
            continue
        k = tgt[0]
        row = f"{q[:28]:30s}"
        for name, fn in variants:
            ws = idf3 if "idf≤3" in name else idf8
            r = rank(texts, [fn(t, ws, cr) for t in texts], k)
            totals[name] += r
            row += f"{r:14d}"
        print(row)

    print("-" * len(hdr))
    print(f"{'排名总和（越小越好）':30s}"
          + "".join(f"{totals[v[0]]:14d}" for v in variants))
    print()
    best = min(totals, key=totals.get)
    print(f"最优：{best}")
    print()
    print("""
读法：排名总和越小说明整体越好。
- 若「只放宽到8」就变差，说明放宽 idf 本身有害（可能让个别
  生僻词权重过高，把真正相关的片段压下去）。
- 若「idf≤3+平方」不输于 A，说明平方是可保留的改进。
- 打折那一档若单独也变差，说明 _QUERY_CUES 收得太宽
  （把"消防"这种实词误判成虚词了）。
""")


if __name__ == "__main__":
    main()