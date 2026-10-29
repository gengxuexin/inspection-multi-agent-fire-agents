"""验证最终的精排覆盖率实现（IDF 超线性加成 + 疑问句折扣）。

对照三种实现，同一批候选、同一组查询：
  A. 旧：idf≤3.0 + bool 命中
  B. 失败的第一版：idf≤8.0 + tf 饱和递增 idf*(1+ln(1+tf))
  C. 现在：idf≤8.0 + idf^2 + 疑问句折扣

B 之所以留着，是因为它的失败教训必须能被复核 ——
不把失败的版本一起放进对照，就无法证明 C 的改进来自哪一步。
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


def score_A(text, ws):
    tot = sum(ws.values()) or 1.0
    return sum(w for t, w in ws.items() if t in text) / tot


def score_B(text, ws):
    tot = sum(w * (1 + math.log1p(5.0)) for w in ws.values()) or 1.0
    got = 0.0
    for t, w in ws.items():
        if t in text:
            got += w * (1 + math.log1p(text.count(t)))
    return got / tot


def score_C(text, ws, cue_ratio):
    tot = sum(w * w for w in ws.values()) or 1.0
    cov = sum(w * w for t, w in ws.items() if t in text) / tot
    if cue_ratio > 0:
        cov *= (1.0 - CUE_D * cue_ratio)
    return cov


def main() -> None:
    with open(ROOT / "agent1" / "faiss_full" / "index.pkl", "rb") as f:
        data = pickle.load(f)
    texts = [d.page_content for d in data[0]._dict.values()]
    n = len(texts)
    print(f"索引 {n} 个片段\n")

    idf3 = build_idf(texts, 3.0)
    idf8 = build_idf(texts, 8.0)

    print("=" * 76)
    print("三种实现的目标片段排名（越小越好）")
    print("=" * 76)
    print(f"{'查询':32s} {'A旧':>7s} {'B失败版':>8s} {'C现在':>7s}")
    print("-" * 76)

    wins = losses = 0
    for q, target in CASES:
        tset = {t for t in jieba.cut(q) if t.strip()}
        ws3 = {t: idf3.get(t, 0.15) for t in tset}
        ws8 = {t: idf8.get(t, 0.15) for t in tset}
        cue_ratio = sum(1 for t in tset if t in CUES) / len(tset) if tset else 0.0

        tgt = [i for i, t in enumerate(texts) if target in t]
        if not tgt:
            print(f"{q[:30]:32s}  (索引中无「{target}」)")
            continue
        k = tgt[0]

        rA = sorted(range(n), key=lambda i: -score_A(texts[i], ws3)).index(k) + 1
        rB = sorted(range(n), key=lambda i: -score_B(texts[i], ws8)).index(k) + 1
        rC = sorted(range(n),
                    key=lambda i: -score_C(texts[i], ws8, cue_ratio)).index(k) + 1

        mark = ""
        if rC < rA:
            wins += 1
            mark = " ✓"
        elif rC > rA:
            losses += 1
            mark = " !"
        print(f"{q[:30]:32s} {rA:7d} {rB:8d} {rC:7d}{mark}")

    print()
    print(f"相对旧实现 A：改善 {wins} 例，退化 {losses} 例")


if __name__ == "__main__":
    main()