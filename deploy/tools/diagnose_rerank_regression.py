"""诊断：为什么后两个查询的目标片段排名反而后退了。

现象（idf≤3+bool → idf≤8+饱和）：
    电线电缆类    47 → 13    ✓ 前移 34
    火灾报警产品 202 → 285   ! 后退 83
    建筑消防设施  212 → 565   ! 后退 353

怀疑：IDF 上限提高后，"属于/的/哪些"这类查询垃圾词的 idf 被放大，
它们在噪声片段里反复出现，靠 tf 饱和拿到了不成比例的分数。

验证方式：把查询分词的 idf 逐个列出来，并统计每个词
在"目标片段"和"排在前面的噪声片段"里的tf 分布。
只有看清分布才能判断是idf 放大了噪声，还是饱和项惩罚了稀疏答案。
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


def main() -> None:
    with open(ROOT / "agent1" / "faiss_full" / "index.pkl", "rb") as f:
        data = pickle.load(f)
    texts = [d.page_content for d in data[0]._dict.values()]
    n = len(texts)

    df: dict[str, int] = {}
    for t in texts:
        for tok in {x for x in jieba.cut(t) if x.strip()}:
            df[tok] = df.get(tok, 0) + 1
    idf = {w: min(8.0, max(0.15, math.log(n / (1.0 + c)))) for w, c in df.items()}

    for q, target in (("建筑消防设施包括哪些", "消防设施"),
                      ("消防产品目录里有哪些火灾报警产品", "火灾报警")):
        toks = list(jieba.cut(q))
        tset = {t for t in toks if t.strip()}
        print("=" * 74)
        print(f"查询：{q}")
        print(f"分词：{toks}")
        print()
        print(f"  {'词':12s} {'df':>6s} {'idf':>7s}  {'目标片段tf':>10s}")
        tgt_idx = [i for i, t in enumerate(texts) if target in t]
        if not tgt_idx:
            print("  索引里找不到目标片段")
            continue
        tgt = texts[tgt_idx[0]]

        for t in sorted(tset, key=lambda x: -idf.get(x, 0.15)):
            d_ = sum(1 for x in texts if t in x)
            tf_ = tgt.count(t) if t in tgt else 0
            print(f"  {t:12s} {d_:6d} {idf.get(t, 0.15):7.3f}  {tf_:10d}")

        print()
        print(f"  目标片段长度 {len(tgt)} 字")
        # 目标片段命中了几个查询词
        hit = [t for t in tset if t in tgt]
        print(f"  目标片段只命中 {len(hit)}/{len(tset)} 个查询词：{hit}")
        print(f"  目标片段前 90 字：{tgt[:90]}")

        # 排名前 5 的片段命中了什么 —— 看噪声长什么样
        tot = sum(idf.get(t, 0.15) * (1 + math.log1p(5.0))
                  for t in tset) or 1.0

        def sc(s):
            return sum(idf.get(t, 0.15) * (1 + math.log1p(s.count(t)))
                       for t in tset if t in s) / tot

        top = sorted(range(n), key=lambda i: -sc(texts[i]))[:5]
        print()
        print("  排在前 5 的片段（噪声画像）：")
        for i in top:
            s = texts[i]
            hits = [t for t in tset if t in s]
            tfs = {t: s.count(t) for t in hits}
            print(f"    得分 {sc(s):.4f}  命中 {hits}")
            print(f"              tf={tfs}  长度={len(s)}")
            print(f"              {s[:70]}")
        print()


if __name__ == "__main__":
    main()