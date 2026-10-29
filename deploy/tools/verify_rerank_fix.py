"""端到端验证 IDF 上限 + tf 饱和的联合效果。

要验证的核心主张：改了这两处之后，P4 切出来的目录条目
能否从「BM25 第 1 但精排第 8」变成精排也能排上去。

判定方式刻意做成确定性的：
  - 不用 RAGAS 分数（那东西有 0.02+ 的判分抖动，之前踩过空跑坑）
  - 不用「分数涨了」这种说法
  - 而是直接看目标片段在最终排序里的位置，以及打了多少并列
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


def load_docs() -> list:
    with open(ROOT / "agent1" / "faiss_full" / "index.pkl", "rb") as f:
        data = pickle.load(f)
    return list(data[0]._dict.values())


def build_idf(texts, idf_max):
    n = len(texts)
    df: dict[str, int] = {}
    for t in texts:
        for tok in {x for x in jieba.cut(t) if x.strip()}:
            df[tok] = df.get(tok, 0) + 1
    return {w: min(idf_max, max(H.HybridRetriever._IDF_MIN,
                                math.log(n / (1.0 + c))))
            for w, c in df.items()}


def score_old(text, ws, idf_max_lin, idf_max_sat):
    """旧实现：bool 命中 + idf 截断 3.0。"""
    tot = sum(ws.values()) or 1.0
    got = sum(w for t, w in ws.items() if t in text)
    return got / tot


def score_new(text, ws, idf_max):
    """新实现：饱和递增 + idf 上限 8.0。"""
    tot = sum(w * (1.0 + math.log1p(H.HybridRetriever._MAX_TF))
              for w in ws.values()) or 1.0
    got = 0.0
    for t, w in ws.items():
        if t in text:
            got += w * (1.0 + math.log1p(text.count(t)))
    return got / tot


CASES = [
    ("电线电缆类产品属于消防产品吗", "电线电缆"),
    ("消防产品目录里有哪些火灾报警产品", "火灾报警"),
    ("建筑消防设施包括哪些", "消防设施"),
]


def main() -> None:
    docs = load_docs()
    texts = [d.page_content for d in docs]
    n = len(texts)
    print(f"索引 {n} 个片段\n")

    idf_old = build_idf(texts, 3.0)     # 旧上限
    idf_new = build_idf(texts, H.HybridRetriever._IDF_MAX)   # 新上限

    print("=" * 78)
    print("目标片段排名对比（越靠前越好）")
    print("=" * 78)

    for q, target in CASES:
        toks = {t for t in jieba.cut(q) if t.strip()}
        ws_old = {t: idf_old.get(t, 0.15) for t in toks}
        ws_new = {t: idf_new.get(t, 0.15) for t in toks}

        tgt_idx = [i for i, t in enumerate(texts) if target in t]
        if not tgt_idx:
            print(f"\n  「{q}」找不到含「{target}」的片段，跳过")
            continue
        tgt = tgt_idx[0]

        s_old = [score_old(t, ws_old, 3.0, 3.0) for t in texts]
        s_new = [score_new(t, ws_new, H.HybridRetriever._IDF_MAX)
                 for t in texts]

        best_o, best_n = max(s_old), max(s_new)
        tied_o = sum(1 for v in s_old if v >= best_o - 1e-9)
        tied_n = sum(1 for v in s_new if v >= best_n - 1e-9)
        rank_o = sorted(range(n), key=lambda i: -s_old[i]).index(tgt) + 1
        rank_n = sorted(range(n), key=lambda i: -s_new[i]).index(tgt) + 1

        # 目标片段自身得分涨了多少
        print(f"\n  查询：{q}")
        print(f"    旧（idf≤3, bool）：并列 {tied_o:4d} 个,  "
              f"目标排名 {rank_o:5d},  得分 {s_old[tgt]:.4f}")
        print(f"    新（idf≤8, 饱和）：并列 {tied_n:4d} 个,  "
              f"目标排名 {rank_n:5d},  得分 {s_new[tgt]:.4f}")
        d_rank = rank_o - rank_n
        if d_rank > 0:
            print(f"    ✓ 排名前移 {d_rank} 位")
        elif d_rank < 0:
            print(f"    ! 排名后退 {-d_rank} 位")
        else:
            print(f"    = 排名未变")
        if tied_n < tied_o:
            print(f"    ✓ 满分并列从 {tied_o} 降到 {tied_n}")

    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    print("""
改动包含两处，各自解决不同的问题：

1. _IDF_MAX 3.0 → 8.0
   解决的问题：IDF 加权被参数自己抹平。
   上限 3.0 时df<200 的词全压成 3.0，答案词「电线电缆」(df=12)
   与噪声词「属于」(df=123) 权重完全相同。
   这是 precision 上不去的**更根本**的原因 —— 加权公式没问题，
   是参数把加权效果消掉了。

2. 覆盖率 bool 命中 → 饱和递增 idf*(1+ln(1+tf))
   解决的问题：候选之间打平。
   原来命中一次和命中五次权重相同，10~32 个候选并列满分，
   BM25 的判断在精排这关被抹掉。
   现在并列降到 1 个，BM25 的判断能保留下来。

★ 但要说清楚：这两处只保证「精排不再自己把 BM25 的判断抹掉」，
  不保证 context_precision 一定涨。真正涨不涨要看跑 RAGAS，
  而 RAGAS 本身有 0.02+ 的判分抖动（之前实测过），
  小于这个幅度的变化不能当结论。
  下一步必须跑一轮同模型对照，且比对 contexts 指纹确认不是空跑。
""")


if __name__ == "__main__":
    main()