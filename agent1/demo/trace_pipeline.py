"""检索链路追踪：把 RAG 各阶段的真实中间态抓出来。

★ 为什么需要它：
  演示页最忌讳"看起来很厉害但数据是编的"。
  而 RAG 链路天然难以展示 ——
  LangChain 的 chain.invoke() 只有一个最终答案，
  中间"检索到了什么、精排排第几、引用了哪条法规"全在黑箱里。

  这个模块不改链路，只"旁路观测"：
    - 直接调 retrieval_backend.search() 拿真实命中片段
    - 各阶段自己计时（不用 hook —— hook 会改变链路行为）
    - 弃答判定按 rag_module 的 prompt 约定做文本识别

  代价：检索会跑两次（一次给自己看，一次给 chain）。
  演示场景下多几十毫秒没关系，换来的是"每段展示都是真实发生的"。

★ 演示页真正该展示什么（先想清楚再写代码）：
  不是"我用了 RAG"，而是三件事：
    1. **它知道自己不知道** —— 弃答判断。
       消防执法场景下乱答的代价远大于不答。
    2. **检索过程可解释** —— 命中了哪些片段、引用了哪部法规第几条
    3. **回答规则是显式约束** —— prompt 里写死了
       "不编造条款号""完全无关才答未检索到"
  这三点恰好是 FDE 面试里最想讲的。
"""
from __future__ import annotations

import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "agent1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ============================================================
# 弃答约定
#
# ★ 重要事实：弃答**不是代码判定**，而是 prompt 里的约定 ——
#   rag_module.py 的 system prompt 第 4 条写的是
#     "只有在检索结果和问题完全无关时，才回答'未检索到相关内容'"
#   LLM 按这个约定输出固定文案。
#
#   所以判定方式是文本识别，不是找某个 should_abstain() 函数。
#   之前我写了个 _judge_abstain 去 getattr 找 should_abstain，
#   找不到就退回"空答案才算弃答"——
#   于是 "未检索到相关内容。" 这种真实弃答被判成了 False。
#   演示页上出现"明明没答还说没弃答"，比不做演示更糟。
#
#   识别不到任何约定文案时，返回 None（三态：True/False/None）
#   而不是硬给一个False —— "不知道"要说成"不知道"。
ABSTAIN_MARKERS = (
    "未检索到相关内容",
    "未检索到相关",
    "没有检索到",
    "暂无相关内容",
    "未找到相关",
)

# 法条引用抽取
_CITE_RE = re.compile(
    r"第[一二三四五六七八九十百零〇\d]+条"
    r"(?:第[一二三四五六七八九十\d]+款)?"
    r"(?:第[一二三四五六七八九十\d]+项)?"
)
_LAW_RE = re.compile(r"《([^》]{2,40})》")


def detect_abstain(answer: str) -> tuple[bool | None, str]:
    """识别弃答。

    :return: (是否弃答, 命中的文案)
             第一项为 None 表示**没识别出任何弃答标记**，
             此时答案里可能真的有内容，也可能是漏了新文案 —— 如实说不确定。
    """
    a = (answer or "").strip()
    if not a:
        return True, "(空回答)"
    for m in ABSTAIN_MARKERS:
        if m in a:
            return True, m
    # 答案很短且没有引用 —— 多半是弃答的变体，
    # 但不能确定，所以标 None 而不是判True
    if len(a) <= 20 and not _CITE_RE.search(a):
        return None, "短回答且无条款引用（疑似弃答变体）"
    return False, ""


_CN_NUM = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100,
}


def _cn_to_int(s: str) -> int | None:
    """中文数字转阿拉伯数字（覆盖到「一百二十三条」这种量级）。

    只为溯源比对用，转不动就返回 None（退化成严格字符串匹配）。
    """
    s = (s or "").strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    total, section, last = 0, 0, None
    for ch in s:
        if ch not in _CN_NUM:
            return None
        v = _CN_NUM[ch]
        if v == 100:
            section = (section + (last or 0)) * 100
            last = 0
        elif v == 10:
            section += (last or 1) * 10
            last = 0
        else:
            last = v
    return total + section + (last or 0)


def _articles_in(text: str) -> set[str]:
    """抽取文本里出现过的所有条款号（统一成整数，便于比对）。"""
    out = set()
    for m in _CITE_RE.findall(text or ""):
        n = _cn_to_int(re.sub(r"^第|条$|款$|项$", "", m))
        if n:
            out.add(n)
    return out


def extract_citations(answer: str, contexts: list | None = None) -> list[dict]:
    """抽法条引用，并判定每条是否真能在检索片段里找到。

    :return: [{article, law, source_excerpt, verified, risk}]
      risk: "ok" / "fuzzy" / "fabricated"
        ok        —— 条款号在检索片段中原样出现
        fuzzy     —— 数字一致但表述有差异（"第十五条" vs "第十五条之一"）
        fabricated —— 检索片段里根本没这个条款号 → **模型编造**
    """
    if not answer:
        return []

    laws = _LAW_RE.findall(answer)
    ctx_articles: set[int] = set()
    for c in (contexts or []):
        ctx_articles |= _articles_in(c or "")

    seen, out = set(), []
    for a in _CITE_RE.findall(answer)[:8]:
        if a in seen:
            continue
        seen.add(a)
        n = _cn_to_int(re.sub(r"^第|条$|款$|项$", "", a))

        # 找出处
        src = ""
        for c in (contexts or []):
            if a in (c or ""):
                src = (c or "")[:100]
                break

        if src:
            risk = "ok"
        elif n is not None and n in ctx_articles:
            #数字对得上，表述不同（可能片段里写的是"第十五条之一"）
            risk = "fuzzy"
            for c in (contexts or []):
                if re.search(rf"第{n}\s*条" + r"[^\n]{{0,6}}" + a,
                             (c or "").replace(" ", "")):
                    src = (c or "")[:100]
                    break
        else:
            risk = "fabricated"

        out.append({
            "article": a,
            "law": laws[0] if laws else "",
            "source_excerpt": src,
            "verified": risk in ("ok", "fuzzy"),
            "risk": risk,
        })
    return out


# ============================================================
# 检索阶段
# ============================================================
def trace_retrieval(question: str, top_k: int = 5) -> dict:
    """跑一次检索，把每一步的真实数据抓出来。"""
    stages: list[dict] = []
    out: dict = {"question": question, "top_k": top_k, "stages": stages}

    # ---- 阶段 1：查询改写 ----
    t0 = time.perf_counter()
    expanded = question
    if len(question) < 15:
        expanded = f"《中华人民共和国消防法》相关规定：{question}"
    stages.append({
        "name": "查询改写",
        "detail": ("短查询自动补法规前缀（< 15 字）"
                   if expanded != question else "长度足够，无需改写"),
        "input": question,
        "output": expanded,
        "ms": round((time.perf_counter() - t0) * 1000, 2),
    })

    # ---- 阶段 2：混合检索 ----
    hits, ms, err = [], 0.0, None
    try:
        from retrieval_backend import search as backend_search
        t1 = time.perf_counter()
        hits = backend_search(expanded, top_k=top_k) or []
        ms = time.perf_counter() - t1
    except Exception as e:                          # noqa: BLE001
        err = f"{type(e).__name__}: {e}"

    stages.append({
        "name": "混合检索",
        "detail": "BM25 + 向量 + RRF 融合 → 精排 → 取 top-%d" % top_k,
        "ms": round(ms * 1000, 2),
        "count": len(hits),
        "error": err,
    })

    # ---- 阶段 3：命中片段 ----
    frags = []
    for i, h in enumerate(hits, 1):
        meta = h.get("metadata") or {}
        content = h.get("content") or ""
        frags.append({
            "rank": i,
            "score": round(float(h.get("score", 0.0)), 4),
            "source": (meta.get("source") or meta.get("law") or "未知")[:70],
            "law": (meta.get("law") or "")[:40],
            "preview": content[:260],
            "chars": len(content),
            "cited_articles": _CITE_RE.findall(content)[:3],
        })
    if frags:
        stages.append({
            "name": "精排结果",
            "detail": "按IDF 加权覆盖率 + 结构匹配 + 长度惩罚排序",
            "fragments": frags,
        })

    out["fragments"] = frags
    out["contexts"] = [(h.get("content") or "") for h in hits]
    return out


# ============================================================
# 完整链路
# ============================================================
def trace_answer(question: str, top_k: int = 5) -> dict:
    """跑完整 RAG：检索 → 生成 → 弃答判定 → 引用抽取。"""
    ret = trace_retrieval(question, top_k)
    stages = ret["stages"]
    contexts = ret["contexts"]

    if not contexts:
        out = dict(ret)
        out.update({
            "answer": "", "abstained": True,
            "abstain_marker": "检索无命中",
            "citations": [], "total_ms": 0,
        })
        return out

    # ---- 生成 ----
    t0 = time.perf_counter()
    answer, err = "", None
    try:
        from rag_module import rag_ask
        answer = rag_ask(question, top_k=top_k) or ""
    except Exception as e:                          # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    gen_ms = (time.perf_counter() - t0) * 1000

    stages.append({
        "name": "生成",
        "detail": "prompt 显式约束：不编造条款号 / 完全无关才答未检索到",
        "ms": round(gen_ms, 2),
        "chars": len(answer),
        "error": err,
    })

    abstained, marker = detect_abstain(answer)
    cites = extract_citations(answer, contexts)

    # 前端要显示「检索返回 N 个片段里都没有这个条款号」，
    # 所以片段数必须由后端传，不能让前端硬编码。
    frag_count = len(contexts)

    if abstained:
        stages.append({
            "name": "弃答判定",
            "detail": f"命中弃答文案「{marker}」—— 宁可不答，不可乱答",
            "abstained": True,
        })
    else:
        stages.append({
            "name": "弃答判定",
            "detail": f"正常作答，抽出 {len(cites)} 处条款引用",
            "abstained": False,
        })

    out = dict(ret)
    out.update({
        "answer": answer,
        "abstained": abstained,
        "abstain_marker": marker,
        "citations": cites,
        "verified_citations": sum(1 for c in cites if c["verified"]),
        "fabricated_citations": sum(1 for c in cites
                                    if c["risk"] == "fabricated"),
        "fragment_count": frag_count,
        "total_ms": round(sum(s.get("ms", 0) for s in stages), 2),
    })
    return out


# ============================================================
# 样例问题（现场演示用）
# ============================================================
SAMPLE_QUESTIONS = [
    {
        "q": "建筑消防设施包括哪些",
        "note": "引用可溯源 —— 抽查的条款号确实在检索片段里",
        "expect": "ok",
    },
    {
        "q": "公众聚集场所投入使用需要什么手续",
        "note": "引用可溯源 —— 手续类问题条款对应清晰",
        "expect": "ok",
    },
    {
        "q": "消防设施维护保养要求是什么",
        "note": "混合 —— 一处可溯源、一处编造，最能说明为什么要人工核对",
        "expect": "mixed",
    },
    {
        "q": "消防救援机构有哪些职责",
        "note": "⚠ 6 处引用里5 处编造 —— 这是实测数据，用来讲兜底机制",
        "expect": "fabricated",
    },
    {
        "q": "消防安全重点单位有哪些消防安全义务？",
        "note": "跨文档综合类 —— 上一轮 A/B 暴露的难点，可讲取舍",
        "expect": "none",
    },
    {
        "q": "消防技术文书应当使用什么式样？",
        "note": "文书模板类，答案在 agent1/output/templates",
        "expect": "none",
    },
]


def main() -> int:
    """命令行自检：确认链路跑通，且拿到真实数据。"""
    ok = 0
    for s in SAMPLE_QUESTIONS:
        print("=" * 72)
        print(f"[{s['q']}]")
        print(f"  {s['note']}")
        try:
            r = trace_answer(s["q"], top_k=5)
        except Exception as e:                      # noqa: BLE001
            print(f"    ERROR {type(e).__name__}: {e}")
            continue

        for st in r["stages"]:
            bits = []
            for k in ("ms", "count", "chars"):
                if k in st:
                    bits.append(f"{k}={st[k]}")
            err = f"  ERROR={st['error']}" if st.get("error") else ""
            print(f"    {st['name']:10s} {' '.join(bits)}{err}")

        mark = {True: "是", False: "否", None: "不确定"}[r["abstained"]]
        print(f"    弃答={mark}  引用={len(r['citations'])} 条"
              f"（可溯源 {r.get('verified_citations', 0)} 条）")
        if r["answer"]:
            print(f"    答案：{r['answer'][:100]}")
            ok += 1
        print()

    print("=" * 72)
    print(f"{ok}/{len(SAMPLE_QUESTIONS)} 条跑出内容")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())