# test_hybrid.py
"""hybrid.py 单元测试：查询解析 + 法律线索反查 + 分级精排 + RRF 融合

不依赖真实 FAISS 索引 / embedding API，用内存桩件覆盖检索策略的关键决策。
背景：RAGAS 逐样本诊断暴露 4 个检索缺陷，本文件逐一对应。
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "agent1"))

# ---- 桩掉重依赖，让纯检索逻辑可离线测试 ----
# jieba 优先用真的；没有就用 bigram 近似分词。
# 不能按字符切 —— 那会让"电动自行车在高层民用建筑…"这类查询的词面
# 命中统计完全失真，测不出真实的排序行为。
try:
    import jieba as _real_jieba

    def _cut(s, **k):
        return _real_jieba.cut(s, **k)
except ImportError:
    def _cut(s, **k):
        s = s.strip()
        return [s[i:i + 2] for i in range(len(s) - 1)] or [s]

if "jieba" not in sys.modules:
    _jb = types.ModuleType("jieba")
    _jb.cut = _cut
    _jb.setLogLevel = lambda *a: None
    sys.modules["jieba"] = _jb
if "rank_bm25" not in sys.modules:
    _bm = types.ModuleType("rank_bm25")
    _bm.BM25Okapi = object
    sys.modules["rank_bm25"] = _bm
for _m in ("langchain_core", "langchain_core.documents",
           "langchain_community", "langchain_community.docstore",
           "langchain_community.vectorstores"):
    if _m not in sys.modules:
        _mod = types.ModuleType(_m)
        _mod.__path__ = []
        sys.modules[_m] = _mod
if not hasattr(sys.modules["langchain_community.vectorstores"], "FAISS"):
    sys.modules["langchain_community.vectorstores"].FAISS = type(
        "FAISS", (), {"load_local": classmethod(lambda *a, **k: None)})

from hybrid import (  # noqa: E402
    CONTENT_HINTS,
    HybridRetriever,
    LAW_ALIASES,
    doc_key,
    doc_law,
    parse_query,
)


# ==================== 桩件 ====================

class Doc:
    def __init__(self, page_content="", metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


def make_docs():
    """构造一批贴近真实分布的 chunk（复刻评估里出现过的噪声形态）。"""
    fire_law = "中华人民共和国消防法"
    return [
        Doc("第六十条 单位违反本法规定，有下列行为之一的，责令改正："
            "（六）人员密集场所在门窗上设置影响逃生和灭火救援的障碍物的；"
            "（七）对火灾隐患经消防救援机构通知后不及时采取措施消除的。",
            {"law": fire_law, "source": f"{fire_law}.pdf",
             "article": "第60条", "chunk_id": "c60"}),
        # 噪声：同一条的邻居，写法是"第一款第六项"
        Doc("按照《消防法》第六十条第一款第六项，责令改正并处罚款，经责令"
            "改正拒不改正的，强制执行。",
            {"law": fire_law, "source": f"{fire_law}.pdf",
             "article": "第60条", "chunk_id": "n60_6"}),
        Doc("依据《消防法》第六十条第一款第三项至第六项的规定责令改正，"
            "并依法予以处罚后，拒不改正的，应当依法实施强制执行。",
            {"law": fire_law, "source": f"{fire_law}.pdf",
             "article": "第60条", "chunk_id": "n60_36"}),
        # 噪声：同法名但不同文件（短名污染的来源）
        Doc("公众聚集场所投入使用、营业前消防安全检查实行告知承诺管理。",
            {"law": "贯彻实施新修改消防法通知",
             "source": "（应急[2021]34号）贯彻实施新修改"
                       "«中华人民共和国消防法»的通知.pdf",
             "article": None, "chunk_id": "gongshi"}),
        # 目标只有条号时的一条
        Doc("第六十一条 消防技术服务机构应当具备相应条件。",
            {"law": fire_law, "source": f"{fire_law}.pdf",
             "article": "第61条", "chunk_id": "c61"}),
    ]


def make_retriever(docs):
    """绕开 __init__（需真实 FAISS），直接注入检索器状态。"""
    r = HybridRetriever.__new__(HybridRetriever)
    r.all_docs = docs
    r.rrf_k = 60
    r.tokenized = [list(_cut(d.page_content)) for d in docs]
    r.bm25 = type("BM25", (), {"get_scores": lambda self, q: [1.0] * len(docs)})()
    r.vs = type("VS", (), {"similarity_search": lambda self, q, k=50: docs})()
    # IDF 表：与 __init__ 同一口径，语料小所以 idf 直接给 1.0
    # （单测只验证"加权覆盖率"的调用链能跑通，不验证权重数值）
    r._idf = {w: 1.0 for toks in r.tokenized for w in set(toks)}
    # 精排现在自己从 df 计算 idf（不再只读 _idf），
    # 所以 mock 必须补这两个属性，否则 _rerank 会 AttributeError。
    r._df = {w: 1 for toks in r.tokenized for w in set(toks)}
    r._n_docs = max(len(docs), 1)
    r._IDF_MIN = HybridRetriever._IDF_MIN
    r._IDF_MAX_RERANK = HybridRetriever._IDF_MAX_RERANK
    r._IDF_MAX_BM25 = HybridRetriever._IDF_MAX_BM25
    from collections import defaultdict
    r._law_index = defaultdict(dict)
    for d in docs:
        src = str(d.metadata.get("source", ""))
        law = str(d.metadata.get("law") or "")
        ident = (law, src)
        for key in LAW_ALIASES:
            if key in law:
                r._law_index[key][ident] = 3
            elif key in src:
                r._law_index[key].setdefault(ident, 2)
        for key in CONTENT_HINTS:
            if key in d.page_content:
                r._law_index[key].setdefault(ident, 1)
    return r


# ==================== parse_query ====================

def test_parse_query_article_normalized():
    law, art = parse_query("《消防法》第六十条第一款第七项规定了什么？")
    assert art == "第60条第1款第7项"
    assert law == "消防法"


def test_parse_query_article_only():
    _, art = parse_query("中华人民共和国消防法第六十条是怎么规定的")
    assert art == "第60条"


def test_parse_query_no_article():
    law, art = parse_query("消防行政处罚的裁量基准是什么？")
    assert art is None
    assert law == "裁量基准"


def test_parse_query_empty():
    assert parse_query("") == (None, None)
    assert parse_query(None) == (None, None)


@pytest.mark.parametrize("q,expect", [
    # 缺陷 1：这两条原本解析不出法律名，精准定位完全没触发
    ("公众聚集场所投入使用营业前消防安全检查实行什么制度？", "公众聚集场所"),
    ("消防行政处罚的裁量基准是什么？", "裁量基准"),
    ("浙江省消防行政处罚裁量规定对轻微违法行为如何处理？", "行政处罚裁量"),
    ("高层民用建筑消防安全管理规定对电动自行车有什么要求？", "高层民用建筑"),
])
def test_parse_query_recovers_law_keyword(q, expect):
    """这4 条是评估里的真实查询，必须都能抽出线索。"""
    law, _ = parse_query(q)
    assert law == expect


def test_parse_query_longer_alias_wins():
    """长别名更有指向性，必须优先于其子串短别名。"""
    law, _ = parse_query("浙江省消防行政处罚裁量规定")
    assert law == "行政处罚裁量"
    assert LAW_ALIASES["行政处罚裁量"] == "行政处罚裁量"


# ==================== doc_law / doc_key ====================

def test_doc_law_prefers_law_field():
    d = Doc("x", {"law": "浙江省消防行政处罚裁量规定",
                  "source": "（浙消〔2022〕67号）.pdf"})
    assert doc_law(d) == "浙江省消防行政处罚裁量规定"


def test_doc_law_fallback_to_source():
    assert doc_law(Doc("x", {"source": "a.pdf"})) == "a.pdf"


def test_doc_key_uses_chunk_id():
    a = Doc("内容相同", {"chunk_id": "aaa"})
    b = Doc("内容相同", {"chunk_id": "bbb"})
    assert doc_key(a) != doc_key(b), "chunk_id 不同必须视为不同文档"


def test_doc_key_fallback_md5():
    a = Doc("完全一样的内容")
    assert doc_key(a) == doc_key(Doc("完全一样的内容"))


# ==================== 线索反查 ====================

def test_resolve_docs_tiers_law_field_above_filename():
    """缺陷 2：短名"消防法"在文件名里也会命中通知文件，必须分层。

    真正的兜底是 _precise_search 的分级过滤（通知里的 chunk 匹配不到条款），
    但候选排序仍应让权威法律名的文档排在前面。
    """
    docs = make_docs()
    r = make_retriever(docs)
    cands = r._resolve_docs("消防法")
    assert cands, "消防法必须能召回候选"
    # 权威 law 字段命中的"中华人民共和国消防法"应排在前面
    first_src = cands[0].metadata["source"]
    assert first_src == "中华人民共和国消防法.pdf"


def test_resolve_docs_falls_back_to_filename_tier():
    """law 字段没命中时，仍能靠文件名召回（tier=1）。"""
    docs = [Doc("告知承诺管理内容", {"law": "", "source": "公众聚集场所通知.pdf"})]
    r = make_retriever(docs)
    assert len(r._resolve_docs("公众聚集场所")) == 1


def test_resolve_docs_unknown_keyword_returns_empty():
    r = make_retriever(make_docs())
    assert r._resolve_docs("不存在于索引中的法律名称") == []
    assert r._resolve_docs(None) == []


def test_resolve_docs_matches_law_field_content():
    """线���靠law 内容匹配，不只靠文件名。"""
    docs = [Doc("告知承诺管理内容", {"law": "公众聚集场所消防安全要求",
                                     "source": "附件1.docx"})]
    r = make_retriever(docs)
    assert len(r._resolve_docs("公众聚集场所")) == 1


# ==================== 精准定位 ====================

def test_precise_search_keeps_only_level_ge_2():
    """缺陷 3：必须丢掉"第六十条的邻居"（lv=1 噪声）。"""
    docs = make_docs()
    r = make_retriever(docs)
    out = r._precise_search("消防法", "第60条第1款第7项", 5)
    assert out is not None
    ids = {d.metadata["chunk_id"] for d, _ in out}
    assert "c60" in ids, "真正的原文 chunk 必须保留"
    assert "n60_6" not in ids, "只到条的噪声必须剔除"
    assert "n60_36" not in ids, "只到条的噪声必须剔除"


def test_precise_search_ranks_level3_first():
    docs = make_docs()
    r = make_retriever(docs)
    out = r._precise_search("消防法", "第60条第1款第7项", 5)
    assert out[0][0].metadata["chunk_id"] == "c60"
    assert out[0][1] > 0


def test_precise_search_rejects_when_best_level_too_low():
    """最高只到"条"级时宁可不走强先验，避免噪声污染结果。"""
    docs = [Doc("第六十条违反本法规定的，责令改正。",
                {"law": "中华人民共和国消防法", "source": "a.pdf"})]
    r = make_retriever(docs)
    assert r._precise_search("消防法", "第60条第1款第7项", 5) is None


def test_precise_search_needs_article():
    """没有条款号时不走精准强先验（否则等于按文件顺序返回噪声）。"""
    r = make_retriever(make_docs())
    assert r._precise_search("消防法", None, 5) is None


def test_precise_search_unknown_keyword():
    r = make_retriever(make_docs())
    assert r._precise_search("不存在于索引中的法律名称", "第60条", 5) is None


# ==================== 精排打分 ====================

def test_score_orders_by_match_level():
    r = make_retriever(make_docs())
    docs = make_docs()
    s_orig = r._score("消防法", "第60条第1款第7项", docs[0])
    s_noise = r._score("消防法", "第60条第1款第7项", docs[1])
    assert s_orig > s_noise, "精确到项必须重于只到条"
    assert s_noise > 0


def test_score_without_article_only_law_signal():
    r = make_retriever(make_docs())
    docs = make_docs()
    assert r._score("消防法", None, docs[0]) == HybridRetriever.W_LAW


def test_w_match_weight_ordering():
    w = HybridRetriever.W_MATCH
    assert w[3] > w[2] > w[1] > w[0]


def test_rerank_short_chunk_penalised():
    """过短 chunk 信息量不足，应降权。"""
    r = make_retriever(make_docs())
    short = Doc("短", {"law": "消防法", "source": "a.pdf", "chunk_id": "s"})
    long_ = Doc("短" * 200, {"law": "消防法", "source": "a.pdf", "chunk_id": "l"})
    picked = {d.metadata["chunk_id"] for d in r._rerank("短", "消防法", None,
                                                      [short, long_], 1)}
    assert "l" in picked


def test_rerank_returns_requested_count():
    r = make_retriever(make_docs())
    assert len(r._rerank("消防法", "消防法", None, make_docs(), 3)) == 3


# ==================== 端到端 search ====================

def test_search_puts_precise_hit_in_top_k():
    """缺陷 4：精准命中必须稳定进入 top-k，且排在最前。"""
    docs = make_docs()
    r = make_retriever(docs)
    hits = r.search("《消防法》第六十条第一款第七项规定了什么？", k=3)
    assert hits, "必须有结果"
    assert hits[0].metadata["chunk_id"] == "c60"


def test_search_survives_no_precise_hit():
    """没有精准命中时应优雅回退到混合检索，不能抛异常。"""
    r = make_retriever(make_docs())
    hits = r.search("什么是消防安全责任人", k=5)
    assert isinstance(hits, list)


def test_search_empty_query():
    r = make_retriever(make_docs())
    assert r.search("") == []
    assert r.search(None) == []


# ==================== 内容线索兜底 ====================
# 真实案例：查询"高层民用建筑消防安全管理规定对电动自行车有什么要求？"
#   - 元数据里没有任何文件的 law/source 含"高层民用建筑"
#   -《高层民用建筑消防安全管理规定》本身不在知识库
#   - 只按 metadata 反查 -> 候选数 0 -> 精准定位完全不触发
# 必须允许从**正文**兜底召回。


def test_content_hint_recovers_when_metadata_misses():
    """元数据无命中时，正文含线索也要能召回。"""
    docs = [
        Doc("当事人当场上报电动自行车违规停放行为，适用《高层民用建筑消防"
            "安全管理规定》第三十七条。",
            {"law": "全省消防法制业务研讨会纪要", "source": "纪要.pdf",
             "chunk_id": "hint1"}),
        Doc("电动自行车充停场所应当设置引导标识，充换区域标识清晰。",
            {"law": "浙江省电动自行车充停、充换场所建设技术导则",
             "source": "导则.pdf", "chunk_id": "hint2"}),
    ]
    r = make_retriever(docs)
    assert r._resolve_docs("高层民用建筑"), "正文线索必须能召回"
    assert len(r._resolve_docs("电动自行车")) == 2


def test_content_hint_ranked_by_query_overlap():
    """tier 相同时，同时命中更多查询词的 chunk 应排前面。"""
    docs = [
        Doc("电动自行车充停场所应设置引导标识。",
            {"law": "A", "source": "A.pdf", "chunk_id": "weak"}),
        Doc("电动自行车在高层民用建筑的公共门厅停放，依据《高层民用建筑消防"
            "安全管理规定》第三十七条处理。",
            {"law": "B", "source": "B.pdf", "chunk_id": "strong"}),
    ]
    r = make_retriever(docs)
    got = r._resolve_docs("电动自行车", query="高层民用建筑电动自行车")
    assert got[0].metadata["chunk_id"] == "strong"


def test_metadata_tier_beats_content_tier():
    """权威法律名命中（tier3）必须排在仅正文命中（tier1）之前。"""
    docs = [
        Doc("电动自行车充电管理要求。",  # 仅正文
            {"law": "不相干文件", "source": "x.pdf", "chunk_id": "weak"}),
        Doc("公众聚集场所消防安全要求。",  # law 字段命中
            {"law": "公众聚集场所消防安全要求", "source": "y.pdf",
             "chunk_id": "strong"}),
    ]
    r = make_retriever(docs)
    got = r._resolve_docs("公众聚集场所")
    assert got[0].metadata["chunk_id"] == "strong"


def test_resolve_docs_without_query_still_works():
    r = make_retriever(make_docs())
    assert r._resolve_docs("消防法") != []


def test_score_uses_tier_not_substring():
    """内容线索不出现在 law 字段里 —— 用子串判断会恒为 0。

    回归防护：查"电动自行车"时，law='浙江省电动自行车充停…导则' 能命中，
    但 law='不相干文件' + 正文含"电动自行车" 的 chunk 也应拿到非零信号。
    """
    docs = [
        Doc("电动自行车停放管理要求。",  # 仅正文命中，tier=1
            {"law": "不相干文件", "source": "x.pdf", "chunk_id": "body"}),
        Doc("电动自行车停放管理要求。",  # law 字段命中，tier=3
            {"law": "电动自行车管理规定", "source": "y.pdf", "chunk_id": "meta"}),
    ]
    r = make_retriever(docs)
    body = r._score("电动自行车", None, docs[0])
    meta = r._score("电动自行车", None, docs[1])
    assert body > 0, "仅正文命中也必须有信号"
    assert meta > body, "权威 law 字段命中必须更强"
    assert meta == pytest.approx(HybridRetriever.W_LAW, abs=1e-6)


def test_tier_of_unknown_keyword_is_zero():
    r = make_retriever(make_docs())
    assert r._tier_of("不存在于索引中的法律名称", make_docs()[0]) == 0
    assert r._tier_of(None, make_docs()[0]) == 0
