# hybrid.py
"""BM25 + 向量 混合检索 + 查询解析精准定位"""
import re
import jieba
from collections import defaultdict
from langchain_community.vectorstores import FAISS
import logging

logger = logging.getLogger("fire_agents.agent1")

jieba.setLogLevel(20)


# 已知法律名称（可扩展）
KNOWN_LAWS = [
    "中华人民共和国消防法", "消防法",
    "中华人民共和国行政处罚法", "行政处罚法",
    "中华人民共和国行政强制法", "行政强制法",
    "中华人民共和国安全生产法", "安全生产法",
    "中华人民共和国行政复议法", "行政复议法",
    "中华人民共和国治安管理处罚法", "治安管理处罚法",
    "中华人民共和国民事诉讼法", "民事诉讼法",
    "中华人民共和国立法法", "立法法",
    "浙江省消防条例", "高层民用建筑消防安全管理规定",
]


def parse_query(query: str):
    """从查询中提取 (法律名, 条款号)"""
    law = None
    for l in sorted(KNOWN_LAWS, key=len, reverse=True):
        if l in query:
            law = l
            break

    # 匹配 "第六十条" / "第60条" / "第六十条第一款"
    m = re.search(r"第([一二三四五六七八九十百零\d]+)条(第[一二三四五六七八九十\d]+款)?",
                  query)
    article = None
    if m:
        article = "第" + m.group(1) + "条"
        if m.group(2):
            article += m.group(2)

    return law, article


class HybridRetriever:
    def __init__(self, index_dir, embeddings):
        self.vs = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )
        self.all_docs = list(self.vs.docstore._dict.values())

        logger.info(f"🔨 构建 BM25（{len(self.all_docs)} 条）...")
        self.tokenized = [list(jieba.cut(d.page_content)) for d in self.all_docs]

        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.tokenized)
        logger.info("✅ HybridRetriever 就绪")

    def _precise_search(self, law, article, k):
        """精准定位：先按文件名筛，再按条款号匹配"""
        if not law:
            return None

        # 1) 按文件名筛选
        candidates = [
            d for d in self.all_docs
            if law in d.metadata.get("source", "")
        ]
        if not candidates:
            return None
        logger.info(f"    🎯 精准定位：'{law}' 命中 {len(candidates)} 个 chunk")

        # 2) 优先返回含条款号的
        if article:
            with_article = [d for d in candidates if article in d.page_content]
            if with_article:
                logger.info(f"    🎯 精确匹配 '{article}'：{len(with_article)} 条")
                return with_article[:k]

            # 3) 没有含完整条款号的，放宽到含"第X条"里的数字
            m = re.search(r"第([一二三四五六七八九十百零\d]+)条", article)
            if m:
                num = m.group(1)
                fuzzy = [d for d in candidates
                         if f"第{num}条" in d.page_content]
                if fuzzy:
                    logger.info(f"    🎯 模糊匹配 '第{num}条'：{len(fuzzy)} 条")
                    return fuzzy[:k]

        return candidates[:k]

    def search(self, query, k=5):
        # ★ 1) 先尝试精准定位
        law, article = parse_query(query)
        if law:
            precise = self._precise_search(law, article, k)
            if precise:
                return precise

        # ★ 2) 回退到混合检索
        pool = k * 10
        vec_results = self.vs.similarity_search(query, k=pool)
        tokens = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_top = sorted(range(len(bm25_scores)),
                          key=lambda i: -bm25_scores[i])[:pool]

        rrf = defaultdict(float)
        key_to_doc = {}
        for rank, d in enumerate(vec_results, 1):
            key = d.page_content[:80]
            rrf[key] += 1.0 / (60 + rank)
            key_to_doc[key] = d
        for rank, i in enumerate(bm25_top, 1):
            d = self.all_docs[i]
            key = d.page_content[:80]
            rrf[key] += 1.0 / (60 + rank)
            key_to_doc[key] = d

        sorted_keys = sorted(rrf.keys(), key=lambda x: -rrf[x])[:k]
        return [key_to_doc[k] for k in sorted_keys if k in key_to_doc]


_hybrid_cache = {}


def get_hybrid(index_dir, embeddings):
    key = (index_dir, getattr(embeddings, "model", "unknown"))
    if key not in _hybrid_cache:
        _hybrid_cache[key] = HybridRetriever(index_dir, embeddings)
    return _hybrid_cache[key]