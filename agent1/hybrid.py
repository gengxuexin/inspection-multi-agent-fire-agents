# hybrid.py
"""BM25 + 向量混合检索 + 法条精准定位 + 规则精排（rerank）

设计背景（RAGAS 实测驱动）
-------------------------
改造前：faithfulness=0.986 / answer_relevancy=0.88（LLM 层没问题）
        context_precision=0.165 / context_recall=0.60（检索层崩了）

三个根因：
    1. `_precise_search` 三处 `return xxx[:k]` **完全无相关性排序**，
       按 docstore 物理顺序截断 —— 相当于随机返回。
    2. 精准路径命中后**直接 return**，跳过混合检索与 RRF，
       误召回无法被语义信号纠正。
    3. dedup key用 `page_content[:80]`：法条引用重叠（"依照本法第X条"）
       会让多个 chunk 前 80 字符相同 → **误合并**，把正确 chunk 挤掉。

本次改造：
    - 精准定位加**分层打分**（法名命中 > 条款命中 > 款/项命中 > 词面重叠）
    - 精准结果**不再短路**，而是作为"强先验"与混合检索**融合排序**
    - dedup key 改为 `chunk_id`（内容哈希）→ 缺失时回退 md5(全文)
    - 新增**规则精排**：法条结构匹配加权 + 长度惩罚 + 去冗余（MMR）
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import sys
from collections import defaultdict

import jieba
from langchain_community.vectorstores import FAISS

# 允许从任意目录 import 本模块（retrieval_backend 会先注入 sys.path，
# 但 CLI 脚本 / 测试直接 import hybrid 时需要自己兜底）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from chunking import (
    article_match_level,
    article_matches,
    is_standard_catalog,
    is_tier_table_chunk,
    normalize_article,
)

logger = logging.getLogger("fire_agents.agent1")

jieba.setLogLevel(20)


# ==================== 法律法规名识别 ====================
# 实测修复（来自 RAGAS 逐样本诊断）：
#   "公众聚集场所投入使用营业前消防安全检查实行什么制度？" -> 法律名=None
#   "消防行政处罚的裁量基准是什么？"-> 法律名=None
# 精准定位完全不会触发，只能靠向量硬碰。
#
# 问题不在"词典没收录"，而在**索引里的law 字段才是权威**：
#   浙江省消防行政处罚裁量规定(54 chunk)
#   浙江省消防救援总队关于推行消防领轻微违法行为不予处罚制度的实施意见
# 而短名"消防法"会同时匹配 4 个文件，候选被污染。
#
# 方案：**词典只做粗筛，最终以索引 metadata.law/doc_id 精确锁定**。
# 元数据匹配优先于字符串包含，避免短名污染。

# 粗筛词典：命中即认为"这个查询大概率在问这部法律"
LAW_ALIASES: dict[str, str] = {
    "消防法": "消防法",
    "行政处罚法": "行政处罚法",
    "行政强制法": "行政强制法",
    "安全生产法": "安全生产法",
    "行政复议法": "行政复议法",
    "治安管理处罚法": "治安管理处罚法",
    "民事诉讼法": "民事诉讼法",
    "立法法": "立法法",
    "标准化法": "标准化法",
    "消防条例": "消防条例",
    "行政处罚裁量": "行政处罚裁量",
    "裁量基准": "裁量基准",
    "告知承诺": "告知承诺",
    "公众聚集场所": "公众聚集场所",
    "消防监督检查": "消防监督检查",
    "高层民用建筑": "高层民用建筑",
    "电动自行车": "电动自行车",
    "火灾隐患": "火灾隐患",
    "轻微违法": "轻微违法",
    "裁量": "裁量",
}


def doc_law(doc) -> str:
    """取文档的权威法律标识（优先 law 字段，回退 source）。"""
    md = doc.metadata or {}
    law = md.get("law")
    if law:
        return str(law)
    return str(md.get("source", ""))


# 内容线索：某些主题词**只出现在正文里**，metadata.law / source 完全没有。
#
# 真实案例：查询"高层民用建筑消防安全管理规定对电动自行车有什么要求？"
#   - law 字段：'浙江省电动自行车充停、充换场所建设技术导则'、'…超高层建筑…通知'
#     —— 都不含"高层民用建筑"
#   - 而《高层民用建筑消防安全管理规定》**根本不在知识库里**，
#     只有其他文件在正文中"引用"它（如"适用《高层民用建筑消防安全管理规定》
#     第四十七条第六项"）
#
# 只按 metadata 反查会让候选数为 0。必须允许从**正文**兜底召回。
CONTENT_HINTS = [
    "电动自行车",
    "高层民用建筑",
    "裁量基准",
    "轻微违法",
    "告知承诺",
    "公众聚集场所",
]

# tier 先验的权重（见 _resolve_docs 的排序说明）。
# 刻意小于 1：够用来在同一cover 分数下破同分，不够压过内容本身。
_TIER_WEIGHT = 0.3

# 精准候选上限：tier1 正文线索可能召回数百条，截断以控制精排开销。
PRECISE_CAND_CAP = 200


# 法条引用：第六十条 / 第60条 / 第六十条第一款第七项
_RE_ARTICLE_REF = re.compile(r"第\s*[一二三四五六七八九十百千零〇两\d]+\s*条"
                             r"(?:\s*第\s*[一二三四五六七八九十百千零〇两\d]+\s*款)?"
                             r"(?:\s*第\s*[一二三四五六七八九十百千零〇两\d]+\s*项)?")


def parse_query(query: str):
    """从查询中提取 (法律线索, 条款号)。

    条款号已归一为阿拉伯数字形式（"第六十条第一款" -> "第60条第1款"），
    与 chunking.normalize_article 输出格式一致。

    :param query: 用户查询
    :return: (law_keyword|None, article|None)
        law_keyword 是**粗筛线索**（可能多位命中），由调用方拿它去索引里
        按 metadata 反查真正的法律文档。返回 None 表示不该走精准定位。
    """
    if not query:
        return None, None

    law = None
    # 长别名优先（"行政处罚裁量" 比 "裁量" 更有指向性）
    for key in sorted(LAW_ALIASES, key=len, reverse=True):
        if key in query:
            law = key
            break

    m = _RE_ARTICLE_REF.search(query)
    article = normalize_article(m.group(0)) if m else None

    return law, article


def doc_key(doc) -> str:
    """文档去重 key：优先 chunk_id，否则全文 md5。

    原实现用 page_content[:80]，法条互相引用时多个 chunk 前缀相同会被误合并。
    """
    cid = (doc.metadata or {}).get("chunk_id")
    if cid:
        return str(cid)
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


class HybridRetriever:
    """混合检索器：精准定位 + BM25/向量 RRF + 规则精排。"""

    # 条款匹配分级权重（3=精确到项 / 2=到款 / 1=到条）
    #
    # 为什么分级权重差这么大：真实失败案例中，查"第六十条第一款第七项"时，
    # "第六十条第一款第一项"这类解读片段有 11 条，而精确到项的原文只有 2 条。
    # 权重必须让精确项**压倒性胜出**，否则噪声把答案挤出 top-k。
    W_MATCH = {3: 12.0, 2: 8.0, 1: 4.0, 0: 0.0}

    # 精排权重
    W_LAW = 3.0        # 法律名/来源命中
    W_OVERLAP = 1.5    # 查询词覆盖率
    LEN_PENALTY = 0.6  # 过短 chunk 惩罚（信息量不足）
    SHORT_CHUNK = 80

    # IDF 截断范围
    #
    # ★ _IDF_MAX 从 3.0 提到 8.0，这是一个定位出来的实质修复。
    #
    #   实测（N=4019）查「电线电缆类产品属于消防产品吗」的 5 个查询词：
    #       词          df     raw_idf   截断后(MAX=3)  截断后(MAX=8)
    #       电线电缆    12      5.734        3.000          5.734
    #       类产品      17      5.408        3.000          5.408
    #       消防产品    108     3.607        3.000          3.607
    #       属于        123     3.479        3.000          3.479
    #   上限是 3.0 时，**df<200 的词全部压成同一个值**——
    #   真正的答案词「电线电缆」(df=12) 和纯噪声词「属于」(df=123)
    #   权重完全相同，IDF 加权等于没加。
    #
    #   这解释了为什么加了 IDF 加权之后 precision 依然上不去：
    #   问题不在加权公式，在参数把加权效果自己抹平了。
    #   log(4019)=8.30，所以 8.0 是这个规模下的自然上界，
    #   既能让稀有词拉开差距，又不至于让单个生僻词压垮其他信号
    #   （_IDF_MIN 下限仍在，df=N 的"消防"这类词 idf≈0.7，仍然几乎不算）。
    _IDF_MIN = 0.15

    # ★ IDF 上限拆成两个，因为 BM25 与精排共用同一份 idf 字典。
    #
    #   原先只有 _IDF_MAX 一个，改它会**同时**改动
    #     - BM25 召回打分（__init__ 里构建 self._idf）
    #     - 精排覆盖率（_rerank 里 weights 直接读 self._idf）
    #
    #   实测教训（47 条 A/B）：改 _IDF_MAX 3.0->8.0 后
    #     top5 完全相同   8/47  (17%)
    #     集合大变       13/47  (28%)
    #   也就是说**召回阶段早就变了**，而我当时以为自己"只改了精排"。
    #   于是把时间花在了精排覆盖率上（改 B 方案），
    #   结果分层数字逐位不变 —— 瓶颈根本不在那里。
    #
    #   教训：**「改一个常量」不等于「改一个变量」。**
    #   共享的可配置项是调参时最大的陷阱，必须先确认它的作用域。
    #
    # 现在两者独立，才能做真正的变量隔离实验。

    # ★ 三轮 A/B 实测后的最终状态（详见 docs/rerank-ab-log.md）
    #
    #   BM25 保持 3.0：这是已验证过的召回配置，不动。
    #   精排也是 3.0：试过 8.0 + idf²，机制上消除了覆盖率饱和，
    #   但把「跨文档综合」类的 recall 压下去了
    #   （-0.063 ~ -0.085，是唯一超出噪声底的指标），
    #   而这类样本占 45%（21/47）。代价 > 收益，回退。
    #
    #   **保留拆分的意义**：BM25 与精排不再共用 idf 上限，
    #   今后调精排不会连带影响召回 —— 这是本轮最重要的结构性收获。
    _IDF_MAX_BM25 = 3.0
    _IDF_MAX_RERANK = 3.0

    def __init__(self, index_dir, embeddings, rrf_k: int = 60):
        self.vs = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )
        self.all_docs = list(self.vs.docstore._dict.values())
        self.rrf_k = rrf_k

        # BM25 语料：预先分词一次，避免每次 search 重复切词
        logger.info("构建 BM25（%d 条）...", len(self.all_docs))
        self.tokenized = [list(jieba.cut(d.page_content)) for d in self.all_docs]

        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.tokenized)

        # ★ 文档频次表（供精排做 IDF 加权，见 _rerank）
        #
        # 为什么需要：47 条评测里"消防产品目录中的电线电缆类产品包括哪些？"
        # 一直是 rec=0，而BM25 明确把它排在**第 1 名**。逐层排查后确认：
        #   - 分块没问题：答案完整存在于 1 个 306 字的 chunk
        #   - 降权没问题：P2 已按查询意图豁免
        #   - 候选池没问题：BM25 rank=1，必进pool=50
        # 真正的杀手是**词面覆盖率在比谁复述查询里的烂大街词更多**：
        #   答案 chunk 覆盖 0.28（命中"电线电缆""包括"等稀有词）
        #   噪声 chunk 覆盖 0.50（命中"消防""产品""目录""认证"等高频词）
        # 语料 3990 条里"消防"几乎处处都在，它的命中不构成相关性证据；
        # 而"电线电缆"只出现在 12 条里，命中一次就是强证据。
        #
        # 所以覆盖率必须按 IDF 加权，而不是数命中个数。
        df: dict = defaultdict(int)
        for toks in self.tokenized:
            for w in set(toks):
                df[w] += 1
        n_docs = max(len(self.all_docs), 1)
        # idf = log(N / (1 + df))，再截到 [_IDF_MIN, _IDF_MAX] 防止极端值
        # 精排要用的原始 df 与文档数（自己保留一份，不从截断后的 idf 反推）
        self._df = df
        self._n_docs = n_docs

        # BM25 与精排不再共用同一份 idf —— 见 _IDF_MAX_BM25 的注释
        self._idf = {
            w: min(self._IDF_MAX_BM25,
                  max(self._IDF_MIN, math.log(n_docs / (1.0 + c))))
            for w, c in df.items()
        }

        # law_keyword -> {(law, source): tier} 反向表
        #
        # tier 语义：
        #   3=线索出现在 law 字段（权威法律名）
        #   2=线索只出现在文件名
        #   1=线索只出现在正文（其他文件引用了该主题，最弱的证据）
        #
        # 分层的必要性：短名"消防法"在文件名里也会命中
        # "（应急[2021]34号）贯彻实施新修改«中华人民共和国消防法»的通知.pdf"，
        # 只靠集合无法区分主从关系。真正的兜底是 _precise_search 的分级
        # 过滤 —— 通知里的 chunk 匹配不到条款，会被剔除。
        self._law_index = defaultdict(dict)
        for d in self.all_docs:
            md = d.metadata or {}
            src = str(md.get("source", ""))
            law = str(md.get("law") or "")
            ident = (law, src)
            for key in LAW_ALIASES:
                if key in law:
                    self._law_index[key][ident] = 3
                elif key in src:
                    self._law_index[key].setdefault(ident, 2)
            for key in CONTENT_HINTS:
                if key in d.page_content:
                    self._law_index[key].setdefault(ident, 1)
        logger.info("HybridRetriever 就绪（%d 个检索线索）", len(self._law_index))

    # ---------- 精准定位 ----------
    def _resolve_docs(self, law_keyword, query=None, cap=None):
        """粗筛线索 -> 候选 chunk 列表。

        排序：**词面相关性主导，tier 只作弱先验**。

        早期版本让 tier 绝对优先，实测把正确 chunk 压到 rank 6~10
        （查询"消防行政处罚的裁量基准是什么"时，权威 law 名命中的
        裁量基准表格排在"第二条 裁量定义"之前）。但 tier 只是"这个文件
        名义上相关"的弱证据，真正决定相关性的是 chunk 本身讲不讲这个问题。

        因此用加权综合分：tier 贡献一个有界的小常数，词面命中率才是主项。
        这样既保留了元数据先验，又不会让它压过实际内容。

        :param law_keyword: LAW_ALIASES / CONTENT_HINTS 里的线索
        :param query: 原始查询，用于按词面命中率排序
        :param cap: 候选上限，防止 tier1 线索召回数百条拖慢精排
        """
        if not law_keyword:
            return []
        tiers = self._law_index.get(law_keyword)
        if not tiers:
            # 线索在索引里找不到对应文档，不做无谓降级
            return []

        out = []
        for d in self.all_docs:
            md = d.metadata or {}
            key = (str(md.get("law") or ""), str(md.get("source", "")))
            if key in tiers:
                out.append(d)

        if not out:
            return []

        qtokens = [t for t in jieba.cut(query or "") if t.strip() and len(t) > 1]

        def rank_key(d):
            md = d.metadata or {}
            tier = tiers[(str(md.get("law") or ""),
                          str(md.get("source", "")))]
            hits = sum(1 for t in set(qtokens) if t in d.page_content)
            cover = hits / len(qtokens) if qtokens else 0.0
            # tier 权重刻意设小：够用来破同分，不够用来压过内容
            return -(_TIER_WEIGHT * tier / 3.0 + cover)

        out.sort(key=rank_key)
        return out[:cap] if cap else out

    def _precise_search(self, law_keyword, article, k, query=None):
        """法条精准定位：**按匹配级别分层**，而不是布尔匹配。

        :return: 命中返回 [(doc, score), ...] 降序；未命中返回 None
        """
        if not law_keyword:
            return None

        candidates = self._resolve_docs(law_keyword, query=query,
                                        cap=PRECISE_CAND_CAP)
        if not candidates:
            return None
        logger.info("精准定位线索 '%s' 命中 %d 个 chunk", law_keyword, len(candidates))

        if not article:
            # 无条款号：按查询词覆盖率取前 k，避免整部法律的文件序噪声
            return None

        # ★ 按分级筛选：只要有 lv>=2 的命中，就不要 lv=1 的
        # （lv=1 是"第六十条的邻居"，正是把答案挤掉的噪声源）
        levels = [(article_match_level(d.page_content, article), d)
                  for d in candidates]
        best = max((lv for lv, _ in levels), default=0)
        if best < 2:
            logger.info("条款 '%s' 最高仅匹配到 lv=%d，不走精准强先验",
                        article, best)
            return None

        pool = [d for lv, d in levels if lv >= 2]
        scored = [(d, self._score(law_keyword, article, d)) for d in pool]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    # ---------- 规则精排 ----------
    # 结构性噪声降权（P1 表格型/ P2 引用名录型）
    #
    # 为什么是"降权"而不是"过滤"：
    #   裁量基准表格恰恰是问"怎么处罚"时的正确答案，标准名录里的
    #   GB/T 号也可能是问题主体（如"依据哪个规范验收"）。一刀切删除会
    #   直接砸掉 recall（已经 1.0 的指标不能再掉）。所以只压排序权重，
    #   让强信号（条款精确命中）依然能把它们捞回top-k。
    #
    # 幅度刻意小于 W_MATCH[2]=8.0，也小于 W_OVERLAP 满档 1.5：
    # 只压得动"语义相近但没答到点上"的噪声，压不动真答案。
    PENALTY_TIER_TABLE = 0.6      # P1：档位组仍并列多档位
    PENALTY_STD_CATALOG = 1.2     # P2：引用清单，不含处罚结论
    # 表格主题词：查询命中这些词时，说明用户问的就是处罚裁量本身
    _TIER_TABLE_CUES = ("处罚", "裁量", "档", "情节", "后果", "从重", "从轻")

    def _tier_of(self, law_keyword, doc) -> int:
        """取该文档对当前线索的证据强度（0~3），查预建索引，不做子串扫描。"""
        if not law_keyword:
            return 0
        tiers = self._law_index.get(law_keyword)
        if not tiers:
            return 0
        md = doc.metadata or {}
        key = (str(md.get("law") or ""), str(md.get("source") or ""))
        return tiers.get(key, 0)

    def _score(self, law_keyword, article, doc) -> float:
        """对单个 chunk 打相关性分（越大越相关）。"""
        text = doc.page_content
        score = 0.0

        # 法律线索证据强度：tier3=权威法律名 / 2=文件名 / 1=仅正文提及
        #
        # 用 tier 而不是 `law_keyword in doc_law(doc)`：内容线索（如"电动自行车"）
        # 压根不出现在 law 字段里，子串判断会恒为 0，白白丢掉这路信号。
        score += self.W_LAW * self._tier_of(law_keyword, doc) / 3.0

        if article:
            score += self.W_MATCH.get(article_match_level(text, article), 0.0)

        return score

    def _rerank(self, query, law_keyword, article, docs, k):
        """规则精排：结构匹配 + 词面覆盖率 + 长度惩罚 + MMR 去冗余。

        :param docs: 候选 Document 列表（已去重）
        :return: 重排后 top-k
        """
        tokens = [t for t in jieba.cut(query) if t.strip()]
        token_set = set(tokens)
        scored: list[tuple[float, object]] = []

        # ★ IDF 加权覆盖率
        #
        # 旧实现数"命中了几个词"，等价于让高频词和稀有词等权。实测后果：
        # 查"电线电缆类产品"，答案 chunk 命中 2 个稀有词得 0.28，
        # 而噪声 chunk 复述"消防/产品/目录/认证"得 0.50，直接把答案挤掉。
        #
        # 改成按 idf 加权后，"电线电缆"（12/3990）的权重远高于
        # "消防"（近乎全库都有），答案凭**稀有词命中**就能反超。
        # 精排用自己的 idf（上限 _IDF_MAX_RERANK），不复用 BM25 那份。
        # 不复用是必须的：两者服务于不同目标 ——
        # BM25 要的是"词频统计上的区分度"，精排要的是"稀有词优先"，
        # 上限不同才能各自成立。
        if token_set:
            default_idf = self._IDF_MIN
            weights = {}
            for t in token_set:
                c = self._df.get(t, 0)
                raw = math.log(self._n_docs / (1.0 + c)) if self._n_docs else default_idf
                weights[t] = min(self._IDF_MAX_RERANK,
                                 max(self._IDF_MIN, raw))
            # 归一化基准必须用同一个映射，否则分子多出 ln 项、分母还是线性
            # idf 和，覆盖率会整体超过 1，把 W_OVERLAP 的量纲撑破
            # （分数看似变高，但相对排序被压缩，等于白改）。
            total_w = sum(weights.values()) or 1.0
        else:
            weights, total_w = {}, 1.0

        for d in docs:
            score = self._score(law_keyword, article, d)
            text = d.page_content

            # 词面覆盖率（IDF 超线性加成）
            #
            # ★ 这里踩了两次坑才找对方向，完整记录以免再犯：
            #
            # 【坑 1】原来是 `if t in text` —— bool 命中，命中一次和五次同权。
            #   后果：覆盖率迅速饱和。查"电线电缆类产品"时 10 个候选的 ovl
            #   全顶到同一个值，BM25 把 P4 标准条目排到第 1 的判断
            #   在精排这关被抹平，P4 目录切分等于白做。
            #   教训：**召回侧的改进会被精排侧的判据设计抵消。**
            #   改分块前必须问："切出来的更好片段，精排会不会把它排上去？"
            #
            # 【坑 2】第一版改成 tf 饱和递增 idf*(1+ln(1+tf))，
            #   想奖励"命中多次"。结果更糟：法条文本里重复的往往是虚词。
            #   实测"建筑消防设施包括哪些"的噪声片段
            #       tf={'消防设施':1, '包括':7, '建筑':4}
            #   光靠"包括"重复 7 次就刷到 0.85 分，
            #   答案片段只命中 1 词 tf=1，排名 212 → 565（大幅后退）。
            #   —— 重复次数不是相关性的证据。
            #
            # 【现在】只按 idf 计权，稀有词用 idf^2 做超线性加成：
            #   - 稀有词命中一次就能压过所有常见词的命中总和；
            #   - 常见词重复不再有额外收益，"包括"刷 7 次也没用；
            #   - 不依赖 tf，也就无需长度归一化 —— 若按 tf 除以文本长度，
            #     会出现"短片段更容易满分"的副作用，
            #     那样89 字的 P4 目录条目会被长段落比下去（正好和目标相反）。
            # 词面覆盖率（IDF 加权，线性）
            #
            # ★ 这里试过 idf² 超线性加成，三轮 A/B 后回退（见 docs/rerank-ab-log.md）：
            #
            #   动机：原来的 `if t in text` 是 bool 命中，
            #   命中 1 次和5 次同权 → 10~32 个候选并列满分，
            #   BM25 把 P4 标准条目排第1 的判断被抹平。
            #   idf² 能让「稀有词命中一次」压过「常见词命中多次」。
            #
            #   实测后果（47 条）：
            #       直查法条     n=20   recall +0.050   ← 单文档精确查找，确实受益
            #       跨文档综合    n=21   recall -0.143   ← 答案横跨多份文件受损
            #       陷阱题       n= 6   recall -0.167
            #   5 条退化样本的 top5 重叠仅 1~3/5 —— 答案片段被压出 top5。
            #
            #   原因：跨文档类答案分布在多份文件里，
            #   **每个片段都只命中 1 个查询词**，
            #   而单词命中在 idf² 下权重极高，压过了命中 2-3 词的片段。
            #   idf² 优化的是"找准那一条"，代价是"找齐那几条"——
            #   跨文档类占样本 45%，代价超过收益。
            #
            #   保留 bool→线性加权这一段的原因：
            #   IDF 加权本身是对的（答案词"电线电缆"df=12 vs
            #   噪声词"消防"df≈2000，本就该有区别）。
            #   错的是"超线性"这个选择，不是"加权"本身。
            if token_set:
                got = sum(w for t, w in weights.items() if t in text)
                score += self.W_OVERLAP * (got / total_w)


            # 过短 chunk 信息量不足，降权（治"precision 低"的常见噪声源）
            if len(text) < self.SHORT_CHUNK:
                score -= self.LEN_PENALTY

            # ---- P1：裁量基准档位表降权 ----
            # 一段里并列"严重/一般/较轻"多个档位，语义中心分散，
            # 通用查询下容易"沾边即召回"。但查询若点名要处罚/裁量情节，
            # 说明它就是答案，此时不降权。
            if is_tier_table_chunk(d) and not any(
                c in query for c in self._TIER_TABLE_CUES
            ):
                score -= self.PENALTY_TIER_TABLE

            # ---- P2：标准引用名录降权 ----
            # 必须把 query 传进去：用户问"产品目录有哪些"时，
            # 这些 chunk 是唯一答案，is_standard_catalog 会按查询意图豁免。
            if is_standard_catalog(text, d.metadata, query=query):
                score -= self.PENALTY_STD_CATALOG

            scored.append((score, d))

        scored.sort(key=lambda x: -x[0])
        return self._mmr_select(scored, k)

    @staticmethod
    def _mmr_select(scored, k, lam: float = 0.75):
        """MMR（最大边际相关性）选top-k，抑制同文档的重复片段。

        :param lam: 相关性权重（1=纯相关，0=纯多样性）
        """
        if not scored:
            return []
        vals = [s for s, _ in scored]
        vmin, vmax = min(vals), max(vals)
        rng = (vmax - vmin) or 1.0

        # 预先缓存 token 集合，避免 O(n^2) 重复分词
        remain = [((s - vmin) / rng, d, set(jieba.cut(d.page_content)))
                  for s, d in scored]
        selected: list[tuple[float, object, set]] = []

        while remain and len(selected) < k:
            best_i, best_val = -1, float("-inf")
            for i, (rel, d, dt) in enumerate(remain):
                if not selected:
                    score = rel
                else:
                    # 与已选内容的最大 Jaccard 相似度
                    sim = 0.0
                    for _, _, st in selected:
                        union = len(dt | st)
                        if union:
                            j = len(dt & st) / union
                            if j > sim:
                                sim = j
                    score = lam * rel - (1 - lam) * sim
                if score > best_val:
                    best_i, best_val = i, score
            selected.append(remain.pop(best_i))
        return [d for _, d, _ in selected]

    # ---------- 主入口 ----------
    def search(self, query, k=5):
        """混合检索主入口。

        流程：精准定位得强先验 → BM25 + 向量 RRF → 融合 → 规则精排 → top-k

        :param query: 查询文本
        :param k:     返回条数
        :return: Document 列表
        """
        if not query:
            return []

        law_kw, article = parse_query(query)

        # ★ 1) 精准定位 —— 只作为强先验，不短路
        precise = self._precise_search(law_kw, article, k, query=query) or []
        precise_docs = [d for d, _ in precise]
        precise_scores = {doc_key(d): s for d, s in precise}

        # ★ 2) BM25 + 向量 RRF
        pool = max(k * 10, 50)
        vec_results = self.vs.similarity_search(query, k=pool)
        tokens = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_top = sorted(range(len(bm25_scores)),
                          key=lambda i: -bm25_scores[i])[:pool]

        rrf = defaultdict(float)
        key_to_doc = {}
        for rank, d in enumerate(vec_results, 1):
            key = doc_key(d)
            rrf[key] += 1.0 / (self.rrf_k + rank)
            key_to_doc.setdefault(key, d)
        for rank, i in enumerate(bm25_top, 1):
            d = self.all_docs[i]
            key = doc_key(d)
            rrf[key] += 1.0 / (self.rrf_k + rank)
            key_to_doc.setdefault(key, d)

        # ★ 3) 精准先验注入
        #
        # 关键：RRF 自身的量级是 1/(60+rank) ≈ 0.016 ~ 0.008，
        # 两条向量+BM25 双命中也就 0.03 左右。而精准分数是 12~15 量级。
        # 若按比例归一化再乘系数（旧的 0.02 * ratio），boost 只有 0.02，
        # **与 RRF 原分同量级，起不到任何排序作用** —— 这正是"精准命中
        # 仍然被噪声挤掉"的直接原因。
        #
        # 改为直接加一个绝对量级的 bonus：LV3 命中必须排在所有仅靠
        # 语义相似度得分的 chunk 之前，无论语义相似度多高。
        PRECISE_BONUS_BASE = 1.0   # 归一化用
        PRECISE_BONUS_LV = {3: 0.30, 2: 0.15}
        if precise_scores:
            base = max(precise_scores.values()) or 1.0
            for d in precise_docs:
                key = doc_key(d)
                key_to_doc.setdefault(key, d)
                ratio = precise_scores[key] / base
                lv = article_match_level(d.page_content, article) if article else 0
                bonus = PRECISE_BONUS_LV.get(lv, 0.05) * ratio
                rrf[key] += bonus

        # ★ 4) 候选池：RRF Top + 全部精准命中
        ordered = sorted(rrf.keys(), key=lambda x: -rrf[x])[:max(pool, len(precise_docs))]
        candidates = []
        seen = set()
        for key in ordered:
            if key in seen:
                continue
            seen.add(key)
            candidates.append(key_to_doc[key])
        for d in precise_docs:
            key = doc_key(d)
            if key not in seen:
                seen.add(key)
                candidates.append(d)

        # ★ 5) 规则精排 → top-k
        return self._rerank(query, law_kw, article, candidates, k)


_hybrid_cache = {}


def get_hybrid(index_dir, embeddings, rrf_k: int = 60):
    """获取（并缓存）混合检索器。"""
    key = (index_dir, getattr(embeddings, "model", "unknown"), rrf_k)
    if key not in _hybrid_cache:
        _hybrid_cache[key] = HybridRetriever(index_dir, embeddings, rrf_k=rrf_k)
    return _hybrid_cache[key]