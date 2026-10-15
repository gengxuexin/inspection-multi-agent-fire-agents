# test_chunking.py
"""chunking.py 单元测试：PDF 清洗 + 条款解析 + 法条感知分块

纯函数逻辑，不依赖 langchain / FAISS /模型，可直接运行。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "agent1"))

from chunking import (  # noqa: E402
    LegalChunker,
    all_articles_of,
    article_matches,
    article_match_level,
    article_of,
    clean_pdf_text,
    cn_to_int,
    is_standard_catalog,
    is_tier_table_chunk,
    normalize_article,
    regroup_tier_table,
    regroup_catalog_entries,
)


# ==================== 中文数字 ====================
@pytest.mark.parametrize("raw,expected", [
    ("六十", 60), ("60", 60), ("一", 1), ("十", 10), ("十五", 15),
    ("二十三", 23), ("一百", 100), ("一百一十", 110), ("两百", 200),
    ("两百零五", 205), ("一千零五", 1005), ("五百六十", 560),
])
def test_cn_to_int(raw, expected):
    assert cn_to_int(raw) == expected


@pytest.mark.parametrize("bad", ["", "abc", "第X条", None])
def test_cn_to_int_invalid(bad):
    assert cn_to_int(bad) is None


# ==================== PDF 清洗 ====================
def test_clean_removes_page_marker():
    assert "第 14 页" not in clean_pdf_text("正文\n--- 第 14 页 ---\n后文")


def test_clean_removes_pdf_private_mark():
    out = clean_pdf_text("条文\x14\x00 0 0 3 2 \x14内容")
    assert "\x14" not in out
    assert "0 0 3 2" not in out
    assert "条文" in out and "内容" in out


def test_clean_removes_repeated_footer():
    """跨页重复的页脚必须清掉（否则纯噪声进chunk，precision 被拉低）。"""
    raw = "第一条正文内容甲。\n版权所有\n第二条正文内容乙。\n版权所有"
    out = clean_pdf_text(raw)
    assert out.count("版权所有") == 0
    assert "第一条正文内容甲" in out


def test_clean_keeps_chapter_heading():
    """章节标题虽短但语义重要，不能被当页脚删掉。"""
    raw = "正文甲。\n第一章总则\n正文乙。\n第一章总则"
    assert "第一章总则" in clean_pdf_text(raw)


def test_clean_collapses_blank_lines():
    out = clean_pdf_text("甲。\n\n\n\n\n乙。")
    assert "\n\n\n" not in out


def test_clean_empty_input():
    assert clean_pdf_text("") == ""
    assert clean_pdf_text(None) == ""


# ==================== 条款解析 ====================
@pytest.mark.parametrize("raw,expected", [
    ("第六十条", "第60条"),
    ("第60条", "第60条"),
    ("第一百一十条", "第110条"),
    ("本法第15条", "第15条"),
    ("第六十条第一款第七项", "第60条第1款第7项"),
    ("第60条第1款", "第60条第1款"),
])
def test_normalize_article(raw, expected):
    assert normalize_article(raw) == expected


def test_normalize_article_invalid():
    assert normalize_article("没有条款") is None
    assert normalize_article("") is None


def test_article_of():
    assert article_of("第六十条违反本法") == "第60条"
    assert article_of("前面是引用第五十条，然后第六十条才是本条") == "第50条"
    assert article_of("无条款文本") is None


def test_all_articles_of():
    assert all_articles_of("依照第五十条和第六十条的规定") == ["第50条", "第60条"]
    assert all_articles_of("") == []


@pytest.mark.parametrize("text,target,expected", [
    # 中文数字查询 vs 中文数字文本
    ("第十五条公众聚集场所", "第15条", True),
    ("第十五条公众聚集场所", "第十五条", True),
    # 数字查询 vs 中文数字文本 —— 归一化后必须能配上
    ("第六十条第一款第七项的规定处罚", "第60条第一款", True),
    ("第六十条违反本法规定", "第60条第一款", True),   # 降级到"条"级
    # 不相关
    ("第二条本法适用于灭火", "第60条", False),
    ("听证程序应当告知当事人", "第15条", False),
])
def test_article_matches(text, target, expected):
    assert article_matches(text, target) is expected


def test_article_matches_invalid():
    assert article_matches("", "第60条") is False
    assert article_matches("第六十条", "") is False


# ==================== 法条感知分块 ====================
LAW_TEXT = """中华人民共和国消防法
第一章总则
第二条本法适用于灭火、火灾预防、扑救和火灾救援工作。
第三章火灾预防
第十五条公众聚集场所投入使用、营业前消防安全检查实行告知承诺管理。
第六十条违反本法规定，有下列行为之一的，责令改正：
（一）消防设施配置不符合国家标准；
（七）其他消防安全违法行为依照本法第六十条第一款第七项的规定处罚。
"""


def test_chunker_splits_on_article_boundary():
    """核心断言：法条不能被拦腰截断，每块首条条款号必须自洽。

    期望 5 块：法律名+第一章（前言段）、第二条、第三章标题、第十五条、第六十条。
    章标题独立成块，不粘在法条尾部造成跨界污染。
    """
    docs = LegalChunker(law="中华人民共和国消防法").split_text(
        LAW_TEXT, source="中华人民共和国消防法.pdf")
    assert len(docs) == 5
    articles = [d.metadata["article"] for d in docs]
    assert "第2条" in articles and "第15条" in articles and "第60条" in articles
    # 元数据条款号必须与正文首条条款一致（归一后比对，原文是中文数字）
    for d in docs:
        assert d.metadata["article"] == article_of(d.page_content)


def test_chunker_no_cross_contamination():
    """第二章标题不能粘在第二条的 chunk 尾部。"""
    docs = LegalChunker(law="中华人民共和国消防法").split_text(
        LAW_TEXT, source="中华人民共和国消防法.pdf")
    art2 = next(d for d in docs if d.metadata["article"] == "第2条")
    assert "第三章" not in art2.page_content
    # 章标题应独立成块
    assert any("第三章" in d.page_content for d in docs)


def test_chunker_metadata_fields():
    docs = LegalChunker(law="中华人民共和国消防法").split_text(
        LAW_TEXT, source="中华人民共和国消防法.pdf")
    for d in docs:
        assert d.metadata["law"] == "中华人民共和国消防法"
        assert d.metadata["source"] == "中华人民共和国消防法.pdf"
        assert d.metadata["chunk_id"]and len(d.metadata["chunk_id"]) == 16
        assert d.metadata["chars"] == len(d.page_content)


def test_chunker_ids_unique():
    """chunk_id 必须在块间唯一 —— 检索层靠它做精确去重。"""
    docs = LegalChunker(law="消防法").split_text(LAW_TEXT, source="消防法.pdf")
    ids = [d.metadata["chunk_id"] for d in docs]
    assert len(ids) == len(set(ids))


def test_chunker_keeps_short_article():
    """短法条本身就是完整语义单元，不能被 min_chunk 过滤掉（否则 recall 受损）。"""
    text = "第十五条公众聚集场所投入使用、营业前消防安全检查实行告知承诺管理。"
    docs = LegalChunker(min_chunk=200).split_text(text, source="消防法.pdf")
    assert len(docs) == 1
    assert docs[0].metadata["article"] == "第15条"


def test_chunker_drops_pure_noise():
    """无中文、无实质内容的短片段应被丢弃。"""
    docs = LegalChunker(min_chunk=200).split_text(
        "版权所有\n2021-10-01\n- 14 -", source="消防法.pdf")
    assert len(docs) == 0


def test_chunker_long_article_respects_size():
    """超长法条应带 overlap 二次切分，且每块不超 chunk_size 太多。"""
    long_article = "第五十条" + "规定当事人应当履行消防安全义务。" * 60+ "。"
    docs = LegalChunker(chunk_size=300, chunk_overlap=50).split_text(
        long_article, source="消防法.pdf")
    assert len(docs) > 1
    # overlap 会让块略超 chunk_size，控制在 1.6倍以内即合理
    for d in docs:
        assert len(d.page_content) <= 300 * 1.6


def test_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        LegalChunker(chunk_size=100, chunk_overlap=100)


def test_chunker_preserves_parent_metadata():
    """权限元数据必须继承下来 —— 丢了会造成越权召回。"""
    from chunking import _Doc
    parent = _Doc(
        page_content=LAW_TEXT,
        metadata={"dept": "消防支队", "secret_level": 2, "source": "消防法.pdf"},
    )
    docs = LegalChunker().split_documents([parent])
    assert docs
    for d in docs:
        assert d.metadata["dept"] == "消防支队"
        assert d.metadata["secret_level"] == 2


def test_chunker_empty_text():
    assert LegalChunker().split_text("", source="x.pdf") == []

# ==================== 条款匹配分级 ====================
# 背景（真实 RAGAS 失败案例）：评估问题"《消防法》第六十条第一款第七项规定了什么？"
# 在 top_k=5 下 precision=0 / recall=0，召回的 5 条全是"第六十条的邻居"
# （第一项、第六项、第二项…），唯独缺第七项本身。
#
# 旧 article_matches 的层级降级只比到"条"级，把款/项差异全丢了，于是
# "第六十条第一款第一项"与"第六十条第一款第七项"同权，解读性片段把答案挤出 top-k。
#
# 反直觉点：法条原文的项写作"（七）"而不是"第7项"，所以纯字面匹配对
# **真正的答案**返回 False、对噪声返回 True —— 区分度完全反了。

_TGT_ITEM = "第六十条第一款第七项"


def test_int_to_cn():
    from chunking import _int_to_cn
    assert _int_to_cn(1) == "一"
    assert _int_to_cn(7) == "七"
    assert _int_to_cn(9) == "九"
    assert _int_to_cn(10) == "十"
    assert _int_to_cn(15) == "十五"
    assert _int_to_cn(20) == "二十"
    assert _int_to_cn(99) == "九十九"


def test_clause_no_must_strip_suffix():
    """split("第") 后每段仍带 "1款"/"7项" 后缀，不剥离则解析失败返回 None。"""
    from chunking import _clause_no
    assert _clause_no("第60条第1款第7项", 1) == 60
    assert _clause_no("第60条第1款第7项", 2) == 1
    assert _clause_no("第60条第1款第7项", 3) == 7
    assert _clause_no("第60条", 2) is None
    assert _clause_no("第60条", 3) is None


def test_all_item_markers():
    from chunking import _all_item_markers
    assert _all_item_markers("（一）甲；（七）乙") == {1, 7}
    assert _all_item_markers("(一)甲 (二)乙") == {1, 2}
    assert _all_item_markers("无枚举") == set()


def test_match_level_original_enumerated_item_is_3():
    """法条原文形态："第六十条 …… （七）对火灾隐患……"  必须判3级。

    这是真正答案的典型写法，纯字面匹配会漏掉。
    """
    text = ("第六十条 单位违反本法规定，有下列行为之一的，责令改正："
            "（六）人员密集场所在门窗上设置影响逃生和灭火救援的障碍物的；"
            "（七）对火灾隐患经消防救援机构通知后不及时采取措施消除的。")
    assert article_match_level(text, _TGT_ITEM) == 3


def test_match_level_normalized_full_string_is_3():
    assert article_match_level("依据第60条第1款第7项的规定", "第60条第1款第7项") == 3


def test_match_level_other_item_demoted_to_1():
    """引用别项的解读片段只能算"到条"，不能与原文同权。"""
    for text in (
        "依据《消防法》第六十条第一款第一项、第二项的规定已依法实施行政处罚",
        "按照《消防法》第六十条第一款第六项，责令改正并处罚款",
        "依据《消防法》第六十条第一款第三项至第六项的规定责令改正",
    ):
        assert article_match_level(text, _TGT_ITEM) == 1, text


def test_match_level_article_only_is_1():
    """只提条号、既不提款也不提项，不能被误升到"到款"级。"""
    assert article_match_level(
        "第六十条违反本法规定的，责令改正并处罚款", _TGT_ITEM) == 1


def test_match_level_mentions_clause_is_2():
    """明确提到目标款号但非该项-> 2 级。"""
    assert article_match_level(
        "违反《消防法》第六十条第一款规定的，处五千元以上五万元以下罚款",
        _TGT_ITEM) == 2


def test_match_level_unrelated_is_0():
    assert article_match_level(
        "高层民用建筑因为维修需要停用建筑消防设施，但未制定预案", _TGT_ITEM) == 0


def test_match_level_empty_and_garbage():
    assert article_match_level("", _TGT_ITEM) == 0
    assert article_match_level("第六十条", "") == 0
    assert article_match_level("第六十条", " nonsense ") == 0
    assert article_match_level(None, _TGT_ITEM) == 0


def test_match_level_article_only_target():
    """目标只有条号时，命中同条即 3 级（无更细层级可区分）。"""
    assert article_match_level("第六十条 单位违反本法规定的", "第六十条") == 3
    assert article_match_level("第六十一条 其他规定", "第六十条") == 0


def test_article_matches_agrees_with_level():
    """布尔接口必须等价于 level >= 1，避免两套语义打架。"""
    assert article_matches("第六十条 单位违反本法规定的", _TGT_ITEM) is True
    assert article_matches("无关内容", _TGT_ITEM) is False


def test_match_level_brackets_halfwidth():
    """半角括号枚举项同样要识别。"""
    text = "第六十条单位违反本法规定的，(七)对火灾隐患经通知后不及时消除的。"
    assert article_match_level(text, _TGT_ITEM) == 3


# ============================================================
# P1 裁量基准档位表重组
# 样本取自生产index.pkl 的真实 chunk（PDF 折行形态），
# 不手工编造 —— 手编样本会掩盖真实折行模式（已踩坑）。
# ============================================================

_TIER_TABLE_REAL = (
    "第二十三条\n"
    "定》第五十条\n"
    "严重1.申请人以欺骗、贿赂等不正当手段取得注册消防工程师资格注册的。\n"
    "\n"
    "对未经准予变\n"
    "更注册而继续\n"
    "执业的行政处\n"
    "师管理规定》 第二\n"
    "十二条\n"
    "定》第五十二\n"
    "严重 1.变更后仍以原注册人名义从事执业活动的。\n"
    "一般 1.未按规定办理变更注册手续的。\n"
    "较轻 1.变更后未及时办理备案的。\n"
    "对消防安全技\n"
    "术文件未经注\n"
    "册消防工程师\n"
    "签名、加盖执\n"
    "业印章的行\n"
    "为\n"
    "严重 1.技术文件上全部签字均非本所注册工程师的。\n"
    "较轻 1.个别签字非本所注册工程师的。"
)


def test_regroup_tier_table_splits_multiple_offenses():
    """459 字混 5 个主题的表格 -> 4 个独立语义单元。"""
    groups = regroup_tier_table(_TIER_TABLE_REAL)
    assert groups is not None
    assert len(groups) >= 3, "应拆出多个违法行为，实际 %d 组" % len(groups)


def test_regroup_tier_table_restores_wrapped_names():
    """PDF 折行打散的左列名称必须还原，否则语义单元没有主语。"""
    groups = regroup_tier_table(_TIER_TABLE_REAL)
    joined = "\n".join(groups)
    # "对未经准予变" + "更注册而继续" + "执业的行政处" 应还原成一个名称
    assert "更注册" in joined
    assert any("未经准予变更" in g for g in groups), \
        "名称行未还原：%r" % (groups,)
    assert any("消防安全技" in g and "签名" in g for g in groups), \
        "第二组违法行为丢失"


def test_regroup_tier_table_keeps_tier_with_its_offense():
    """档位行必须归属到正确的违法行为组，不能串组。

    这是第三次迭代修掉的 bug：曾把第二组的"较轻"分给第一组。
    """
    groups = regroup_tier_table(_TIER_TABLE_REAL)
    # 找含"未经准予变更"的那组，它应含"变更后仍以原注册人名义"
    grp = next(g for g in groups if "未经准予变更" in g)
    assert "变更后仍以原注册人名义" in grp
    # 而"个别签字非本所"属于另一组，不应出现在这组里
    assert "个别签字非本所" not in grp


def test_regroup_tier_table_no_duplicate_head():
    """表头只应出现一次，不能被重复拼进名称行。"""
    groups = regroup_tier_table(_TIER_TABLE_REAL)
    assert groups[0].count("定》第五十条") <= 1, \
        "表头重复：%r" % (groups[0][:60],)


def test_regroup_tier_table_returns_none_for_plain_text():
    assert regroup_tier_table("第二十三条注册消防工程师应当依法执业。") is None
    assert regroup_tier_table("") is None


def test_regroup_tier_table_rejects_too_many_rows():
    """档位行过多（>14）说明不是一张裁量表，应放弃重组而非硬切。"""
    text = "\n".join(["严重 1.某违法行为。"] * 15)
    assert regroup_tier_table(text) is None


# ============================================================
# P2 目录/清单型 chunk 识别
# ============================================================

def test_is_standard_catalog_detects_model_listing():
    """消防产品目录类：纯型号罗列 -> 应判True。"""
    text = ("一、电线电缆（3 种）\n"
            "1.不包括阻燃电线电缆、耐火电线电缆。\n"
            "适用标准：GB/T5013.4 JB/T8735.2")
    md = {"law": "消防产品目录（2022年修订本）"}
    assert is_standard_catalog(text, md) is True


def test_is_standard_catalog_spares_normative_content():
    """同为目录类文档，但正文含义务性表述 -> 不得降权。

    这是 P2 最关键的防误杀闸门：宁可漏降权，也不能压掉真答案
    （recall 已经 1.0，压掉就是净损失）。
    """
    text = ("7.13 防烟排烟系统及通风、空调系统防火\n"
            "防烟、排烟系统的设置应符合国家及地方工程建设消防技术标准的规定，"
            "不符合的判定为不合格。\n"
            "适用标准：GB50016 GB50019")
    md = {"law": "浙江省建筑工程消防验收操作技术导则（试行）"}
    assert is_standard_catalog(text, md) is False


def test_is_standard_catalog_spares_penalty_chunk():
    """含处罚后果的段落永远不算名录。"""
    text = ("依据GB50016、GB50045 的规定，违反的给予行政处罚，"
            "并处五千元以上五万元以下罚款。")
    md = {"law": "消防产品目录（2022年修订本）"}
    assert is_standard_catalog(text, md) is False


def test_is_standard_catalog_requires_catalog_title():
    """非目录类文档即便有标准号也不降权。"""
    text = "适用标准：GB50016、GB50045 的规定。"
    md = {"law": "浙江省消防救援总队关于推行消防轻微违法不予处罚制度的实施意见"}
    assert is_standard_catalog(text, md) is False


def test_is_standard_catalog_empty_inputs():
    assert is_standard_catalog("", {"law": "消防产品目录"}) is False
    assert is_standard_catalog("一、电线电缆（3 种）", None) is False


def test_is_standard_catalog_exempts_catalog_intent_query():
    """查询意图豁免：用户问目录本身时，目录 chunk 是答案，不能降权。

    这是 47 条评测打出来的补丁：4/6 陷阱题 recall=0，根因就是 P2 把
    《消防产品目录》的 chunk 降权挤出 top-k，而这些 chunk 里恰恰写着
    "一、电线电缆（3 种）"——正是那几题的答案。
    """
    text = ("一、电线电缆（3 种）\n"
            "1.不包括阻燃电线电缆、耐火电线电缆。\n"
            "适用标准：GB/T5013.4 JB/T8735.2")
    md = {"law": "消防产品目录（2022年修订本）"}

    # 无查询意图 -> 降权
    assert is_standard_catalog(text, md) is True
    assert is_standard_catalog(text, md, query="电动自行车违规怎么处罚？") is True

    # 目录意图 -> 豁免
    assert is_standard_catalog(text, md, query="消防产品目录涵盖哪些大类？") is False
    assert is_standard_catalog(text, md, query="产品需要哪些认证程序？") is False


def test_is_standard_catalog_exemption_does_not_widen_scope():
    """豁免只对目录类 chunk 生效，不能波及其他内容。

    豁免判断放在标题/义务性两道闸门**之后**，顺序不能颠倒。
    """
    # 非目录类文档：任何查询都不该被判为名录
    text = "适用标准：GB50016、GB50045 的规定。"
    md = {"law": "浙江省消防行政处罚裁量规定"}
    assert is_standard_catalog(text, md, query="消防产品目录有哪些") is False


# ============================================================
# P4 目录条目切分
#
# 样本按生产 PDF 的**真实规模**构造（早期用 11 行小样本，导致
# 新判据的最小规模闸门被误判为"不含目录结构"，测试与实现同时失真）。
# 两份样本分别对应两份真实目录文档的排版形态：
#   _CATALOG_REAL_A —— 《关于优化强制性产品认证目录的公告（2020年第18号）》
#                      序号型："一、电线电缆（3 种）" / "1.电线组件（0101）"
#   _CATALOG_REAL_B —— 《（应急消〔2022〕136号）消防产品目录（2022年修订本）》
#                      无序号型，靠"标准号独占整行"锚定条目收尾
# ============================================================

_CATALOG_REAL_A_HEAD = (
    "一、电线电缆（3 种）\n"
    "1.不包括阻燃电线电缆、耐火电线电缆、裸电线、电力电缆、控制电缆、\n"
    "架空绝缘电缆、通信电缆和光缆、绕组线产品；\n"
    "2.不包括认证依据标准中未列明的型号、规格。\n"
)
_CATALOG_REAL_A_ITEM = (
    "1.电线组件（0101） 适用于家用和类似一般设备所用的\n"
    "电线组件（即，由带不可拆线插头和\n"
    "不可拆线的连接器的软缆或软线构\n"
    "成的组件）和互连电线组件\n"
)
_CATALOG_REAL = (
    "列举的产品可根据具体情况参照相应描述界定。\n"
    "6.*所标记产品为实施自我声明程序 A 的产品，\n"
    "**所标记产品为实施自我声明程序 B 的产品。\n"
    + _CATALOG_REAL_A_HEAD
    + "产品种类及代码 对产品种类的描述 产品适用范围 说明\n"
    + _CATALOG_REAL_A_ITEM
    + "2.插头插座（0201） 适用于户内或户外使用的、家用和类似用途的装置\n"
    "3.电缆接线端子（0202） 用于与电线连接的那种装置\n"
    "二、电路开关及保护或连接用电器装置（5 种）\n"
    "4.插头插座（0301） 适用于户内或户外使用的装置\n"
    "5.电缆接线端子（0302） 用于与电线连接的那种装置\n"
    "6.熔断器（0303） 用来在电路中保护元件\n"
    "三、低压电器（2 种）\n"
    "7.低压配电箱（0401） 用于配电的箱体\n"
    "8.低压电动机（0402） 用于驱动的电动机\n"
)

# 体系 B：产品名 + 说明（可多行）+ 标准号独占整行，全程无任何序号
_CATALOG_REAL_B = "\n".join(
    ["本目录所列产品均按国家标准逐项判定。"]
    + [
        f"{n}、用于第 {n} 类场所的灭火器组，"
        f"符合本目录第 {n} 项技术要求。\nGB {1000 + n}"
        for n in range(1, 13)
    ]
)

# 导则类正文：标准号是**句内引用**，行首匹配会把整篇切碎。
# 这是此前 P4 误切《浙江省建筑工程消防验收操作技术导则》的根因。
_PROSE_WITH_INLINE_STD = "\n".join(
    f"{n}.5.{n} 该项目的施工质量应符合 GB 5035{n % 10} 等技术标准检查验收"
    f"合格基础上进行，抽样检验比例不应低于百分之{n}。" for n in range(1, 15)
)


def test_regroup_catalog_entries_isolates_answer():
    """答案条目必须被单独切出。

    这是 47 条评测打出来的第四个补丁：问"电线电缆类产品"，
    答案"一、电线电缆（3 种）1.不包括阻燃…"被夹在自我声明说明与
    "电线组件"条目之间，bag-of-words 无法对齐，rec 恒为 0。
    """
    parts = regroup_catalog_entries(_CATALOG_REAL)
    assert parts is not None
    assert len(parts) >= 2
    hit = [p for p in parts if "电线电缆（3 种）" in p and "不包括阻燃" in p]
    assert hit, "答案条目未被独立切出：%r" % (parts,)
    # 答案条目里不应混入下一个条目
    assert "电线组件" not in hit[0]
    # 也不应混入前一段自我声明说明
    assert "自我声明" not in hit[0]


def test_regroup_catalog_entries_splits_type_b_by_solo_std():
    """体系 B：无序号型目录，靠独占标准号行切出条目。"""
    parts = regroup_catalog_entries(_CATALOG_REAL_B)
    assert parts is not None
    assert len(parts) >= 10, "体系 B 应切出 12 个条目，实际 %d" % len(parts)
    for n in range(1, 13):
        hit = [p for p in parts if "第 %d 项技术要求" % n in p]
        assert hit, "第 %d 项条目未独立：%r" % (n, parts)
        # 标准号必须与说明行同块（它是条目的收尾）
        assert "GB %d" % (1000 + n) in hit[0]


def test_regroup_catalog_entries_spares_prose_with_inline_std():
    """正文里句内引用标准号**不得**触发目录切分（防误切导则）。"""
    assert regroup_catalog_entries(_PROSE_WITH_INLINE_STD) is None


def test_regroup_catalog_entries_returns_none_for_prose():
    assert regroup_catalog_entries("第二十三条注册消防工程师应当依法执业。") is None
    assert regroup_catalog_entries("") is None


def test_chunker_marks_catalog_entries():
    """LegalChunker 必须给目录条目打 is_catalog 标记，供检索层豁免降权。"""
    ck = LegalChunker(chunk_size=200, min_chunk=20)
    out = ck.split_text(_CATALOG_REAL, source="消防产品目录.pdf")
    assert out
    assert any(d.metadata.get("is_catalog") for d in out)
    # is_tier_table 不应被误标
    assert not any(d.metadata.get("is_tier_table") for d in out)


def test_catalog_priority_over_tier_table():
    """P4 必须优先于 P1。

    实测《关于优化强制性产品认证目录的公告》每个条目都是
    "若干行说明 + 一个处罚档位"排版，regroup_tier_table 会把整篇
    判成档位表（45 块全标 tier_table），目录条目就拿不到 is_catalog。
    """
    src = "关于优化强制性产品认证目录的公告（2020年第18号）.pdf"
    ck = LegalChunker(chunk_size=200, min_chunk=20)
    out = ck.split_text(_CATALOG_REAL, source=src)
    assert any(d.metadata.get("is_catalog") for d in out)
    assert not any(d.metadata.get("is_tier_table") for d in out)


def test_tier_table_still_works_when_not_catalog():
    """非目录文档的档位表重组不能被 P4 抢走。"""
    ck = LegalChunker(chunk_size=300, min_chunk=20)
    out = ck.split_text(_TIER_TABLE_REAL, source="浙江省消防行政处罚裁量规定.pdf")
    assert out
    assert any(d.metadata.get("is_tier_table") for d in out), (
        "P4 抢占导致 P1 档位表失效：%r" % [d.page_content[:40] for d in out]
    )


def test_is_standard_catalog_spares_marked_catalog_entries():
    """已切成单条目的 chunk 不该再被判名录降权。"""
    md = {"law": "消防产品目录（2022年修订本）", "is_catalog": True}
    text = "一、电线电缆（3 种）1.不包括阻燃电线电缆、耐火电线电缆。"
    assert is_standard_catalog(text, md) is False
