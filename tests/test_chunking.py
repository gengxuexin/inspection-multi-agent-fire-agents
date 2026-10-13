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
    article_of,
    clean_pdf_text,
    cn_to_int,
    normalize_article,
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