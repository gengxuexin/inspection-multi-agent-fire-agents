# test_change_detector.py
"""change_detector.py 单元测试：diff / 风险分级 / 报告落盘

不依赖 LLM（use_llm=False）与 docx，纯文本即可跑。
"""
import json
import os
import sys

import pytest

_AGENT1 = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "agent1")
sys.path.insert(0, _AGENT1)

from change_detector import (  # noqa: E402
    ChangeReport,
    _pair_similar,
    _render_markdown,
    _split_units,
    _trunc,
    analyze_changes,
    assess_risk,
    diff_units,
    extract_text,
)


ORIGINAL = """当事人存在消防设施配置不符合标准的违法行为。
依据《中华人民共和国消防法》第六十条第一款的规定，责令改正，并处五千元以上五万元以下罚款。
当事人应当自收到处罚决定书之日起十五日内向指定银行缴纳罚款。
本决定书一式二份，一份交当事人，一份存档。"""

MODIFIED = """当事人存在消防设施、器材或者消防安全标志的配置、设置不符合国家标准、行业标准的违法行为。
依据《中华人民共和国消防法》第六十条第一款第（七）项的规定，责令改正，并处五万元以上五十万元以下罚款。
当事人应当自收到处罚决定书之日起十五日内向指定银行缴纳罚款。逾期不缴纳的，每日按罚款数额的百分之三加处罚款。
本决定书一式二份，一份交当事人，一份存档。"""


@pytest.fixture
def doc_pair(tmp_path):
    a, b = tmp_path / "origin.txt", tmp_path / "modified.txt"
    a.write_text(ORIGINAL, encoding="utf-8")
    b.write_text(MODIFIED, encoding="utf-8")
    return str(a), str(b)


# ==================== 文本切分 ====================
def test_split_units_keeps_sentences():
    units = _split_units("第一句内容。第二句内容！第三句内容？")
    assert len(units) == 3


def test_split_units_splits_long_paragraph():
    long_p = "前置说明，" * 30 + "最终结论。"
    units = _split_units(long_p)
    assert len(units) > 1
    assert "".join(units) == long_p


def test_split_units_empty():
    assert _split_units("") == []
    assert _split_units(None) is None or _split_units(None) == []


# ==================== diff ====================
def test_diff_identical_text_has_no_changes():
    assert diff_units(ORIGINAL, ORIGINAL) == []


def test_diff_detects_replace():
    cs = diff_units(ORIGINAL, MODIFIED)
    assert any(c["type"] == "replace" for c in cs)


def test_diff_detects_insert():
    cs = diff_units("第一行内容。", "第一行内容。\n新增的一行内容。")
    assert any(c["type"] == "insert" for c in cs)


def test_diff_detects_delete():
    cs = diff_units("第一行内容。\n将被删除的一行内容。", "第一行内容。")
    assert any(c["type"] == "delete" for c in cs)


def test_diff_no_duplicate_reporting():
    """同一段内容不得既报 replace 又报 insert —— 曾出现的真实缺陷。"""
    cs = diff_units(ORIGINAL, MODIFIED)
    for c in cs:
        if c["type"] != "insert":
            continue
        # 插入项的 modified 不应与任何 replace 项的 modified 重复
        modified = c["modified"].strip()
        for r in cs:
            if r["type"] == "replace" and r["modified"].strip() == modified:
                pytest.fail(f"重复计数：{modified[:40]}")


def test_diff_changes_are_minimal():
    """只改一句话时，不应把整篇都算成改动。"""
    a = "第一段内容。\n第二段内容。\n第三段内容。\n第四段内容。\n第五段内容。"
    b = "第一段内容。\n第二段内容。\n第三段改了。\n第四段内容。\n第五段内容。"
    cs = diff_units(a, b)
    assert len(cs) == 1
    assert cs[0]["type"] == "replace"
    assert "第三段改了" in cs[0]["modified"]


def test_pair_similar_threshold():
    """相似度低于 0.3 的不配对，应视为"删一段+加一段"。"""
    old = ["完全不相干的内容甲乙丙丁"]
    new = ["毫不相关的另一段文字戊己庚辛"]
    pairs, used = _pair_similar(old, new)
    assert pairs == []
    assert used == set()


# ==================== 风险分级 ====================
def test_risk_amount_change_is_high():
    """篡改罚款金额 —— 必须高风险。"""
    lvl, reasons = assess_risk("处五千元罚款。", "处五十万元罚款。")
    assert lvl == "high"
    assert any("金额" in r for r in reasons)


def test_risk_article_change_is_high():
    lvl, reasons = assess_risk("依据第六十条处理。", "依据第六十一条处理。")
    assert lvl == "high"
    assert any("条款" in r for r in reasons)


def test_risk_law_name_change_is_high():
    lvl, reasons = assess_risk("依据《消防法》。", "依据《安全生产法》。")
    assert lvl == "high"
    assert any("法律名称" in r for r in reasons)


def test_risk_pure_wording_change_is_low():
    """纯措辞调整不应误报高风险。"""
    lvl, _ = assess_risk("当事人不服处罚的，可申请复议。",
                        "当事人对处罚不服的，可以申请行政复议。")
    assert lvl in ("low", "medium")


def test_risk_insert_with_amount_is_medium_not_high():
    """纯新增含要素 -> 中风险，不能和越权改罚同等对待。"""
    lvl, _ = assess_risk("", "逾期不缴纳的，每日加处罚款。")
    assert lvl == "medium"


def test_risk_identical_is_low():
    lvl, reasons = assess_risk("同样的内容。", "同样的内容。")
    assert lvl == "low"
    assert reasons == []


def test_risk_delete_with_amount_is_high():
    lvl, reasons = assess_risk("处五千元罚款。", "")
    assert lvl == "high"


def test_risk_empty_both():
    lvl, reasons = assess_risk("", "")
    assert lvl == "low"


# ==================== 主流程 ====================
def test_analyze_changes_no_llm(doc_pair):
    rep = analyze_changes(*doc_pair, review_id="RV-T1", use_llm=False)
    assert rep.total_changes > 0
    assert rep.high_risk_count >= 1
    assert rep.review_id == "RV-T1"
    assert rep.changes


def test_analyze_changes_identical_files(doc_pair):
    a, _ = doc_pair
    rep = analyze_changes(a, a, review_id="RV-T2", use_llm=False)
    assert rep.total_changes == 0
    assert rep.high_risk_count == 0
    assert "未检测到人工改动" in rep.summary


def test_analyze_changes_missing_file(doc_pair):
    a, _ = doc_pair
    with pytest.raises(FileNotFoundError):
        analyze_changes(a, "/nonexistent/nope.txt", use_llm=False)


def test_report_saved_to_disk(doc_pair):
    rep = analyze_changes(*doc_pair, review_id="RV-T3", use_llm=False)
    assert os.path.isfile(rep.json_path)
    assert os.path.isfile(rep.markdown_path)

    with open(rep.json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["review_id"] == "RV-T3"
    assert data["total_changes"] == rep.total_changes

    with open(rep.markdown_path, encoding="utf-8") as f:
        md = f.read()
    assert "改动分析报告" in md
    assert "高风险改动" in md


def test_report_paths_unique(doc_pair):
    """两次运行不得覆盖同一份报告。"""
    r1 = analyze_changes(*doc_pair, review_id="RV-U", use_llm=False)
    r2 = analyze_changes(*doc_pair, review_id="RV-U", use_llm=False)
    assert r1.json_path != r2.json_path


def test_summary_mentions_high_risk(doc_pair):
    rep = analyze_changes(*doc_pair, use_llm=False)
    if rep.high_risk_count:
        assert "高风险" in rep.summary


# ==================== 报告渲染 ====================
def test_render_markdown_no_crashes_on_empty():
    rep = ChangeReport(
        report_id="CR-X", review_id="RV-X",
        created_at="2026-10-14 10:00:00",
        original_file="a", modified_file="b",
        total_changes=0, high_risk_count=0,
        summary="无改动",
    )
    md = _render_markdown(rep)
    assert "无高风险改动" in md
    assert "未检测到改动" in md


def test_render_markdown_section_numbering():
    """有根因分析时，模板建议章节应编号为五而非四。"""
    rep = ChangeReport(
        report_id="CR-Y", review_id="RV-Y",
        created_at="2026-10-14 10:00:00",
        original_file="a", modified_file="b",
        total_changes=3, high_risk_count=1,
        summary="摘要",
        llm_analysis="根因分析正文",
        template_suggestions=["建议一"],
    )
    md = _render_markdown(rep)
    assert "## 四、根因分析" in md
    assert "## 五、模板改进建议" in md


def test_render_markdown_section_numbering_without_llm():
    rep = ChangeReport(
        report_id="CR-Z", review_id="RV-Z",
        created_at="2026-10-14 10:00:00",
        original_file="a", modified_file="b",
        total_changes=3, high_risk_count=1,
        summary="摘要",
        template_suggestions=["建议一"],
    )
    md = _render_markdown(rep)
    assert "## 四、模板改进建议" in md
    assert "## 五、模板改进建议" not in md


def test_trunc_shortens_and_strips_newline():
    assert len(_trunc("啊" * 500, 50)) <= 51
    assert "\n" not in _trunc("第一行\n第二行")
    assert "|" not in _trunc("a|b|c")   # 表格分隔符不能破坏 Markdown 表格


# ==================== 文本提取 ====================
def test_extract_text_txt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("内容内容", encoding="utf-8")
    assert extract_text(str(p)) == "内容内容"


def test_extract_text_unsupported_type(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="不支持"):
        extract_text(str(p))


def test_extract_text_missing_file(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        extract_text(str(tmp_path / "nope.txt"))


# ==================== 文件名安全 ====================
def test_safe_stem_sanitizes_review_id():
    """review_id 含路径分隔符时不得逃出报告目录 —— 曾是真实崩溃点。"""
    from change_detector import REPORT_DIR, _safe_stem
    stem = _safe_stem("N/A", "CR-1")
    assert "/" not in stem and "\\" not in stem
    assert os.path.dirname(os.path.join(REPORT_DIR, stem)) == REPORT_DIR


@pytest.mark.parametrize("bad", ["../../etc/passwd", "a/b", "a\\b",
                                 "N/A", "..", "  ", None])
def test_safe_stem_handles_hostile_ids(bad):
    from change_detector import _safe_stem
    stem = _safe_stem(bad, "CR-1")
    assert stem and not any(ch in stem for ch in '<>:"/\\|?*')
    assert ".." not in stem


def test_analyze_changes_default_review_id_saves_ok(doc_pair):
    """review_id 缺省为 'N/A' 时也必须能落盘，且文件名不含路径分隔符。"""
    from change_detector import REPORT_DIR
    rep = analyze_changes(*doc_pair, use_llm=False)
    assert os.path.isfile(rep.json_path)
    assert os.path.dirname(os.path.abspath(rep.json_path)) == \
        os.path.abspath(REPORT_DIR)
    assert "N/A" not in os.path.basename(rep.json_path)