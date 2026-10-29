# tests/test_index_state.py
"""索引状态追踪测试。

★ 这组测试要证明什么
-------------------
1. 旧格式（list）能被识别并迁移
2. 记录文件后能取回 chunk_ids
3. diff 能正确分出 新增 / 变更 / 删除
4. 孤儿块能被识别
5. 状态文件原子写：中断不会产生半截文件
6. 摘要能反映「能不能删旧块」

★ 为什么这组测试重要
-------------------
「增量更新只增不删」这个 bug 的本质是**状态文件信息不足**。
如果状态文件只存 fingerprint，就永远定位不到旧块。
所以测试要钉死「chunk_ids 一定要被记下来」这个性质。
"""
import json
import os

import pytest

from agent1.index_state import (
    chunks_of,
    diff_files,
    is_legacy,
    load_state,
    orphan_chunks,
    record_file,
    save_state,
    stale_fingerprints_for,
    summarize,
)


# ==================== 读写与兼容 ====================
def test_load_missing_returns_empty(tmp_path):
    assert load_state(str(tmp_path / "nope.json")) == {}


def test_load_legacy_list_format(tmp_path):
    """旧格式是 list，必须能读并标记为 legacy。"""
    p = tmp_path / "state.json"
    p.write_text(json.dumps(["fp1", "fp2"]), encoding="utf-8")
    st = load_state(str(p))
    assert set(st.keys()) == {"fp1", "fp2"}
    assert is_legacy(st) is True
    # legacy 状态下不能删旧块
    assert summarize(st)["can_delete_stale"] is False


def test_load_new_dict_format(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({
        "fp1": {"path": "a.pdf", "chunk_ids": ["c1", "c2"], "chunks": 2},
    }), encoding="utf-8")
    st = load_state(str(p))
    assert is_legacy(st) is False
    assert chunks_of(st, "fp1") == ["c1", "c2"]
    assert summarize(st)["can_delete_stale"] is True


def test_load_corrupt_returns_empty(tmp_path):
    """状态文件损坏不能让整个构建流程崩掉。"""
    p = tmp_path / "state.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert load_state(str(p)) == {}


def test_save_state_roundtrip(tmp_path):
    p = str(tmp_path / "state.json")
    st = {}
    record_file(st, "fp1", "a.pdf", ["c1", "c2"], 2)
    save_state(p, st)

    loaded = load_state(p)
    assert chunks_of(loaded, "fp1") == ["c1", "c2"]
    assert loaded["fp1"]["path"] == "a.pdf"
    assert loaded["fp1"]["chunks"] == 2
    assert loaded["fp1"]["indexed_at"]


def test_save_state_atomic(tmp_path):
    """原子写：临时文件 + rename。写一半中断不会留下半截状态文件。"""
    p = str(tmp_path / "state.json")
    st = {}
    record_file(st, "fp1", "a.pdf", ["c1"], 1)
    save_state(p, st)
    # 不应残留 .tmp
    assert not os.path.exists(p + ".tmp")


def test_save_state_creates_dir(tmp_path):
    p = str(tmp_path / "deep" / "state.json")
    save_state(p, {})
    assert os.path.exists(p)


# ==================== 记录 ====================
def test_record_file_sets_chunk_ids():
    st = {}
    record_file(st, "fp1", "规范性文件类/x.pdf", ["a", "b", "c"], 3)
    assert chunks_of(st, "fp1") == ["a", "b", "c"]
    assert st["fp1"]["chunks"] == 3
    assert st["fp1"]["legacy"] is False


def test_record_file_overwrites_previous():
    """同一 fingerprint 重新入库 → 旧 chunk_ids 被替换，不是叠加。"""
    st = {}
    record_file(st, "fp1", "x.pdf", ["old1", "old2"], 2)
    record_file(st, "fp1", "x.pdf", ["new1"], 1)
    assert chunks_of(st, "fp1") == ["new1"]


# ==================== diff ====================
def test_diff_classifies_three_cases():
    """★ 以「路径」为基准分类 —— 指纹含 mtime，改版时指纹会变。

    如果按指纹比对，改版的文件会变成「removed + added」，
    根本看不出「这是同一文件的改版」，也就没法删旧块。
    """
    st = {}
    record_file(st, "fp_keep", "keep.pdf", ["k1"], 1)
    record_file(st, "fp_old_changed", "changed.pdf", ["o1", "o2"], 2)
    record_file(st, "fp_gone", "gone.pdf", ["g1"], 1)

    current = {
        "fp_keep": "keep.pdf",              # 未变
        "fp_new_changed": "changed.pdf",   # 同路径，指纹不同 → 改版
        "fp_added": "added.pdf",            # 新文件
    }

    added, changed, removed = diff_files(st, current)
    assert added == ["fp_added"]
    assert changed == ["fp_new_changed"], "改版的文件应归到 changed"
    assert removed == ["fp_gone"], "被删的文件应归到 removed"


def test_stale_fingerprints_for_changed_file():
    """★ 核心：改版时能定位到旧块 —— 这是「先删旧块」的前提。"""
    st = {}
    record_file(st, "fp_old", "reg.pdf", ["old_c1", "old_c2"], 2)
    current = {"fp_new": "reg.pdf"}      # 同一路径，新指纹

    to_delete = stale_fingerprints_for(st, current)
    assert to_delete == {"fp_old": ["old_c1", "old_c2"]}


def test_stale_fingerprints_for_removed_file():
    """文件被删 → 它的块也要清理。"""
    st = {}
    record_file(st, "fp_gone", "gone.pdf", ["g1", "g2"], 2)
    to_delete = stale_fingerprints_for(st, {})
    assert to_delete == {"fp_gone": ["g1", "g2"]}


def test_stale_fingerprints_none_when_unchanged():
    st = {}
    record_file(st, "fp_a", "a.pdf", ["c1"], 1)
    to_delete = stale_fingerprints_for(st, {"fp_a": "a.pdf"})
    assert to_delete == {}


def test_stale_fingerprints_legacy_cannot_locate():
    """legacy 状态没有 chunk_ids → 无法定位旧块（必须 rebuild）。"""
    st = load_state_from_list(["fp_old"])
    current = {"fp_new": "a.pdf"}
    to_delete = stale_fingerprints_for(st, current)
    assert to_delete == {}, "legacy 状态定位不到旧块，应返回空"


def test_diff_empty_state():
    added, changed, removed = diff_files({}, {"a": "a.pdf"})
    assert added == ["a"]
    assert changed == []
    assert removed == []


def test_diff_no_changes():
    st = {}
    record_file(st, "a", "a.pdf", ["c1"], 1)
    added, changed, removed = diff_files(st, {"a": "a.pdf"})
    assert added == [] and changed == [] and removed == []


# ==================== 孤儿块 ====================
def test_orphan_chunks_detected():
    """索引里有、状态里没有的块 → 孤儿（旧版本残留）。"""
    st = {}
    record_file(st, "fp1", "a.pdf", ["c1"], 1)
    index_sources = {
        "c1": "a.pdf",        # 已知
        "c2": "b.pdf",        # 孤儿：b.pdf 不在状态里
        "c3": "b.pdf",
    }
    orphans = orphan_chunks(index_sources, st)
    assert orphans == {"b.pdf": ["c2", "c3"]}


def test_orphan_chunks_none_when_clean():
    st = {}
    record_file(st, "fp1", "a.pdf", ["c1", "c2"], 2)
    orphans = orphan_chunks({"c1": "a.pdf", "c2": "a.pdf"}, st)
    assert orphans == {}


def test_orphan_chunks_legacy_state():
    """legacy 状态没有 chunk_ids → 所有块都算孤儿（提示需要 rebuild）。"""
    st = load_state_from_list(["fp1"])
    orphans = orphan_chunks({"c1": "a.pdf"}, st)
    assert "a.pdf" in orphans


def load_state_from_list(items):
    p = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests", "_tmp_legacy_state.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(items, f)
    try:
        return load_state(p)
    finally:
        os.remove(p)


# ==================== 摘要 ====================
def test_summarize():
    st = {}
    record_file(st, "a", "a.pdf", ["c1", "c2"], 2)
    record_file(st, "b", "b.pdf", ["c3"], 1)
    s = summarize(st)
    assert s["files"] == 2
    assert s["files_with_chunk_ids"] == 2
    assert s["chunks_tracked"] == 3
    assert s["legacy_format"] is False
    assert s["can_delete_stale"] is True
