# tests/test_index_prune.py
"""旧块删除测试。

★ 这组测试要证明什么
-------------------
1. FAISS 删除用 doc_id（UUID），传 chunk_id 会失败 —— 两套 ID 不能混
2. Milvus 删除用业务 chunk_id 做 expr
3. 删除失败必须抛 PruneError（不能静默继续）
4. 收集待删 ID 时两套 ID 都能拿到

★ 为什么「删除失败必须抛异常」这么重要
------------------------------------
删不掉旧块 → 新旧两版条文并存 → 检索可能命中**已作废**的条款 →
模型把它写进处罚决定书。

这是会出执法事故的静默失败。所以宁可让构建流程中断，
也不能带着新旧并存继续跑 —— 后者看起来「成功」了。
"""
import pytest

from agent1.index_prune import (
    PruneError,
    collect_ids_to_delete,
    delete_from_faiss,
    delete_from_milvus,
)
from agent1.index_state import record_file


# ==================== 收集 ID ====================
def test_collect_ids_both_sets():
    state = {}
    record_file(state, "fp1", "a.pdf", ["c1", "c2"], 2,
                doc_ids=["u1", "u2"])
    stale = {"fp1": ["c1", "c2"]}
    ids = collect_ids_to_delete(state, stale)
    assert ids["chunk_ids"] == ["c1", "c2"]
    assert ids["doc_ids"] == ["u1", "u2"]


def test_collect_ids_without_doc_ids():
    """旧状态没有 doc_ids → FAISS 会提示需要 rebuild。"""
    state = {}
    record_file(state, "fp1", "a.pdf", ["c1"], 1)
    ids = collect_ids_to_delete(state, {"fp1": ["c1"]})
    assert ids["chunk_ids"] == ["c1"]
    assert ids["doc_ids"] == []


# ==================== FAISS ====================
class _FakeFAISS:
    def __init__(self, fail=False):
        self.deleted = []
        self.fail = fail

    def delete(self, ids=None, **kwargs):
        if self.fail:
            raise ValueError("Some specified ids do not exist in the current store")
        self.deleted.extend(ids or [])
        return True


def test_faiss_delete_uses_doc_ids():
    vs = _FakeFAISS()
    n = delete_from_faiss(vs, ["uuid-1", "uuid-2"])
    assert n == 2
    assert vs.deleted == ["uuid-1", "uuid-2"]


def test_faiss_delete_empty_noop():
    vs = _FakeFAISS()
    assert delete_from_faiss(vs, []) == 0
    assert vs.deleted == []


def test_faiss_delete_failure_raises():
    """删除失败必须抛 —— 不能静默继续。"""
    vs = _FakeFAISS(fail=True)
    with pytest.raises(PruneError) as ei:
        delete_from_faiss(vs, ["bad-id"])
    msg = str(ei.value)
    assert "UUID" in msg, "错误信息要提示用 UUID 而不是 chunk_id"


def test_faiss_delete_none_store_raises():
    with pytest.raises(PruneError):
        delete_from_faiss(None, ["x"])


# ==================== Milvus ====================
class _FakeMilvus:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def delete(self, collection_name=None, filter=None):
        if self.fail:
            raise RuntimeError("collection not found")
        self.calls.append((collection_name, filter))
        n = filter.count('"')
        return {"delete_count": n // 2}


def test_milvus_delete_uses_chunk_ids():
    c = _FakeMilvus()
    n = delete_from_milvus(c, "fire_laws", ["c1", "c2"])
    assert n == 2
    assert c.calls[0][0] == "fire_laws"
    assert 'chunk_id in ["c1", "c2"]' in c.calls[0][1]


def test_milvus_delete_batches():
    """超过 100 个要分批，避免 expr 过长。"""
    c = _FakeMilvus()
    ids = [f"c{i}" for i in range(250)]
    delete_from_milvus(c, "fire_laws", ids)
    assert len(c.calls) == 3, "250 个应分成 3 批"


def test_milvus_delete_failure_raises():
    c = _FakeMilvus(fail=True)
    with pytest.raises(PruneError):
        delete_from_milvus(c, "fire_laws", ["c1"])


def test_milvus_delete_none_raises():
    with pytest.raises(PruneError):
        delete_from_milvus(None, "", ["c1"])


def test_milvus_escape_quotes_in_ids():
    """chunk_id 里含引号会破坏 expr —— 至少不能崩。"""
    c = _FakeMilvus()
    delete_from_milvus(c, "fire_laws", ['weird"id'])
    assert c.calls
