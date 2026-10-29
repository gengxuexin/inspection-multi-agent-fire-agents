# agent1/index_prune.py
"""按 chunk 删除已入库的块 —— 支撑「先删旧块、再插新块」的增量更新。

★★ 为什么需要
--------------
法规改版后，如果不删旧块，检索会同时返回新旧两版条文，
模型可能引用**已作废的条款**。在消防执法场景这直接导致文书出错。

★★★ 两套 ID 的坑（实测踩过）
---------------------------
FAISS 和 Milvus 用的「块 ID」不是同一个东西：

- Milvus：按 ``metadata.chunk_id``（业务 ID，如 "e1b3e10695acde79"）做 expr 删除
- FAISS：docstore 的 key 是 **UUID**（如 "bc33d1ad-e37a-4937-a92f-309e25daf895"），
  ``FAISS.delete(ids=...)`` **只认 UUID**，传业务 chunk_id 会直接报
  "Some specified ids do not exist"

所以状态文件里两套 ID 都要记（见 index_state.record_file 的 doc_ids 参数）。
本模块负责把「要删的 ID」翻译成各后端能接受的形式。

★ 失败不静默
------------
删不掉旧块 → 新旧条文并存 → 可能引用作废条款。
这是**会出执法事故**的静默失败，所以删除失败必须抛异常中断流程，
而不是打条日志继续跑。
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("fire_agents.index_prune")


class PruneError(RuntimeError):
    """删除旧块失败 —— 流程必须中断，不能带着新旧并存继续。"""


def delete_from_faiss(vectorstore: Any, doc_ids: list[str]) -> int:
    """从 FAISS 索引删除指定 doc_id。

    返回实际删除数量。失败抛 PruneError。
    """
    if not doc_ids:
        return 0
    if vectorstore is None:
        raise PruneError("FAISS vectorstore 为空，无法删除旧块")

    try:
        ok = vectorstore.delete(ids=list(doc_ids))
    except Exception as e:  # noqa: BLE001
        raise PruneError(
            f"FAISS 删除旧块失败（{len(doc_ids)} 个）：{e}。"
            f"请确认 doc_ids 来自 index_state 记录的 doc_ids（UUID），"
            f"而不是业务 chunk_id。") from e

    if ok is False:
        raise PruneError("FAISS.delete 返回 False，旧块未删除")
    logger.info("[prune] FAISS 已删除 %d 个旧块", len(doc_ids))
    return len(doc_ids)


def delete_from_milvus(mv_client: Any, collection: str,
                       chunk_ids: list[str]) -> int:
    """从 Milvus 删除指定业务 chunk_id。

    用 expr 按 metadata.chunk_id 匹配，精确且不依赖主键。
    """
    if not chunk_ids:
        return 0
    if mv_client is None or not collection:
        raise PruneError("Milvus 客户端或 collection 为空，无法删除旧块")

    # 分批删除，避免 expr 过长
    batch = 100
    deleted = 0
    for i in range(0, len(chunk_ids), batch):
        part = chunk_ids[i:i + batch]
        ids = ", ".join(f'"{cid}"' for cid in part)
        expr = f'chunk_id in [{ids}]'
        try:
            res = mv_client.delete(collection_name=collection, filter=expr)
        except Exception as e:  # noqa: BLE001
            raise PruneError(
                f"Milvus 删除旧块失败（批次 {i // batch + 1}）：{e}") from e
        n = res.get("delete_count", 0) if isinstance(res, dict) else 0
        deleted += int(n or 0)
    logger.info("[prune] Milvus 已删除 %d 个旧块", deleted)
    return deleted


def collect_ids_to_delete(state: dict, stale: dict[str, list[str]]) -> dict:
    """从 stale 映射里收集两套 ID。

    stale: {旧 fingerprint: [业务 chunk_id, ...]}
    返回 {"chunk_ids": [...], "doc_ids": [...]}
    """
    from agent1.index_state import doc_ids_of

    chunk_ids: list[str] = []
    doc_ids: list[str] = []
    for fp in stale:
        chunk_ids.extend(stale.get(fp) or [])
        doc_ids.extend(doc_ids_of(state, fp))
    return {"chunk_ids": chunk_ids, "doc_ids": doc_ids}
