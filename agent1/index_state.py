# agent1/index_state.py
"""索引状态追踪：支持「先删旧块、再插新块」的增量更新。

★★ 为什么需要这个模块
----------------------
原实现的 `_processed_files.json` 只是一份 **fingerprint 的集合**：

    ["013725fd...", "04cd2874...", ...]     # 103 条

fingerprint = md5(path | mtime | size)，只能回答
「这个文件处理过吗 / 变过吗」。它**无法回答**：

    「这个文件在索引里对应哪些 chunk？」

于是增量更新变成**只增不删**：

    某份法规改了 → fingerprint 变了 → 新块插进索引
                  → 但**旧块还在** → 检索时新旧两版条文同时返回
                  → 模型可能引用已作废的条款

**在消防执法场景这直接导致文书出错。**
一份被废止的处罚规定如果还能被检索到并写进处罚决定书，
那是会出执法事故的。

★★★ 解法
--------
把状态文件从「fingerprint 集合」升级为**结构化映射**：

    {
      "<fingerprint>": {
        "path": "规范性文件类/xxx.pdf",     # 相对知识库根目录
        "chunk_ids": ["e1b3e10695acde79", ...],
        "indexed_at": "2026-10-29T10:00:00",
        "chunks": 42
      }
    }

有了 chunk_ids 就能：
1. 文件改版 → 先按旧 fingerprint 查出 chunk_ids → 删除这些块 → 再插新块
2. `--prune` → 扫描索引，发现「有块但状态文件里没有对应记录」的孤儿块 → 清理

★ 向后兼容
----------
旧的状态文件（纯 list）会被自动识别并迁移：
读出来转成 dict，chunk_ids 置空（此时无法删除旧块，只能提示用 --rebuild 重建一次）。
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any


# ==================== 读写 ====================
def load_state(path: str) -> dict:
    """读状态文件；兼容旧的 list 格式。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    if isinstance(raw, list):
        # 旧格式：只有 fingerprint，无 chunk_ids
        return {fp: {"path": "", "chunk_ids": [], "indexed_at": "",
                     "chunks": 0, "legacy": True}
                for fp in raw}
    if isinstance(raw, dict):
        return raw
    return {}


def save_state(path: str, state: dict) -> None:
    """写状态文件（原子写：先写临时文件再 rename，避免中断产生半截文件）。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def is_legacy(state: dict) -> bool:
    """状态是否还是旧格式（无 chunk_ids，无法删除旧块）。"""
    return any(v.get("legacy") for v in state.values()) if state else False


# ==================== 记录 ====================
def record_file(state: dict, fingerprint: str, rel_path: str,
                chunk_ids: list[str] | None = None,
                chunks: int = 0,
                doc_ids: list[str] | None = None) -> None:
    """记录一个文件已入库及其块 ID。

    ★ 两套 ID 的区别（重要）
    ----------------------
    - `chunk_ids` —— 业务 ID，来自 metadata.chunk_id（如 "e1b3e10695acde79"）。
      Milvus 后端用它做 expr 删除。
    - `doc_ids` —— 向量库内部 ID。FAISS 的 docstore key 是 UUID
      （如 "bc33d1ad-e37a-4937-a92f-309e25daf895"），
      FAISS.delete(ids=...) 只认这个。

    两个都记下来，才能对两个后端都做精确删除。
    """
    state[fingerprint] = {
        "path": rel_path,
        "chunk_ids": list(chunk_ids or []),
        "doc_ids": list(doc_ids or []),
        "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunks": chunks,
        "legacy": False,
    }


def chunks_of(state: dict, fingerprint: str) -> list[str]:
    """取某 fingerprint 对应的业务 chunk_ids。"""
    entry = state.get(fingerprint) or {}
    return list(entry.get("chunk_ids") or [])


def doc_ids_of(state: dict, fingerprint: str) -> list[str]:
    """取某 fingerprint 对应的向量库内部 doc_ids（FAISS 用）。"""
    entry = state.get(fingerprint) or {}
    return list(entry.get("doc_ids") or [])


def files_of(state: dict) -> dict:
    """fingerprint → 相对路径 的映射（用于报告）。"""
    return {fp: (v.get("path") or "") for fp, v in state.items()}


# ==================== 差异计算 ====================
def diff_files(
    state: dict,
    current: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """对比状态与当前文件，算出三类。

    ★ 关键：以「文件路径」为基准，而不是指纹
    -----------------------------------------
    指纹 = md5(path | mtime | size)，**文件一改，指纹就变**。
    所以如果按指纹比对，改版的文件会同时表现为
    「旧指纹 removed」+「新指纹 added」，
    而看不出「这是同一个文件的改版」——
    也就没法在改版时先删掉旧块。

    正确做法：以路径为基准。
      - 路径在状态里、当前也在、但**指纹不同** → changed（改版，需删旧块）
      - 路径只在当前              → added（新增文件）
      - 路径只在状态里            → removed（文件被删，索引里的块应清理）

    参数
    ----
    state        状态文件内容
    current      {fingerprint: 相对路径} —— 当前磁盘上的文件及其指纹

    返回
    ----
    (added, changed, removed) —— 每项是**当前/历史的 fingerprint**
      added    —— 新文件（磁盘有、状态无）
      changed  —— 内容变了（同一路径，指纹不同）
      removed  —— 文件被删了（状态有、磁盘无）
    """
    # 路径 → 指纹（当前磁盘）
    cur_by_path: dict[str, str] = {}
    for fp, path in current.items():
        cur_by_path.setdefault(path, fp)

    # 路径 → 指纹（历史状态）
    old_by_path: dict[str, str] = {}
    for fp, entry in state.items():
        path = (entry or {}).get("path") or ""
        if path:
            old_by_path.setdefault(path, fp)

    added: list[str] = []
    changed: list[str] = []
    removed: list[str] = []

    for path, fp in cur_by_path.items():
        if path not in old_by_path:
            added.append(fp)
        elif old_by_path[path] != fp:
            changed.append(fp)

    for path, fp in old_by_path.items():
        if path not in cur_by_path:
            removed.append(fp)

    return added, changed, removed


def stale_fingerprints_for(state: dict, current: dict[str, str]) -> dict[str, list[str]]:
    """返回「需要删除的旧块」：路径 → 旧 chunk_ids。

    这是「先删旧块、再插新块」的关键一步。

    - 改版的文件（changed）：旧指纹对应的 chunk_ids 要删
    - 被删的文件（removed）：它的 chunk_ids 也要删

    返回 {旧指纹: [chunk_id, ...]}
    """
    added, changed, removed = diff_files(state, current)
    to_delete: dict[str, list[str]] = {}

    # 改版：找同路径的旧指纹
    # current 是 {fingerprint: path}，所以：
    #   fp → path 用 current 直接查
    #   path → fp 需要反向建表
    cur_by_path: dict[str, str] = {}
    for fp, path in current.items():
        cur_by_path.setdefault(path, fp)

    old_by_path: dict[str, str] = {}
    for fp, entry in state.items():
        path = (entry or {}).get("path") or ""
        if path:
            old_by_path.setdefault(path, fp)

    for fp in changed:
        path = current.get(fp, "")        # ← fp → path
        old_fp = old_by_path.get(path)    # ← path → 旧 fp
        if old_fp:
            ids = chunks_of(state, old_fp)
            if ids:
                to_delete[old_fp] = ids

    # 被删的文件
    for fp in removed:
        ids = chunks_of(state, fp)
        if ids:
            to_delete[fp] = ids

    return to_delete


def orphan_chunks(index_chunk_sources: dict[str, str],
                  state: dict) -> dict[str, list[str]]:
    """找出孤儿块：索引里有、但状态文件里没有对应记录。

    index_chunk_sources: {chunk_id: 相对路径} —— 索引中每个块的来源
    返回 {相对路径: [chunk_id, ...]}
    """
    known: dict[str, list[str]] = {}
    for entry in state.values():
        for cid in entry.get("chunk_ids") or []:
            known.setdefault(entry.get("path") or "?", []).append(cid)
    known_ids = {cid for ids in known.values() for cid in ids}

    orphans: dict[str, list[str]] = {}
    for cid, src in index_chunk_sources.items():
        if cid in known_ids:
            continue
        orphans.setdefault(src, []).append(cid)
    return orphans


# ==================== 摘要（给人看的） ====================
def summarize(state: dict) -> dict:
    """状态摘要，给运维端点/CLI 用。"""
    total_chunks = sum(len(v.get("chunk_ids") or []) for v in state.values())
    tracked = sum(1 for v in state.values() if v.get("chunk_ids"))
    return {
        "files": len(state),
        "files_with_chunk_ids": tracked,
        "chunks_tracked": total_chunks,
        "legacy_format": is_legacy(state),
        "can_delete_stale": tracked > 0,
    }
