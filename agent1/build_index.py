
"""
分批次构建 FAISS 向量库
- 支持增量添加
- 支持断点续传
- 保存阶段显示向量化进度条
"""
import os
import sys
import json
import hashlib
import time
import argparse
import httpx
import yaml
from typing import ClassVar
from tqdm import tqdm
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ★ 让脚本能找到同目录的 config / file_analyzer
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import get_embedding_config
from file_analyzer import extract_text
from chunking import LegalChunker

# ==================== 配置 ====================
MODE = "full"     # single / multi / full

MODE_CONFIG = {
    "single": {"kb_dir": "knowledge_base/single", "index_dir": "faiss_single"},
    "multi":  {"kb_dir": "knowledge_base/multi",  "index_dir": "faiss_multi"},
    "full":   {"kb_dir": "knowledge_base/full",   "index_dir": "faiss_full"},
}

# ---- 分块参数 ----
# 改造说明：原先统一用 RecursiveCharacterTextSplitter 按 500 字硬切，
# 法条被拦腰截断（第六十条第一款劈成两半，任一片段都不完整），
# 直接导致 context_recall 低。现改为 LegalChunker —— 法条边界优先切分。
# 法条本身普遍较短，故调小窗口以提升单块语义纯度。
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_CHUNK = 40          # 短于此且不含法条号的片段视为噪声
BATCH_SIZE = 100
EMBED_SUB_BATCH = 64      # ★ 每多少条 chunk 调一次 Ollama（越小进度条越细，但调用越频繁）
SUPPORTED_EXT = ("pdf", "docx", "doc", "pptx", "ppt", "txt", "md", "json", "csv", "chm", "jpg", "jpeg", "png")

_KB_DIR = os.path.join(_SCRIPT_DIR, MODE_CONFIG[MODE]["kb_dir"])
_INDEX_DIR = os.path.join(_SCRIPT_DIR, MODE_CONFIG[MODE]["index_dir"])
_STATE_FILE = os.path.join(_INDEX_DIR, "_processed_files.json")

# kb_meta.yaml 路径
_KB_META_FILE = os.path.join(_SCRIPT_DIR, "kb_meta.yaml")
_kb_meta_cache = None


def _load_kb_meta() -> dict:
    """加载 kb_meta.yaml，返回 {defaults, dirs, files}。"""
    global _kb_meta_cache
    if _kb_meta_cache is not None:
        return _kb_meta_cache
    if not os.path.isfile(_KB_META_FILE):
        _kb_meta_cache = {"defaults": {}, "dirs": {}, "files": {}}
        return _kb_meta_cache
    with open(_KB_META_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _kb_meta_cache = {
        "defaults": data.get("defaults", {}),
        "dirs": data.get("dirs", {}),
        "files": data.get("files", {}),
    }
    return _kb_meta_cache


def _get_meta_for_file(file_path: str) -> dict:
    """根据文件相对路径解析 metadata（defaults → dirs → files 逐级覆盖）。"""
    meta = _load_kb_meta()
    result = dict(meta["defaults"])
    result.setdefault("dept", "通用")
    result.setdefault("role", "officer")
    result.setdefault("secret_level", 0)
    result.setdefault("valid_until", 0)

    rel = os.path.relpath(file_path, _KB_DIR).replace("\\", "/")
    parts = rel.split("/")
    # 逐级匹配目录
    for i in range(1, len(parts) + 1):
        dir_key = "/".join(parts[:i])
        if dir_key in meta["dirs"]:
            result.update(meta["dirs"][dir_key])

    basename = os.path.basename(file_path)
    if basename in meta["files"]:
        result.update(meta["files"][basename])

    result.setdefault("doc_id", hashlib.md5(
        basename.encode()).hexdigest()[:16])
    result["source"] = basename
    return result


# ==================== ★ Embeddings（云端批量 + 进度条）====================
class ProgressEmbeddings(OpenAIEmbeddings):
    """
    包裹 OpenAIEmbeddings，为云端 Embedding API 做三件事：

    1. **批量调用**：云端单请求有文本数上限（DashScope 为 10），按批切分，
       相比逐条调用吞吐提升一个数量级。
    2. **指数退避重试**：云端有速率限制与偶发 5xx，遇到429/5xx 自动重试，
       不让整库构建因为单次抖动前功尽弃。
    3. **维度自检**：首批返回时校验维度，与配置不符立即报错 ——
       维度错了但索引建成，检索时结果是乱的且很难排查。

    本地 Ollama（EMBEDDING_PROVIDER=ollama）也走这里，
    只是批量大小由配置决定，逻辑一致。
    """

    # ★ Pydantic 字段声明：未在类里声明的名字，__init__ 里赋值会被 __setattr__ 拦截
    #   （报错 "object has no field ..."）。必须在此显式声明为字段。
    expect_dim: int | None = None
    batch_size: int = 10

    # 需要重试的瞬时错误关键词（ClassVar 明确告诉 Pydantic 这是类常量、不是字段）
    _RETRYABLE: ClassVar[tuple] = ("rate limit", "429", "500", "502", "503", "504",
                                   "timeout", "temporarily", "overloaded")

    def __init__(self, *args, expect_dim: int | None = None,
                 batch_size: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.expect_dim = expect_dim
        self.batch_size = max(1, int(batch_size or 10))

    def _embed_with_retry(self, batch: list[str], max_retries: int = 4):
        """单批向量化，指数退避重试。"""
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return super().embed_documents(batch)
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if not any(k in msg for k in self._RETRYABLE):
                    raise  # 非瞬时错误（如 key 失效、维度不合法），直接抛
                if attempt >= max_retries:
                    break
                wait = min(2 ** attempt, 16)
                print(f"\n  ⚠️ 向量批失败（{type(e).__name__}），"
                      f"{wait}s 后第 {attempt + 2} 次重试...")
                time.sleep(wait)
        raise RuntimeError(
            f"向量化重试 {max_retries} 次仍失败：{last_err}"
        )

    def embed_documents(self, texts):
        total = len(texts)
        if total == 0:
            return []

        pbar = tqdm(total=total, desc="🧠 向量化", ncols=90,
                    unit="chunk", leave=False)
        results: list[list[float]] = []
        try:
            for i in range(0, total, self.batch_size):
                batch = texts[i:i + self.batch_size]
                vecs = self._embed_with_retry(batch)

                # 维度自检：只在首批做一次
                if self.expect_dim and not results:
                    got = len(vecs[0])
                    if got != self.expect_dim:
                        raise RuntimeError(
                            f"Embedding 维度不符：配置 {self.expect_dim} 维，"
                            f"实际返回 {got} 维。\n"
                            f"换模型后必须重建索引（--rebuild），否则检索结果无意义。"
                        )

                results.extend(vecs)
                pbar.update(len(batch))
        finally:
            pbar.close()

        return results


def _precheck_embedding(embeddings, expect_dim: int) -> None:
    """建库前连通性自检 —— 避免跑了 3 小时才发现 key 填错。

    顺序很关键：先验连通（1 次调用），再验维度（关系到索引正确性）。
    """
    print("🔌 预检 Embedding 服务...")
    probe = "这是一段用于连通性检查的短文本。"
    try:
        vec = embeddings.embed_query(probe)
    except Exception as e:
        raise SystemExit(
            f"\n❌ Embedding 服务不可达：{e}\n\n"
            f"排查清单：\n"
            f"  1) .env 里 EMBEDDING_API_KEY / DASHSCOPE_API_KEY 是否已填\n"
            f"  2) 密钥是否已失效（去阿里云百炼控制台确认）\n"
            f"  3) EMBEDDING_BASE_URL 是否正确（应为空或 compatible-mode 地址）\n"
            f"  4) 若走代理，确认 HTTP_PROXY / HTTPS_PROXY 环境变量已设置\n"
            f"  5) 想退回本地：把 EMBEDDING_PROVIDER 改为 ollama\n"
        ) from e

    got = len(vec)
    if got != expect_dim:
        raise SystemExit(
            f"\n❌ 向量维度不符：配置 {expect_dim} 维，实际 {got} 维\n"
            f"   换 Embedding 模型后索引维度会变，必须用 --rebuild 重建，"
            f"否则检索结果无意义。\n"
            f"   当前模型：{embeddings.model}"
        )
    print(f"✅ 预检通过：{embeddings.model}  {got} 维\n")


# ==================== 状态文件 ====================
# ★ 2026-10 从「fingerprint 集合」升级为「结构化映射」
#
# 旧格式（list）只能回答「这个文件处理过吗」，
# 定位不到「它对应索引里哪些 chunk」——
# 于是增量更新变成只增不删：法规改版后旧块残留在索引里，
# 检索会同时返回新旧两版条文，模型可能引用已作废的条款。
# 在消防执法场景这直接导致文书出错。
#
# 新格式（含 chunk_ids）才能实现「先删旧块、再插新块」。

from index_state import (  # noqa: E402
    load_state as _istate_load,
    save_state as _istate_save,
    record_file as _istate_record,
    chunks_of as _istate_chunks,
    summarize as _istate_summarize,
    is_legacy as _istate_is_legacy,
    stale_fingerprints_for as _istate_stale,
    diff_files as _istate_diff,
)


def _load_state() -> dict:
    """读状态文件。旧 list 格式会被自动识别（标记 legacy）。"""
    return _istate_load(_STATE_FILE)


def _save_state(state: dict):
    _istate_save(_STATE_FILE, state)


# ==================== 旧块删除 ====================
def _collect_faiss_doc_ids(processed: dict, added_docs, batch_docs) -> None:
    """把 FAISS 生成的 doc_id（UUID）回填到状态记录里。

    added_docs 是 add_documents 的返回值 —— **list[str]（UUID 字符串）**，
    与 batch_docs 一一对应 —— 靠位置配对。

    为什么必须记 UUID：FAISS.delete(ids=...) 只认 docstore 的 UUID，
    传业务 chunk_id 会报 "Some specified ids do not exist"，
    于是改版时旧块永远删不掉。

    processed 显式传参 —— 它是 main() 的局部变量，
    模块级函数拿不到（第一次写这版时踩了这个坑）。
    """
    if not added_docs or not batch_docs:
        return
    # 先建 chunk_id → entry 的反查表，避免 O(n²)
    chunk_owner: dict[str, list] = {}
    for _fp, entry in processed.items():
        for cid in entry.get("chunk_ids") or []:
            chunk_owner.setdefault(cid, []).append(entry)

    # ★ added_docs 是 **id 字符串列表**（UUID），不是 Document 对象。
    #   实测：vectorstore.add_documents(docs) -> list[str]
    #   第一版误写成 getattr(doc, "id", None)，永远拿到 None，
    #   导致 FAISS 后端永远删不掉旧块（单元测试测不到这种接线错误）。
    for doc_id, original in zip(added_docs, batch_docs):
        if not isinstance(doc_id, str) or not doc_id:
            continue
        meta = getattr(original, "metadata", {}) or {}
        chunk_id = meta.get("chunk_id")
        if not chunk_id:
            continue
        for entry in chunk_owner.get(chunk_id, []):
            entry.setdefault("doc_ids", []).append(doc_id)



def _delete_stale_chunks(stale: dict, state: dict, backend: str,
                         mv_client, collection, vectorstore) -> None:
    """删除改版/删除文件留下的旧块。

    ★ 删除失败必须中断，不能继续
    --------------------------
    删不掉旧块 → 新旧两版条文并存 → 检索可能命中已作废的条款 →
    模型把它写进处罚决定书。**这是会出执法事故的静默失败**，
    所以这里抛异常中断，而不是打条日志继续跑。

    两套 ID：
      - Milvus 按业务 chunk_id（metadata.chunk_id）做 expr 删除
      - FAISS 按 docstore 的 UUID 删除（FAISS.delete 只认 UUID）
    """
    from index_prune import (
        PruneError, collect_ids_to_delete,
        delete_from_faiss, delete_from_milvus,
    )

    ids = collect_ids_to_delete(state, stale)
    chunk_ids = ids["chunk_ids"]
    doc_ids = ids["doc_ids"]

    try:
        if backend == "milvus":
            n = delete_from_milvus(mv_client, collection, chunk_ids)
        else:
            if not doc_ids:
                raise PruneError(
                    "FAISS 需要 doc_ids（UUID）才能删除，但状态文件里没有。"
                    "说明该索引是用旧版脚本建的，请执行一次 --rebuild。")
            n = delete_from_faiss(vectorstore, doc_ids)
    except PruneError:
        raise
    except Exception as e:  # noqa: BLE001
        raise PruneError(f"删除旧块时发生未预期错误：{e}") from e

    print(f"   ✅ 已删除 {n} 个旧块")
    # 从状态里移除这些旧记录（改版的会在插入新块后重新写入）
    for fp in stale:
        state.pop(fp, None)


def _file_fingerprint(path: str) -> str:
    stat = os.stat(path)
    raw = f"{path}|{stat.st_mtime}|{stat.st_size}"
    return hashlib.md5(raw.encode()).hexdigest()


def _scan_files() -> list:
    result = []
    for root, _, files in os.walk(_KB_DIR):
        for fname in files:
            if fname.startswith("~$") or fname.startswith("."):
                continue
            ext = fname.lower().split(".")[-1]
            if ext not in SUPPORTED_EXT:
                continue
            result.append(os.path.join(root, fname))
    return result


def _load_file(path: str) -> list:
    with open(path, "rb") as f:
        file_bytes = f.read()
    text = extract_text(file_bytes, os.path.basename(path))
    if not text.strip():
        return []
    # 注入 kb_meta.yaml 元数据（dept/role/secret_level/valid_until/doc_id）
    meta = _get_meta_for_file(path)
    meta["path"] = path
    return [Document(page_content=text, metadata=meta)]


# ==================== Milvus 后端 ====================
def _milvus_init(uri: str, collection_name: str, dim: int = 1024):
    """连接 Milvus，不存在则创建 collection + HNSW(COSINE) 索引。"""
    from pymilvus import (
        MilvusClient, DataType,
    )
    client = MilvusClient(uri=uri)

    if client.has_collection(collection_name):
        client.load_collection(collection_name)
        return client

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("source", DataType.VARCHAR, max_length=500)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=200)
    schema.add_field("dept", DataType.VARCHAR, max_length=100)
    schema.add_field("role", DataType.VARCHAR, max_length=100)
    schema.add_field("secret_level", DataType.INT8)
    schema.add_field("valid_until", DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    print(f"🗄️  Milvus collection '{collection_name}' 已创建 (dim={dim})")
    return client


def _milvus_insert(client, collection_name: str, chunks: list, embeddings):
    """把 chunks 向量化后插入 Milvus。"""
    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts)

    data = {
        "vector": vectors,
        "text": texts,
        "source": [c.metadata.get("source", "") for c in chunks],
        "doc_id": [c.metadata.get("doc_id", "") for c in chunks],
        "dept": [c.metadata.get("dept", "通用") for c in chunks],
        "role": [c.metadata.get("role", "officer") for c in chunks],
        "secret_level": [int(c.metadata.get("secret_level", 0)) for c in chunks],
        "valid_until": [int(c.metadata.get("valid_until", 0)) for c in chunks],
    }
    client.insert(collection_name=collection_name, data=data)
    return len(vectors)


# ==================== 分块标记落库核查 ====================
def _report_chunk_marks():
    """重建后报告索引里 is_catalog / is_tier_table 的实际数量。

    这是"索引重建是否真的生效"的**唯一可信证据**。
    改动分块策略后评测分数常常纹丝不动，肉眼无法区分
    "策略无效"和"索引没重建"—— 只能直接查落库的标记。

    预期基线（103 文件 / 4019 chunk 实测）：
        is_catalog     268  —— 两份消防产品目录
        is_tier_table    9  —— 两份裁量规定
    is_tier_table 从早期的 79 降到 9 是**正确的**：其中 45 个原本是
    目录公告被误判成档位表，目录条目切分（P4）后不再走档位分支。
    """
    pkl = os.path.join(_INDEX_DIR, "index.pkl")
    if not os.path.isfile(pkl):
        return
    try:
        import pickle as _pk
        with open(pkl, "rb") as f:
            obj = _pk.load(f)
        store = obj[0] if isinstance(obj, tuple) else obj
        raw = getattr(store, "_dict", store)
        docs = list(raw.values()) if hasattr(raw, "values") else list(raw)
        n_cat = sum(1 for d in docs if (getattr(d, "metadata", None) or {})
                    .get("is_catalog"))
        n_tier = sum(1 for d in docs if (getattr(d, "metadata", None) or {})
                     .get("is_tier_table"))
        print(f"   分块标记：is_catalog={n_cat}  is_tier_table={n_tier}")
        if n_cat == 0 and n_tier == 0:
            print("   ⚠️ 两个标记都是 0，说明新分块策略未生效"
                  "（检查 chunking.py 是否为最新版）")
    except Exception as e:
        print(f"   分块标记核查跳过：{e}")


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description="构建向量知识库")
    parser.add_argument("--backend", choices=["faiss", "milvus"], default="faiss",
                        help="向量库后端（默认 faiss）")
    parser.add_argument("--mode", choices=["single", "multi", "full"], default=MODE,
                        help="知识库模式（默认 full）")
    parser.add_argument("--prune", action="store_true",
                    help="只清理孤儿块（索引里有、状态里没有），不重建")
    parser.add_argument("--rebuild", "--force", dest="rebuild",
                        action="store_true",
                        help="强制全量重建（分块策略变更后必须加，否则会跳过已处理文件）")
    args = parser.parse_args()

    global _KB_DIR, _INDEX_DIR, _STATE_FILE
    mode_cfg = MODE_CONFIG[args.mode]
    _KB_DIR = os.path.join(_SCRIPT_DIR, mode_cfg["kb_dir"])
    _INDEX_DIR = os.path.join(_SCRIPT_DIR, mode_cfg["index_dir"])
    _STATE_FILE = os.path.join(_INDEX_DIR, "_processed_files.json")

    backend = args.backend
    milvus_uri = ""
    collection = ""
    print(f"📂 模式：{args.mode}  后端：{backend}")
    print(f"📂 知识库目录：{_KB_DIR}")
    if backend == "faiss":
        print(f"📂 索引目录：{_INDEX_DIR}\n")

        # ★ 分块策略自检
        #
        # 这个检查是被真实事故逼出来的：连续两轮改了分块策略（P1 表格
        # 重组、P4 目录条目切分），评测分数纹丝不动，排查半天才发现
        # **索引根本没重建** —— 检索跑的还是三个月前的旧 chunk。
        #
        # 症状识别：改了 chunking.py 但所有检索结果完全不变。
        # 因此这里主动检查"索引元数据是否包含新分块策略的标记字段"。
        #
        # 关键：带 --rebuild 时旧索引马上就要被删，此时报"疑似未重建"
        # 是**必然为真的假警报**（实测已误导过一次：刚加的自检第一次
        # 跑出来就是报警，而同一次运行结尾成功写了 4019 块新索引）。
        # 所以重建模式下只提示、不报警，把判定交给重建后的实际统计。
        _rebuilding = bool(getattr(args, "rebuild", False))
        _pkl = os.path.join(_INDEX_DIR, "index.pkl")
        if _rebuilding and os.path.isfile(_pkl):
            print("🔄 --rebuild：旧索引即将删除，跳过前置标记检查"
                  "（重建后报告实际标记数）\n")
        elif os.path.isfile(_pkl):
            try:
                import pickle as _pk
                with open(_pkl, "rb") as _f:
                    _obj = _pk.load(_f)
                _store = _obj[0] if isinstance(_obj, tuple) else _obj
                _raw = getattr(_store, "_dict", _store)
                _docs = list(_raw.values()) if hasattr(_raw, "values") \
                    else list(_raw)
                # 新分块策略的标记字段（LegalChunker 写入）
                _need = {"is_catalog", "is_tier_table"}
                _have = set()
                for _d in _docs[:50]:
                    _have |= set((getattr(_d, "metadata", None) or {}).keys())
                _missing = _need - _have
                if _missing:
                    print("⚠️⚠️ 索引疑似未重建！")
                    print(f"   元数据缺少分块标记：{sorted(_missing)}")
                    print("   说明当前 chunking.py 与索引内容不同步，")
                    print("   检索结果不会反映新的分块策略。")
                    print(f"   请执行：python build_index.py --rebuild\n")
                else:
                    print(f"✅ 分块标记检查通过（{len(_docs)} chunk）\n")
            except Exception as _e:
                print(f"⚠️ 索引自检跳过：{_e}\n")
        elif not args.rebuild:
            print("ℹ️  未找到既有索引，将执行全量构建\n")
    else:
        try:
            from common.config import get_settings
            settings = get_settings()
            milvus_uri = settings.milvus_uri
            collection = settings.milvus_collection
        except Exception:
            milvus_uri = os.getenv("MILVUS_URI", "")
            collection = os.getenv("MILVUS_COLLECTION", "fire_laws")
        if not milvus_uri:
            print("❌ Milvus URI 未配置（设 .env MILVUS_URI 或环境变量）")
            return
        print(f"🗄️  Milvus: {milvus_uri}  collection: {collection}\n")

    if not os.path.exists(_KB_DIR):
        print(f"❌ 目录不存在: {_KB_DIR}")
        return

    # 1. 扫描
    all_files = _scan_files()
    print(f"📊 共扫描到 {len(all_files)} 个支持的文件\n")

    # 2. 断点续传
    if args.rebuild:
        print("🔄 --rebuild：忽略断点记录，全量重建\n")
        processed = {}
        state = {}
        todo = list(all_files)
        if backend == "faiss":
            for fn in ("index.faiss", "index.pkl"):
                p = os.path.join(_INDEX_DIR, fn)
                if os.path.isfile(p):
                    os.remove(p)
                    print(f"   已删除旧索引文件：{fn}")
    else:
        state = _load_state()
        if state and _istate_is_legacy(state):
            print("⚠️ 检测到旧格式状态文件（只有指纹、没有块 ID）。\n"
                  "   旧格式无法定位已入库的块，因此**无法删除改版残留**。\n"
                  "   建议执行一次 --rebuild 重建，之后增量更新才能正确清理旧块。\n")

        # ★ 按「路径」而非指纹做差异计算
        #   指纹含 mtime，文件一改指纹就变 —— 按指纹比对的话，
        #   改版的文件会变成「removed + added」，看不出是同一文件的改版，
        #   也就无法定位并删除它的旧块。
        current_map = {}
        fp_to_path = {}
        for f in all_files:
            fp = _file_fingerprint(f)
            current_map[fp] = os.path.relpath(f, _KB_DIR).replace(os.sep, "/")
            fp_to_path[fp] = f

        added_fps, changed_fps, removed_fps = _istate_diff(state, current_map)

        # 需要删除的旧块：改版的 + 被删的
        stale = _istate_stale(state, current_map)
        if stale:
            total_stale = sum(len(v) for v in stale.values())
            print(f"🧹 检测到 {len(stale)} 个文件有旧块残留"
                  f"（共 {total_stale} 块）待清理")
            for fp, ids in stale.items():
                src_path = (state.get(fp) or {}).get("path") or "?"
                print(f"   - {src_path}: {len(ids)} 块")
            _delete_stale_chunks(stale, state, backend, mv_client,
                                 collection, vectorstore)
        else:
            print("✅ 无旧块残留")

        # 待处理 = 新增 + 改版（改版的要先删旧块再插新块）
        todo = [fp_to_path[fp] for fp in added_fps + changed_fps if fp in fp_to_path]
        processed = state

    index_complete = True
    if backend == "faiss":
        index_file = os.path.join(_INDEX_DIR, "index.faiss")
        pkl_file = os.path.join(_INDEX_DIR, "index.pkl")
        index_complete = os.path.isfile(index_file) and os.path.isfile(pkl_file)
        if not index_complete and processed:
            print("⚠️ 检测到处理记录存在，但 FAISS 索引文件不完整，将重新构建\n")
            processed = {}
            state = {}
            todo = list(all_files)

    print(f"✅ 已处理：{len(processed)} 个")
    print(f"⏳ 待处理：{len(todo)} 个\n")

    if not todo:
        print("🎉 无需重建" if index_complete
              else "⚠️ 没有待处理文件，但索引不存在")
        return

    # 3. 分块器（法条感知：按"第X条"边界切，chunk 带law/article 元数据）
    splitter = LegalChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk=MIN_CHUNK,
    )

    # 4. 嵌入模型（云端批量 + 进度条）
    cfg = get_embedding_config()
    # ★ 网络策略：云端 API 必须走系统代理（trust_env=True），
    #   本地 Ollama 才需要 trust_env=False 避免代理劫持内网请求。
    #   这里按provider 分开，是换API 后最容易踩的坑。
    is_local = cfg.get("provider") == "ollama"
    print(f"📐Embedding: {cfg['model']}  "
          f"provider={cfg.get('provider')}  dim={cfg.get('dim')}  "
          f"batch={cfg.get('batch_size')}  代理={'关闭' if is_local else '开启'}")
    embeddings = ProgressEmbeddings(
        model=cfg["model"], api_key=cfg["api_key"],
        base_url=cfg["base_url"], check_embedding_ctx_length=False,
        expect_dim=cfg.get("dim"),
        batch_size=cfg.get("batch_size", 10),
        http_client=httpx.Client(
            trust_env=not is_local,
            timeout=cfg.get("timeout", 120),
        ),
    )
    # 连通性 + 维度预检：跑3 小时才发现 key 填错是最糟的失败方式
    _precheck_embedding(embeddings, cfg.get("dim", 1024))

    # 5. 后端初始化
    mv_client = None
    vectorstore = None
    if backend == "milvus":
        if args.rebuild:
            try:
                from pymilvus import MilvusClient
                probe = MilvusClient(uri=milvus_uri)
                if probe.has_collection(collection):
                    probe.drop_collection(collection)
                    print(f"🗑️  --rebuild：已删除 collection '{collection}'")
            except Exception as e:
                print(f"⚠️ 清理旧 collection 失败（继续重建）: {e}")
        mv_client = _milvus_init(milvus_uri, collection)
    else:
        if os.path.isfile(index_file) and os.path.isfile(pkl_file):
            print("📥 加载已有索引，进行增量添加...")
            vectorstore = FAISS.load_local(
                _INDEX_DIR, embeddings, allow_dangerous_deserialization=True
            )
        else:
            print("🆕 创建新索引")

    # 6. 分批处理
    start_time = time.time()
    total_chunks = 0
    batch_docs = []

    for i, path in enumerate(todo, 1):
        try:
            docs = _load_file(path)
            if not docs:
                print(f"  ⏭️ [{i}/{len(todo)}] 跳过（无文本）: {os.path.basename(path)}")
                processed[_file_fingerprint(path)] = {
                    "path": os.path.relpath(path, _KB_DIR).replace(os.sep, "/"),
                    "chunk_ids": [], "doc_ids": [],
                    "indexed_at": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(timespec="seconds"),
                    "chunks": 0, "legacy": False,
                }
                continue

            chunks = splitter.split_documents(docs)
            batch_docs.extend(chunks)
            time.sleep(0.05)

            # ★ 记录本文件的块 ID（业务 chunk_id + 向量库 doc_id）
            #   没有这两个 ID，改版时就定位不到旧块，增量更新会退化成
            #   「只增不删」——法规改版后旧条文仍可被检索并写进处罚文书。
            _fp = _file_fingerprint(path)
            _chunk_ids = [str(c.metadata.get("chunk_id", "")) for c in chunks
                          if getattr(c, "metadata", None)]
            _chunk_ids = [c for c in _chunk_ids if c]
            processed[_fp] = {
                "path": os.path.relpath(path, _KB_DIR).replace(os.sep, "/"),
                "chunk_ids": _chunk_ids,
                "doc_ids": [],
                "indexed_at": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(timespec="seconds"),
                "chunks": len(chunks),
                "legacy": False,
            }

            print(f"  ✅ [{i}/{len(todo)}] {os.path.basename(path)}  ({len(chunks)} 块)")

            # 每 BATCH_SIZE 个文件存一次
            if i % BATCH_SIZE == 0 or i == len(todo):
                if not batch_docs:
                    continue
                n = len(batch_docs)
                print(f"\n💾 开始向量化并保存（{n} 块，累计 {total_chunks + n}）...")
                t0 = time.time()

                if backend == "milvus":
                    _milvus_insert(mv_client, collection, batch_docs, embeddings)
                else:
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(
                            batch_docs, embedding=embeddings
                        )
                    else:
                        # ★ 收集 FAISS 生成的 doc_id（UUID）
                        #   add_documents 返回带 id 的文档列表 —— 这是唯一
                        #   能拿到 UUID 的时机，事后无法从 chunk_id 反推。
                        added = vectorstore.add_documents(batch_docs)
                        _collect_faiss_doc_ids(processed, added, batch_docs)
                    os.makedirs(_INDEX_DIR, exist_ok=True)
                    vectorstore.save_local(_INDEX_DIR)

                dt = time.time() - t0
                speed = n / dt if dt > 0 else 0
                total_chunks += n
                batch_docs = []
                _save_state(processed)
                dst = f"Milvus {collection}" if backend == "milvus" else _INDEX_DIR
                print(f"   ✅ 保存完成：{n} 块，用时 {dt:.1f}s（{speed:.1f} 块/秒）")
                print(f"   索引已更新到 {dst}\n")

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
            if batch_docs:
                if backend == "milvus" and mv_client is not None:
                    print(f"   正在插入 {len(batch_docs)} 块残留数据...")
                    _milvus_insert(mv_client, collection, batch_docs, embeddings)
                elif backend == "faiss" and vectorstore is not None:
                    print(f"   正在保存 {len(batch_docs)} 块残留数据...")
                    vectorstore.add_documents(batch_docs)
                    vectorstore.save_local(_INDEX_DIR)
                _save_state(processed)
            return
        except Exception as e:
            print(f"  ❌ [{i}/{len(todo)}] 失败 {os.path.basename(path)}: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"🎉 完成！")
    print(f"   本次处理：{len(todo)} 个文件")
    print(f"   累计块数：{total_chunks}")
    print(f"   耗时：{elapsed/60:.1f} 分钟")
    if backend == "milvus":
        print(f"   Milvus collection: {collection}")
    else:
        print(f"   索引目录：{_INDEX_DIR}")
        _report_chunk_marks()
    print("=" * 60)


if __name__ == "__main__":
    main()