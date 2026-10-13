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
from tqdm import tqdm
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ★ 让脚本能找到同目录的 config / file_analyzer
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config import get_embedding_config
from file_analyzer import extract_text

# ==================== 配置 ====================
MODE = "full"     # single / multi / full

MODE_CONFIG = {
    "single": {"kb_dir": "knowledge_base/single", "index_dir": "faiss_single"},
    "multi":  {"kb_dir": "knowledge_base/multi",  "index_dir": "faiss_multi"},
    "full":   {"kb_dir": "knowledge_base/full",   "index_dir": "faiss_full"},
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
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


# ==================== ★ 带进度条的 Embeddings ====================
class ProgressEmbeddings(OpenAIEmbeddings):
    """
    包裹 OpenAIEmbeddings：
    在 embed_documents 里分批调用 + tqdm 进度条，
    这样 FAISS.from_documents / add_documents 触发向量化时，
    终端就会显示实时进度。
    """
    def embed_documents(self, texts):
        total = len(texts)
        pbar = tqdm(
            total=total,
            desc="🧠 向量化",
            ncols=90,
            unit="chunk",
            leave=False,          # 结束后不残留
        )
        results = []
        try:
            for i in range(0, total, EMBED_SUB_BATCH):
                sub = texts[i:i + EMBED_SUB_BATCH]
                vecs = super().embed_documents(sub)
                results.extend(vecs)
                pbar.update(len(sub))
        finally:
            pbar.close()
        return results


# ==================== 状态文件 ====================
def _load_state() -> set:
    if os.path.exists(_STATE_FILE):
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def _save_state(processed: set):
    os.makedirs(_INDEX_DIR, exist_ok=True)
    with open(_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(processed), f, ensure_ascii=False, indent=2)


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


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description="构建向量知识库")
    parser.add_argument("--backend", choices=["faiss", "milvus"], default="faiss",
                        help="向量库后端（默认 faiss）")
    parser.add_argument("--mode", choices=["single", "multi", "full"], default=MODE,
                        help="知识库模式（默认 full）")
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
    processed = _load_state()
    todo = [f for f in all_files if _file_fingerprint(f) not in processed]

    index_complete = True
    if backend == "faiss":
        index_file = os.path.join(_INDEX_DIR, "index.faiss")
        pkl_file = os.path.join(_INDEX_DIR, "index.pkl")
        index_complete = os.path.isfile(index_file) and os.path.isfile(pkl_file)
        if not index_complete and processed:
            print("⚠️ 检测到处理记录存在，但 FAISS 索引文件不完整，将重新构建\n")
            processed = set()
            todo = list(all_files)

    print(f"✅ 已处理：{len(processed)} 个")
    print(f"⏳ 待处理：{len(todo)} 个\n")

    if not todo:
        print("🎉 无需重建" if index_complete
              else "⚠️ 没有待处理文件，但索引不存在")
        return

    # 3. 分块器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )

    # 4. 嵌入模型（★ 用带进度条的类）
    cfg = get_embedding_config()
    embeddings = ProgressEmbeddings(
        model=cfg["model"], api_key=cfg["api_key"],
        base_url=cfg["base_url"], check_embedding_ctx_length=False,
        http_client=httpx.Client(trust_env=False, timeout=300),
    )

    # 5. 后端初始化
    mv_client = None
    vectorstore = None
    if backend == "milvus":
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
                processed.add(_file_fingerprint(path))
                continue

            chunks = splitter.split_documents(docs)
            batch_docs.extend(chunks)
            time.sleep(0.05)
            processed.add(_file_fingerprint(path))

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
                        vectorstore.add_documents(batch_docs)
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
    print("=" * 60)


if __name__ == "__main__":
    main()