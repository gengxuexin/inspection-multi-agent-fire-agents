# retrieval_backend.py
"""
统一检索接口：Milvus 优先（含 expr 权限过滤），FAISS 回退。
对外签名：search(query, top_k, user_ctx) -> list[dict]

返回格式：
    [
        {
            "content": "条文原文...",
            "source": "中华人民共和国消防法.pdf",
            "score": 0.95,
            "metadata": {"doc_id": "...", "dept": "通用", "role": "officer",
                         "secret_level": 0, "valid_until": 0},
        },
        ...
    ]

权限过滤规则（两种后端一致）：
    dept in [user_dept, "通用"]
    secret_level <= user_clearance
    valid_until == 0 (永久) 或 valid_until > now
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
from typing import Any, Optional

# ---- 确保能导入 common 包和 agent1 本地模块 ----
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import httpx

logger = logging.getLogger("fire_agents.retrieval")

# 清算等级含义：0=公开 1=内部 2=秘密 3=机密 4=绝密
DEFAULT_CLEARANCE = 2
DEFAULT_DEPT = "消防支队"

# 检索条数默认值。RAGAS 实测：top_k=20 时 context_precision 仅 0.165，
# 大量噪声片段进入 prompt。默认压到 5，可由调用方按需上调。
DEFAULT_TOP_K = 5
MAX_TOP_K = 10


# ==================== UserCtx 兼容 ====================
def _normalize_user_ctx(user_ctx: Any = None) -> tuple[str, int]:
    """从各种类型的 user_ctx 提取 (dept, clearance)。"""
    if user_ctx is None:
        return DEFAULT_DEPT, DEFAULT_CLEARANCE
    # common.auth.UserCtx (pydantic BaseModel)
    dept = getattr(user_ctx, "dept", DEFAULT_DEPT) or DEFAULT_DEPT
    clearance = getattr(user_ctx, "clearance", DEFAULT_CLEARANCE)
    try:
        clearance = int(clearance)
    except (TypeError, ValueError):
        clearance = DEFAULT_CLEARANCE
    return str(dept), clearance


# ==================== Embedding 单例 ====================
_embed_client = None


def _get_embed_client():
    """惰性创建 OpenAI 兼容 embedding 客户端。

    ★ 网络策略：云端 provider 必须 trust_env=True 走系统代理；
      本地 Ollama 用 trust_env=False 避免代理劫持内网请求。
    """
    global _embed_client
    if _embed_client is not None:
        return _embed_client
    from config import get_embedding_config
    cfg = get_embedding_config()
    is_local = cfg.get("provider") == "ollama"
    from openai import OpenAI
    _embed_client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=cfg.get("timeout", 120),
        http_client=httpx.Client(
            trust_env=not is_local,
            timeout=cfg.get("timeout", 120),
        ),
    )
    return _embed_client


def _embed_query(query: str) -> list[float]:
    """把查询文本转向量。维度由 embedding 配置决定（不硬编码）。"""
    from config import get_embedding_config
    cfg = get_embedding_config()
    client = _get_embed_client()
    resp = client.embeddings.create(model=cfg["model"], input=query)
    return resp.data[0].embedding


# ==================== Milvus 后端 ====================
_milvus_client = None
_milvus_state: Optional[bool] = None  # None=未探测 True=可用 False=不可用


def _milvus_connect():
    """连接 Milvus 并返回 client；不可用返回 None（全局缓存探测结果）。"""
    global _milvus_client, _milvus_state
    if _milvus_state is False:
        return None
    if _milvus_client is not None:
        return _milvus_client

    from common.config import get_settings
    settings = get_settings()
    if not settings.milvus_uri:
        _milvus_state = False
        return None

    try:
        from pymilvus import MilvusClient
        _milvus_client = MilvusClient(uri=settings.milvus_uri)
        if not _milvus_client.has_collection(settings.milvus_collection):
            logger.warning(
                "[retrieval] Milvus collection '%s' 不存在，回退 FAISS",
                settings.milvus_collection,
            )
            _milvus_state = False
            _milvus_client = None
            return None
        _milvus_state = True
        logger.info("[retrieval] Milvus 已连接: %s", settings.milvus_uri)
        return _milvus_client
    except Exception as e:
        logger.warning("[retrieval] Milvus 连接失败，回退 FAISS: %s", e)
        _milvus_state = False
        return None


def _build_milvus_expr(dept: str, clearance: int) -> str:
    """构造 Milvus 权限过滤 expr。"""
    now_ts = int(time.time())
    return (
        f'dept in ["{dept}", "通用"] && '
        f'secret_level <= {clearance} && '
        f'(valid_until == 0 || valid_until > {now_ts})'
    )


def _milvus_search(
    query: str, top_k: int, dept: str, clearance: int
) -> list[dict] | None:
    """Milvus 向量检索 + 权限过滤；失败返回 None 触发 FAISS 回退。"""
    client = _milvus_connect()
    if client is None:
        return None

    from common.config import get_settings
    settings = get_settings()

    try:
        vec = _embed_query(query)
    except Exception as e:
        logger.warning("[retrieval] embedding 失败，回退 FAISS: %s", e)
        return None

    expr = _build_milvus_expr(dept, clearance)
    output_fields = [
        "text", "source", "doc_id", "dept", "role", "secret_level", "valid_until",
    ]

    try:
        results = client.search(
            collection_name=settings.milvus_collection,
            data=[vec],
            limit=top_k,
            filter=expr,
            output_fields=output_fields,
        )
    except Exception as e:
        logger.warning("[retrieval] Milvus search 失败，回退 FAISS: %s", e)
        return None

    hits = results[0] if results else []
    out: list[dict] = []
    for hit in hits:
        entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
        score = hit.get("distance", 0.0) if isinstance(hit, dict) else 0.0
        out.append({
            "content": entity.get("text", ""),
            "source": entity.get("source", "?"),
            "score": float(score),
            "metadata": {
                "doc_id": entity.get("doc_id", ""),
                "dept": entity.get("dept", ""),
                "role": entity.get("role", ""),
                "secret_level": int(entity.get("secret_level", 0)),
                "valid_until": int(entity.get("valid_until", 0)),
            },
        })
    return out


# ==================== FAISS 回退后端 ====================
_hybrid_cache: dict = {}


def _get_faiss_hybrid():
    """惰性加载 FAISS + BM25 混合检索器（复用 hybrid.py）。"""
    cache_key = "faiss_full"
    if cache_key in _hybrid_cache:
        return _hybrid_cache[cache_key]

    index_dir = os.path.join(_SCRIPT_DIR, "faiss_full")
    index_file = os.path.join(index_dir, "index.faiss")
    pkl_file = os.path.join(index_dir, "index.pkl")
    if not (os.path.isfile(index_file) and os.path.isfile(pkl_file)):
        raise FileNotFoundError(f"向量库不完整: {index_dir}")

    import httpx as _httpx
    from langchain_openai import OpenAIEmbeddings
    from config import get_embedding_config
    from hybrid import get_hybrid

    emb_cfg = get_embedding_config()
    is_local = emb_cfg.get("provider") == "ollama"
    embeddings = OpenAIEmbeddings(
        model=emb_cfg["model"], api_key=emb_cfg["api_key"],
        base_url=emb_cfg["base_url"], check_embedding_ctx_length=False,
        http_client=_httpx.Client(
            trust_env=not is_local,
            timeout=emb_cfg.get("timeout", 120),
        ),
    )
    hybrid = get_hybrid(index_dir, embeddings)
    _hybrid_cache[cache_key] = hybrid
    return hybrid


def _passes_filter(metadata: dict, dept: str, clearance: int) -> bool:
    """检查单条文档的 metadata 是否通过权限过滤。"""
    doc_dept = metadata.get("dept", "通用")
    if doc_dept not in (dept, "通用"):
        return False
    doc_secret = int(metadata.get("secret_level", 0))
    if doc_secret > clearance:
        return False
    valid_until = int(metadata.get("valid_until", 0))
    if valid_until and valid_until < int(time.time()):
        return False
    return True


# ==================== 片段质量过滤 ====================
# 检索侧的第二道闸门：即便精排把噪声排后了，
# 仍需拦住"纯页眉/目录/空白"这类无信息量片段，避免它们占用 top-k 名额。
_MIN_USEFUL_CHARS = 24
_RE_CJK = re.compile(r"[一-鿿]")


def _is_useful(content: str) -> bool:
    """片段是否含足够有效信息。

    判据（任一不满足即丢弃）：
      - 长度 < 24 字符
      - 中文字符 < 8 个（纯英文/数字/符号页码片段）
    """
    if not content:
        return False
    if len(content.strip()) < _MIN_USEFUL_CHARS:
        return False
    if len(_RE_CJK.findall(content)) < 8:
        return False
    return True


def _faiss_search(
    query: str, top_k: int, dept: str, clearance: int
) -> list[dict]:
    """FAISS + BM25 混合检索 + metadata 权限过滤 + 片段质量过滤。"""
    hybrid = _get_faiss_hybrid()
    # 扩大候选池以补偿过滤后的损失
    pool = max(top_k * 5, 20)
    docs = hybrid.search(query, k=pool)

    now_ts = int(time.time())
    out: list[dict] = []
    for d in docs:
        meta = dict(d.metadata or {})
        if not _passes_filter(meta, dept, clearance):
            continue
        if not _is_useful(d.page_content):
            continue
        out.append({
            "content": d.page_content,
            "source": meta.get("source", "?"),
            "score": 0.0,  # FAISS hybrid 不返回精确分数
            "metadata": meta,
        })
        if len(out) >= top_k:
            break

    # 如果过滤后结果不足，放宽 dept 限制兜底（但保留密级与时效约束）
    if len(out) < top_k:
        seen = {r["content"] for r in out}
        for d in docs:
            meta = dict(d.metadata or {})
            doc_secret = int(meta.get("secret_level", 0))
            if doc_secret > clearance:
                continue
            valid_until = int(meta.get("valid_until", 0))
            if valid_until and valid_until < now_ts:
                continue
            content = d.page_content
            if content in seen or not _is_useful(content):
                continue
            seen.add(content)
            out.append({
                "content": content,
                "source": meta.get("source", "?"),
                "score": 0.0,
                "metadata": meta,
            })
            if len(out) >= top_k:
                break

    return out


# ==================== 统一入口 ====================
def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    user_ctx: Any = None,
) -> list[dict]:
    """统一检索：Milvus 优先，FAISS 回退。

    :param query:     查询文本
    :param top_k:     返回条数（默认 5，上限 MAX_TOP_K=10；
                      实测top_k 越大噪声越多，RAGAS context_precision 越低）
    :param user_ctx:  用户上下文（common.auth.UserCtx 或含 dept/clearance 属性的对象）
    :return:          [{content, source, score, metadata}, ...]
    """
    if not query or not query.strip():
        return []
    # 收敛到合理区间：过多片段会把噪声一起塞进 prompt
    top_k = max(1, min(int(top_k), MAX_TOP_K))

    dept, clearance = _normalize_user_ctx(user_ctx)

    # 1) 尝试 Milvus
    milvus_result = _milvus_search(query, top_k, dept, clearance)
    if milvus_result is not None:
        return milvus_result

    # 2) 回退 FAISS
    try:
        return _faiss_search(query, top_k, dept, clearance)
    except FileNotFoundError:
        logger.error("[retrieval] FAISS 向量库不存在，返回空结果")
        return []
    except Exception as e:
        logger.error("[retrieval] FAISS 检索失败: %s", e)
        return []


def search_docs(query: str, top_k: int = DEFAULT_TOP_K, user_ctx: Any = None):
    """返回 langchain Document 列表（兼容 rag_module.py / rag_legal_advisor.py）。"""
    from langchain_core.documents import Document
    results = search(query, top_k, user_ctx)
    return [
        Document(page_content=r["content"], metadata=r["metadata"])
        for r in results
    ]


def is_milvus_enabled() -> bool:
    """当前是否使用 Milvus 后端（仅查询，不触发连接）。"""
    return _milvus_state is True


def reset_backend():
    """重置后端状态（测试用）。"""
    global _milvus_client, _milvus_state
    _milvus_client = None
    _milvus_state = None
