# debug_retrieval.py
"""对比有/无 BGE 前缀的检索结果"""
import httpx
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import get_embedding_config

cfg = get_embedding_config()
PREFIX = "为这个句子生成表示以用于检索相关文章："


def make_emb(with_prefix: bool):
    if with_prefix:
        class E(OpenAIEmbeddings):
            def embed_query(self, text):
                return super().embed_query(PREFIX + text)
        return E(
            model=cfg["model"], api_key=cfg["api_key"],
            base_url=cfg["base_url"], check_embedding_ctx_length=False,
            http_client=httpx.Client(trust_env=False, timeout=60),
        )
    return OpenAIEmbeddings(
        model=cfg["model"], api_key=cfg["api_key"],
        base_url=cfg["base_url"], check_embedding_ctx_length=False,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


query = "消防法第六十条 单位违反本法规定 责令改正 罚款"

for with_prefix in [False, True]:
    tag = "有前缀" if with_prefix else "无前缀"
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")
    emb = make_emb(with_prefix)
    vs = FAISS.load_local("./faiss_full", emb,
                         allow_dangerous_deserialization=True)
    for i, d in enumerate(vs.similarity_search(query, k=5), 1):
        src = d.metadata.get("source", "?")
        content = d.page_content[:70].replace("\n", " ")
        print(f"  {i}. [{src}]")
        print(f"     {content}...")