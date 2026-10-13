# check_sources.py
from collections import Counter
from config import get_embedding_config
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import httpx

cfg = get_embedding_config()
emb = OpenAIEmbeddings(
    model=cfg["model"], api_key=cfg["api_key"],
    base_url=cfg["base_url"], check_embedding_ctx_length=False,
    http_client=httpx.Client(trust_env=False, timeout=60),
)
vs = FAISS.load_local("./faiss_full", emb,
                     allow_dangerous_deserialization=True)

sources = Counter()
for doc in vs.docstore._dict.values():
    src = doc.metadata.get("source", "?")
    sources[src] += 1

print(f"★ 索引总块数：{len(vs.docstore._dict)}")
print(f"★ 来源文件数：{len(sources)}\n")

print("=== 含'消防法'的文件 ===")
found = False
for src, cnt in sorted(sources.items()):
    if "消防法" in src and "立法" not in src:
        print(f"  {src}: {cnt} 条")
        found = True
if not found:
    print("  ❌ 一个都没有！《消防法》没进索引")

print("\n=== 全部来源（前 20） ===")
for src, cnt in sources.most_common(20):
    print(f"  {cnt:4d}  {src}")