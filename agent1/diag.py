# diag.py
import httpx
import traceback
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import get_embedding_config

cfg = get_embedding_config()
print(f"模型: {cfg['model']}")
print(f"URL:  {cfg['base_url']}\n")

emb = OpenAIEmbeddings(
    model=cfg["model"], api_key=cfg["api_key"],
    base_url=cfg["base_url"], check_embedding_ctx_length=False,
    http_client=httpx.Client(trust_env=False, timeout=60),
)

# 1. 测 query 向量维度
try:
    v = emb.embed_query("测试")
    print(f"✅ Query 维度: {len(v)}")
except Exception as e:
    print(f"❌ Query 向量化失败: {type(e).__name__}: {e}")
    traceback.print_exc()
    raise SystemExit

# 2. 加载 FAISS
try:
    vs = FAISS.load_local("./faiss_full", emb,
                         allow_dangerous_deserialization=True)
    print(f"✅ 索引条目数: {len(vs.docstore._dict)}")
    print(f"✅ 索引维度: {vs.index.d}")
except Exception as e:
    print(f"❌ 加载 FAISS 失败: {type(e).__name__}: {e}")
    traceback.print_exc()
    raise SystemExit

# 3. 试检索
try:
    docs = vs.similarity_search("消防法第六十条", k=20)
    print(f"\n✅ 检索到 {len(docs)} 条:")
    for i, d in enumerate(docs, 1):
        print(f"  {i}. [{d.metadata.get('source','?')}]")
except Exception as e:
    print(f"\n❌ 检索失败: {type(e).__name__}: {e}")
    traceback.print_exc()