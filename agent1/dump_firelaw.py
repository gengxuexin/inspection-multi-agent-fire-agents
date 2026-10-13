# dump_firelaw.py
import httpx
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import get_embedding_config

cfg = get_embedding_config()
emb = OpenAIEmbeddings(
    model=cfg["model"], api_key=cfg["api_key"],
    base_url=cfg["base_url"], check_embedding_ctx_length=False,
    http_client=httpx.Client(trust_env=False, timeout=60),
)
vs = FAISS.load_local("./faiss_full", emb, allow_dangerous_deserialization=True)

# 只看这两个源文件的 chunk
TARGET = ["中华人民共和国消防法.pdf", "消防法（2021年4月29日）.docx"]

for src in TARGET:
    print(f"\n{'='*70}")
    print(f"  {src}")
    print(f"{'='*70}")
    n = 0
    for d in vs.docstore._dict.values():
        if d.metadata.get("source") != src:
            continue
        n += 1
        text = d.page_content
        # 只显示跟第六十条有关的
        if "第六十条" in text:
            print(f"\n--- chunk #{n} 含'第六十条' ---")
            print(text[:800])
    print(f"\n【{src}】共 {n} 个 chunk")