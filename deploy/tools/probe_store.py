import pickle

d = pickle.load(open("agent1/faiss_full/index.pkl", "rb"))
store = d[0]
print("store:", type(store).__name__)
print("attrs:", [a for a in dir(store) if not a.startswith("__")])

inner = getattr(store, "_dict", None)
print("_dict:", type(inner).__name__,
      f"len={len(inner)}" if hasattr(inner, "__len__") else "")

if isinstance(inner, dict) and inner:
    k0 = list(inner.keys())[0]
    v0 = inner[k0]
    print(f"  key类型={type(k0).__name__}")
    print(f"  值类型={type(v0).__name__}")
    print(f"  有 page_content? {hasattr(v0, 'page_content')}")
    if hasattr(v0, "page_content"):
        pc = v0.page_content
        print(f"  文本长度: {len(pc)}")
        print(f"  前60 字: {pc[:60]}")
    else:
        print(f"  值内容: {str(v0)[:80]}")