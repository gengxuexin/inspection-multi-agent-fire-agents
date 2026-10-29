"""探查 index.pkl 的真实结构 —— 只打印结构，不加载 embedding 模型。"""
import pickle
import sys

pkl = sys.argv[1] if len(sys.argv) > 1 else "agent1/faiss_full/index.pkl"

with open(pkl, "rb") as f:
    d = pickle.load(f)

found = []


def walk(o, depth=0, path="root"):
    pad = "  " * depth
    if depth > 5:
        return
    if isinstance(o, dict):
        keys = list(o.keys())[:5]
        print(f"{pad}{path}: dict len={len(o)} keys={keys[:5]}")
        for k in keys[:2]:
            v = o[k]
            print(f"{pad}   [{k!r}] -> {type(v).__name__}")
            if isinstance(v, (dict, list, tuple)):
                walk(v, depth + 1, f"{path}[{k!r}]")
    elif isinstance(o, (list, tuple)):
        print(f"{pad}{path}: {type(o).__name__} len={len(o)}")
        for i, v in enumerate(o[:3]):
            print(f"{pad}   [{i}] -> {type(v).__name__}")
            if isinstance(v, (dict, list, tuple)):
                walk(v, depth + 1, f"{path}[{i}]")
    elif isinstance(o, str):
        if len(o) > 40:
            found.append((path, o[:60]))
        print(f"{pad}{path}: str len={len(o)}")
    else:
        print(f"{pad}{path}: {type(o).__name__} = {str(o)[:50]}")


walk(d)
print()
print(f"长度 >40 的字符串：{len(found)} 个")
for p, s in found[:5]:
    print(f"  {p}: {s}...")