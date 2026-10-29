"""适配 tests/test_hybrid.py 到拆分后的 idf 结构。

★ 为什么改测试而不是给产品代码加兼容别名：
  加别名 `_IDF_MAX = _IDF_MAX_RERANK` 会掩盖问题 ——
  测试明明想验证"精排用哪个上限"，别名让它验证不到真实路径，
  于是拆分的意义（区分 BM25 与精排）就丢了。
  改测试让它显式声明"我模拟的是精排，所以给 RERANK 的上限"，
  测试才真正测到要测的东西。

要改两处：
  1. r._IDF_MAX -> r._IDF_MAX_RERANK
  2. mock 里补 _df / _n_docs（精排现在自己算 idf，不再只读 _idf）
"""
import pathlib
import re

p = pathlib.Path("tests/test_hybrid.py")
src = p.read_text(encoding="utf-8")

# ---- 1. 常量名 ----
src = src.replace(
    "    r._IDF_MAX = HybridRetriever._IDF_MAX",
    "    r._IDF_MAX_RERANK = HybridRetriever._IDF_MAX_RERANK\n"
    "    r._IDF_MAX_BM25 = HybridRetriever._IDF_MAX_BM25",
)

# ---- 2. 补 _df / _n_docs ----
old_idf = "    r._idf = {w: 1.0 for toks in r.tokenized for w in set(toks)}"
new_idf = (
    "    r._idf = {w: 1.0 for toks in r.tokenized for w in set(toks)}\n"
    "    # 精排现在自己从 df 计算 idf（不再只读 _idf），\n"
    "    # 所以 mock 必须补这两个属性，否则 _rerank 会 AttributeError。\n"
    "    r._df = {w: 1 for toks in r.tokenized for w in set(toks)}\n"
    "    r._n_docs = max(len(docs), 1)"
)
if old_idf in src:
    src = src.replace(old_idf, new_idf)
    print("✓ 已补 _df / _n_docs")
else:
    print("! 未找到 _idf 锚点")

# ---- 3. 兜底：若别处还引用旧名，全部改掉 ----
left = re.findall(r"_IDF_MAX(?![_A-Z])", src)
if left:
    print(f"! 仍有 {len(left)} 处引用旧名 _IDF_MAX，需手工检查")
    for i, ln in enumerate(src.splitlines(), 1):
        if re.search(r"_IDF_MAX(?![_A-Z])", ln):
            print(f"    {i}: {ln.strip()[:70]}")

p.write_text(src, encoding="utf-8")
print("✓ 测试已适配")