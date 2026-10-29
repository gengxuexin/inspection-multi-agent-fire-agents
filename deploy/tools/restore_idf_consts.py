"""补回被误删的两个 IDF 常量定义。

重建时清理疑问词集合残留，删除区间 228-241 把
`_IDF_MAX_BM25` 与 `_IDF_MAX_RERANK` 两行定义一起删掉了 ——
注释留下了，赋值没了。

★ 教训（第二次栽在同类问题上）：
  按行号区间删除代码时，区间边界必须以「结构标记」为准
  （这里是 `def __init__` 这类明确的下界），不能靠"往上找注释块起点"
  这种启发式 —— 注释块可能属于上一段（IDF 常量的说明）而非下一段。
  **删除区间的下界应当是下一个明确的结构起点，而不是上一段的尾巴。**
"""
import ast
import pathlib

p = pathlib.Path("agent1/hybrid.py")
src = p.read_text(encoding="utf-8")

if "_IDF_MAX_BM25 =" in src:
    print("· 常量已存在，跳过")
    raise SystemExit(0)

ANCHOR = "    def __init__(self, index_dir, embeddings, rrf_k: int = 60):"

if ANCHOR not in src:
    print("✗ 找不到 __init__ 锚点")
    raise SystemExit(1)

BLOCK = """    # ★ 三轮 A/B 实测后的最终状态（详见 docs/rerank-ab-log.md）
    #
    #   BM25 保持 3.0：这是已验证过的召回配置，不动。
    #   精排也是 3.0：试过 8.0 + idf²，机制上消除了覆盖率饱和，
    #   但把「跨文档综合」类的 recall 压下去了
    #   （-0.063 ~ -0.085，是唯一超出噪声底的指标），
    #   而这类样本占 45%（21/47）。代价 > 收益，回退。
    #
    #   **保留拆分的意义**：BM25 与精排不再共用 idf 上限，
    #   今后调精排不会连带影响召回 —— 这是本轮最重要的结构性收获。
    _IDF_MAX_BM25 = 3.0
    _IDF_MAX_RERANK = 3.0

"""

src = src.replace(ANCHOR, BLOCK + ANCHOR, 1)

try:
    ast.parse(src)
except SyntaxError as e:
    print(f"✗ 插入后语法错误 line {e.lineno}: {e.msg}")
    raise SystemExit(1)

p.write_text(src, encoding="utf-8")
print("✓ 已补回 _IDF_MAX_BM25 / _IDF_MAX_RERANK")