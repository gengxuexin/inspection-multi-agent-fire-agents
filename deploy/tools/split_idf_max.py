"""把 _IDF_MAX 拆成「BM25 用」与「精排用」两个独立参数。

★ 为什么必须拆（这是本轮 A/B 最重要的一条发现）：
  `_IDF_MAX` 原先是 HybridRetriever 的类属性，
  而 `self._idf` 在 __init__ 里构建 BM25 时就定型了，
  精排（第 461行 `weights = {t: self._idf.get(t, ...)}`）复用的是**同一个字典**。

  所以我改 `_IDF_MAX: 3.0 -> 8.0` 时，
  **同时改了 BM25 召回与精排排序两处**。

  实测证据（47 条 A/B）：
      top5 完全相同   8/47  (17%)
      部分相同      26/47  (55%)
      集合大变      13/47  (28%)
  只有 17% 的样本候选集没变 —— 召回阶段早就动了。

  而我A/B 脚本里宣称的"只切精排这一个变量"是**错的**。
  这会导致误判责任：recall 掉 0.063，我第一反应去查精排覆盖率，
  改了 B 方案（疑问词折扣），结果分层数字逐位不变——
  因为瓶颈根本不在精排。

  教训：**"改一个常量"不等于"改一个变量"。**
  要先确认那个常量的作用域到底覆盖几处。
  共享的可配置项在调参时是最大的陷阱。

拆完之后：
  _IDF_MAX_BM25    BM25 召回用（保持 3.0不动，避免扰动召回）
  _IDF_MAX_RERANK  精排覆盖率用（8.0）
这样才能真正做到"只改精排"。
"""
import pathlib

p = pathlib.Path("agent1/hybrid.py")
src = p.read_text(encoding="utf-8")

# ---- 1. 常量拆分 ----
OLD_CONST = "    _IDF_MIN = 0.15\n    _IDF_MAX = 8.0"
NEW_CONST = """    _IDF_MIN = 0.15

    # ★ IDF 上限拆成两个，因为 BM25 与精排共用同一份 idf 字典。
    #
    #   原先只有 _IDF_MAX 一个，改它会**同时**改动
    #     - BM25 召回打分（__init__ 里构建 self._idf）
    #     - 精排覆盖率（_rerank 里 weights 直接读 self._idf）
    #
    #   实测教训（47 条 A/B）：改 _IDF_MAX 3.0->8.0 后
    #     top5 完全相同   8/47  (17%)
    #     集合大变       13/47  (28%)
    #   也就是说**召回阶段早就变了**，而我当时以为自己"只改了精排"。
    #   于是把时间花在了精排覆盖率上（改 B 方案），
    #   结果分层数字逐位不变 —— 瓶颈根本不在那里。
    #
    #   教训：**「改一个常量」不等于「改一个变量」。**
    #   共享的可配置项是调参时最大的陷阱，必须先确认它的作用域。
    #
    # 现在两者独立，才能做真正的变量隔离实验。
    _IDF_MAX_BM25 = 3.0     # BM25 召回：保持原值，不扰动已验证的召回
    _IDF_MAX_RERANK = 8.0   # 精排覆盖率：放宽到 log(4019)=8.30附近"""

if OLD_CONST not in src:
    print("✗ 未找到 _IDF_MAX 常量定义")
    raise SystemExit(1)
src = src.replace(OLD_CONST, NEW_CONST, 1)

# ---- 2. BM25 构建处用 _IDF_MAX_BM25 ----
src = src.replace(
    """        self._idf = {
            w: min(self._IDF_MAX,""",
    """        # BM25 与精排不再共用同一份 idf —— 见 _IDF_MAX_BM25 的注释
        self._idf = {
            w: min(self._IDF_MAX_BM25,""",
    1,
)

# ---- 3. 精排处单独构建一份 idf ----
OLD_W = '''        if token_set:
            default_idf = self._IDF_MIN
            weights = {t: self._idf.get(t, default_idf) for t in token_set}'''
NEW_W = '''        # 精排用自己的 idf（上限 _IDF_MAX_RERANK），不复用 BM25 那份。
        # 不复用是必须的：两者服务于不同目标 ——
        # BM25 要的是"词频统计上的区分度"，精排要的是"稀有词优先"，
        # 上限不同才能各自成立。
        if token_set:
            default_idf = self._IDF_MIN
            weights = {}
            for t in token_set:
                c = self._df.get(t, 0)
                raw = math.log(self._n_docs / (1.0 + c)) if self._n_docs else default_idf
                weights[t] = min(self._IDF_MAX_RERANK,
                                 max(self._IDF_MIN, raw))'''

if OLD_W not in src:
    print("✗ 未找到精排 weights 构建处")
    raise SystemExit(1)
src = src.replace(OLD_W, NEW_W, 1)

# ---- 4. 需要 _df 与 _n_docs ----
OLD_INIT = """        # BM25 与精排不再共用同一份 idf —— 见 _IDF_MAX_BM25 的注释
        self._idf = {
            w: min(self._IDF_MAX_BM25,"""
NEW_INIT = """        # 精排要用的原始 df 与文档数（自己保留一份，不从截断后的 idf 反推）
        self._df = df
        self._n_docs = n_docs

        # BM25 与精排不再共用同一份 idf —— 见 _IDF_MAX_BM25 的注释
        self._idf = {
            w: min(self._IDF_MAX_BM25,"""

src = src.replace(OLD_INIT, NEW_INIT, 1)

p.write_text(src, encoding="utf-8")
print("✓ _IDF_MAX 已拆分为 BM25(3.0) 与 RERANK(8.0)")
print("✓ 精排改为独立构建 idf")
print("✓ 已保存 _df 与 _n_docs 供精排计算原始 idf")