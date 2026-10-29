"""按三轮 A/B 的结论回退：保留结构改进，回退效果不佳的打分逻辑。

决策依据（47 条实测，三轮 A/B，详见 docs/rerank-ab-log.md）：
    轮次  改动                        precision   recall      判定
    A     idf≤8 + idf² + 折扣         +0.021    -0.063      退化
    B     + IDF加权判据 + 剔除实词+0.017    -0.063      无变化
    C     拆分 idf 上限（BM25 3.0）+0.029    -0.085      退化加剧

保留：
  - _IDF_MAX 拆成 _IDF_MAX_BM25 / _IDF_MAX_RERANK
    → 结构上正确。拆分后 BM25 回到 3.0 证实「共享常量会破坏变量隔离」
    这个教训本身有价值，且今后调参不会再"改一处动两处"。
  - _QUERY_CUES 剔除「包含」「包括」「属于」
    → 那是实词不是虚词，真实 idf 分别 4.538 / 2.569 / 3.479。
    保留这个修正是纯收益（不改打分行为，只修正认知错误的常量）。

回退：
  - idf² 超线性加成 → 回到线性 idf
    → 机制上它确实消除了 bool 饱和，但代价是把跨文档类问题的
      recall 压下去了（5 条退化样本 top5 重叠仅 1~3/5）。
      跨文档类占样本 45%，代价 > 收益。
  - 疑问词折扣 → 整个移除
    → B 方案证明它对结果零影响，而复杂度是实打实的。
      「零影响 + 高复杂度」的代码应当删掉。

回退后 _IDF_MAX_RERANK 设为 3.0（与 BM25 一致，等价于旧行为）。
留着 8.0 会继续压跨文档 recall。
"""
import pathlib

p = pathlib.Path("agent1/hybrid.py")
src = p.read_text(encoding="utf-8")

# ---- 1. _IDF_MAX_RERANK 回到 3.0 ----
old_const = "    _IDF_MAX_RERANK = 8.0   # 精排覆盖率：放宽到 log(4019)=8.30附近"
new_const = """    # ★ 三轮 A/B 实测后回退到 3.0（详见 docs/rerank-ab-log.md）：
    #   放开到 8.0 + idf² 确实消除了 bool 饱和，
    #   但把「跨文档综合」类的recall 压下去了（-0.063 ~ -0.085，
    #   唯一超出噪声底的指标），而这类样本占 45%。
    #   代价 > 收益，回退。
    #
    #   保留拆分的意义：今后调精排不会再连带影响 BM25 召回。
    _IDF_MAX_RERANK = 3.0"""
if old_const in src:
    src = src.replace(old_const, new_const, 1)
    print("✓ _IDF_MAX_RERANK 回退到 3.0")
else:
    print("! 未找到 _IDF_MAX_RERANK 定义")

# ---- 2. 覆盖率回到线性 idf ----
old_cov = '''            if token_set:
                got = sum(w * w for t, w in weights.items() if t in text)
                cov = got / total_w
                if cue_ratio > 0:
                    cov *= (1.0 - self._CUE_DISCOUNT * cue_ratio)
                score += self.W_OVERLAP * cov'''

new_cov = '''            # 词面覆盖率（IDF 加权，线性）
            #
            # ★ 这里试过 idf² 超线性加成，三轮 A/B 后回退（见 docs/rerank-ab-log.md）：
            #
            #   动机：原来的 `if t in text` 是 bool 命中，
            #   命中 1 次和5 次同权 → 10~32 个候选并列满分，
            #   BM25 把 P4 标准条目排第1 的判断被抹平。
            #   idf² 能让「稀有词命中一次」压过「常见词命中多次」。
            #
            #   实测后果（47 条）：
            #       直查法条     n=20   recall +0.050   ← 单文档精确查找，确实受益
            #       跨文档综合    n=21   recall -0.143   ← 答案横跨多份文件受损
            #       陷阱题       n= 6   recall -0.167
            #   5 条退化样本的 top5 重叠仅 1~3/5 —— 答案片段被压出 top5。
            #
            #   原因：跨文档类答案分布在多份文件里，
            #   **每个片段都只命中 1 个查询词**，
            #   而单词命中在 idf² 下权重极高，压过了命中 2-3 词的片段。
            #   idf² 优化的是"找准那一条"，代价是"找齐那几条"——
            #   跨文档类占样本 45%，代价超过收益。
            #
            #   保留 bool→线性加权这一段的原因：
            #   IDF 加权本身是对的（答案词"电线电缆"df=12 vs
            #   噪声词"消防"df≈2000，本就该有区别）。
            #   错的是"超线性"这个选择，不是"加权"本身。
            if token_set:
                got = sum(w for t, w in weights.items() if t in text)
                score += self.W_OVERLAP * (got / total_w)'''
if old_cov in src:
    src = src.replace(old_cov, new_cov, 1)
    print("✓ 覆盖率回退到线性 idf")
else:
    print("! 未找到覆盖率块")

# ---- 3. total_w 回到线性 ----
src = src.replace(
    "            total_w = sum(w * w for w in weights.values()) or 1.0",
    "            total_w = sum(weights.values()) or 1.0",
    1,
)

# ---- 4. 移除 cue_ratio 相关代码 ----
import re
cue_block = re.search(
    r"\n        # 查询里疑问句/连接词的 \*\*IDF 加权\*\*占比。\n"
    r".*?\n        cue_ratio = 0\.0\n"
    r"        if token_set:\n"
    r".*?\n"
    r"            cue_ratio = min\(1\.0, cue_w / total_w\) if total_w else 0\.0\n",
    src, re.S)
if cue_block:
    src = src[:cue_block.start()] + "\n" + src[cue_block.end():]
    print("✓ 已移除疑问词折扣")
else:
    print("! 未找到 cue_ratio 块")

# ---- 5. 清理不再使用的常量 ----
for dead in ("_CUE_DISCOUNT", "_QUERY_CUES"):
    m = re.search(rf"\n(?:    # [^\n]*\n)*    {dead} = .*?\n", src)
    if m:
        src = src[:m.start()] + "\n" + src[m.end():]
        print(f"✓ 已移除常量 {dead}")

p.write_text(src, encoding="utf-8")

import ast
try:
    ast.parse(src)
    print("\n✓ 语法正常")
except SyntaxError as e:
    print(f"\n✗ 语法错误 line {e.lineno}: {e.msg}")