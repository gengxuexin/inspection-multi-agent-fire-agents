"""诊断 _IDF_MAX 截断是否让 IDF 失去了区分度。

背景：实测查「电线电缆类产品属于消防产品吗」时，5 个查询词的 idf
全部被截到 3.0，其中「电线电缆」只出现在 12 个片段（真正的答案词），
而「属于」出现在 123 个、「消防产品」108 个、「类产品」17 个（噪声词）。

如果截断把它们全压成同一个值，IDF 加权就完全失效了 ——
这比 tf 饱和更根本：饱和只是让候选之间打平，
截断则让「稀有词」和「常见词」直接等权。
"""
import math

N = 4019
IDF_MIN, IDF_MAX = 0.15, 3.0

rows = [
    ("电线电缆", 12),      # 答案词
    ("类产品", 17),       # 噪声
    ("消防产品", 108),     # 噪声
    ("属于", 123),        # 噪声（"属于"几乎不该进查询词）
    ("消防", 2000),       # 极高频
]

print(f"N = {N}, _IDF_MIN={IDF_MIN}, _IDF_MAX={IDF_MAX}")
print()
print(f"{'词':12s} {'df':>6s} {'raw_idf':>9s} {'截断后':>8s}  是否被截断")
print("-" * 56)

clamped_vals = []
for word, df in rows:
    raw = math.log(N / (1.0 + df))
    cl = min(IDF_MAX, max(IDF_MIN, raw))
    clamped_vals.append(cl)
    flag = "★是" if raw > IDF_MAX else ""
    print(f"  {word:10s} {df:6d} {raw:9.3f} {cl:8.3f}  {flag}")

print()
uniq = len(set(round(v, 4) for v in clamped_vals))
print(f"截断后不同取值数：{uniq} / {len(clamped_vals)}")
if uniq == 1:
    print("  ★ IDF 加权已完全失效——所有查询词等权，"
          "跟没加 IDF 一模一样")
else:
    print(f"  仍有 {uniq} 种取值，但最高/最低只差 "
          f"{max(clamped_vals) - min(clamped_vals):.3f}")
    if max(clamped_vals) - min(clamped_vals) < 0.5:
        print("  ★ 差距过小：稀有词与高频词的权重差异被截断抹平，")
        print("    IDF 实际起不到区分作用")

print()
print("=== 反事实：把 _IDF_MAX 提到 8会怎样 ===")
for word, df in rows:
    raw = math.log(N / (1.0 + df))
    cl = min(8.0, max(IDF_MIN, raw))
    print(f"  {word:10s} df={df:5d}  idf={cl:6.3f}")
print()
print("  结论：_IDF_MAX=3.0 对N=4019 来说太紧了。")
print("  log(4019) =", round(math.log(N), 3), "，而上限只有 3.0，")
print("  意味着 df < 约 200 的词全部压到同一个值。")
print(f"  df=200 时raw_idf = {math.log(N/201):.3f}")
print(f"  df=12  时 raw_idf = {math.log(N/13):.3f}  ← 应该比 df=200 高 {math.log(201/13):.2f} 倍")
print("  但两者都被截到 3.0 → 权重完全相同")