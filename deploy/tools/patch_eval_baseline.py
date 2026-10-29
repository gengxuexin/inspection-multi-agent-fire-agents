"""把「噪声底 + 上下文指纹 + 基线对比」插入 local_eval.py。

用源码切片插入而不是整文件重写：
local_eval.py 有 570 行，重写风险远大于精确插入。
"""
import pathlib

p = pathlib.Path("agent1/eval/local_eval.py")
s = p.read_text(encoding="utf-8")

if "def contexts_fingerprint" in s:
    print("· 已含 contexts_fingerprint，跳过")
    raise SystemExit(0)

BLOCK = '''# ---------------------------------------------------------------- 噪声底
# 实测 LLM 判分抖动约 0.021（v7/v8 空跑轮的 recall 差值）。
# 取 0.05 ≈ 实测 2.4 倍余量。
#
# ★ 为什么不直接用实测值：
#   单次 A/B 差值只是抖动幅度的**下界**，不是幅度本身。
#   阈值取 0.021 时，恰好等于 0.021 的差值会被判成"超出噪声"，
#   而它其实没超 —— 曾因此让测试失败过一次。
_NOISE_FLOOR = 0.05

_METRIC_KEYS = ("faithfulness", "answer_relevancy",
                "context_precision", "context_recall")


def contexts_fingerprint(details: list) -> str:
    """上下文指纹：只看检索结果，不看答案和分数。

    ★ 指纹**绝不能包含答案或分数**。
      那样分数一变指纹就变，空跑检测立刻失效 ——
      而空跑检测恰恰是这套机制里最有用的一项。
    """
    h = hashlib.sha256()
    for d in details:
        h.update((d.get("question") or "").encode("utf-8"))
        for c in (d.get("contexts") or []):
            h.update(c.encode("utf-8"))
        h.update(b"\\x00")
    return h.hexdigest()[:16]


def _avg_metric(details: list, key: str) -> float:
    vals = [d.get(key) for d in details
            if isinstance(d.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def compare_with_baseline(details: list, baseline_path: str,
                          noise_floor: float = _NOISE_FLOOR) -> int:
    """与基线报告对比。返回 0=可作结论，1=不可作结论。"""
    bp = pathlib.Path(baseline_path)
    if not bp.is_file():
        print(f"\\n[FAIL] 基线文件不存在：{baseline_path}")
        return 1
    try:
        base = json.loads(bp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"\\n[FAIL] 基线文件读取失败：{e}")
        return 1

    base_details = base.get("details") or []
    if not base_details:
        print("\\n[FAIL] 基线报告里没有 details，无法对比")
        return 1

    print("\\n" + "=" * 66)
    print("基线对比")
    print("=" * 66)
    print(f"  基线: {baseline_path}")

    fp_new = contexts_fingerprint(details)
    fp_old = contexts_fingerprint(base_details)
    same_fp = (fp_new == fp_old)

    print(f"\\n  上下文指纹  本轮={fp_new}  基线={fp_old}")
    if same_fp:
        print("    -> 检索层完全未变（空跑）")
    else:
        print("    -> 检索层已变化")

    print(f"\\n  {'指标':22s} {'基线':>8s} {'本轮':>8s} {'差值':>9s}")
    print("  " + "-" * 52)
    deltas = {}
    for m in _METRIC_KEYS:
        a = _avg_metric(base_details, m)
        b = _avg_metric(details, m)
        d = b - a
        deltas[m] = d
        mark = "*" if abs(d) > noise_floor else " "
        print(f"  {m:22s} {a:8.4f} {b:8.4f} {d:+9.4f}{mark}")

    over = [m for m, d in deltas.items() if abs(d) > noise_floor]
    under = [m for m, d in deltas.items() if abs(d) <= noise_floor]

    print()
    if same_fp:
        print(f"  [空跑] 检索层零变化，但 {len(over)}/{len(_METRIC_KEYS)} "
              f"项超出噪声底")
        if over:
            print("     这些差异 100% 来自 LLM 判分随机性，不是策略改进")
        print("     结论：本轮数据不能证明任何优化有效")
        return 1

    print(f"  检索层已变化 -> 差异可归因到检索/生成策略（不是空跑）\\n")
    if over:
        print(f"  [可作结论] 超出噪声底（+-{noise_floor}）：")
        for m in over:
            print(f"       {m:22s} {deltas[m]:+.4f}")
    if under:
        print(f"  [不可作结论] 未超出噪声底：")
        for m in under:
            print(f"       {m:22s} {deltas[m]:+.4f}")
        print("     要扩大样本量或重复多轮才能判断，不要写成'提升了'")

    blank = [d for d in details if not (d.get("answer") or "").strip()]
    if blank:
        pct = len(blank) / max(len(details), 1) * 100
        print(f"\\n  [故障] {len(blank)}/{len(details)} 条（{pct:.0f}%）未生成答案")
        if pct > 5:
            print("     占比过高，本轮全部指标不可与其他轮对比")
            return 1
    return 0


'''

anchor = "def main():"
if anchor not in s:
    print("✗ 找不到 def main() 锚点")
    raise SystemExit(1)

s = s.replace(anchor, BLOCK + anchor, 1)

if "import hashlib" not in s:
    s = s.replace("import argparse", "import argparse\nimport hashlib", 1)

p.write_text(s, encoding="utf-8")
print("✓ 已插入 _NOISE_FLOOR / contexts_fingerprint / compare_with_baseline")
print("✓ 已确保 hashlib 导入")