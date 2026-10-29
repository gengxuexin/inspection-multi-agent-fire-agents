"""修正 compare_with_baseline：指标在 scores 里，不在 details 里。

★ 踩坑记录：
  第一版按「details 每条含四个指标」来读，结果四项全是 0.0000 ——
  因为报告的真实结构是：
      scores.faithfulness / answer_relevancy / context_precision / context_recall
      scores._per_sample[]逐条分数
      details[]              只有 question/answer/contexts/ground_truth/...

  读错位置的危害比报错更隐蔽：它不会崩，只会让所有差值变成 0，
  于是「未超出噪声底」看起来像正常结论。
  —— 和当初 v7/v8 空跑被误读成"调优有效果"是同一类错误：
  **工具没有失败，是读数的人（或代码）读错了地方。**
"""
import pathlib

p = pathlib.Path("agent1/eval/local_eval.py")
s = p.read_text(encoding="utf-8")

old = '''def _avg_metric(details: list, key: str) -> float:
    vals = [d.get(key) for d in details
            if isinstance(d.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0'''

new = '''def _pick_scores(report: dict) -> dict:
    """从报告里取四个指标的均值。

    ★ 报告的真实结构（第一版读错过这里）：
        scores.faithfulness / answer_relevancy /
                context_precision / context_recall   <- 均值在这里
        scores._per_sample[]                       <- 逐条分数
        details[]                                   <- 只有问答与检索结果

    读错位置不会报错，只会让所有差值变成 0，
    于是"未超出噪声底"看起来像正常结论 —— 这比直接崩溃危险得多。
    """
    sc = report.get("scores") or {}
    out = {}
    for k in _METRIC_KEYS:
        v = sc.get(k)
        out[k] = float(v) if isinstance(v, (int, float)) else None
    return out


def _avg_metric(details: list, key: str) -> float:
    """兜底：若 scores 缺失，从 _per_sample 求均值。"""
    vals = [d.get(key) for d in details
            if isinstance(d.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0'''

if old in s:
    s = s.replace(old, new, 1)
    print("✓ 已加 _pick_scores")
else:
    print("! 未找到 _avg_metric 锚点")

# 改 compare_with_baseline 的签名与读数逻辑
s = s.replace(
    "def compare_with_baseline(details: list, baseline_path: str,\n"
    "                          noise_floor: float = _NOISE_FLOOR) -> int:",
    "def compare_with_baseline(details: list, baseline_path: str,\n"
    "                          cur_scores: dict | None = None,\n"
    "                          noise_floor: float = _NOISE_FLOOR) -> int:",
    1,
)

s = s.replace(
    """    deltas = {}
    for m in _METRIC_KEYS:
        a = _avg_metric(base_details, m)
        b = _avg_metric(details, m)
        d = b - a
        deltas[m] = d
        mark = "*" if abs(d) > noise_floor else " "
        print(f"  {m:22s} {a:8.4f} {b:8.4f} {d:+9.4f}{mark}")""",
    """    # ★ 从 scores 读，不是从 details 读 —— 见 _pick_scores 的注释
    base_scores = _pick_scores(base)
    new_scores = (cur_scores if cur_scores is not None
                  else _pick_scores({"scores": {}}))

    deltas = {}
    for m in _METRIC_KEYS:
        a = base_scores.get(m)
        b = new_scores.get(m)
        if a is None or b is None:
            print(f"  {m:22s} {'(基线缺失)' if a is None else '(本轮缺失)'}")
            deltas[m] = None
            continue
        d = b - a
        deltas[m] = d
        mark = "*" if abs(d) > noise_floor else " "
        print(f"  {m:22s} {a:8.4f} {b:8.4f} {d:+9.4f}{mark}")

    usable = {m: v for m, v in deltas.items() if v is not None}
    if not usable:
        print("\\n  [FAIL] 两份报告都取不到指标，无法对比")
        print("         请确认报告是 local_eval.py 生成的（含 scores 段）")
        return 1""",
    1,
)

# 后面的 over/under 要过滤 None
s = s.replace(
    "    over = [m for m, d in deltas.items() if abs(d) > noise_floor]\n"
    "    under = [m for m, d in deltas.items() if abs(d) <= noise_floor]",
    "    over = [m for m, d in usable.items() if abs(d) > noise_floor]\n"
    "    under = [m for m, d in usable.items() if abs(d) <= noise_floor]",
    1,
)

# 调用点补 cur_scores
s = s.replace(
    "        rc = compare_with_baseline(\n"
    "            eval_data, args.baseline,\n"
    "            noise_floor=args.noise_floor)",
    "        rc = compare_with_baseline(\n"
    "            eval_data, args.baseline,\n"
    "            cur_scores=report.get(\"scores\"),\n"
    "            noise_floor=args.noise_floor)",
    1,
)

p.write_text(s, encoding="utf-8")
print("✓ 已修正读数位置与调用签名")