"""对比两份评测报告，输出空跑检测与噪声判定结论。

用途：
  1. 独立对比：python baseline_diff.py old.json new.json
     —— 不需要跑评测就能比对已有报告
  2. 接入 local_eval.py：python baseline_diff.py install

★ 两个独立模式踩过的坑：
  - 直接 import local_eval 会因它的平铺import（from ragas_eval import ...）
    找不到模块，必须先把 agent1/eval 加进 sys.path。
  - 相对路径在 cwd 变化后解析错，所以所有路径都按本文件位置推导绝对路径。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_EVAL_DIR = _HERE                 # agent1/eval
_AGENT1 = _EVAL_DIR.parent         # agent1
_ROOT = _AGENT1.parent             # 项目根

# 噪声底：实测 LLM 判分抖动约 0.021，取 2.4 倍余量。
#
# ★ 不用实测值的原因：单次 A/B 差值只是抖动幅度的**下界**，不是幅度本身。
#   阈值取 0.021 时，恰好等于 0.021 的差值会被误判为"超出噪声"。
_NOISE_FLOOR = 0.05

_METRIC_KEYS = ("faithfulness", "answer_relevancy",
                "context_precision", "context_recall")


def contexts_fingerprint(details: list) -> str:
    """上下文指纹：只看检索结果，不看答案和分数。

    ★ 指纹绝不能含答案或分数。含了的话，
      分数一变指纹就变，空跑检测立刻失效 ——
      而空跑检测是这套机制里最有用的一项。
    """
    h = hashlib.sha256()
    for d in details:
        h.update((d.get("question") or "").encode("utf-8"))
        for c in (d.get("contexts") or []):
            h.update(c.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _pick_scores(report: dict) -> dict:
    """从报告里取四个指标均值。

    ★ 报告的真实结构（这里读错过）：
        scores.<metric>        均值在这里
        scores._per_sample[]   逐条分数
        details[]              只有问答与检索结果，不含指标

      读错位置不会报错，只会让所有差值变成 0，
      于是"未超出噪声底"看起来像正常结论 —— 比直接崩溃危险得多。
    """
    sc = report.get("scores") or {}
    return {k: (float(sc[k]) if isinstance(sc.get(k), (int, float)) else None)
            for k in _METRIC_KEYS}


def _resolve(p: str) -> pathlib.Path:
    """把用户给的路径解析成绝对路径。

    依次尝试：原样 -> 相对项目根 -> 相对本目录 -> 相对 agent1。
    这样不管从哪个 cwd 调用都能找到文件。
    """
    cand = pathlib.Path(p)
    if cand.is_absolute() and cand.is_file():
        return cand
    for base in (pathlib.Path.cwd(), _ROOT, _EVAL_DIR, _AGENT1):
        q = (base / p)
        if q.is_file():
            return q
        # 也试试 p 已经带了 ../ 前缀的情况
        q2 = (base / p).resolve()
        if q2.is_file():
            return q2
    return cand     # 找不到就原样返回，让读文件时报出真实路径


def compare(new_path: str, old_path: str,
            noise_floor: float = _NOISE_FLOOR) -> int:
    nb = _resolve(new_path)
    ob = _resolve(old_path)

    if not ob.is_file():
        print(f"[FAIL] 基线文件不存在：{ob}")
        return 1
    if not nb.is_file():
        print(f"[FAIL] 本轮文件不存在：{nb}")
        return 1

    try:
        base = json.loads(ob.read_text(encoding="utf-8"))
        cur = json.loads(nb.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[FAIL] 报告读取失败：{e}")
        return 1

    base_details = base.get("details") or []
    cur_details = cur.get("details") or []
    if not base_details or not cur_details:
        print("[FAIL] 报告里没有 details")
        return 1

    print("=" * 66)
    print("基线对比")
    print("=" * 66)
    print(f"  基线: {ob}")
    print(f"  本轮: {nb}")
    print(f"  条数: 基线 {len(base_details)} / 本轮 {len(cur_details)}")

    fp_cur = contexts_fingerprint(cur_details)
    fp_base = contexts_fingerprint(base_details)
    same_fp = (fp_cur == fp_base)

    print(f"\n  上下文指纹  本轮={fp_cur}  基线={fp_base}")
    print("    -> 检索层完全未变（空跑）" if same_fp
          else "    -> 检索层已变化（不是空跑）")

    base_sc = _pick_scores(base)
    cur_sc = _pick_scores(cur)

    print(f"\n  {'指标':22s} {'基线':>8s} {'本轮':>8s} {'差值':>9s}")
    print("  " + "-" * 52)
    deltas: dict[str, float] = {}
    for m in _METRIC_KEYS:
        a, b = base_sc.get(m), cur_sc.get(m)
        if a is None or b is None:
            miss = "基线" if a is None else "本轮"
            print(f"  {m:22s} {'?':>8s} {b if b is not None else '?':>8s}"
                  f"   ({miss}缺失)")
            continue
        d = b - a
        deltas[m] = d
        print(f"  {m:22s} {a:8.4f} {b:8.4f} {d:+9.4f}"
              f"{'*' if abs(d) > noise_floor else ' '}")

    if not deltas:
        print("\n  [FAIL] 两份报告都取不到指标")
        return 1

    over = [m for m, d in deltas.items() if abs(d) > noise_floor]
    under = [m for m, d in deltas.items() if abs(d) <= noise_floor]

    print()
    if same_fp:
        print(f"  [空跑] 检索层零变化，但 {len(over)}/{len(_METRIC_KEYS)} "
              f"项超出噪声底")
        if over:
            print("     这些差异 100% 来自 LLM 判分随机性，不是策略改进")
        print("     结论：本轮数据不能证明任何优化有效")
        print()
        print("     要得到有效结论，必须先让检索层真的变化：")
        print("       - 改分块规则 -> 重建索引")
        print("       - 改精排逻辑 -> 确认进程用的是新代码")
        print("       - 换embedding 模型 -> 必须重建索引（向量不兼容）")
        return 1

    print(f"  检索层已变化 -> 差异可归因到检索/生成策略（不是空跑）\n")
    if over:
        print(f"  [可作结论] 超出噪声底（+-{noise_floor}）：")
        for m in over:
            print(f"       {m:22s} {deltas[m]:+.4f}")
    if under:
        print(f"  [不可作结论] 未超出噪声底：")
        for m in under:
            print(f"       {m:22s} {deltas[m]:+.4f}")
        print("     要扩大样本量或重复多轮才能判断，不要写成'提升了'")

    blank = [d for d in cur_details if not (d.get("answer") or "").strip()]
    if blank:
        pct = len(blank) / len(cur_details) * 100
        print(f"\n  [故障] {len(blank)}/{len(cur_details)} 条"
              f"（{pct:.0f}%）未生成答案")
        if pct > 5:
            print("     占比过高，本轮全部指标不可与其他轮对比")
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("new", nargs="?", help="本轮报告")
    ap.add_argument("old", nargs="?", help="基线报告")
    ap.add_argument("--noise-floor", type=float, default=_NOISE_FLOOR)
    args = ap.parse_args()

    if args.new and args.old:
        return compare(args.new, args.old, args.noise_floor)

    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())