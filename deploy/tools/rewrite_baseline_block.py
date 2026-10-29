"""按 tests/test_eval_baseline.py 的确切契约重写「基线对比」区块。

★ 为什么要按测试契约重写，而不是让测试迁就我的实现：
  我第一版重建把它拍平成一层（{"comparable": ..., "null_run": ...}），
  而原实现是三层结构：
      {
        "error": str,                       # 顶层错误
        "checks": {
          "comparable": {"passed": bool, "baseline_num_samples": int, ...},
          "null_run":   {"detected": bool, "passed": bool, "note": str, ...},
          "noise":      {...},                # 每个指标 {floor, delta, significant}
          "overall":    {...},
        }
      }

  这是明显的"原实现更完整"的信号：
    - comparable 里带baseline_num_samples / current_top_k 等诊断字段
    - null_run 分 detected 与 passed 两个概念
      （上下文一致但分数一致时，detected=False 且 passed=True）
    - 不可比时**根本不返回 null_run 检查**
    - 噪声判定是逐指标的

  测试比实现更接近"需求文档"。让实现去满足测试，
  比改30 个断言去迁就一个临时实现划算得多。

★ 顺带从测试注释里学到一条 v8 的历史误读：
  "实测 v8 的 cp 从 0.520 掉到 0.325，完全是 top_k 5→8 造成的，
   不是检索退化。"
  —— 这也是为什么 comparable 必须检查 top_k。
"""
import ast
import pathlib
import re

p = pathlib.Path("agent1/eval/local_eval.py")
src = p.read_text(encoding="utf-8")

START = "# ==================== 基线对比 ===================="
END = "# ==================== 主流程 ===================="

i = src.find(START)
j = src.find(END)
if i < 0 or j < 0 or j <= i:
    print("✗ 找不到区块边界")
    raise SystemExit(1)

BLOCK = '''# ==================== 基线对比 ====================
#
# ★ 这套机制的全部理由（2026-10-15 实测踩坑）：
#   v6/v7/v8 三轮跑完，总分"有涨有跌"，很自然会解读成调优时灵时不灵。
#   但逐条比对 contexts 发现三轮 47 条上下文**逐字完全相同**——
#   索引没动过，v8 是次空跑。而即便变量隔离了，LLM-as-judge 本身也会抖：
#   同输入两次跑，context_recall 能差 0.021。
#
#   v8 还有另一种误读：cp 从 0.520 掉到 0.325，
#   看起来像检索退化，实际是 **top_k 从 5 改成 8**，
#   precision 的分母变了。所以可比性必须查 top_k。
#
#   因此必须用机制锁住三件事，缺一不可：
#     Q1 可比吗？      样本数 / top_k 是否一致
#     Q2 是空跑吗？    contexts 指纹一致却分数变了 → 差异 100% 来自 LLM 随机性
#     Q3 超噪声吗？    逐指标差值是否大于该指标的抖动幅度

# 抖动幅度未知时的保守阈值。
#
# ★ 为什么不能写死一个数：
#   0.021 是在特定数据集 + 特定 judge 上测的，
#   换数据集后这个数就不一定成立。
#   而"不知道抖动多大"时仍套用 0.05，
#   等于把未经验证的数当常量用——换数据集就会误判显著性。
#   所以基线里带测量值就用它，否则用这个保守值。
_NOISE_FLOOR_UNKNOWN = 0.10

_METRIC_KEYS = ("faithfulness", "answer_relevancy",
                "context_precision", "context_recall")


def _contexts_fingerprint(details: list) -> str:
    """上下文指纹：只哈希 question + contexts。

    ★ 绝不能包含答案与分数。
      含了的话分数一变指纹就变，空跑检测立刻失效——
      而空跑检测是这三项机制里最有用的一项。

    实测：v7/v8/v9 三份报告指纹完全相同（cb941a0116de7cef），
    正是空跑的铁证。
    """
    h = hashlib.sha256()
    for d in details:
        h.update((d.get("question") or "").encode("utf-8"))
        for c in (d.get("contexts") or []):
            h.update(c.encode("utf-8"))
        h.update(b"\\x00")
    return h.hexdigest()[:16]


def _fingerprint_of_report(rep: dict, details: list) -> str:
    """优先用报告里存的指纹；没有就算一遍。

    ★ 不能只信报告里的字段：万一是旧版本生成的报告（没有该字段）
      或字段被手改过——那就该自己算，不该盲信。
    """
    fp = rep.get("_contexts_fingerprint")
    if isinstance(fp, str) and fp:
        return fp
    return _contexts_fingerprint(details)


def _read_scores(rep: dict) -> dict:
    """取四指标均值。

    ★ 报告的真实结构：
        scores.<metric>       均值在这里
        scores._per_sample[]  逐条分数
        details[]             只有问答与检索结果，不含指标

    读错位置不会崩，只会让所有差值变成 0，
    于是"未超出噪声底"看起来像正常结论——比直接崩溃危险得多。
    """
    sc = rep.get("scores") or {}
    out = {}
    for k in _METRIC_KEYS:
        v = sc.get(k)
        out[k] = float(v) if isinstance(v, (int, float)) else None
    return out


def _noise_floor_for(base: dict, key: str) -> float:
    """取基线里该指标的噪声测量值；没有则用保守值。"""
    src = base.get("_noise_floor")
    if isinstance(src, dict) and isinstance(src.get(key), (int, float)):
        return float(src[key])
    sc = base.get("scores") or {}
    src = sc.get("_noise_floor")
    if isinstance(src, dict) and isinstance(src.get(key), (int, float)):
        return float(src[key])
    return _NOISE_FLOOR_UNKNOWN


def compare_with_baseline(cur_report: dict, baseline_path: str) -> dict:
    """与基线报告对比。

    :return: {"error": str|None,
              "checks": {"comparable": {...}, ["null_run": {...}],
                         "noise": {...}, ["overall": {...}]}}
             不可比时不返回 null_run —— 因为"空不空跑"这个问题
             在样本数/top_k 不一致时没有意义。
    """
    result: dict = {"error": None, "checks": {}}

    bp = pathlib.Path(baseline_path)
    if not bp.is_file():
        result["error"] = f"基线文件不存在：{baseline_path}"
        return result
    try:
        base = json.loads(bp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        result["error"] = f"基线读取失败：{e}"
        return result

    cur_details = cur_report.get("details") or []
    base_details = base.get("details") or []
    if not cur_details or not base_details:
        result["error"] = "报告缺少 details，无法对比"
        return result

    # ---------- Q1 可比性 ----------
    cur_n = cur_report.get("num_samples") or len(cur_details)
    base_n = base.get("num_samples") or len(base_details)
    cur_k = cur_report.get("top_k")
    base_k = base.get("top_k")
    k_differs = (cur_k is not None and base_k is not None and cur_k != base_k)

    comparable = {
        "passed": (cur_n == base_n) and not k_differs,
        "baseline_num_samples": base_n,
        "current_num_samples": cur_n,
        "baseline_top_k": base_k,
        "current_top_k": cur_k,
        "note": "",
    }
    if cur_n != base_n:
        comparable["note"] = (
            f"样本数不同（基线 {base_n} vs 本轮 {cur_n}），分数不可比")
    elif k_differs:
        # v8 踩过的坑：top_k 变化会改变 precision 的分母
        comparable["note"] = (
            f"top_k 不同（基线 {base_k} vs 本轮 {cur_k}），"
            f"precision 分母变了，分数不可比（v8 的 cp 0.520→0.325 "
            f"就是纯 top_k 5→8 造成的，不是检索退化）")
    result["checks"]["comparable"] = comparable

    if not comparable["passed"]:
        return result

    # ---------- Q2 空跑检测 ----------
    fp_cur = _fingerprint_of_report(cur_report, cur_details)
    fp_base = _fingerprint_of_report(base, base_details)
    same_fp = (fp_cur == fp_base)

    cur_sc = _read_scores(cur_report)
    base_sc = _read_scores(base)
    deltas = {}
    score_changed = False
    for k in _METRIC_KEYS:
        a, b = cur_sc.get(k), base_sc.get(k)
        if a is None or b is None:
            continue
        deltas[k] = a - b
        if abs(a - b) > 1e-9:
            score_changed = True

    result["checks"]["variable_isolation"] = {
        "passed": not same_fp,
        "current_fingerprint": fp_cur,
        "baseline_fingerprint": fp_base,
        "contexts_identical": same_fp,
    }

    #上下文一致且分数也变-> 空跑。
    # 上下文一致但分数也一致 -> 可复现，不算空跑。
    if same_fp and score_changed:
        result["checks"]["null_run"] = {
            "detected": True,
            "passed": False,
            "note": ("上下文指纹完全一致，检索层零变化，"
                     "但分数变了 → 差异 100% 来自 LLM 判分随机性，"
                     "**不是策略改进**。本轮数据不能证明任何优化有效。"),
            "deltas": deltas,
        }
    else:
        result["checks"]["null_run"] = {
            "detected": False,
            "passed": True,
            "note": ("上下文与分数均未变，可复现"
                     if same_fp else
                     "检索层已变化，差异可归因到检索/生成策略"),
            "deltas": deltas,
        }

    # ---------- Q3 逐指标噪声判定 ----------
    noise = {}
    significant = []
    for k in _METRIC_KEYS:
        if k not in deltas:
            continue
        floor = _noise_floor_for(base, k)
        sig = abs(deltas[k]) > floor
        if sig:
            significant.append(k)
        noise[k] = {
            "delta": deltas[k],
            "noise_floor": floor,
            "significant": sig,
        }
    result["checks"]["noise"] = noise

    # ---------- 汇总 ----------
    if not same_fp:
        result["checks"]["overall"] = {
            "passed": True,
            "verdict": (
                "检索层已变化，差异可归因"
                + (f"；{len(significant)}/{len(noise)} 项超出噪声底"
                   if noise else "")),
            "significant_metrics": significant,
        }

    return result


def print_baseline_report(result: dict) -> None:
    """把 verdict 打成人类可读的表格。"""
    if result.get("error"):
        print(f"\\n[基线对比] 失败：{result['error']}")
        return

    checks = result.get("checks") or {}
    cmp_ = checks.get("comparable") or {}

    print("\\n" + "=" * 66)
    print("基线对比")
    print("=" * 66)

    print(f"  样本数 基线={cmp_.get('baseline_num_samples')} "
          f"本轮={cmp_.get('current_num_samples')}   "
          f"top_k 基线={cmp_.get('baseline_top_k')} "
          f"本轮={cmp_.get('current_top_k')}")

    if not cmp_.get("passed"):
        print(f"  [不可比] {cmp_.get('note')}")
        return

    vi = checks.get("variable_isolation") or {}
    print(f"\\n  指纹  本轮={vi.get('current_fingerprint')}  "
          f"基线={vi.get('baseline_fingerprint')}")

    nr = checks.get("null_run") or {}
    print(f"\\n  {'指标':22s} {'差值':>9s} {'噪声底':>8s}")
    print("  " + "-" * 44)
    for k in _METRIC_KEYS:
        n = (checks.get("noise") or {}).get(k)
        if not n:
            continue
        mark = "*" if n["significant"] else " "
        print(f"  {k:22s} {n['delta']:+9.4f} "
              f"{n['noise_floor']:8.3f}{mark}")

    print()
    if nr.get("detected"):
        print("  [空跑] " + nr.get("note", ""))
    else:
        sig = (checks.get("overall") or {}).get("significant_metrics") or []
        if sig:
            print(f"  [可作结论] 超出噪声底：{', '.join(sig)}")
        else:
            print("  [不可作结论] 全部指标均在噪声底内，"
                  "不要写成'提升了'")


'''

src = src[:i] + BLOCK + src[j:]

try:
    ast.parse(src)
except SyntaxError as e:
    print(f"✗ 语法错误 line {e.lineno}: {e.msg}")
    raise SystemExit(1)

p.write_text(src, encoding="utf-8")
print("✓ 已按测试契约重写基线对比区块")