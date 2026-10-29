"""恢复被覆盖的「基线对比」区块（tests/test_eval_baseline.py 期望的接口）。

★ 事故经过（必须写下来）：
  2026-10-16 早上我检查 local_eval.py 时用 grep 找
  fingerprint / baseline / noise_floor，零命中，
  于是判断"这套能力完全不在了"，从零重写了一套。

  但实际上：**tests/test_eval_baseline.py（30 个测试）还在**，
  被测的代码区块是被我的 patch 脚本覆盖掉的。
  grep 只搜了 local_eval.py，没搜 tests/——
  **于是"实现消失"被误判成"从未存在"，而实际上是有 30 个测试在盯着它的。**

  更糟的是，重写版本比原实现退化。原实现有我完全没做的：
    - _NOISE_FLOOR_UNKNOWN：抖动幅度未知时的保守阈值
      （我写死单一 0.05）
    - 样本数 / top_k 不可比检查（我完全没做）
    - 逐指标抖动幅度（我只有一个全局值）
    - 真实空跑回归测试（直接用 v7/v8 报告断言）

  教训：**判断"某个能力是否被移除"时，
  必须连测试一起搜。**
  测试的存在是"这里本来有东西"的最强信号 ——
  30 个测试指向一个不存在的符号，比 grep 零命中更有说服力，
  而我当时只看了后者。

本脚本按测试期望的接口重建区块：
  - 分节标记：# ==================== 基线对比 ... 主流程
  - 符号：_contexts_fingerprint / compare_with_baseline /
          _NOISE_FLOOR_UNKNOWN
  - 行为：见 tests/test_eval_baseline.py 的 30 个断言
"""
import ast
import pathlib

p = pathlib.Path("agent1/eval/local_eval.py")
src = p.read_text(encoding="utf-8")

if "# ==================== 基线对比" in src:
    print("· 基线对比区块已存在，跳过")
    raise SystemExit(0)

BLOCK = '''# ==================== 基线对比 ====================
#
# ★ 这套机制的全部理由（2026-10-15 实测踩坑）：
#   v6/v7/v8 三轮跑完，总分"有涨有跌"，很自然会解读成调优时灵时不灵。
#   但逐条比对 contexts 发现三轮 47 条上下文**逐字完全相同**——
#   索引没动过，v8 是次空跑。而即便变量隔离了，LLM-as-judge 本身也会抖：
#   同输入两次跑，context_recall 能差 0.021。
#
#   所以必须用机制锁住三件事，缺一不可：
#     Q1 变量隔离了吗？  contexts 指纹是否与基线一致
#     Q2 是空跑吗？      上下文一致却分数变了 → 差异 100% 来自 LLM 随机性
#     Q3 变化超噪声吗？  差值是否大于该指标的抖动幅度
#
#   要锁的重点是**别再把空跑/噪声误读成优化效果**。

# 抖动幅度未知时的保守阈值。
#
# ★ 为什么要分「已知」与「未知」两种：
#   实测抖动约 0.021 是在特定数据集 + 特定 judge 上测的，
#   换数据集后这个数就不一定成立。
#   而"不知道抖动多大"时若仍套用 0.05，
#   等于把一个未经验证的数当常量用——
#   换个数据集就会得出错误的显著性判断。
#   所以基线里若带抖动测量值就用它，否则用这个保守值。
_NOISE_FLOOR_UNKNOWN = 0.10


def _contexts_fingerprint(details: list) -> str:
    """上下文指纹：只哈希 question + contexts。

    ★ 绝不能包含答案与分数。
      含了的话分数一变指纹就变，空跑检测立刻失效——
      而空跑检测是这三项机制里最有用的一项。

    实测：v7/v8/v9 三份报告的指纹完全相同
    （cb941a0116de7cef），正是空跑的铁证。
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

    ★ 不能只信报告里的字段：
      万一是旧版本生成的报告（还没有该字段），
      或者字段被手改过——那就该自己算，不该盲信。
    """
    fp = rep.get("_contexts_fingerprint")
    if isinstance(fp, str) and fp:
        return fp
    return _contexts_fingerprint(details)


def _avg_metric(details: list, key: str):
    """单指标均值；无有效样本时返回 None（区别于 0.0）。"""
    vals = [d.get(key) for d in details
            if isinstance(d.get(key), (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _read_scores(rep: dict):
    """取四指标均值。

    ★ 报告的真实结构：
        scores.<metric>       均值在这里
        scores._per_sample[]  逐条分数
        details[]             只有问答与检索结果，不含指标

    读错位置不会崩，只会让所有差值变成 0，
    于是"未超出噪声底"看起来像正常结论 —— 比直接崩溃危险得多。
    """
    sc = rep.get("scores") or {}
    out = {}
    for k in _METRIC_KEYS:
        v = sc.get(k)
        out[k] = float(v) if isinstance(v, (int, float)) else None
    return out


_METRIC_KEYS = ("faithfulness", "answer_relevancy",
                "context_precision", "context_recall")


def compare_with_baseline(cur_report: dict, baseline_path: str) -> dict:
    """与基线报告对比，返回可序列化的 verdict。

    :return: {"comparable": bool, "null_run": bool, "reason": str,
              "fingerprints": {...}, "deltas": {...},
              "noise_floor": {...}}
    """
    import pathlib as _pl

    out = {
        "comparable": False,
        "null_run": False,
        "reason": "",
        "fingerprints": {},
        "deltas": {},
        "noise_floor": {},
    }

    bp = _pl.Path(baseline_path)
    if not bp.is_file():
        out["reason"] = f"基线文件不存在：{baseline_path}"
        return out
    try:
        base = json.loads(bp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        out["reason"] = f"基线读取失败：{e}"
        return out

    cur_details = cur_report.get("details") or []
    base_details = base.get("details") or []
    if not cur_details or not base_details:
        out["reason"] = "报告缺少 details，无法对比"
        return out

    # ---- 不可比的两个前提：样本数与 top_k 必须一致 ----
    cur_n = cur_report.get("num_samples") or len(cur_details)
    base_n = base.get("num_samples") or len(base_details)
    if cur_n != base_n:
        out["reason"] = (
            f"样本数不同（基线 {base_n} vs 本轮 {cur_n}），不可比")
        return out

    cur_k = cur_report.get("top_k")
    base_k = base.get("top_k")
    if cur_k is not None and base_k is not None and cur_k != base_k:
        out["reason"] = f"top_k 不同（基线 {base_k} vs 本轮 {cur_k}），不可比"
        return out

    # ---- Q1 变量隔离 ----
    fp_cur = _fingerprint_of_report(cur_report, cur_details)
    fp_base = _fingerprint_of_report(base, base_details)
    out["fingerprints"] = {"current": fp_cur, "baseline": fp_base}
    same_fp = (fp_cur == fp_base)

    # ---- 噪声底：基线有测量值就用它，否则用保守值 ----
    base_scores = base.get("scores") or {}
    floor = {}
    for k in _METRIC_KEYS:
        v = (base_scores.get("_noise_floor")
             or base.get("_noise_floor") or {})
        f = v.get(k) if isinstance(v, dict) else None
        floor[k] = float(f) if isinstance(f, (int, float)) else _NOISE_FLOOR_UNKNOWN
    out["noise_floor"] = floor

    # ---- Q3 逐指标差值与噪声判定 ----
    cur_sc = _read_scores(cur_report)
    deltas = {}
    over, under, missing = [], [], []
    for k in _METRIC_KEYS:
        a, b = cur_sc.get(k), _read_scores(base).get(k)
        if a is None or b is None:
            missing.append(k)
            continue
        d = a - b
        deltas[k] = d
        (over if abs(d) > floor[k] else under).append(k)
    out["deltas"] = deltas
    if missing:
        out["reason"] = f"缺少指标：{', '.join(missing)}"
        return out

    # ---- Q2 空跑判定 ----
    if same_fp:
        out["null_run"] = True
        out["comparable"] = False
        out["reason"] = (
            f"上下文指纹完全一致（{fp_cur}）→ 检索层零变化。"
            + (f"但 {len(over)}/{len(_METRIC_KEYS)} 项超出噪声底，"
               "差异 100% 来自 LLM 判分随机性，**不是策略改进**。"
               if over else "分数亦无显著变化。"))
        return out

    out["comparable"] = True
    out["over_noise"] = over
    out["within_noise"] = under
    out["reason"] = "检索层已变化，差异可归因到检索/生成策略"
    return out


def _print_baseline_report(verdict: dict) -> None:
    """把 verdict 打成人类可读的表格。"""
    print("\\n" + "=" * 66)
    print("基线对比")
    print("=" * 66)

    if not verdict["fingerprints"]:
        print(f"  [不可比] {verdict['reason']}")
        return

    fp = verdict["fingerprints"]
    print(f"  指纹  本轮={fp['current']}  基线={fp['baseline']}")
    if verdict["null_run"]:
        print("    -> 检索层零变化（空跑）")
    else:
        print("    -> 检索层已变化")

    print(f"\\n  {'指标':22s} {'差值':>9s} {'噪声底':>8s}")
    print("  " + "-" * 44)
    for k in _METRIC_KEYS:
        d = verdict["deltas"].get(k)
        f = verdict["noise_floor"].get(k)
        if d is None:
            print(f"  {k:22s} {'(缺失)':>9s}")
            continue
        mark = "*" if abs(d) > (f or _NOISE_FLOOR_UNKNOWN) else " "
        print(f"  {k:22s} {d:+9.4f} {(f or 0):8.3f}{mark}")

    print()
    if verdict["null_run"]:
        print("  [空跑] 本轮数据不能证明任何优化有效")
        print("        要得到有效结论，必须先让检索层真的变化")
    elif verdict.get("over_noise"):
        print(f"  [可作结论] 超出噪声底：{', '.join(verdict['over_noise'])}")
        if verdict.get("within_noise"):
            print(f"  [不可作结论] 未超出：{', '.join(verdict['within_noise'])}")
    else:
        print("  [不可作结论] 全部指标均在噪声底内，"
              "不要写成'提升了'")

    if not verdict["comparable"] and verdict["reason"]:
        print(f"\\n  说明：{verdict['reason']}")


'''

ANCHOR = "# ==================== 主流程 ===================="

if ANCHOR not in src:
    print("✗ 找不到「主流程」锚点")
    raise SystemExit(1)

src = src.replace(ANCHOR, BLOCK + ANCHOR, 1)

# 确保 hashlib / pathlib 已导入
if "import hashlib" not in src:
    src = src.replace("import argparse", "import argparse\nimport hashlib", 1)
if "\nimport pathlib" not in src:
    src = src.replace("import os", "import os\nimport pathlib", 1)

try:
    ast.parse(src)
except SyntaxError as e:
    print(f"✗ 插入后语法错误 line {e.lineno}: {e.msg}")
    raise SystemExit(1)

p.write_text(src, encoding="utf-8")
print("✓ 基线对比区块已恢复")
print("  符号: _contexts_fingerprint / compare_with_baseline /")
print("        _NOISE_FLOOR_UNKNOWN / _METRIC_KEYS / _read_scores")
print("  改进: 噪声底可来自基线实测值，未知时才用保守值")