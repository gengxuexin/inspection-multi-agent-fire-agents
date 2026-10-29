"""按 tests/test_eval_baseline.py 的完整契约重写「基线对比」区块。

★ 这一版与前一版的区别：
  前一版我在**盲猜**契约 —— 改了 3 次、测试通过数从 11 掉到 8。
  这次先把测试文件完整读完，逐条对齐字段名与嵌套层级。

契约（从 30 个断言里反推）：
  返回 {
    "error": str|None,
    "baseline_file": str,                     # print 要用
    "checks": {
      "comparable": {"passed", "baseline_num_samples",
                     "current_num_samples", "baseline_top_k",
                     "current_top_k", ...},
      "variable_isolation": {"passed", "current_fingerprint",
                             "baseline_fingerprint", "contexts_identical"},
      "null_run": {...},        # 仅当可比时才出现
      "noise": {"passed", "measured", "source", "noise_floor": {...},
                "note"},
    },
    "deltas": {4 个指标},
    "verdict": {4 个指标: {"delta", "exceeds_noise", "verdict"}},
  }

  指标名常量为 KEYS（测试里直接用了 KEYS）。
  verdict 文本约定：
    超出噪声 -> "真实变化"
    未超出   -> 含"不可归因"
  打印约定：
    空跑时输出含 "空跑"
    不可比时输出含 "不可比" 且两个 top_k 值都要出现
    全部在噪声内时输出含 "0/4"
    残缺 dict 不能崩
"""
import ast
import pathlib

p = pathlib.Path("agent1/eval/local_eval.py")
src = p.read_text(encoding="utf-8")

START = "# ==================== 基线对比 ===================="
END = "# ==================== 主流程 ===================="
i, j = src.find(START), src.find(END)
if i < 0 or j <= i:
    print("✗ 找不到区块边界")
    raise SystemExit(1)

BLOCK = r'''# ==================== 基线对比 ====================
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
#   要锁住的重点是**别再把空跑/噪声误读成优化效果**。

KEYS = ("faithfulness", "answer_relevancy",
        "context_precision", "context_recall")

# 抖动幅度未知时的保守阈值。
#
# ★ 为什么必须是 0.10 而不是实测的 0.021：
#   实测抖动是在特定数据集 + 特定 judge 上测的，换数据集就不一定成立。
#   更重要的是：**单次 A/B 差值只是抖动幅度的下界，不是幅度本身。**
#   曾拿 v7/v8 的差值（recall 0.021）当阈值设成 0.02，
#   结果 v7→v8 的 recall 差值恰好 0.021，卡在边界上，
#   `abs(0.021) > 0.02` 判真，测试直接失败。
#
#   差值分布有方差，两次恰好接近不代表常态如此。
#   tests 里有一条专门锁定这个决策：若有人调回 0.02，
#   必须先解释为什么单次观测可以当阈值用。
_NOISE_FLOOR_UNKNOWN = 0.10


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
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _fingerprint_of(rep: dict, details: list) -> str:
    """优先用报告里存的指纹，没有才算一遍。

    ★ 不能只信报告字段：万一是旧版本报告（没这字段）或被手改过，
      那就该自己算。存指纹的意义是省一次全量哈希，不是免于校验。
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
    return {k: (float(sc[k]) if isinstance(sc.get(k), (int, float)) else None)
            for k in KEYS}


def _noise_from_baseline(base: dict):
    """取基线里的实测噪声底。

    :return: ({指标: 值}, measured: bool)
    """
    for src in (base.get("_noise_floor"),
                (base.get("scores") or {}).get("_noise_floor")):
        if isinstance(src, dict):
            vals = {k: float(src[k]) for k in KEYS
                    if isinstance(src.get(k), (int, float))}
            if vals:
                return vals, True
    return {}, False


def compare_with_baseline(cur_report: dict, baseline_path: str) -> dict:
    """与基线报告对比，返回结构化 verdict（供打印与断言）。"""
    out = {
        "error": None,
        "baseline_file": baseline_path,
        "checks": {},
        "deltas": {},
        "verdict": {},
    }

    bp = pathlib.Path(baseline_path)
    if not bp.is_file():
        out["error"] = f"基线文件不存在：{baseline_path}"
        return out
    try:
        base = json.loads(bp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        out["error"] = f"基线读取失败：{e}"
        return out

    cur_details = cur_report.get("details") or []
    base_details = base.get("details") or []
    if not cur_details or not base_details:
        out["error"] = "报告缺少 details，无法对比"
        return out

    # ---------- Q1 可比性 ----------
    cur_n = cur_report.get("num_samples") or len(cur_details)
    base_n = base.get("num_samples") or len(base_details)
    cur_k = cur_report.get("top_k")
    base_k = base.get("top_k")
    k_differs = (cur_k is not None and base_k is not None and cur_k != base_k)

    note = ""
    if cur_n != base_n:
        note = (f"样本数不同（基线 {base_n} vs 本轮 {cur_n}），"
                "分数不可比")
    elif k_differs:
        note = (f"top_k 不同（基线 {base_k} vs 本轮 {cur_k}），"
                "precision 分母变了，分数不可比"
                "（v8 的 cp 0.520→0.325 就是纯 top_k 5→8 造成的）")

    out["checks"]["comparable"] = {
        "passed": (cur_n == base_n) and not k_differs,
        "baseline_num_samples": base_n,
        "current_num_samples": cur_n,
        "baseline_top_k": base_k,
        "current_top_k": cur_k,
        "note": note,
    }
    if note:
        return out          # 不可比时不给 verdict——比什么都不报更危险

    # ---------- 差值 ----------
    cur_sc = _read_scores(cur_report)
    base_sc = _read_scores(base)
    deltas = {}
    for k in KEYS:
        a, b = cur_sc.get(k), base_sc.get(k)
        if a is None or b is None:
            continue
        deltas[k] = a - b
    out["deltas"] = deltas

    # ---------- Q2 空跑检测 ----------
    fp_cur = _fingerprint_of(cur_report, cur_details)
    fp_base = _fingerprint_of(base, base_details)
    same_fp = (fp_cur == fp_base)
    score_changed = any(abs(v) > 1e-9 for v in deltas.values())

    out["checks"]["variable_isolation"] = {
        "passed": not same_fp,
        "contexts_identical": same_fp,
        "current_fingerprint": fp_cur,
        "baseline_fingerprint": fp_base,
    }

    if same_fp and score_changed:
        out["checks"]["null_run"] = {
            "detected": True,
            "passed": False,
            "note": ("上下文指纹完全一致（" + fp_cur + "），检索层零变化，"
                     "但分数变了 → 差异 100% 来自 LLM 判分随机性，"
                     "**不是策略改进**。本轮数据不能证明任何优化有效。"),
        }
    else:
        out["checks"]["null_run"] = {
            "detected": False,
            "passed": True,
            "note": ("上下文与分数均未变，结果可复现" if same_fp
                     else "检索层已变化，差异可归因到检索/生成策略"),
        }

    # ---------- Q3 噪声判定 ----------
    measured, floors = _noise_from_baseline(base)
    if measured:
        floor_map = {k: floors.get(k, _NOISE_FLOOR_UNKNOWN) for k in KEYS}
        source = "基线实测"
        nz_note = "使用基线报告里的实测噪声底"
        passed = True
    else:
        floor_map = {k: _NOISE_FLOOR_UNKNOWN for k in KEYS}
        source = f"经验保守值 {_NOISE_FLOOR_UNKNOWN}"
        nz_note = ("基线未实测噪声底，用经验保守值。"
                   "建议在基线上跑两次同输入评测，实测抖动后再对比，"
                   "否则只能得到保守结论")
        passed = False

    out["checks"]["noise"] = {
        "passed": passed,
        "measured": measured,
        "source": source,
        "noise_floor": floor_map,
        "note": nz_note,
    }

    # ---------- 逐指标 verdict ----------
    for k in KEYS:
        if k not in deltas:
            continue
        d = deltas[k]
        sig = abs(d) > floor_map[k]
        out["verdict"][k] = {
            "delta": d,
            "exceeds_noise": sig,
            "verdict": "真实变化" if sig else "在噪声内，不可归因",
        }

    return out


def print_baseline_report(out: dict) -> None:
    """把 verdict 打成人类可读的表格。

    ★ 必须能扛各种残缺输入 —— 它崩了等于整轮评测白跑，
      所以全程用 .get() 链而不是下标访问。
    """
    print("\n" + "=" * 66)
    print("基线对比")
    print("=" * 66)
    print(f"  基线文件：{out.get('baseline_file', '(未提供)')}")

    if out.get("error"):
        print(f"  [FAIL] {out['error']}")
        return

    checks = out.get("checks") or {}
    cmp_ = checks.get("comparable") or {}
    print(f"  样本数 基线={cmp_.get('baseline_num_samples')} "
          f"本轮={cmp_.get('current_num_samples')}   "
          f"top_k 基线={cmp_.get('baseline_top_k')} "
          f"本轮={cmp_.get('current_top_k')}")

    if not cmp_.get("passed"):
        print(f"  [不可比] {cmp_.get('note')}")
        return

    vi = checks.get("variable_isolation") or {}
    print(f"\n  指纹  本轮={vi.get('current_fingerprint')}  "
          f"基线={vi.get('baseline_fingerprint')}")

    nr = checks.get("null_run") or {}
    noise = checks.get("noise") or {}
    verdict = out.get("verdict") or {}
    deltas = out.get("deltas") or {}

    if verdict:
        print(f"\n  {'指标':22s} {'差值':>9s} {'噪声底':>8s}")
        print("  " + "-" * 44)
        for k in KEYS:
            v = verdict.get(k)
            if not v:
                continue
            f = (noise.get("noise_floor") or {}).get(k, _NOISE_FLOOR_UNKNOWN)
            mark = "*" if v.get("exceeds_noise") else " "
            print(f"  {k:22s} {v.get('delta', 0.0):+9.4f} "
                  f"{f:8.3f}{mark}")

    total = len(verdict)
    sig = sum(1 for v in verdict.values() if v.get("exceeds_noise"))
    print()
    if nr.get("detected"):
        print("  [空跑] " + (nr.get("note") or ""))
        return

    print(f"  噪声底来源：{noise.get('source', '(未知)')}")
    if not noise.get("measured"):
        print(f"  ⚠ {noise.get('note', '')}")

    if sig:
        names = [k for k, v in verdict.items() if v.get("exceeds_noise")]
        print(f"  [可作结论] {sig}/{total} 项超出噪声底：{', '.join(names)}")
        rest = [k for k, v in verdict.items() if not v.get("exceeds_noise")]
        if rest:
            print(f"  [不可作结论] 未超出：{', '.join(rest)}")
    else:
        print(f"  [不可作结论] {sig}/{total} 项超出噪声底，"
              "全部在噪声内，不要写成'提升了'")


'''

src = src[:i] + BLOCK + src[j:]

if "import hashlib" not in src:
    src = src.replace("import argparse", "import argparse\nimport hashlib", 1)
if "\nimport pathlib" not in src:
    src = src.replace("import os", "import os\nimport pathlib", 1)

try:
    ast.parse(src)
except SyntaxError as e:
    print(f"✗ 语法错误 line {e.lineno}: {e.msg}")
    raise SystemExit(1)

p.write_text(src, encoding="utf-8")
print("✓ 已按测试契约重写（这次先读完了 30 个断言）")