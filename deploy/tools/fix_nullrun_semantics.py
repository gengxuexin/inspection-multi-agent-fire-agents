"""按测试最终契约修正空跑语义与 noise 结构（目标 30/30）。

★ 契约（这次逐条对过测试原文，不再猜）：
  1. 空跑 = 指纹一致 且 分数有任何变化
     —— 即使变化只有 +0.001 也算空跑。
     理由：指纹一致就意味着检索层零变化，此时分数的**任何**变化
     都不可能来自策略改进，只能来自 LLM 抖动。
     阈值是给"检索层确实变了"的情况用的二级过滤，
     不能拿来判定空跑 —— 否则 v7/v8 那种 rec +0.021 就会被放过。

  2. contexts 变化时，checks 里**完全没有** null_run 这个键。

  3. checks["noise"]["noise_floor"] 是 {指标: 值} 的逐指标 dict。
  4. checks["noise"]["passed"]：用了基线实测值才为 True。
"""
import pathlib

p = pathlib.Path("agent1/eval/local_eval.py")
s = p.read_text(encoding="utf-8")

# ---- 1) 撤掉我加错的那段注释 ----
s = s.replace(
    '''    # ★ 空跑判定放在噪声判定之后，且语义是「超出噪声」而非「有变化」。
    #   指纹一致 + prec 0.221→0.222（+0.001）：那是抖动，
    #   既不是空跑的有效信号，也不该报空跑 ——
    #   否则每次跑都会因为那点抖动被拦下，机制就成了摆设。
    #   真正要拦的是：指纹一致**却**分数大幅变化（v7/v8 那种）。
''',
    '''    # 空跑判定见下方（放在噪声判定之后，但要区分两件事，见那处注释）
''', 1)

# ---- 2) 空跑：改回「有任何变化」，且只依赖指纹 ----
OLD = '''    # ---------- Q2 空跑（依赖噪声判定结果）----------
    # 只在 contexts 一致时才有这个问题；检索层变了就不必问。
    if same_fp:
        if sig_count:
            names = [k for k, v in out["verdict"].items()
                     if v["exceeds_noise"]]
            out["checks"]["null_run"] = {
                "detected": True,
                "passed": False,
                "note": ("上下文指纹完全一致（" + fp_cur + "），"
                         "检索层零变化，但 " + f"{sig_count}/{len(out['verdict'])}"
                         f" 项（{', '.join(names)}）超出噪声底 → "
                         "差异 100% 来自 LLM 判分随机性，"
                         "**不是策略改进**。本轮数据不能证明任何优化有效。"),
            }
        else:
            out["checks"]["null_run"] = {
                "detected": False,
                "passed": True,
                "note": ("上下文一致且各项变化均在噪声底内 —— "
                         "结果可复现，没有可归因的改进"),
            }

    return out
'''

NEW = '''    # ---------- Q2 空跑 ----------
    # ★ 语义澄清（这里我一度改错，值得记下来）：
    #   空跑的判据是「指纹一致 **且** 分数有任何变化」，
    #   **不是**「指纹一致且变化超出噪声底」。
    #
    #   为什么不用噪声底当条件：指纹一致已经意味着检索层零变化，
    #   此时分数的任何变化都不可能来自策略改进 —— 只能是 LLM 抖动。
    #   阈值是给"检索层确实变了"的情况做二级过滤用的；
    #   拿它来判空跑，v7/v8 那种 rec +0.021 就会被放过，
    #   而那正是最初被骗的那一轮。
    #
    #   而且指纹一致 + prec 0.221→0.222 也该报空跑：
    #   它说明"什么都没改却动了分数"，这本身就是需要知道的事实。
    if same_fp:
        if score_changed:
            out["checks"]["null_run"] = {
                "detected": True,
                "passed": False,
                "note": ("上下文指纹完全一致（" + fp_cur + "），"
                         "检索层零变化，但分数变了 → "
                         "差异 100% 来自 LLM 判分随机性，"
                         "**不是策略改进**。本轮数据不能证明任何优化有效。"),
            }
        else:
            out["checks"]["null_run"] = {
                "detected": False,
                "passed": True,
                "note": "上下文与分数均未变，结果可复现",
            }

    return out
'''

assert OLD in s, "未找到空跑块"
s = s.replace(OLD, NEW, 1)

# ---- 3) 打印函数：噪声块缺键防护 ----
s = s.replace(
    '''            f = (noise.get("noise_floor") or {}).get(k, _NOISE_FLOOR_UNKNOWN)''',
    '''            floor_map = noise.get("noise_floor")
            if not isinstance(floor_map, dict):
                floor_map = {}
            f = floor_map.get(k, _NOISE_FLOOR_UNKNOWN)''', 1)

p.write_text(s, encoding="utf-8")

import ast
ast.parse(s)
print("✓ 空跑语义与打印防护已修正")
print("✓ 语法正常")