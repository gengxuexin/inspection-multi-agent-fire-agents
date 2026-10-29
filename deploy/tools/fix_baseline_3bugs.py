"""修三处与测试契约的偏差（26/30 -> 30/30）。

1. `null_run` 在 contexts 变化时应**完全不出现**
   （测试: assert "null_run" not in r["checks"]）
   —— 语义更对：检索层变了，就不存在"空跑"这个问题，
   不该输出一个 detected=False 的空检查项。
   只在「contexts 一致」时才给 null_run。

2. `checks["noise"]["noise_floor"]` 期望是逐指标 dict
   我已是这样，但 `checks["noise"]` 里还多塞了别的键导致
   `'bool' object has no attribute 'get'` —— 是打印函数取键路径错了。

3. ★ 逻辑错误（最重要）：空跑判定不该看「分数有没有变」，
   而该看「变化是否超出噪声」。
   指纹一致 + prec 从 0.221 到 0.222（+0.001）
   —— 那是抖动，不是空跑，也不该报警。
   改成：指纹一致 且 至少一项超出噪声底 -> 空跑。
"""
import pathlib

p = pathlib.Path("agent1/eval/local_eval.py")
s = p.read_text(encoding="utf-8")

# ---- 修正 1&3：空跑判定挪到噪声判定之后，且语义改为「超出噪声」----
OLD_NR = '''    if same_fp and score_changed:
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
'''
NEW_NR = '''    # ★ 空跑判定放在噪声判定之后，且语义是「超出噪声」而非「有变化」。
    #   指纹一致 + prec 0.221→0.222（+0.001）：那是抖动，
    #   既不是空跑的有效信号，也不该报空跑 ——
    #   否则每次跑都会因为那点抖动被拦下，机制就成了摆设。
    #   真正要拦的是：指纹一致**却**分数大幅变化（v7/v8 那种）。
'''

OLD_VERDICT = '''    # ---------- 逐指标 verdict ----------
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
'''
NEW_VERDICT = '''    # ---------- 逐指标 verdict ----------
    sig_count = 0
    for k in KEYS:
        if k not in deltas:
            continue
        d = deltas[k]
        sig = abs(d) > floor_map[k]
        if sig:
            sig_count += 1
        out["verdict"][k] = {
            "delta": d,
            "exceeds_noise": sig,
            "verdict": "真实变化" if sig else "在噪声内，不可归因",
        }

    # ---------- Q2 空跑（依赖噪声判定结果）----------
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

if OLD_NR in s:
    s = s.replace(OLD_NR, NEW_NR, 1)
    print("✓ 修正 1&3：空跑判定改为「超出噪声」且只在 contexts 一致时出现")
else:
    print("! 未找到 null_run 块")

if OLD_VERDICT in s:
    s = s.replace(OLD_VERDICT, NEW_VERDICT, 1)
    print("✓ 修正 3b：空跑判定挪到噪声之后")
else:
    print("! 未找到 verdict 块")

# ---- 修正 2：打印函数对缺键的防护 ----
OLD_PRINT = '''    nr = checks.get("null_run") or {}
    noise = checks.get("noise") or {}'''
NEW_PRINT = '''    nr = checks.get("null_run")
    if not isinstance(nr, dict):
        nr = {}
    noise = checks.get("noise")
    if not isinstance(noise, dict):
        noise = {}'''
if OLD_PRINT in s:
    s = s.replace(OLD_PRINT, NEW_PRINT, 1)
    print("✓ 修正 2：打印函数对非 dict 键做防护")
else:
    print("! 未找到打印防护点")

p.write_text(s, encoding="utf-8")

import ast
ast.parse(s)
print("✓ 语法正常")