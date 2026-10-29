"""修 tuple 解包顺序颠倒（最后 3 个失败的根因）。

`_noise_from_baseline` 返回 `({指标: 值}, measured_bool)`，
但调用处写成了 `measured, floors = ...` —— 两个变量互换了。

★ 为什么这个 bug 特别难发现：
  它**不报错**，只是让 measured 拿到一个 dict（空 dict 是 falsy），
  于是永远走"未实测"分支，用保守值 0.10 而不是基线里的实测值。

  后果是**所有显著性判定都偏保守**：
  明明基线里存着 _noise_floor=0.001，
  代码却用 0.10 去比，于是 +0.004 被判为"在噪声内"——
  而测试期望它是超出噪声的。

  也就是说：机制静默降级成"几乎什么都不报"。
  这与整个评测可信度机制的目的正好相反。

  教训：**tuple 返回值的顺序必须在解包处显式核对。**
  dict 有键名兜底，tuple 没有 —— 写反了编译器不会吭声。
  宁可返回 dict/具名元组，也不要裸 tuple。
"""
import pathlib

p = pathlib.Path("agent1/eval/local_eval.py")
s = p.read_text(encoding="utf-8")

OLD = "    measured, floors = _noise_from_baseline(base)"
NEW = '''    # ★ 顺序要对齐函数的返回值 ({指标: 值}, measured) —— 曾写反过，
    #   导致 measured 拿到空 dict（falsy）而永远走保守分支，
    #   所有显著性判定被静默降级。
    floors, measured = _noise_from_baseline(base)'''

assert OLD in s, "未找到解包点"
s = s.replace(OLD, NEW, 1)

# 顺手把函数返回改成具名 dict，彻底消除这类风险
OLD_FN = '''def _noise_from_baseline(base: dict):
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
    return {}, False'''

NEW_FN = '''def _noise_from_baseline(base: dict) -> dict:
    """取基线里的实测噪声底。

    :return: {"floors": {指标: 值}, "measured": bool}
    ★ 用 dict 而不是 tuple —— tuple 的解包顺序写错不会报错，
      只会让「measured」拿到空 dict（falsy），
      机制静默降级成"几乎什么都不报"，与设计目的正好相反。
    """
    for src in (base.get("_noise_floor"),
                (base.get("scores") or {}).get("_noise_floor")):
        if isinstance(src, dict):
            vals = {k: float(src[k]) for k in KEYS
                    if isinstance(src.get(k), (int, float))}
            if vals:
                return {"floors": vals, "measured": True}
    return {"floors": {}, "measured": False}'''

assert OLD_FN in s, "未找到函数定义"
s = s.replace(OLD_FN, NEW_FN, 1)

# 调用处改成键访问
s = s.replace(
    '''    # ★ 顺序要对齐函数的返回值 ({指标: 值}, measured) —— 曾写反过，
    #   导致 measured 拿到空 dict（falsy）而永远走保守分支，
    #   所有显著性判定被静默降级。
    floors, measured = _noise_from_baseline(base)''',
    '''    _nz = _noise_from_baseline(base)
    floors = _nz["floors"]
    measured = _nz["measured"]''', 1)

p.write_text(s, encoding="utf-8")

import ast
ast.parse(s)
print("✓ 解包顺序已修正，函数改为返回 dict")
print("✓ 语法正常")