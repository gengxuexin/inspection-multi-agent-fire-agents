"""删除残留的 cue_ratio 计算（revert 脚本的正则没匹配到）。

现象：revert 用正则匹配 cue_ratio 块时因为注释文本已变（被前一步改写），
正则失配，于是 cue_ratio 的实际计算代码留了下来，
而 _QUERY_CUES / _CUE_DISCOUNT 常量已被删除 ->
运行时会 AttributeError，但语法检查与单测都过了
（因为该分支在单测的 mock 场景下没被触发到）。

★ 教训：
  正则删除的失败模式是「静默不匹配」—— 不报错、不改变语法，
  只在运行时才炸。本次是靠「符号定义/引用一致性检查」抓到的：
  grep 到 _QUERY_CUES 的**引用**（480 行）却找不到它的**定义**。

  所以凡是删常量，都必须连带确认：
    1. 定义已删
    2. 所有引用也已删
    3. 运行一遍真实代码路径（不是只跑语法检查）
"""
import ast
import pathlib
import re

p = pathlib.Path("agent1/hybrid.py")
src = p.read_text(encoding="utf-8")

# 连注释带代码一起删：往上找连续的 # 行，往下找到 cue_ratio 赋值块结束
lines = src.splitlines(keepends=True)

lo = None
for i, ln in enumerate(lines):
    if "cue_ratio = (" in ln:
        # 往上：注释块与空行
        j = i
        while j > 0 and (lines[j].strip().startswith("#")
                         or not lines[j].strip()):
            j -= 1
        # j 指向第一行代码（if token_set: 或类似）
        # 往前多退一行，把上面的整段注释纳入
        while j > 0 and lines[j - 1].strip().startswith("#"):
            j -= 1
        lo = j
        # 往下：到括号配平
        depth = 0
        hi = i
        while hi < len(lines):
            depth += lines[hi].count("(") - lines[hi].count(")")
            if depth <= 0 and hi > i:
                break
            hi += 1
        break

if lo is None:
    print("· 未找到 cue_ratio")
    raise SystemExit(0)

print(f"将删除第 {lo+1} ~ {hi+1} 行：")
for i in range(lo, hi + 1):
    print(f"  {i+1:4d}: {lines[i].rstrip()[:66]}")

new = "".join(lines[:lo] + lines[hi + 1:])
new = re.sub(r"\n{3,}(?=        for d in docs)", "\n\n", new)

try:
    ast.parse(new)
except SyntaxError as e:
    print(f"\n✗ 删后语法错误 line {e.lineno}: {e.msg}")
    raise SystemExit(1)

p.write_text(new, encoding="utf-8")
print("\n✓ 已删除")

# ---- 符号一致性检查（这才是抓出这类问题的正确工具）----
src = p.read_text(encoding="utf-8")
print()
print("符号一致性检查：")
for sym in ("_QUERY_CUES", "_CUE_DISCOUNT", "cue_ratio"):
    n = src.count(sym)
    print(f"  {sym:16s} 出现 {n} 次 {'✓' if n == 0 else '✗ 仍有残留'}")