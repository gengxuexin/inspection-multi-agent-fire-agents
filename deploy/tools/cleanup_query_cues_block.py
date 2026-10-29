"""清理上一版正则删除残留的孤立代码块。

现象：revert 脚本用正则删 `_QUERY_CUES = frozenset({` 那一行，
结果集合内容与上方注释全都留了下来，变成孤立表达式 → SyntaxError
（line 261 unexpected indent）。

教训：正则删除多行结构（集合字面量）时，
`^    NAME = ` 只匹配到首行就停，
剩下的元素行与注释全部孤立。**正则不适合删多行结构。**
这次用行号区间精确删。
"""
import ast
import pathlib

p = pathlib.Path("agent1/hybrid.py")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)

# 找出所有「以 _QUERY_CUES 注释块开头、到 }) 结束」的区间
start = None
for i, ln in enumerate(lines):
    # 注释块起点：提到 _QUERY_CUES 的注释行
    if "_QUERY_CUES" in ln and ln.strip().startswith("#"):
        if start is None:
            start = i
    # 集合结束
    if start is not None and ln.strip() in ("})", "})  # 疑问词集合"):
        end = i
        break
    # 或残留的集合元素行（说明首行被删了）
    if start is not None and 'frozenset' not in ln and \
            ln.strip().startswith('"') and '})' not in ln:
        pass  # 继续找结束标记
else:
    end = None

if start is None or end is None:
    print("未定位到残留块，逐行显示 235-265 供检查：")
    for i in range(230, min(268, len(lines))):
        print(f"{i+1:4d}: {lines[i].rstrip()}")
    raise SystemExit(1)

print(f"将删除第 {start+1} ~ {end+1} 行：")
for i in range(start, end + 1):
    print(f"  {i+1:4d}: {lines[i].rstrip()[:70]}")

new_lines = lines[:start] + lines[end + 1:]
src = "".join(new_lines)

# 清掉可能残留的多余空行
import re
src = re.sub(r"\n{3,}(?=    def )", "\n\n", src)

try:
    ast.parse(src)
except SyntaxError as e:
    print(f"\n✗ 删后仍语法错误 line {e.lineno}: {e.msg}")
    for i in range(max(0, e.lineno - 4), min(len(new_lines), e.lineno + 2)):
        print(f"  {i+1:4d}: {new_lines[i].rstrip()[:70]}")
    raise SystemExit(1)

p.write_text(src, encoding="utf-8")
print("\n✓ 已删除并语法正常")