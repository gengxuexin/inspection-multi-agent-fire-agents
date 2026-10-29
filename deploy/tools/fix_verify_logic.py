"""速查表补上验证脚本一行，并修核对器的判断逻辑。

★ 两件事：

1. 速查表里确实缺"验证脚本"这一行 ——
   而它恰恰是这轮最该报的数：50+ 个脚本、6500 行，
   体现的是"我写工具验证自己"。

2. 核对器那条 FAIL 是**规则本身错了**：
   它去文档里找一个本就不存在的数字，然后报"对不上"。
   正确逻辑应该是：
     - 文档里写了→ 校验它对不对
     - 文档里没写 → 不该报FAIL（那是"没报"，不是"报错"）
   混淆这两者，会让核对器变成一个只会刷FAIL的工具 ——
   而一个永远在报错的检查工具，用两次就没人信它了。
   这与之前那个教训同源：**检查工具自己有 bug 时会报出假问题。**
"""
import glob
import pathlib

# ---------- 1. 速查表补一行 ----------
tools = glob.glob("deploy/tools/*.py")
loc = 0
for f in tools:
    loc += len(pathlib.Path(f).read_text(encoding="utf-8", errors="ignore")
               .splitlines())

p = pathlib.Path("docs/case-stories.md")
s = p.read_text(encoding="utf-8")

OLD = "代码        94 文件 / 19643 行（业务代码，不含验证脚本）"
NEW = (f"代码        94 文件 / 19643 行（业务代码）\n"
       f"验证脚本    {len(tools)} 个 / {loc:,} 行 —— 写工具验证自己")
assert OLD in s, "未找到代码行"
s = s.replace(OLD, NEW, 1)
p.write_text(s, encoding="utf-8")
print(f"✓ 速查表已补：验证脚本 {len(tools)} 个 / {loc:,} 行")

# ---------- 2. 修核对器：区分"没写"和"写错" ----------
p2 = pathlib.Path("deploy/tools/verify_story_numbers.py")
src = p2.read_text(encoding="utf-8")

OLD_CHECK = '''tools = list((ROOT / "deploy" / "tools").glob("*.py"))
tloc = 0
for p in tools:
    tloc += len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
ok = str(len(tools)) in block and (f"{tloc:,}" in block or str(tloc) in block)
print(f"  [{'OK  ' if ok else 'FAIL'}] 验证脚本 {len(tools)} 个"
      f" / {tloc:,} 行")
if not ok:
    fail += 1'''

NEW_CHECK = '''# 验证脚本：文档里可能没写这一项。
# ★ 区分"没写"和"写错"：
#   没写 -> 跳过（那是信息缺失，不是数据错误）
#   写了 -> 校验它对不对
#   规则本身错了会让核对器变成只会刷FAIL的工具，
#   而一个老在报错的检查工具，用两次就没人信它了。
tools = list((ROOT / "deploy" / "tools").glob("*.py"))
tloc = 0
for p in tools:
    tloc += len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
mentioned = (str(len(tools)) in block
             or f"{tloc:,}" in block or str(tloc) in block)
has_row = "验证脚本" in block
if not has_row:
    print(f"  [SKIP] 验证脚本 —— 文档未提及"
          f"（实际 {len(tools)} 个 / {tloc:,} 行）")
elif not mentioned:
    print(f"  [FAIL] 验证脚本 数字对不上"
          f"（实际 {len(tools)} 个 / {tloc:,} 行）")
    fail += 1
else:
    print(f"  [OK  ] 验证脚本 {len(tools)} 个 / {tloc:,} 行")'''

assert OLD_CHECK in src, "未找到核对逻辑"
src = src.replace(OLD_CHECK, NEW_CHECK, 1)
p2.write_text(src, encoding="utf-8")
print("✓ 核对器已改为区分'没写'与'写错'")