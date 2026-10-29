"""重建 hybrid.py 的最终状态（拆分 IDF 上限 + 按 A/B 结论回退）。

★ 为什么需要重建：
  sync-all.sh 的第一遍是「桌面 → WSL」，而桌面侧的 hybrid.py
  一直没跟上 WSL 侧的改动（拆分 IDF + 回退），于是被覆盖成旧版。
  `/tmp/hybrid_before_split.py` 是拆分前的备份，不含最终状态。

  好在每一步改动都是脚本，可以按顺序重放：
      1. split_idf_max.py         拆分 _IDF_MAX -> _IDF_MAX_BM25/_IDF_MAX_RERANK
      2. revert_rerank_per_ab.py  按 A/B 结论回退打分逻辑
      3. cleanup_query_cues_block.py  清掉疑问词集合残留（手工按行号处理）

  这就是把每一步都写成脚本而不是 sed 的回报 ——
  出错时能一键重建，而不用凭记忆重做三次改动。
"""
import ast
import pathlib
import subprocess
import sys

ROOT = pathlib.Path("/home/geng/fire_agent_v2/fire_agents")
TOOLS = ROOT / "deploy" / "tools"
PY = "/home/geng/.venv/bin/python"
HYBRID = ROOT / "agent1" / "hybrid.py"


def run(name):
    p = TOOLS / name
    if not p.is_file():
        print(f"  跳过 {name}（不存在）")
        return
    print(f"\n>>> {name}")
    r = subprocess.run([PY, str(p)], cwd=str(ROOT),
                       capture_output=True, text=True)
    print((r.stdout or "").rstrip())
    if r.returncode != 0:
        print(f"  失败：{(r.stderr or '').strip()[:300]}")


print("=" * 60)
print("重建 hybrid.py")
print("=" * 60)

# ---- 步骤 1：拆分 ----
run("split_idf_max.py")

# ---- 步骤 2：回退打分逻辑 ----
run("revert_rerank_per_ab.py")

# ---- 步骤 3：清残留（脚本的定位逻辑不可靠，改用行号定位）----
print("\n>>> 清理疑问词集合残留")
src = HYBRID.read_text(encoding="utf-8")
lines = src.splitlines(keepends=True)

# 找孤立残留：出现字符串列表元素但没有 frozenset({ 的上下文
lo = None
for i, ln in enumerate(lines):
    st = ln.strip()
    # 元素行，如  "哪些", "哪个", ...
    if st.startswith('"哪些"') or st.startswith('"哪个"'):
        # 往上找注释块起点
        j = i
        while j > 0 and not lines[j].strip().startswith("#"):
            j -= 1
        # 再往上找到空行或上一段结束
        while j > 0 and lines[j].strip().startswith("#"):
            j -= 1
        lo = j
        break

if lo is not None:
    hi = lo
    while hi < len(lines) and lines[hi].strip() != "})":
        hi += 1
    if hi < len(lines):
        print(f"  删除第 {lo+1} ~ {hi+1} 行")
        new = "".join(lines[:lo] + lines[hi + 1:])
        ast.parse(new)
        HYBRID.write_text(new, encoding="utf-8")
        print("  ✓ 已删除")
    else:
        print("  ! 未找到结束标记")
else:
    print("  · 无残留")

# ---- 验证 ----
print("\n" + "=" * 60)
print("验证")
print("=" * 60)
final = HYBRID.read_text(encoding="utf-8")
try:
    ast.parse(final)
    print("  ✓ 语法正常")
except SyntaxError as e:
    print(f"  ✗ 语法错误 line {e.lineno}: {e.msg}")
    lines2 = final.splitlines()
    for i in range(max(0, e.lineno - 4), min(len(lines2), e.lineno + 2)):
        print(f"    {i+1}: {lines2[i][:70]}")
    sys.exit(1)

checks = [
    ("_IDF_MAX_BM25 已拆分", "_IDF_MAX_BM25" in final),
    ("_IDF_MAX_RERANK = 3.0", "_IDF_MAX_RERANK = 3.0" in final),
    ("idf² 已回退", "w * w" not in final),
    ("cue_ratio 已移除", "cue_ratio" not in final),
    ("_QUERY_CUES 已移除", "_QUERY_CUES" not in final),
    ("log1p 已清除", "log1p" not in final),
    ("线性 idf 覆盖率在位", "got = sum(w for t, w in weights.items()" in final),
]
for label, ok in checks:
    print(f"  {'✓' if ok else '✗'} {label}")
if not all(ok for _, ok in checks):
    sys.exit(1)