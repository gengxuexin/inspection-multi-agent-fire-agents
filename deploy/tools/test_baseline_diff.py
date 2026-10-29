"""用真实历史报告反测基线对比功能。

★ 为什么必须反测：
  这个工具的全部价值在于「检出空跑」。
  如果它对着已确认是空跑的两份报告（v7/v8）报不出空跑，
  那它就是个只会说"通过"的工具 —— 比没有更危险。

用法：
    python test_baseline_diff.py
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent1" / "eval"))

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "le", ROOT / "agent1" / "eval" / "local_eval.py")
le = importlib.util.module_from_spec(spec)
spec.loader.exec_module(le)

print("=" * 70)
print("反测 1：v7 vs v8（当时人工确认是空跑的一对）")
print("=" * 70)

v7 = json.loads((ROOT / "report_v7.json").read_text(encoding="utf-8"))
v8 = json.loads((ROOT / "report_v8.json").read_text(encoding="utf-8"))

d7 = v7.get("details") or []
d8 = v8.get("details") or []
print(f"  v7 条数={len(d7)}  v8 条数={len(d8)}")

if not d7 or not d8:
    print("  (报告里没有 details，跳过)")
else:
    fp7 = le.contexts_fingerprint(d7)
    fp8 = le.contexts_fingerprint(d8)
    print(f"  v7 指纹 = {fp7}")
    print(f"  v8 指纹 = {fp8}")
    if fp7 == fp8:
        print("\n  ✓ 指纹一致 —— 工具正确识别出检索层未变")
    else:
        print("\n  ! 指纹不一致（这两轮检索层确实变了，或指纹实现有问题）")

    rc = le.compare_with_baseline(d8, "report_v7.json")
    print(f"\n  compare_with_baseline 返回码 = {rc}")
    if fp7 == fp8 and rc == 1:
        print("  ✓ 正确：检出空跑并返回非零（阻断误用）")
    elif fp7 == fp8 and rc == 0:
        print("  ✗ 错误：指纹一致却判为可作结论 —— 空跑检测失效！")
    else:
        print(f"  （指纹不同，返回码 {rc} 属正常）")

print()
print("=" * 70)
print("反测 2：v8 vs v9（当时也确认是空跑）")
print("=" * 70)

v9 = json.loads((ROOT / "report_v9.json").read_text(encoding="utf-8"))
d9 = v9.get("details") or []
if d9 and d8:
    fp9 = le.contexts_fingerprint(d9)
    print(f"  v8 指纹 = {fp8}")
    print(f"  v9 指纹 = {fp9}")
    print(f"  {'✓ 一致（空跑）' if fp9 == fp8 else '! 不一致'}")
else:
    print("  (数据不足，跳过)")

print()
print("=" * 70)
print("反测 3：指纹对分数变化不敏感（这是设计前提）")
print("=" * 70)
# 造一份「同 contexts、不同分数」的假报告，验证指纹不变
import copy
fake = copy.deepcopy(d8)
for x in fake:
    for k in ("faithfulness", "answer_relevancy",
              "context_precision", "context_recall"):
        x[k] = 0.99
fp_fake = le.contexts_fingerprint(fake)
print(f"  原报告指纹   = {fp8}")
print(f"  改分数后指纹 = {fp_fake}")
if fp_fake == fp8:
    print("\n  ✓ 关键性质成立：分数全改，指纹不变")
    print("    这正是空跑检测能工作的前提 ——")
    print("    若指纹含分数，分数一变指纹就变，检测立刻失效")
else:
    print("\n  ✗ 指纹受分数影响 —— 空跑检测不可靠")

print()
print("=" * 70)
print("反测 4：指纹对 contexts 变化敏感")
print("=" * 70)
fake2 = copy.deepcopy(d8)
if fake2 and fake2[0].get("contexts"):
    fake2[0]["contexts"] = ["完全不同的内容" * 20]
    fp_fake2 = le.contexts_fingerprint(fake2)
    print(f"  改一条 contexts 后指纹 = {fp_fake2}")
    if fp_fake2 != fp8:
        print("\n  ✓ contexts 变化会改变指纹（检测能力正常）")
    else:
        print("\n  ✗ contexts 变了指纹却没变 —— 检测能力失效！")