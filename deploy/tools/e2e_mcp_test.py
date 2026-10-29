"""MCP 工具端到端验证：真实调用 compare_documents / list_reports / get_report。

不是"检查函数存在"，而是真的跑一遍改动检测，看：
  1. 能不能解析 docx
  2. 能不能抓出预设的那几类改动
  3. 风险评级对不对（罚款金额改动应该是高风险）
  4. 报告能不能落盘并被 list_reports / get_report 读回来

第 4 步是很多实现会漏的：报告生成了但存不到约定目录，
或者存了但读回来是空的 —— 只测"调用成功"会漏掉这类问题。
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent1"))

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "srv", ROOT / "mcp" / "change_detector_server.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

TEST_DIR = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mcp_test")
orig = TEST_DIR / "original.docx"
mod = TEST_DIR / "modified.docx"

if not orig.is_file() or not mod.is_file():
    print(f"测试文书不存在，先跑 make_test_docs.py：{TEST_DIR}")
    sys.exit(1)

print("=" * 70)
print("1. compare_documents（关闭 LLM，纯规则分析，避免 API 依赖）")
print("=" * 70)
r = m.compare_documents(str(orig), str(mod),
                       review_id="E2E-TEST-001",
                       use_llm=False, save_report=True)
print(f"  status       : {r.get('status')}")
if r.get("status") != "success":
    print(f"  失败：{r}")
    sys.exit(1)

print(f"  report_id    : {r.get('report_id')}")
print(f"  total_changes: {r.get('total_changes')}")
print(f"  high_risk    : {r.get('high_risk_count')}")
print(f"  json落盘     : {r.get('json_path') or '(未落盘)'}")

changes = r.get("changes") or []
print()
print(f"  抓到的改动（{len(changes)} 条，显示前 8 条）：")
#★ 字段名以 change_detector.ChangeItem 的定义为准：
#   original_text / modified_text / change_type / risk_level
#   之前这里写成了 old_text / new_text，结果每条的文本都是空的，
#   一度被误判成"改动检测没抓到内容"。字段名写错时输出看起来很像"功能坏了"。
for c in changes[:8]:
    kind = c.get("change_type") or "?"
    risk = c.get("risk_level") or "?"
    old = (c.get("original_text") or "")[:38]
    new = (c.get("modified_text") or "")[:38]
    reasons = c.get("risk_reasons") or []
    print(f"    [{risk:6s}] {kind:10s} 相似度={c.get('similarity', '?')}")
    print(f"             - {old}")
    print(f"             + {new}")
    if reasons:
        print(f"             风险依据：{reasons[:2]}")

print()
print("=" * 70)
print("2. 风险判定检查（罚款金额改动应当被判高风险）")
print("=" * 70)
money = [c for c in changes
         if "5000" in (c.get("original_text", "") + c.get("modified_text", ""))]
if money:
    for c in money:
        risk = c.get("risk_level")
        print(f"  找到金额改动："
              f"{(c.get('original_text') or '')[:50]} → "
              f"{(c.get('modified_text') or '')[:50]}")
        print(f"  风险等级：{risk}  依据：{c.get('risk_reasons')}")
        if str(risk).lower() == "high":
            print("  ✓ 金额改动被判为高风险")
        else:
            print(f"  ! 期望 high，实际 {risk}")
else:
    print("  ! 改动明细里没有出现 5000")
    print("    排查方向：改动单元是怎么切分的（按行还是按段）")
    print(f"    实际改动类型分布："
          f"{ {c.get('change_type') for c in changes} }")

print()
print("=" * 70)
print("3. list_reports（报告能否被读回）")
print("=" * 70)
lr = m.list_reports(limit=5)
print(f"  status={lr.get('status')} count={lr.get('count')}")
for it in lr.get("reports", []):
    print(f"    {it.get('report_id')}  review={it.get('review_id')}  "
          f"改动={it.get('total_changes')} 高危={it.get('high_risk_count')}")
    print(f"      {it.get('summary', '')[:70]}")

if lr.get("count", 0) == 0:
    print()
    print(f"  ★ 报告没有落盘到 {m.REVIEWS_DIR}")
    print("    json_path 字段:", r.get("json_path"))
    print("    这类问题只测'调用成功'是发现不了的：")
    print("    函数正常返回，但报告存到了别处，后续查询全都落空。")

print()
print("=" * 70)
print("4. get_report（单份报告读取）")
print("=" * 70)
rid = r.get("report_id") or ""
if rid:
    gr = m.get_report(rid)
    print(f"  status={gr.get('status')}")
    rep = gr.get("report") or {}
    if rep:
        print(f"  report_id={rep.get('report_id')}")
        print(f"  改动数={rep.get('total_changes')} 高危={rep.get('high_risk_count')}")
        print(f"  建议数={len(rep.get('template_suggestions') or [])}")
        print(f"  llm_analysis 长度={len(rep.get('llm_analysis') or '')}")
        for s in (rep.get("template_suggestions") or [])[:3]:
            print(f"    建议：{s[:80]}")
    else:
        print(f"  ★ 读不到内容：{str(gr)[:200]}")
else:
    print("  (无 report_id，跳过)")

print()
print("=" * 70)
print("5. 提示词资源")
print("=" * 70)
p = m.review_changes("")
print("  " + p[:150].replace("\n", "\n  "))
print()
p2 = m.review_changes(rid) if rid else ""
print(f"  带 report_id 的提示词长度：{len(p2)}")
if rid and len(p2) < 200:
    print("  ★ 带 report_id 的提示词过短，可能没把报告内容嵌进去")