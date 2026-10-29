"""修MCP Server 的报告目录不一致问题。

★ Bug：change_detector.analyze_changes 把报告落盘到
      agent1/output/change_reports/，
  而 MCP 工具的 list_reports / get_report 去 agent1/output/reviews/ 找。
  结果：compare_documents 明明返回成功、还给了 json_path，
  但 list_reports 永远返回 0，get_report 永远报"找不到"。

  这类 bug 只测"函数调用成功"是发现不了的 ——
  调用确实成功了，只是把文件写到了没人读的地方。

修法：以 change_detector 实际使用的目录为准，
     不自己猜路径（猜就是这次出错的根源）。
     并且用同一个常量，避免以后再漂移。
"""
import pathlib
import re

ROOT = pathlib.Path(".").resolve()
TARGET = ROOT / "mcp" / "change_detector_server.py"

src = TARGET.read_text(encoding="utf-8")

# 1) 从 change_detector 里挖出它真正用的目录定义
cd_src = (ROOT / "agent1" / "change_detector.py").read_text(encoding="utf-8")
m = re.findall(r'^[A-Z_]*DIR\w*\s*=.*$', cd_src, re.M)
print("change_detector 里的目录常量：")
for line in m:
    print("   ", line)
print()

# 2) 找出报告输出目录（包含 report 的那个）
report_dir_const = None
for line in m:
    if "report" in line.lower() and "=" in line:
        report_dir_const = line.strip()
        break

NEW_BLOCK = '''# ---- 配置 ----
# ★ 报告目录必须与 change_detector 实际落盘的目录一致，不能自己猜。
#   实测踩过这个坑：analyze_changes 写到 output/change_reports/，
#   而这里原本写的是 output/reviews/，结果 compare_documents
#   返回成功、还给了 json_path，但 list_reports 恒为 0、
#   get_report 恒报"找不到" —— 调用成功 ≠ 功能可用。
#
#   下面直接复用 change_detector 的常量，而不是复制一份路径字符串：
#   复制就会出现两份定义漂移，这正是原 bug 的成因。
OUTPUT_DIR = _AGENT1 / "output"
REVIEWS_DIR = cd.OUTPUT_CHANGE_REPORTS
'''

pattern = re.compile(
    r"# ---- 配置 ----\n(?:.*\n)*?REVIEWS_DIR = .*?\n"
)
if pattern.search(src):
    src = pattern.sub(NEW_BLOCK, src)
    TARGET.write_text(src, encoding="utf-8")
    print(f"已把 REVIEWS_DIR 改为复用 change_detector 的常量")
    print("下一步：需要在 change_detector.py 里导出 OUTPUT_CHANGE_REPORTS")
else:
    print("★ 未匹配到配置块，需要手工检查")
    for i, line in enumerate(src.splitlines()[:80], 1):
        if "REVIEWS_DIR" in line or "OUTPUT_DIR" in line:
            print(f"  {i}: {line}")