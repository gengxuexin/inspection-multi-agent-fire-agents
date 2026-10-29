"""修正 case-stories.md 速查表里的指标数字。

★ 为什么要改：
  速查表里写的是早期数据（faithfulness 0.986 / answer_relevancy 0.88 /
  context_precision 0.165），那是几轮调参之前的基线。
  但最新一轮实测（report_rerank_final.json）真实值是：
      faithfulness 0.982 / answer_relevancy 0.698
      context_precision 0.226 / context_recall 0.548

  **面试时记错数字比不记得更糟** —— 对方会顺着追问，
  而说出来的数字如果和自己项目里的报告对不上，一问就露。

  这条本身就是"文档里的数字必须有出处"的一个实例：
  写下来的时候不核实，面试时就成了风险。
  改法不是凭印象改，而是从报告 JSON 里读出来。
"""
import json
import pathlib

p = pathlib.Path("docs/case-stories.md")
src = p.read_text(encoding="utf-8")

# 从真实报告读，不用手写
final = json.loads(
    pathlib.Path("report_rerank_final.json").read_text(encoding="utf-8"))
sc = final["scores"]

OLD = """检索质量    faithfulness 0.986 / answer_relevancy 0.88
            context_precision 0.165（当前瓶颈）/ context_recall 0.51"""

NEW = f"""检索质量    faithfulness {sc['faithfulness']:.3f} / answer_relevancy {sc['answer_relevancy']:.3f}
            context_precision {sc['context_precision']:.3f}（当前瓶颈）
            context_recall {sc['context_recall']:.3f}
            —— 以上出自 report_rerank_final.json（回退后），非早期基线"""

assert OLD in src, "未找到速查表指标行"
src = src.replace(OLD, NEW, 1)

# 补一条数字出处的说明
src = src.replace(
    "> **这些数字要记准。** 面试时记错的数字比不记得更糟——",
    "> ★ **数字必须有出处。** 上面每一项都对应仓库里的具体文件，\n"
    "> 追问时可以直接说\"这个数在 report_rerank_final.json 里\"。\n"
    ">\n"
    "> 我自己写这份速查表时也犯过这个错：\n"
    "> 指标那行凭印象填了早期的 0.88 / 0.165，\n"
    "> 而最新实测是 0.698 / 0.226。**文档里的数字不核实，面试时就成了风险。**\n"
    ">\n"
    "> **这些数字要记准。** 面试时记错的数字比不记得更糟——", 1)

p.write_text(src, encoding="utf-8")
print("✓ 速查表指标已改为报告里的真实值：")
print(f"    faithfulness {sc['faithfulness']:.3f}")
print(f"    answer_relevancy {sc['answer_relevancy']:.3f}")
print(f"    context_precision {sc['context_precision']:.3f}")
print(f"    context_recall {sc['context_recall']:.3f}")