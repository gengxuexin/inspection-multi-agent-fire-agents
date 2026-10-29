#!/usr/bin/env python3
"""
消防文书改动检测 · MCP Server

把 agent1/change_detector.py 的人工改动检测能力封装成 MCP 工具，
让 Claude Desktop / Cursor / 任意 MCP 客户端都能直接调用，
而不是只能在本项目内部通过 HTTP 触发。

对外暴露三组能力：

  1. 工具（tools）—— 可被模型调用的动作
     - compare_documents : 对比两份文书，产出改动分析报告
     - list_reports      : 查历史报告
     - get_report       : 取单份报告详情

  2. 资源（resources）—— 可被读取的上下文
     - reports://index           历史报告索引
     - reports://{report_id}     单份报告的 JSON

  3. 提示（prompts）—— 可复用的分析模板
     - review_changes : 生成一段审查文书改动的提示词

======================================================================
★ MCP SDK 版本坑（必须记下来）：
    FastMCP 在 mcp 2.x 里被重命名为 MCPServer。
    照着网上 v1 的教程写会得到：
        ModuleNotFoundError: No module named 'mcp.server.fastmcp'
        This is mcp 2.x, where FastMCP was renamed to MCPServer
    所以这里用 `from mcp.server.mcpserver import MCPServer`。
    若要兼容 v1，得 pin 'mcp<2' —— 但新写的东西应该跟进新版本。

======================================================================
★ 另一个坑：路径安全策略
    MCPServer 默认 ResourceSecurity(reject_absolute_paths=True,
    reject_path_traversal=True, reject_null_bytes=True)。
    本工具必须读用户给的绝对路径（文书在 /home/... 或 D:/...），
    所以显式放开 reject_absolute_paths，但**保留**路径穿越与空字节防护 ——
    前者会让工具在本场景下完全不可用，后两者是真实的安全边界，不能一起关。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# ---- 让 agent1 内的本地模块可import ----
# ★ 注意这里为什么要指向 agent1/ 而不是项目根：
#   change_detector.py 在 agent1/ 下，且它内部是 `from config import ...`
#   这类平铺导入 —— 只有把 agent1/ 放进 sys.path 才能解析。
#   写成 parents[1]（项目根）会得到 ModuleNotFoundError。
_ROOT = Path(__file__).resolve().parents[1]      # 项目根
_AGENT1 = _ROOT / "agent1"
for _p in (str(_ROOT), str(_AGENT1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mcp.server.mcpserver import MCPServer  # noqa: E402

try:
    from mcp.server.mcpserver import ResourceSecurity
except ImportError:                      # 版本差异兜底
    ResourceSecurity = None              # type: ignore

import change_detector as cd  # noqa: E402

# ---- 配置 ----
# ★ 报告目录必须复用 change_detector 的常量，不能自己写路径。
#   实测踩过这个坑：analyze_changes 把报告写到 output/change_reports/，
#   而这里原本写的是 output/reviews/，结果是
#     - compare_documents 返回 success，还给了 json_path
#     - list_reports 永远返回 0
#     - get_report 永远报「找不到报告」
#   调用成功 ≠ 功能可用 —— 报告存到了没人读的地方，整条链路就是废的。
#   复制一份路径字符串会再次漂移，所以这里直接引用它的 REPORT_DIR。
OUTPUT_DIR = _AGENT1 / "output"
REVIEWS_DIR = Path(cd.REPORT_DIR)

server = MCPServer(
    name="fire-change-detector",
    instructions=(
        "消防文书改动检测工具。用于对比机器生成的执法文书与人工修改后的版本，"
        "识别改动点、评估法律风险、给出文书模板改进建议。\n\n"
        "典型用法：compare_documents 对比两份 docx → "
        "拿到的报告里有 high_risk_count 和 template_suggestions，"
        "可以据此提示审核人哪些改动需要复核。"
    ),
    version="1.0.0",
)

if ResourceSecurity is not None:
    # 只放开绝对路径：文书必然在绝对路径下（本项目根目录之外也有）。
    # reject_path_traversal 与 reject_null_bytes 保持开启 ——
    # 这两个是真实的安全边界，一起关掉等于让工具能被用来读任意文件。
    server.resource_security = ResourceSecurity(
        reject_path_traversal=True,
        reject_absolute_paths=False,
        reject_null_bytes=True,
    )


# ==================== 工具 ====================
@server.tool(
    name="compare_documents",
    description=(
        "对比两份执法文书（机器生成稿 vs 人工修改稿），产出改动分析报告。"
        "返回改动数量、高风险改动数、逐条改动内容与风险评级、"
        "以及基于改动模式的文书模板改进建议。"
    ),
)
def compare_documents(
    original_path: str,
    modified_path: str,
    review_id: str = "",
    use_llm: bool = True,
    save_report: bool = True,
) -> dict[str, Any]:
    """对比文书改动。

    :param original_path: 机器生成稿路径（.docx/.pdf/.txt）
    :param modified_path: 人工修改稿路径
    :param review_id:     审批单号，用于关联归档
    :param use_llm:是否调用 LLM 做语义归因。关闭则纯规则分析（快、无 API 成本）
    :param save_report:   是否落盘。不落盘则只返回结果
    """
    try:
        rep = cd.analyze_changes(
            original_path=original_path,
            modified_path=modified_path,
            review_id=review_id,
            use_llm=use_llm,
            save_report=save_report,
        )
    except FileNotFoundError as e:
        return {"status": "error", "error_type": "FileNotFound", "message": str(e)}
    except Exception as e:                       # noqa: BLE001
        # 工具内部异常必须转成结构化错误返回，不能直接抛。
        # MCP 客户端拿到的是「工具执行失败」这种笼统信息时，
        # 无法判断是该改路径还是该改参数，排查成本很高。
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
        }

    return {
        "status": "success",
        "report_id": rep.report_id,
        "review_id": rep.review_id,
        "total_changes": rep.total_changes,
        "high_risk_count": rep.high_risk_count,
        "summary": rep.summary,
        "changes": rep.changes,
        "llm_analysis": rep.llm_analysis,
        "template_suggestions": rep.template_suggestions,
        "json_path": rep.json_path,
        "markdown_path": rep.markdown_path,
    }


@server.tool(
    name="list_reports",
    description="列出历史改动分析报告，按创建时间倒序。可按 review_id 过滤。",
)
def list_reports(review_id: str = "", limit: int = 20) -> dict[str, Any]:
    """列出历史报告。

    :param review_id: 只看某个审批单的报告；留空则全部
    :param limit:     返回条数上限
    """
    if not REVIEWS_DIR.is_dir():
        return {"status": "success", "count": 0, "reports": [],
                "note": f"报告目录不存在：{REVIEWS_DIR}"}

    items: list[dict[str, Any]] = []
    for p in REVIEWS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue                          # 坏文件跳过，不让一个坏文件搞垮整个列表
        if review_id and data.get("review_id") != review_id:
            continue
        items.append({
            "report_id": data.get("report_id", ""),
            "review_id": data.get("review_id", ""),
            "created_at": data.get("created_at", ""),
            "total_changes": data.get("total_changes", 0),
            "high_risk_count": data.get("high_risk_count", 0),
            "summary": data.get("summary", "")[:200],
            "file": str(p),
        })

    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"status": "success", "count": len(items[:limit]),
            "reports": items[:limit]}


@server.tool(
    name="get_report",
    description="取一份改动分析报告的完整内容（含逐条改动明细）。",
)
def get_report(report_id: str) -> dict[str, Any]:
    """取单份报告详情。"""
    safe = cd.safe_filename(report_id, "")
    if not safe:
        return {"status": "error", "message": "报告 ID 不合法"}

    p = REVIEWS_DIR / f"{safe}.json"
    if not p.is_file():
        # 回退：直接按 id 匹配文件名的前缀，兼容旧命名
        cands = list(REVIEWS_DIR.glob(f"*{safe}*.json"))
        if not cands:
            return {"status": "error",
                    "message": f"找不到报告：{report_id}"}
        p = cands[0]

    try:
        return {"status": "success", "report": json.loads(p.read_text(encoding="utf-8"))}
    except (json.JSONDecodeError, OSError) as e:
        return {"status": "error", "message": f"读取报告失败：{e}"}


# ==================== 资源 ====================
@server.resource("reports://index", name="报告索引",
                mime_type="application/json")
def reports_index() -> str:
    """所有历史报告的索引（JSON 文本）。"""
    return json.dumps(list_reports(limit=200), ensure_ascii=False, indent=2)


@server.resource("reports://{report_id}", name="单份报告",
                mime_type="application/json")
def report_detail(report_id: str) -> str:
    """单份报告详情。"""
    return json.dumps(get_report(report_id), ensure_ascii=False, indent=2)


# ==================== 提示 ====================
@server.prompt(name="review_changes",
              description="生成审查文书改动的提示词模板")
def review_changes(report_id: str = "") -> str:
    """审查文书改动的提示词。

    :param report_id: 指定报告；留空则用通用模板
    """
    if report_id:
        rep = get_report(report_id)
        detail = rep.get("report") or rep
        return (
            f"请审查以下执法文书改动报告，指出哪些改动可能影响法律适用正确性，"
            f"以及文书模板本身有哪些地方需要改进。\n\n"
            f"报告内容：\n{json.dumps(detail, ensure_ascii=False, indent=2)}\n\n"
            f"请从以下角度分析：\n"
            f"1. 事实认定是否有变化（时间、地点、主体、违法行为）\n"
            f"2. 法律适用是否正确（条款引用、裁量档次）\n"
            f"3. 程序是否完备（告知、听证、送达）\n"
            f"4. 结论是否与认定一致（尤其罚款金额是否在裁量范围内）\n"
        )
    return (
        "请审查两份执法文书之间的改动，指出法律风险，"
        "并给出文书模板的改进建议。"
        "先用 compare_documents 工具拿到改动报告再进行分析。"
    )


# ==================== 入口 ====================
def main() -> None:
    # stdout 被 MCP 协议占用，任何 print 都会污染 JSON-RPC 帧。
    # 必须走 stderr。这是最常见的 stdio transport 故障原因：
    # 客户端报"解析响应失败"，实际是服务端往 stdout 打了调试信息。
    print("启动 fire-change-detector MCP Server", file=sys.stderr)

    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    print(f"transport={transport}", file=sys.stderr)
    print(f"报告目录：{REVIEWS_DIR}", file=sys.stderr)

    if transport == "stdio":
        server.run(transport="stdio")
    elif transport == "sse":
        server.run(transport="sse")
    elif transport == "http":
        server.run(transport="streamable-http")
    else:
        raise SystemExit(
            f"未知的 MCP_TRANSPORT={transport}；"
            f"可选：stdio / sse / http"
        )


if __name__ == "__main__":
    main()