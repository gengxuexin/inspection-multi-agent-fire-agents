# orchestrator_langgraph.py
"""
LangGraph 状态图编排器（生产版）

状态图：
    START → classify → [vision | voice | doc_extract]（并行）→ synthesize
          → extract_fields → doc_draft → review_submit → report → END

条件边：
    - classify 后按文件类型 fan-out 到对应模态节点（无文件则直达 synthesize）
    - synthesize 后：无模板/预算超限 → 直达 report（degraded）
    - doc_draft 后：生成失败 → 直达 report（degraded）

特性：
    - SqliteSaver checkpoint（logs/checkpoints.sqlite），thread_id=conversation_id
    - BudgetTracker 每节点入口校验
    - trace_span 包裹每个节点
    - ORCHESTRATOR_MODE=langgraph 时由 gateway/server 调用
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Annotated, TypedDict

import operator
import httpx

# ---- sys.path ----
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.config import get_settings
from common.budget import BudgetTracker, BudgetExceededError
from common.tracing import trace_span, get_langfuse_logger

logger = logging.getLogger("fire_agents.orchestrator")

# ---- LangGraph 可用性探测 ----
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.sqlite import SqliteSaver
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    logger.warning("[orchestrator] langgraph 未安装，ORCHESTRATOR_MODE=langgraph 不可用")

# ==================== 文件类型 ====================
VISION_IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")
VISION_VID = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm")
VOICE_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")
DOC_EXT = (".pdf", ".docx", ".doc", ".txt", ".md")


# ==================== State 定义 ====================
class State(TypedDict):
    query: str
    conversation_id: str
    file_paths: list[str]
    template_path: str
    grouped: dict
    vision_results: list
    voice_results: list
    doc_results: list
    case_summary: str
    case_dict: dict
    review_id: str
    report_md: str
    usage: dict
    hops: int
    errors: Annotated[list[str], operator.add]
    started_at: float


# ==================== 全局单例 ====================
_budget_tracker: BudgetTracker | None = None
_graph_instance = None
_checkpointer = None


def _get_budget() -> BudgetTracker:
    global _budget_tracker
    if _budget_tracker is None:
        s = get_settings()
        _budget_tracker = BudgetTracker(
            max_tokens=s.budget_max_tokens or None,
            max_cost=s.budget_max_cost or None,
        )
    return _budget_tracker


def _internal_headers() -> dict:
    s = get_settings()
    headers = {}
    if s.agent_shared_token:
        headers["X-Internal-Token"] = s.agent_shared_token
    return headers


def _get_checkpointer():
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    log_dir = get_settings().log_dir
    os.makedirs(log_dir, exist_ok=True)
    db_path = os.path.join(log_dir, "checkpoints.sqlite")
    _checkpointer = SqliteSaver.from_conn_string(db_path)
    return _checkpointer


# ==================== 辅助：Agent 调用 ====================
async def _call_vision(img_path: str, is_video: bool) -> dict:
    s = get_settings()
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            with open(img_path, "rb") as f:
                files = {"file": (os.path.basename(img_path), f)}
                if is_video:
                    data = {"segment": "4", "fps": "1", "do_yolo": "false"}
                    r = await client.post(
                        f"{s.agent3_url}/api/vision/analyze_video",
                        files=files, data=data, headers=_internal_headers(),
                        timeout=900,
                    )
                else:
                    data = {"do_yolo": "true", "do_ocr": "true"}
                    r = await client.post(
                        f"{s.agent3_url}/api/vision/analyze_image",
                        files=files, data=data, headers=_internal_headers(),
                        timeout=600,
                    )
        return {"file": os.path.basename(img_path), "result": r.json()}
    except Exception as e:
        return {"file": os.path.basename(img_path),
                "error": f"{type(e).__name__}: {e}"}


async def _call_voice(audio_path: str) -> dict:
    s = get_settings()
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f)}
                r = await client.post(
                    f"{s.agent2_url}/api/voice/transcribe",
                    files=files, data={"engine": "whisper"},
                    headers=_internal_headers(), timeout=900,
                )
        return {"file": os.path.basename(audio_path), "result": r.json()}
    except Exception as e:
        return {"file": os.path.basename(audio_path),
                "error": f"{type(e).__name__}: {e}"}


async def _call_doc_extract(doc_path: str) -> dict:
    s = get_settings()
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            with open(doc_path, "rb") as f:
                files = {"file": (os.path.basename(doc_path), f)}
                r = await client.post(
                    f"{s.agent1_url}/api/case/extract",
                    files=files, headers=_internal_headers(), timeout=300,
                )
        return {"file": os.path.basename(doc_path), "result": r.json()}
    except Exception as e:
        return {"file": os.path.basename(doc_path),
                "error": f"{type(e).__name__}: {e}"}


async def _call_doc_draft(template_path: str, case_dict: dict) -> dict:
    s = get_settings()
    try:
        async with httpx.AsyncClient(trust_env=False, timeout=900) as client:
            with open(template_path, "rb") as tf:
                files = {"template_file": (os.path.basename(template_path), tf)}
                data = {"case_dict_json": json.dumps(case_dict, ensure_ascii=False)}
                r = await client.post(
                    f"{s.agent1_url}/api/doc/generate_from_dict",
                    files=files, data=data, headers=_internal_headers(),
                )
            return r.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ==================== 辅助：LLM 综合 ====================
SYNTH_PROMPT = """你是消防案件分析助手。请根据以下多模态分析结果，综合出一份连贯的案情描述。

【用户文字输入】
{query}

【视觉分析结果】
{vision_text}

【录音转写结果】
{voice_text}

【文档抽取结果】
{doc_text}

要求：
1. 按时间线组织案情
2. 包含时间、地点、当事人、行为、后果等要素
3. 客观描述，不要编造
4. 输出一段完整的案情描述（300-800字）
"""


def _llm_synthesize(vision_results, voice_results, doc_results, query) -> str:
    """用 DeepSeek LLM 综合多模态结果为案情描述。"""
    parts = []
    for v in vision_results:
        if "error" in v:
            parts.append(f"[{v['file']}] 分析失败: {v['error']}")
        else:
            res = v.get("result", {}).get("result", {})
            yolo = res.get("yolo", {}).get("counts", {})
            ocr = res.get("ocr", {}).get("ocr_texts", [])
            judgment = res.get("judgment", {})
            lines = [f"[{v['file']}]"]
            if yolo:
                lines.append(f"  识别: {yolo}")
            if ocr:
                lines.append(f"  OCR: {' / '.join(ocr[:5])}")
            if judgment.get("overall_conclusion"):
                lines.append(f"  结论: {judgment['overall_conclusion']}")
            parts.append("\n".join(lines))

    for v in voice_results:
        if "error" in v:
            parts.append(f"[{v['file']}] 转写失败")
        else:
            text = v.get("result", {}).get("text", "")
            parts.append(f"[{v['file']}]\n  {text}")

    for v in doc_results:
        if "error" in v:
            parts.append(f"[{v['file']}] 抽取失败")
        else:
            parts.append(f"[{v['file']}]\n  {json.dumps(v['result'], ensure_ascii=False)[:500]}")

    vision_text = "\n\n".join(p for p in parts[:len(vision_results)] if p)
    voice_text = "\n\n".join(
        p for p in parts[len(vision_results):len(vision_results)+len(voice_results)])
    doc_text = "\n\n".join(p for p in parts[len(vision_results)+len(voice_results):])

    prompt = SYNTH_PROMPT.format(
        query=query or "（无）",
        vision_text=vision_text or "（无）",
        voice_text=voice_text or "（无）",
        doc_text=doc_text or "（无）",
    )

    s = get_settings()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url,
                        timeout=300)
        resp = client.chat.completions.create(
            model=s.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, timeout=300,
        )
        text = resp.choices[0].message.content.strip()
        usage = resp.usage
        if usage:
            _get_budget().consume(tokens=usage.total_tokens)
            get_langfuse_logger().log_llm_call(
                "synthesize", prompt, text,
                usage={"prompt": usage.prompt_tokens,
                       "completion": usage.completion_tokens,
                       "total": usage.total_tokens},
            )
        return text
    except Exception as e:
        logger.error("[orchestrator] LLM 综合失败: %s", e)
        # 回退到简单拼接
        return f"{vision_text}\n\n{voice_text}\n\n{doc_text}"


# ==================== 节点 ====================
@trace_span("orch.classify")
async def classify_node(state: State) -> dict:
    grouped = {"vision": [], "voice": [], "doc": []}
    for p in state.get("file_paths", []):
        ext = os.path.splitext(p)[1].lower()
        if ext in VISION_IMG or ext in VISION_VID:
            grouped["vision"].append(p)
        elif ext in VOICE_EXT:
            grouped["voice"].append(p)
        elif ext in DOC_EXT:
            grouped["doc"].append(p)
        else:
            grouped["doc"].append(p)
    logger.info("[orch.classify] vision=%d voice=%d doc=%d",
                len(grouped["vision"]), len(grouped["voice"]), len(grouped["doc"]))
    return {"grouped": grouped}


@trace_span("orch.vision")
async def vision_node(state: State) -> dict:
    grouped = state.get("grouped", {})
    results = []
    for p in grouped.get("vision", []):
        ext = os.path.splitext(p)[1].lower()
        results.append(await _call_vision(p, ext in VISION_VID))
    return {"vision_results": results}


@trace_span("orch.voice")
async def voice_node(state: State) -> dict:
    grouped = state.get("grouped", {})
    results = []
    for p in grouped.get("voice", []):
        results.append(await _call_voice(p))
    return {"voice_results": results}


@trace_span("orch.doc_extract")
async def doc_extract_node(state: State) -> dict:
    grouped = state.get("grouped", {})
    results = []
    for p in grouped.get("doc", []):
        results.append(await _call_doc_extract(p))
    return {"doc_results": results}


@trace_span("orch.synthesize")
async def synthesize_node(state: State) -> dict:
    budget = _get_budget()
    if not budget.check():
        return {"case_summary": "（预算超限，跳过综合）",
                "errors": ["预算超限，synthesize 降级"]}

    summary = _llm_synthesize(
        state.get("vision_results", []),
        state.get("voice_results", []),
        state.get("doc_results", []),
        state.get("query", ""),
    )
    case_id = "auto_" + str(int(time.time()))
    return {"case_summary": summary,
            "case_dict": {"case_context": summary, "case_id": case_id}}


@trace_span("orch.extract_fields")
async def extract_fields_node(state: State) -> dict:
    """准备 case_dict（字段抽取在 agent1 generate_from_dict 内部完成）。"""
    case_dict = state.get("case_dict", {})
    if not case_dict:
        summary = state.get("case_summary", "")
        case_dict = {"case_context": summary,
                     "case_id": "auto_" + str(int(time.time()))}
    return {"case_dict": case_dict}


@trace_span("orch.doc_draft")
async def doc_draft_node(state: State) -> dict:
    template_path = state.get("template_path", "")
    if not template_path or not template_path.lower().endswith(".docx"):
        return {"errors": ["模板缺失或非 .docx，跳过文书生成"],
                "review_id": ""}

    budget = _get_budget()
    if not budget.check():
        return {"errors": ["预算超限，跳过文书生成"], "review_id": ""}

    case_dict = state.get("case_dict", {})
    result = await _call_doc_draft(template_path, case_dict)

    if result.get("status") == "success":
        review_id = result.get("review_id", "")
        return {"review_id": review_id}
    return {"errors": [f"文书生成失败: {result.get('error', result)}"],
            "review_id": ""}


@trace_span("orch.review_submit")
async def review_submit_node(state: State) -> dict:
    review_id = state.get("review_id", "")
    if review_id:
        logger.info("[orch.review_submit] 草稿 review_id=%s 等待审批", review_id)
    return {}


@trace_span("orch.report")
async def report_node(state: State) -> dict:
    elapsed = time.time() - state.get("started_at", time.time())
    report = _build_report(
        state.get("case_summary", ""),
        state.get("vision_results", []),
        state.get("voice_results", []),
        state.get("doc_results", []),
        state.get("review_id", ""),
        elapsed,
        state.get("errors", []),
    )
    return {"report_md": report}


# ==================== 条件边 ====================
def route_after_classify(state: State) -> list[str]:
    """按文件类型 fan-out；无文件则直达 synthesize。"""
    grouped = state.get("grouped", {})
    nodes = []
    if grouped.get("vision"):
        nodes.append("vision")
    if grouped.get("voice"):
        nodes.append("voice")
    if grouped.get("doc"):
        nodes.append("doc_extract")
    if not nodes:
        nodes.append("synthesize")
    return nodes


def route_after_synthesize(state: State) -> str:
    """无模板/预算超限 → report（degraded）；否则 → extract_fields。"""
    template = state.get("template_path", "")
    if not template or not template.lower().endswith(".docx"):
        return "report"
    if not _get_budget().check():
        return "report"
    return "extract_fields"


def route_after_doc_draft(state: State) -> str:
    """生成失败 → report（degraded）；否则 → review_submit。"""
    if not state.get("review_id"):
        return "report"
    return "review_submit"


# ==================== 报告生成 ====================
def _build_report(case_summary, vision_results, voice_results,
                  doc_results, review_id, elapsed, errors):
    lines = [
        "# 消防案件综合分析报告",
        f"\n生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"处理耗时：{elapsed:.1f} 秒",
    ]
    if errors:
        lines.append(f"\n⚠️ 降级/错误：{len(errors)} 条")
        for e in errors[:5]:
            lines.append(f"  - {e}")

    lines.append("\n---\n## 一、案情摘要\n")
    lines.append(case_summary or "（无）")

    lines.append("\n---\n## 二、视觉分析明细\n")
    for v in vision_results:
        lines.append(f"### {v.get('file', '?')}\n")
        if "error" in v:
            lines.append(f"❌ {v['error']}\n")
        else:
            res = v.get("result", {}).get("result", {})
            lines.append(f"- YOLO：{res.get('yolo', {}).get('counts', {})}")
            lines.append(f"- OCR：{res.get('ocr', {}).get('ocr_texts', [])}")
            lines.append(f"- 结论：{res.get('judgment', {}).get('overall_conclusion', '')}\n")

    lines.append("\n---\n## 三、录音转写明细\n")
    for v in voice_results:
        if "error" in v:
            lines.append(f"- {v.get('file', '?')}：❌ {v['error']}")
        else:
            lines.append(f"- {v.get('file', '?')}：{v.get('result', {}).get('text', '')}")

    lines.append("\n---\n## 四、文档抽取明细\n")
    for v in doc_results:
        if "error" in v:
            lines.append(f"- {v.get('file', '?')}：❌ {v['error']}")
        else:
            lines.append(f"- {v.get('file', '?')}：{json.dumps(v.get('result', {}), ensure_ascii=False)[:300]}")

    if review_id:
        lines.append(f"\n---\n## 五、文书草稿\n\n✅ review_id: `{review_id}`（待审批）")
    else:
        lines.append("\n---\n## 五、文书草稿\n\n⚠️ 未生成（模板缺失或降级）")

    budget = _get_budget()
    lines.append(f"\n---\n## 六、预算\n\n{budget.to_summary()}")
    return "\n".join(lines)


# ==================== 图构建 ====================
def build_graph():
    """构建并编译 LangGraph 状态图。"""
    if not _LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph 未安装，请 pip install langgraph langgraph-checkpoint-sqlite")

    builder = StateGraph(State)

    builder.add_node("classify", classify_node)
    builder.add_node("vision", vision_node)
    builder.add_node("voice", voice_node)
    builder.add_node("doc_extract", doc_extract_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("extract_fields", extract_fields_node)
    builder.add_node("doc_draft", doc_draft_node)
    builder.add_node("review_submit", review_submit_node)
    builder.add_node("report", report_node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classify)
    # 各模态节点 → synthesize（fan-in）
    builder.add_edge("vision", "synthesize")
    builder.add_edge("voice", "synthesize")
    builder.add_edge("doc_extract", "synthesize")
    builder.add_conditional_edges("synthesize", route_after_synthesize)
    builder.add_edge("extract_fields", "doc_draft")
    builder.add_conditional_edges("doc_draft", route_after_doc_draft)
    builder.add_edge("review_submit", "report")
    builder.add_edge("report", END)

    checkpointer = _get_checkpointer()
    return builder.compile(checkpointer=checkpointer)


def _get_graph():
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance


# ==================== 入口 ====================
async def orchestrate(
    file_paths: list[str],
    template_path: str = "",
    user_query: str = "",
    conversation_id: str = "",
) -> dict:
    """LangGraph 编排入口。

    :return: {case_dict, case_summary, review_id, report_md, elapsed_sec, stats, usage, errors}
    """
    if not _LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph 未安装")

    conv_id = conversation_id or f"task_{int(time.time())}"
    initial_state: State = {
        "query": user_query,
        "conversation_id": conv_id,
        "file_paths": file_paths,
        "template_path": template_path,
        "grouped": {},
        "vision_results": [],
        "voice_results": [],
        "doc_results": [],
        "case_summary": "",
        "case_dict": {},
        "review_id": "",
        "report_md": "",
        "usage": {},
        "hops": 0,
        "errors": [],
        "started_at": time.time(),
    }

    config = {"configurable": {"thread_id": conv_id}}
    graph = _get_graph()

    logger.info("[orchestrate] 开始编排 conversation_id=%s files=%d",
                conv_id, len(file_paths))

    final_state = await graph.ainvoke(initial_state, config=config)

    grouped = final_state.get("grouped", {})
    return {
        "case_dict": final_state.get("case_dict", {}),
        "case_summary": final_state.get("case_summary", ""),
        "review_id": final_state.get("review_id", ""),
        "report_md": final_state.get("report_md", ""),
        "elapsed_sec": round(time.time() - initial_state["started_at"], 1),
        "stats": {
            "vision_files": len(grouped.get("vision", [])),
            "voice_files": len(grouped.get("voice", [])),
            "doc_files": len(grouped.get("doc", [])),
        },
        "usage": _get_budget().usage(),
        "errors": final_state.get("errors", []),
    }


def is_available() -> bool:
    """LangGraph 编排器是否可用。"""
    return _LANGGRAPH_AVAILABLE
