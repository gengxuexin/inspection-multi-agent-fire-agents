# ~/fire_agents/gateway.py
"""
消防智能执法 - 总控 Agent（Gateway）
用户说一句自然语言，自动路由到对应 Agent

生产化改造：
  - JWT 鉴权（所有 /dispatch* 接口挂 Depends(get_current_user)）
  - 审计日志（请求/响应摘要脱敏写入 logs/audit.jsonl）
  - 级联路由（L1 规则 → L2 分类器 → L3 LLM 兜底，common.router.route）
  - 熔断降级（各 Agent 调用包 protected_call，三态机+指数退避+降级路径）
  - 异步任务（长任务走 /dispatch_async + /tasks/{id} 轮询 + /tasks/{id}/events SSE）
  - DRY_RUN 模式（DRY_RUN=1 时只返回路由决策不转发，联调用）
  - Agent URL 从 .env 读取（AGENT1_URL / AGENT2_URL / AGENT3_URL）
"""
import os
import sys
import asyncio
import json
import logging
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import List

from fastapi import FastAPI, Depends, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import uvicorn
import tempfile
import shutil

from common.config import get_settings
from common.auth import get_current_user, UserCtx, create_token
from common.audit import install_audit_middleware, audit_event
from common.router import route, build_route_envelope
from common.resilience import protected_call, CircuitOpenError
from common.tasks import get_task_store, new_task_id
from common.tracing import setup_tracing, trace_span
from common.metrics_middleware import (
    install_http_middleware,
    install_metrics_endpoint,
)


app = FastAPI(title="Fire Agent Gateway")

# ---- Prometheus 指标（指标清单见 common/metrics.py）----
install_metrics_endpoint(app, service="gateway")
install_http_middleware(app, service="gateway")

logger = logging.getLogger("fire_agents.gateway")

# 挂载审计中间件 + OpenTelemetry tracing
install_audit_middleware(app)
setup_tracing(app)
# ==================== 配置 ====================
_settings = get_settings()
AGENT1 = _settings.agent1_url
AGENT2 = _settings.agent2_url
AGENT3 = _settings.agent3_url
DRY_RUN = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")


def _internal_headers() -> dict:
    """构造 Agent 间内部调用头（X-Internal-Token）。"""
    token = _settings.agent_shared_token
    return {"X-Internal-Token": token} if token else {}

# ==================== 请求模型 ====================
class DispatchReq(BaseModel):
    query: str                             # 用户自然语言
    file_path: str = ""                    # 可选：图片/视频/音频的本地路径
    force_kind: str = ""                   # 可选：强制指定 rag/vision/voice/doc
    conversation_id: str = ""              # 会话 ID（每轮重新路由，支持话题切换）


class TokenReq(BaseModel):
    sub: str = "test_user"
    dept: str = "消防支队"
    role: str = "officer"
    clearance: int = 2

# ==================== 降级函数 ====================
async def rag_fallback(*args, **kwargs):
    return {"status": "degraded", "message": "法规 RAG 服务暂不可用，降级为直通 LLM",
            "fallback": "direct_llm"}

async def voice_fallback(*args, **kwargs):
    return {"status": "degraded", "message": "语音服务暂不可用，降级为仅文本回复",
            "fallback": "transcribe_only"}

async def vision_fallback(*args, **kwargs):
    return {"status": "degraded", "message": "视觉服务暂不可用，降级为仅 OCR",
            "fallback": "ocr_only"}

async def doc_fallback(*args, **kwargs):
    return {"status": "degraded", "message": "文书服务暂不可用，降级为法规 RAG",
            "fallback": "rag"}

# ==================== 各路由处理器（带熔断+降级）====================
async def handle_rag(query: str, top_k: int = 8) -> dict:
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post(f"{AGENT1}/api/rag/ask",
                         json={"question": query, "top_k": top_k},
                         headers=_internal_headers())
        return r.json()


async def handle_voice(query: str, file_path: str) -> dict:
    if not file_path:
        return {"status": "error",
                "message": "语音路由需要 file_path 参数（本地音频路径）"}
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(f"{AGENT2}/api/voice/qa",
                         json={"audio_path": file_path,
                               "voice": "zh-CN-XiaoxiaoNeural",
                               "emotion": "正常"},
                         headers=_internal_headers())
        return r.json()


async def handle_vision(query: str, file_path: str) -> dict:
    if not file_path:
        return {"status": "error",
                "message": "视觉路由需要 file_path 参数（本地图片/视频路径）"}
    is_video = file_path.lower().endswith(
        (".mp4", ".avi", ".mkv", ".mov", ".flv")
    )
    async with httpx.AsyncClient(timeout=900) as c:
        if is_video:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f)}
                data = {"segment": "4", "fps": "1", "do_yolo": "false"}
                r = await c.post(f"{AGENT3}/api/vision/analyze_video",
                                 files=files, data=data,
                                 headers=_internal_headers())
        else:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f)}
                data = {"do_yolo": "true", "do_ocr": "true"}
                r = await c.post(f"{AGENT3}/api/vision/analyze_image",
                                 files=files, data=data,
                                 headers=_internal_headers())
        return r.json()


async def handle_doc(query: str) -> dict:
    tpl_map = {
        "通知书": "责令限期改正通知书",
        "审批表": "立案审批表",
        "登记表": "立案登记表",
        "笔录": "询问笔录",
        "决定书": "处罚决定书",
        "告知": "告知笔录",
        "检查记录": "消防监督检查记录",
    }
    templates = []
    for kw, tpl in tpl_map.items():
        if kw in query:
            templates.append(tpl)
    if not templates:
        templates = ["消防监督检查记录"]

    async with httpx.AsyncClient(timeout=900) as c:
        r = await c.post(f"{AGENT1}/api/doc/generate",
                         json={"templates": templates, "also_pdf": False},
                         headers=_internal_headers())
        return r.json()

# ==================== 路由+执行核心 ====================
@trace_span("gateway.dispatch")
async def _route_and_execute(req: DispatchReq) -> dict:
    """级联路由 + 熔断降级执行，返回带路由信封的结果。"""
    decision = route(req.query, req.file_path)
    kind = req.force_kind or decision.kind

    if kind == "vision":
        result = await protected_call(handle_vision, req.query, req.file_path,
                                      name="vision", fallback=vision_fallback)
    elif kind == "voice":
        result = await protected_call(handle_voice, req.query, req.file_path,
                                      name="voice", fallback=voice_fallback)
    elif kind == "doc":
        result = await protected_call(handle_doc, req.query,
                                      name="doc", fallback=doc_fallback)
    else:
        result = await protected_call(handle_rag, req.query,
                                      name="rag", fallback=rag_fallback)

    envelope = build_route_envelope(decision, req.query, req.file_path)
    return {"status": "success", "routed_to": kind,
            "route": envelope, "result": result}

# ==================== 主入口 ====================
@app.get("/")
def root():
    return {"service": "Fire Agent Gateway", "status": "ok",
            "agents": {"rag": AGENT1, "voice": AGENT2, "vision": AGENT3}}

@app.get("/agents/health")
async def health():
    results = {}
    async with httpx.AsyncClient(timeout=5) as c:
        for name, url in [("agent1_rag", AGENT1),
                          ("agent2_voice", AGENT2),
                          ("agent3_vision", AGENT3)]:
            try:
                r = await c.get(url)
                results[name] = "ok" if r.status_code == 200 else f"http {r.status_code}"
            except Exception as e:
                results[name] = f"error: {type(e).__name__}"
    return results

# ==================== Token 签发（联调用）====================
@app.post("/auth/token")
def issue_token(req: TokenReq):
    """签发 JWT（联调/测试用，生产环境应走 OIDC IdP）。"""
    try:
        token = create_token(req.sub, req.dept, req.role, req.clearance)
        return {"access_token": token, "token_type": "Bearer"}
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}

# ==================== 同步 dispatch ====================
@app.post("/dispatch")
async def dispatch(req: DispatchReq, user: UserCtx = Depends(get_current_user)):
    """统一入口：级联路由 + 熔断降级 + 路由信封。"""
    if DRY_RUN:
        decision = route(req.query, req.file_path)
        envelope = build_route_envelope(decision, req.query, req.file_path)
        return {"status": "dry_run", "route": envelope,
                "routed_to": decision.kind}

    try:
        result = await _route_and_execute(req)
        audit_event("dispatch", user=user,
                    routed_to=result.get("routed_to"),
                    query=req.query[:100])
        return result
    except CircuitOpenError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.exception("[gateway] 请求处理异常")
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}

# ==================== 文件上传 dispatch ====================
@app.post("/dispatch_file")
async def dispatch_file(
    file: UploadFile = File(...),
    query: str = Form(""),
    force_kind: str = Form(""),
    user: UserCtx = Depends(get_current_user),
):
    """接收上传文件，级联路由 + 熔断降级。"""
    tmp_dir = os.path.join(tempfile.gettempdir(), "fire_uploads")
    os.makedirs(tmp_dir, exist_ok=True)

    safe_name = f"fire_{int(time.time())}_{file.filename}"
    tmp_path = os.path.join(tmp_dir, safe_name)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 级联路由
    decision = route(query, tmp_path)
    kind = force_kind or decision.kind

    # 文件后缀覆盖（后缀最可信）
    ext = os.path.splitext(safe_name)[1].lower()
    if ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"):
        kind = "voice"
    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp",
                 ".mp4", ".avi", ".mkv", ".mov", ".flv"):
        kind = "vision"

    envelope = build_route_envelope(decision, query, tmp_path)

    if DRY_RUN:
        return {"status": "dry_run", "route": envelope, "routed_to": kind}

    try:
        if kind == "vision":
            result = await protected_call(handle_vision, query, tmp_path,
                                          name="vision", fallback=vision_fallback)
        elif kind == "voice":
            result = await protected_call(handle_voice, query, tmp_path,
                                          name="voice", fallback=voice_fallback)
        else:
            return {"status": "error", "message": f"未知 kind: {kind}"}

        audit_event("dispatch_file", user=user, routed_to=kind,
                    filename=safe_name)
        return {"status": "success", "routed_to": kind,
                "temp_path": tmp_path, "route": envelope, "result": result}
    except CircuitOpenError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.exception("[gateway] 请求处理异常")
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}

# ==================== 异步 dispatch（长任务）====================
@app.post("/dispatch_async")
async def dispatch_async(req: DispatchReq,
                         user: UserCtx = Depends(get_current_user)):
    """异步入口：返回 task_id，后台执行，通过 /tasks/{id} 轮询或 SSE。"""
    task_id = new_task_id()
    store = get_task_store()
    await store.create(task_id, {"query": req.query[:200],
                                 "file_path": req.file_path,
                                 "conversation_id": req.conversation_id})
    asyncio.create_task(_run_async_task(task_id, req))
    audit_event("dispatch_async", user=user, task_id=task_id,
                query=req.query[:100])
    return {"status": "accepted", "task_id": task_id}


async def _run_async_task(task_id: str, req: DispatchReq):
    store = get_task_store()
    try:
        await store.set_running(task_id)
        await store.add_event(task_id, "route_start")
        result = await _route_and_execute(req)
        await store.save_result(task_id, result)
        await store.add_event(task_id, "done",
                              {"routed_to": result.get("routed_to")})
    except Exception as e:
        await store.set_error(task_id, f"{type(e).__name__}: {e}")
        await store.add_event(task_id, "error", {"message": str(e)})

@app.get("/tasks/{task_id}")
async def get_task(task_id: str,
                   user: UserCtx = Depends(get_current_user)):
    """查询任务状态。"""
    store = get_task_store()
    rec = await store.get(task_id)
    if rec is None:
        return {"status": "error", "message": f"任务不存在：{task_id}"}
    return rec

@app.get("/tasks/{task_id}/events")
async def task_events(task_id: str,
                      user: UserCtx = Depends(get_current_user)):
    """SSE 事件流：实时推送任务状态变化。"""
    store = get_task_store()

    async def event_gen():
        last_idx = 0
        while True:
            rec = await store.get(task_id)
            if rec is None:
                yield f"data: {json.dumps({'error': 'task not found'})}\n\n"
                break
            events = rec.get("events", [])
            for ev in events[last_idx:]:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                last_idx += 1
            if rec["status"] in ("success", "failed"):
                yield f"data: {json.dumps({'status': rec['status']})}\n\n"
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_gen(), media_type="text/event-stream")

# ==================== 多模态综合处理 ====================
def _resolve_orchestrator():
    """按 ORCHESTRATOR_MODE 选择编排器实现。

    统一入口签名：orchestrate(file_paths, template_path, user_query, conversation_id) -> dict

    - langgraph（默认，生产）：orchestrator_langgraph.orchestrate
    - legacy：multimodal_orchestrator.orchestrate（仅当 langgraph 不可用时回退）
    """
    settings = get_settings()
    mode = (settings.orchestrator_mode or "langgraph").lower()

    if mode == "legacy":
        try:
            from multimodal_orchestrator import orchestrate as legacy_orchestrate
            return legacy_orchestrate, "legacy"
        except ImportError:
            logger.warning("[orchestrator] legacy 模式不可用，回退 langgraph")

    from orchestrator_langgraph import orchestrate as lg_orchestrate, is_available

    if not is_available():
        raise RuntimeError(
            "ORCHESTRATOR_MODE=langgraph 但 langgraph 未安装，"
            "请执行 pip install 'langgraph>=0.2' 'langgraph-checkpoint-sqlite>=2.0'"
        )
    return lg_orchestrate, "langgraph"

@app.post("/dispatch_multimodal")
async def dispatch_multimodal(
    files: List[UploadFile] = File(default=[]),
    template_file: UploadFile = File(default=None),
    query: str = Form(""),
    conversation_id: str = Form(""),
    user: UserCtx = Depends(get_current_user),
):
    """多模态入口：上传任意文件，自动分发+综合+生成。"""
    orchestrate, mode = _resolve_orchestrator()
    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_paths = []
        template_path = None

        for f in files:
            p = os.path.join(tmp_dir, f.filename)
            with open(p, "wb") as fp:
                shutil.copyfileobj(f.file, fp)
            tmp_paths.append(p)

        if template_file and template_file.filename:
            template_path = os.path.join(tmp_dir, template_file.filename)
            with open(template_path, "wb") as fp:
                shutil.copyfileobj(template_file.file, fp)

        result = await orchestrate(tmp_paths, template_path, query, conversation_id)

        audit_event("dispatch_multimodal", user=user,
                    files=len(tmp_paths), query=query[:100], orchestrator=mode)
        return {"status": "success", "orchestrator": mode, **result}

    except Exception as e:
        logger.exception("[dispatch_multimodal] 失败")
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
