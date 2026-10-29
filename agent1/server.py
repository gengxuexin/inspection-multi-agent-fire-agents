# ~/fire_agents/agent1/server.py
"""
Agent1 消防法规 + 文书服务

生产化改造：
  - 内部 Token 校验（X-Internal-Token，所有 /api/* 接口）
  - 审计日志（install_audit_middleware）
  - 文书草稿审批流：generate → output/pending/ + review_id
                    → approve 移入 output/official/
                    → reject 记录原因
"""
import os
import sys
import glob
import json
import time
import uuid
import shutil
import traceback

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
for p in (_SCRIPT_DIR, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, Form, Depends, UploadFile, File
from pydantic import BaseModel
import uvicorn
from starlette.concurrency import run_in_threadpool

from common.auth import require_internal_token
from common.audit import install_audit_middleware, audit_event
from common.tracing import setup_tracing

from rag_module import rag_ask
from rag_legal_advisor import (
    retrieve_legal_context, extract_legal_fields_from_case,
)
from case_pipeline import (
    generate_one, TEMPLATE_DIR, OUTPUT_DIR, CASE as DEFAULT_CASE,
)
from pdf_converter import docx_to_pdf
from common.metrics_middleware import (
    install_http_middleware,
    install_metrics_endpoint,
)


app = FastAPI(title="Agent1 消防法规+文书服务")

# ---- Prometheus 指标（指标清单见 common/metrics.py）----
install_metrics_endpoint(app, service="agent1")
install_http_middleware(app, service="agent1")

# 挂载审计中间件 + tracing
install_audit_middleware(app)
setup_tracing(app)
# ==================== 草稿审批目录 ====================
PENDING_DIR = os.path.join(OUTPUT_DIR, "pending")
OFFICIAL_DIR = os.path.join(OUTPUT_DIR, "official")
REVIEWS_DIR = os.path.join(OUTPUT_DIR, "reviews")
# 原稿快照：机器生成时的原始内容，供审批时与人工改后稿做 diff
ORIGIN_DIR = os.path.join(OUTPUT_DIR, "pending", ".origin")


def ensure_dirs(dirs) -> None:
    """确保目录存在且可写。

    ★ 为什么要处理 PermissionError：
      Docker 里以非 root（uid 10001）运行，而 agent1/output/ 是从宿主机
      挂载进来的，挂载点属主是宿主用户的 uid。容器启动时
      os.makedirs 会直接抛 PermissionError，整个服务起不来。

      但「起不来」不是好的失败方式 —— 表面看是服务崩溃，
      真实原因却是宿主机目录权限，排查方向会被完全带偏。
      实测踩过：单独 docker run（不挂 output/）一切正常，
      一上 compose 就崩，很容易误判成镜像问题。

    这里降级为警告：目录建不出来时服务仍能启动，
    真正需要写文件的接口会返回明确错误，而不是启动即崩。
    """
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
        except PermissionError:
            print(f"[警告] 目录不可写（可能被宿主机挂载覆盖了属主）：{d}")
            print(f"        排查：ls -ld {d}  看属主是否与容器内运行用户一致")
            print(f"        修复：chown -R 10001:10001 {d}  或改用非 root 之外的挂载方案")
        except OSError as e:
            print(f"[警告] 目录创建失败：{d}  {e}")


ensure_dirs((PENDING_DIR, OFFICIAL_DIR, REVIEWS_DIR, ORIGIN_DIR))

# ==================== 请求模型 ====================
class AskReq(BaseModel):
    question: str
    top_k: int = 5

class RetrieveReq(BaseModel):
    query: str
    top_k: int = 8

class LegalField(BaseModel):
    key: str
    label: str = ""

class LegalReq(BaseModel):
    case_text: str
    legal_fields: list[LegalField]
    top_k: int = 8

class DocReq(BaseModel):
    case: dict | None = None    # 案情字典，不传就用内置示例
    templates: list[str] = []   # 指定模板名（不带 .docx），空=全部
    also_pdf: bool = False

# ==================== 基础接口 ====================
@app.get("/")
def root():
    return {"service": "Agent1", "status": "ok"}

@app.get("/health")
def health():
    """存活探针：只回答「进程还在不在」。

   刻意不查任何外部依赖。Dockerfile 的 HEALTHCHECK 打这个端点，
    如果这里去连Redis/Milvus，那么外部依赖一抖容器就被判定 unhealthy、
    被编排器重启—— 但进程本身没有任何问题，重启也修不好。
    """
    return {"service": "Agent1", "status": "ok"}

@app.get("/ready")
def ready():
    """就绪探针：回答「现在能不能接流量」。

    与 /health 分开，是因为这两件事的处置方式完全不同：
    /health 失败要重启，/ready 失败只需要摘流量。
    混成一个端点是最常见的误配——依赖抖动导致服务被反复重启。

    检查项都是本进程内的静态资源，不做网络调用，避免探针本身成为故障源。
    """
    checks = {}
    ok = True

    # 1) 向量库索引是否挂载到位
    idx_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_full")
    idx_ok = (os.path.isfile(os.path.join(idx_dir, "index.faiss"))
              and os.path.isfile(os.path.join(idx_dir, "index.pkl")))
    checks["vector_index"] = "ok" if idx_ok else "missing"
    ok = ok and idx_ok

    # 2) 模板目录是否挂载到位（生成文书要用）
    tpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    tpl_ok = os.path.isdir(tpl_dir) and bool(os.listdir(tpl_dir))
    checks["templates"] = "ok" if tpl_ok else "missing"
    ok = ok and tpl_ok

    # 3) 输出目录是否可写（只读文件系统会在写文书时才炸）
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    try:
        os.makedirs(out_dir, exist_ok=True)
        writable = os.access(out_dir, os.W_OK)
    except OSError:
        writable = False
    checks["output_writable"] = "ok" if writable else "readonly"
    ok = ok and writable

    return {"service": "Agent1", "status": "ok" if ok else "degraded",
            "checks": checks}

# ==================== 运维：模型供给与熔断状态 ====================
# 落地内网/涉密环境时，运维人员需要回答三个问题：
#   1. 本地模型装了吗？连得上吗？   → /api/ops/model/probe
#   2. 检索精度到底多少？要不要换模型？ → /api/ops/model/check
#   3. LLM 服务现在健康吗？          → /api/ops/llm/breakers
# 这三个端点都是只读或低风险操作，供部署与巡检使用。

class ModelTargetReq(BaseModel):
    model: str = ""


@app.get("/api/ops/model/probe")
def ops_model_probe(_: None = Depends(require_internal_token)):
    """检测本地模型环境：Ollama 是否连通、装了哪些模型、目标模型在不在。"""
    from common.model_provisioner import probe_environment, list_local_models
    env = probe_environment()
    env["models_detail"] = list_local_models()
    return env


@app.get("/api/ops/model/check")
def ops_model_check(sample_limit: int = 20,
                    _: None = Depends(require_internal_token)):
    """实测检索精度并给出建议（换模型 / 下载模型 / 排查检索）。"""
    from common.model_provisioner import check_and_advise
    return check_and_advise(sample_limit=sample_limit)


@app.post("/api/ops/model/download")
def ops_model_download(req: ModelTargetReq,
                       _: None = Depends(require_internal_token)):
    """让 Agent 自动下载模型（仅当本机有 ollama CLI 时可行）。

    不可行时返回 manual 指引 —— 内网环境通常走这条路。
    """
    from common.model_provisioner import autodownload_model, build_download_guidance
    target = req.model or "qwen2.5:7b"
    result = autodownload_model(target)
    if not result.get("ok"):
        result["guidance"] = build_download_guidance(target)
    return result


@app.get("/api/ops/model/guidance")
def ops_model_guidance(model: str = "qwen2.5:7b",
                       _: None = Depends(require_internal_token)):
    """返回手动下载指引（在线 / 离线包两种路径）。"""
    from common.model_provisioner import build_download_guidance
    return build_download_guidance(model)


@app.get("/api/ops/llm/breakers")
def ops_llm_breakers(_: None = Depends(require_internal_token)):
    """熔断器状态快照。熔断器 OPEN 说明上游 LLM 正在不可用。"""
    from common.llm_guard import breaker_snapshot
    return {"breakers": breaker_snapshot()}

@app.post("/api/rag/ask")
async def api_ask(req: AskReq, _: None = Depends(require_internal_token)):
    # TODO: phase5 透传 UserCtx 给检索层做权限过滤
    try:
        # ★ rag_ask 是同步阻塞（内含云端 LLM HTTP 调用，可能 30 秒）。
        #   在 async def 里直接调会卡住整个事件循环 ——
        #   一个请求能把该 worker 的所有其它请求一起堵死。
        #   挪到线程池后事件循环保持自由，并发才真正生效。
        answer = await run_in_threadpool(rag_ask, req.question, top_k=req.top_k)
        audit_event("rag_ask", input=req.question[:100])
        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/rag/retrieve")
async def api_retrieve(req: RetrieveReq,
                        _: None = Depends(require_internal_token)):
    try:
        # ★ 同步阻塞（向量检索 + LLM 改写）→ 线程池，避免卡住事件循环
        snippets = await run_in_threadpool(
            retrieve_legal_context, req.query, top_k=req.top_k)
        return {"status": "success", "count": len(snippets),
                "snippets": snippets}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/rag/legal_extract")
async def api_legal(req: LegalReq, _: None = Depends(require_internal_token)):
    try:
        fields = [f.dict() for f in req.legal_fields]
        # ★ 同步阻塞（含 LLM 调用）→ 线程池，避免卡住事件循环
        result = await run_in_threadpool(
            extract_legal_fields_from_case,
            req.case_text, fields, top_k=req.top_k,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==================== 文书生成（草稿+审批流）====================
@app.get("/api/doc/templates")
def list_templates(_: None = Depends(require_internal_token)):
    """列出 templates/ 目录里所有可用的文书模板。"""
    files = sorted(glob.glob(os.path.join(TEMPLATE_DIR, "*.docx")))
    return {
        "status": "success",
        "templates": [os.path.basename(f) for f in files],
        "count": len(files),
    }

@app.post("/api/doc/generate")
def api_generate_doc(req: DocReq, _: None = Depends(require_internal_token)):
    """
    生成文书草稿（写入 output/pending/），返回 review_id。
    需人工 approve 后才移入 output/official/ 成为终稿。
    """
    try:
        case = req.case or DEFAULT_CASE

        if req.templates:
            tpl_files = [
                os.path.join(TEMPLATE_DIR, f"{t}.docx")
                for t in req.templates
            ]
        else:
            tpl_files = sorted(glob.glob(os.path.join(TEMPLATE_DIR, "*.docx")))

        if not tpl_files:
            return {"status": "error", "message": f"模板目录为空：{TEMPLATE_DIR}"}

        review_id = f"rev_{uuid.uuid4().hex[:10]}"
        results = []
        errors = []

        for tpl in tpl_files:
            if not os.path.isfile(tpl):
                errors.append(f"模板不存在：{os.path.basename(tpl)}")
                continue

            tpl_name = os.path.basename(tpl)
            try:
                # 注：本端点是 `def`（非 async），FastAPI 会自动丢到
                # threadpool 执行 —— 这里的同步阻塞调用不会卡住事件循环。
                # （只有 `async def` 端点里的同步调用才是问题。）
                out_path, meta = generate_one(tpl, case)
                if out_path is None:
                    errors.append(f"跳过（无字段）：{tpl_name}")
                    continue

                # 复制草稿到 pending 目录
                pending_path = os.path.join(PENDING_DIR,
                                            f"{review_id}_{tpl_name}")
                shutil.copy2(out_path, pending_path)

                # ★ 存原稿快照 —— 人工修改后就没法还原了，
                #   审批时靠它与改后的稿做 diff
                origin_path = os.path.join(ORIGIN_DIR,
                                           f"{review_id}_{tpl_name}")
                shutil.copy2(out_path, origin_path)

                item = {
                    "template": tpl_name,
                    "draft_path": pending_path,
                    "origin_path": origin_path,
                    "size_bytes": os.path.getsize(pending_path),
                }

                if req.also_pdf:
                    pdf_path = docx_to_pdf(pending_path)
                    if pdf_path:
                        item["pdf"] = pdf_path

                results.append(item)

            except Exception as e:
                errors.append(f"{tpl_name}: {type(e).__name__}: {e}")
                traceback.print_exc()

        # 创建 review 记录
        review_rec = {
            "review_id": review_id,
            "status": "draft",
            "created_at": time.time(),
            "case_id": case.get("case_id", ""),
            "files": results,
            "errors": errors,
        }
        review_path = os.path.join(REVIEWS_DIR, f"{review_id}.json")
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(review_rec, f, ensure_ascii=False, indent=2)

        audit_event("doc_generate_draft", review_id=review_id,
                    templates=len(results))
        return {
            "status": "success",
            "review_id": review_id,
            "generated": len(results),
            "files": results,
            "errors": errors,
            "message": "草稿已生成，需人工 approve 后成为终稿",
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# ==================== 文书审批接口 ====================
@app.get("/api/doc/reviews")
def list_reviews(_: None = Depends(require_internal_token)):
    """列出所有草稿审批记录。"""
    reviews = []
    for f in sorted(glob.glob(os.path.join(REVIEWS_DIR, "*.json"))):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                reviews.append(json.load(fp))
        except Exception:
            pass
    return {"status": "success", "count": len(reviews), "reviews": reviews}

class ApproveReq(BaseModel):
    """审批请求体。

    confirm_high_risk=True 表示调用方（人或 UI）已确认过高风险改动。
    **不传或传 False 时，若检测到高风险改动会被拦截** ——
    这是「拦截」而非「提示」：提示会被忽略，拦截才会生效。
    """
    confirm_high_risk: bool = False


@app.post("/api/doc/review/{review_id}/approve")
def approve_review(review_id: str,
                   req: ApproveReq | None = None,
                   _: None = Depends(require_internal_token)):
    """approve 草稿：检测人工改动 -> 移动文件到 official/ -> 更新状态。

    ★ 改动检测：生成时已把原稿快照存到 output/pending/.origin/，
      此处拿快照与人工改过的稿逐份对比，产出改动分析报告。
      报告落在 output/change_reports/，供管理员复核。
      use_llm=False —— 审批是同步接口，不能被 LLM 拖慢；
      需要语义归因时用 CLI 跑 change_detector.py 重新生成。
    """
    review_path = os.path.join(REVIEWS_DIR, f"{review_id}.json")
    if not os.path.isfile(review_path):
        return {"status": "error", "message": f"审批记录不存在：{review_id}"}

    with open(review_path, "r", encoding="utf-8") as f:
        rec = json.load(f)

    if rec.get("status") == "approved":
        return {"status": "success", "message": "已审批通过，无需重复操作",
                "review_id": review_id}

    # ---- 人工改动检测（失败不阻断审批）----
    change_reports = []
    try:
        from change_detector import analyze_changes
        for item in rec.get("files", []):
            draft = item.get("draft_path", "")
            origin = item.get("origin_path", "")
            if not (draft and origin and
                    os.path.isfile(draft) and os.path.isfile(origin)):
                continue
            rep = analyze_changes(
                origin, draft, review_id=review_id, use_llm=False,
            )
            change_reports.append({
                "file": os.path.basename(draft),
                "report_id": rep.report_id,
                "total_changes": rep.total_changes,
                "high_risk_count": rep.high_risk_count,
                "summary": rep.summary,
                "json_path": rep.json_path,
                "markdown_path": rep.markdown_path,
            })
    except Exception as e:
        # 报告生成失败不该让审批失败 —— 记录后继续归档
        traceback.print_exc()
        change_reports = [{"error": f"改动检测失败：{e}"}]
        audit_event("change_detect_failed", review_id=review_id,
                    error=str(e)[:200])

    high_risk = sum(r.get("high_risk_count", 0) for r in change_reports
                    if isinstance(r, dict))

    # ==================== 高危改动二次确认 ====================
    # 检测到高风险改动时**不执行审批**，要求调用方确认后重来。
    #
    # ★ 为什么是「拦截」而不是「提示」
    # ------------------------------
    # 原实现检测到高风险后照常批准，只在返回消息里写
    # 「请管理员复核报告」—— 但文书**已经归档成终稿了**。
    # 提示会被忽略，拦截才会生效。
    #
    # ★ 为什么只在有高风险时才拦
    # ----------------------
    # 正常文书（无高风险改动）不该增加额外摩擦 —— 否则运营会为了省事
    # 习惯性勾选确认，机制就白做了。
    confirm = bool(req.confirm_high_risk) if req is not None else False
    if high_risk > 0 and not confirm:
        # ★ 这里**一个文件都没动过** —— 拦截发生在移动之前。
        #   （第一版是「移动 → 拦截 → 回滚」，回滚可能失败导致状态不一致。）
        rec["status"] = "pending"
        rec["pending_confirmation"] = {
            "high_risk_count": high_risk,
            "detected_at": time.time(),
            "note": "检测到高风险改动，需人工确认后重新提交审批",
        }
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

        audit_event("doc_review_confirmation_required", review_id=review_id,
                    high_risk_changes=high_risk)
        logger.warning("[approve] review_id=%s 检测到 %d 处高风险改动，"
                       "已拦截，等待人工确认", review_id, high_risk)
        return {
            "status": "confirmation_required",
            "review_id": review_id,
            "high_risk_count": high_risk,
            "change_reports": change_reports,
            "message": (f"检测到 {high_risk} 处高风险改动，已拦截本次审批。"
                        f"请人工复核改动报告后，带 confirm_high_risk=true "
                        f"重新提交。"),
        }

    # ---- 确认通过（或无高风险）→ 现在才真正落地 ----
    moved = []
    for item in rec.get("files", []):
        src_p = item.get("draft_path", "")
        if src_p and os.path.isfile(src_p):
            dst = os.path.join(OFFICIAL_DIR, os.path.basename(src_p))
            shutil.move(src_p, dst)
            item["official_path"] = dst
            moved.append(dst)

    rec["status"] = "approved"
    rec["approved_at"] = time.time()
    rec["change_reports"] = change_reports
    rec.pop("pending_confirmation", None)
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    audit_event("doc_review_approve", review_id=review_id, files=len(moved),
                high_risk_changes=high_risk, confirmed_high_risk=confirm)
    return {
        "status": "success",
        "review_id": review_id,
        "official_files": moved,
        "change_reports": change_reports,
        "high_risk_count": high_risk,
        "message": "草稿已审批通过，移入 output/official/"
                   + (f"；检测到 {high_risk} 处高风险改动，请管理员复核报告"
                      if high_risk else ""),
    }

@app.get("/api/doc/change-reports/{review_id}")
def get_change_reports(review_id: str, _: None = Depends(require_internal_token)):
    """取某次审批的改动分析报告（含每处改动明细）。"""
    from change_detector import REPORT_DIR, safe_filename
    # review_id 来自 URL，必须用与落盘相同的清洗规则，否则 glob 匹配不到
    safe_id = safe_filename(review_id)
    files = sorted(glob.glob(
        os.path.join(REPORT_DIR, f"{safe_id}_CR-*.json")))
    if not files:
        return {"status": "not_found", "review_id": review_id,
                "message": "未找到改动报告"}
    reports = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                reports.append(json.load(f))
        except Exception:
            continue
    high = sum(r.get("high_risk_count", 0) for r in reports)
    return {"status": "success", "review_id": review_id,
            "count": len(reports), "high_risk_count": high,
            "reports": reports}

@app.post("/api/doc/review/{review_id}/reject")
def reject_review(review_id: str,
                  reason: str = "",
                  _: None = Depends(require_internal_token)):
    """reject 草稿：记录原因，状态改为 rejected。"""
    review_path = os.path.join(REVIEWS_DIR, f"{review_id}.json")
    if not os.path.isfile(review_path):
        return {"status": "error", "message": f"审批记录不存在：{review_id}"}

    with open(review_path, "r", encoding="utf-8") as f:
        rec = json.load(f)

    rec["status"] = "rejected"
    rec["rejected_at"] = time.time()
    rec["reject_reason"] = reason
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    audit_event("doc_review_reject", review_id=review_id, reason=reason[:200])
    return {"status": "success", "review_id": review_id,
            "message": "草稿已驳回"}

# ==================== 抽取文档案情 ====================
@app.post("/api/case/extract")
async def api_case_extract(file: UploadFile = File(...),
                            _: None = Depends(require_internal_token)):
    """从上传文档抽取案情字段。"""
    try:
        from file_analyzer import analyze_case_file

        content = await file.read()
        # ★ 同步阻塞（含 LLM 调用）→ 线程池
        info = await run_in_threadpool(analyze_case_file, content, file.filename)
        audit_event("case_extract", filename=file.filename)
        return {"status": "success", "case_dict": info}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==================== 用 dict 生成文书（草稿）====================
@app.post("/api/doc/generate_from_dict")
async def api_generate_from_dict(
    template_file: UploadFile = File(...),
    case_dict_json: str = Form("{}"),
    _: None = Depends(require_internal_token),
):
    """接收案情 dict + 模板，生成草稿（写入 pending/）。"""
    try:
        case_dict = json.loads(case_dict_json)

        tmp_dir = os.path.join(_SCRIPT_DIR, "tmp_upload")
        os.makedirs(tmp_dir, exist_ok=True)
        tpl_path = os.path.join(tmp_dir, template_file.filename)
        with open(tpl_path, "wb") as f:
            f.write(await template_file.read())

        # ★ 同步阻塞（文书生成，含 LLM 抽字段 + docx 渲染）→ 线程池
        out_path, meta = await run_in_threadpool(generate_one, tpl_path, case_dict)

        review_id = f"rev_{uuid.uuid4().hex[:10]}"
        if out_path is None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {"status": "error", "message": "模板无可用字段"}

        # 复制草稿到 pending
        pending_path = os.path.join(PENDING_DIR,
                                    f"{review_id}_{template_file.filename}")
        shutil.copy2(out_path, pending_path)

        # 创建 review 记录
        review_rec = {
            "review_id": review_id,
            "status": "draft",
            "created_at": time.time(),
            "files": [{"template": template_file.filename,
                       "draft_path": pending_path,
                       "size_bytes": os.path.getsize(pending_path)}],
            "errors": [],
        }
        review_path = os.path.join(REVIEWS_DIR, f"{review_id}.json")
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(review_rec, f, ensure_ascii=False, indent=2)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        audit_event("doc_generate_from_dict", review_id=review_id)
        return {"status": "success", "review_id": review_id,
                "draft_path": pending_path,
                "size_bytes": os.path.getsize(pending_path),
                "message": "草稿已生成，需人工 approve"}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
