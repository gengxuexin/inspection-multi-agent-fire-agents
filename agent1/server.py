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

app = FastAPI(title="Agent1 消防法规+文书服务")

# 挂载审计中间件 + tracing
install_audit_middleware(app)
setup_tracing(app)

# ==================== 草稿审批目录 ====================
PENDING_DIR = os.path.join(OUTPUT_DIR, "pending")
OFFICIAL_DIR = os.path.join(OUTPUT_DIR, "official")
REVIEWS_DIR = os.path.join(OUTPUT_DIR, "reviews")
for _d in (PENDING_DIR, OFFICIAL_DIR, REVIEWS_DIR):
    os.makedirs(_d, exist_ok=True)


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


@app.post("/api/rag/ask")
async def api_ask(req: AskReq, _: None = Depends(require_internal_token)):
    # TODO: phase5 透传 UserCtx 给检索层做权限过滤
    try:
        answer = rag_ask(req.question, top_k=req.top_k)
        audit_event("rag_ask", input=req.question[:100])
        return {"status": "success", "answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/rag/retrieve")
async def api_retrieve(req: RetrieveReq,
                        _: None = Depends(require_internal_token)):
    try:
        snippets = retrieve_legal_context(req.query, top_k=req.top_k)
        return {"status": "success", "count": len(snippets),
                "snippets": snippets}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/rag/legal_extract")
async def api_legal(req: LegalReq, _: None = Depends(require_internal_token)):
    try:
        fields = [f.dict() for f in req.legal_fields]
        result = extract_legal_fields_from_case(
            req.case_text, fields, top_k=req.top_k
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
                out_path, meta = generate_one(tpl, case)
                if out_path is None:
                    errors.append(f"跳过（无字段）：{tpl_name}")
                    continue

                # 复制草稿到 pending 目录
                pending_path = os.path.join(PENDING_DIR,
                                            f"{review_id}_{tpl_name}")
                shutil.copy2(out_path, pending_path)

                item = {
                    "template": tpl_name,
                    "draft_path": pending_path,
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


@app.post("/api/doc/review/{review_id}/approve")
def approve_review(review_id: str, _: None = Depends(require_internal_token)):
    """approve 草稿：移动文件到 official/，更新状态为 approved。"""
    review_path = os.path.join(REVIEWS_DIR, f"{review_id}.json")
    if not os.path.isfile(review_path):
        return {"status": "error", "message": f"审批记录不存在：{review_id}"}

    with open(review_path, "r", encoding="utf-8") as f:
        rec = json.load(f)

    if rec.get("status") == "approved":
        return {"status": "success", "message": "已审批通过，无需重复操作",
                "review_id": review_id}

    moved = []
    for item in rec.get("files", []):
        src = item.get("draft_path", "")
        if src and os.path.isfile(src):
            dst = os.path.join(OFFICIAL_DIR, os.path.basename(src))
            shutil.move(src, dst)
            item["official_path"] = dst
            moved.append(dst)

    rec["status"] = "approved"
    rec["approved_at"] = time.time()
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    audit_event("doc_review_approve", review_id=review_id, files=len(moved))
    return {"status": "success", "review_id": review_id,
            "official_files": moved,
            "message": "草稿已审批通过，移入 output/official/"}


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
        info = analyze_case_file(content, file.filename)
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

        out_path, meta = generate_one(tpl_path, case_dict)

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
