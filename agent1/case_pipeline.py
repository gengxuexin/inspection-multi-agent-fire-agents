"""
logger = logging.getLogger("fire_agents.agent1")

Agent2 文书生成 - CLI 主程序（模板驱动 + RAG 法律推理）

流程：
  模板 → 字段提取 → 字段分流
    ├── 普通字段 → LLM 从案情抽取
    └── 法律字段 → RAG 检索本地知识库 + LLM 归纳
  → LLM 校验 → 回填
"""
import os
import sys
import traceback
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (_SCRIPT_DIR, os.path.dirname(_SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from docx import Document
from document_io import fill_paragraph_placeholders, fill_table_cells
from pdf_converter import docx_to_pdf
from template_schema_extractor import extract_fields_from_template, summarize_fields
from case_field_matcher import (
    extract_fields_from_case, verify_extraction, build_review_table,
)
from rag_legal_advisor import split_fields
from generate_interview import generate_interview_docx
import logging


# ==================== 目录 ====================
TEMPLATE_DIR = os.path.join(_SCRIPT_DIR, "templates")
OUTPUT_DIR   = os.path.join(_SCRIPT_DIR, "output")

ALSO_PDF = True
INTERVIEW_DEMO = False
USE_RAG_LEGAL = True   # 是否对法律字段启用 RAG


# ==================== 内置案情 ====================
CASE = {
    "case_id": "xinghua_2025_000111",
    "time": "2025年11月14日",
    "location": "某市高新区星华工业园3号厂房",
    "interviewer": "小行星",
    "interviewee": "王某",
    "负责人": "王大队长",
    "case_source": "4",
    "unit_name": "星华实业发展有限公司、锐创电子科技有限公司",
    "unit_legal_person": "李某某、张某某",
    "unit_address": "某市高新区星华工业园3号厂房",
    "unit_contact": "0573-88888888、13800000000",
    "unit_credit_code": "91370100MA3TEST0X、91370100MA4TEST0X",
    "contact_person": "李某某、张某某",
    "contact_phone": "0573-88888888、13800000000",
    "case_context": (
        "2025年11月14日凌晨2时15分，位于某市高新区的星华工业园3号厂房发生重大火灾事故。"
        "起火建筑为单层钢结构厂房，建筑面积约4500平方米。火灾造成3人死亡、5人重伤，"
        "直接财产损失约1200万元。起火部位位于3号厂房东南角的夹层仓储区。起火原因为："
        "厂房承租方“锐创电子科技有限公司”在未办理动火审批手续、未清理周边可燃物且无专人监护的情况下，"
        "违规雇佣无特种作业操作证的工人王某进行电焊作业。电焊产生的高温焊渣掉落，"
        "引燃了夹层下方违规堆放的聚氨酯泡沫包装材料，火势迅速蔓延。"
        "出租方星华实业发展有限公司将未经消防验收合格且不符合消防安全条件的厂房出租给锐创公司，"
        "未在租赁合同中明确双方的消防安全管理职责，对共用消防设施缺乏统一管理，"
        "厂房内原有的室内消火栓系统因管网漏水长期处于瘫痪状态，火灾发生时未能正常供水，"
        "违规在厂房东侧搭建彩钢板库房，占用了防火间距，导致火势向相邻2号厂房蔓延。"
        "承租方锐创电子科技有限公司违规在厂房内设置员工宿舍（“三合一”场所），"
        "且住宿区与生产区未进行有效的防火分隔，违规使用易燃可燃材料（聚氨酯泡沫）作为包装及隔音填充物，"
        "严重违反动火作业安全管理规定，违规电焊作业，疏散通道被生产物料堵塞，安全出口上锁，"
        "导致火灾发生时多名夜班员工无法逃生。直接肇事者王某无电工资质，违规进行明火作业，操作不当引发火灾。"
    ),
}


# ==================== 单份文书生成 ====================
def generate_one(template_path: str, case_context: dict):
    tpl_name = os.path.basename(template_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(tpl_name)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_{ts}.docx")

    # 询问笔录走专用生成器
    if "询问笔录" in tpl_name:
        case_copy = dict(case_context)
        case_copy["_generate_answers"] = INTERVIEW_DEMO
        generate_interview_docx(case_copy, out_path)
        return out_path, {"mode": "interview"}

    # 1. 提取模板字段
    fields = extract_fields_from_template(template_path)
    if not fields:
        logger.warning(f"      ⚠️ 未识别到可填字段，跳过")
        return None, {"mode": "empty"}

    normal, legal = split_fields(fields) if USE_RAG_LEGAL else (fields, [])
    logger.info(f"      字段：普通 {len(normal)}，法律 {len(legal)}")

    # 2. 抽取（自动分流）
    case_text = case_context.get("case_context", "")
    logger.info(f"      [1/3] LLM 抽取（含 RAG 法律推理）...")
    extracted = extract_fields_from_case(
        case_text, fields,
        case_dict=case_context,  # ★ 新增
        use_rag_for_legal=USE_RAG_LEGAL,
    )

    # 3. 校验
    logger.info(f"      [2/3] LLM 交叉校验...")
    verified = verify_extraction(case_text, extracted)

    # 4. 汇总
    rows = build_review_table(fields, extracted, verified, legal_fields=legal)
    mapping = {r["字段"]: r["抽取值"] for r in rows if r["抽取值"]}

    missing  = [r["字段"] for r in rows if r["状态"] == "未检索到"]
    mismatch = [r["字段"] for r in rows if r["状态"] == "需人工核对"]
    if missing:
        logger.info(f"      ⚠️ 未检索到：{missing}")
    if mismatch:
        logger.info(f"      ⚠️ 需人工核对：{mismatch}")

    # 5. 回填
    logger.info(f"      [3/3] 回填模板（{len(mapping)} 个字段）...")
    doc = Document(template_path)
    fill_paragraph_placeholders(doc, mapping)
    for table in doc.tables:
        fill_table_cells(table, mapping)
    doc.save(out_path)

    return out_path, {"mode": "template", "rows": rows}


# ==================== 主流程 ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(TEMPLATE_DIR):
        logger.info(f"❌ 模板目录不存在: {TEMPLATE_DIR}")
        return

    templates = [
        os.path.join(TEMPLATE_DIR, f)
        for f in os.listdir(TEMPLATE_DIR)
        if f.endswith(".docx") and not f.startswith("~$")
    ]
    if not templates:
        logger.info(f"❌ 模板目录为空: {TEMPLATE_DIR}")
        return

    logger.info("=" * 60)
    logger.info(f"开始生成 {len(templates)} 份文书")
    logger.info(f"案件编号: {CASE['case_id']}")
    logger.info(f"RAG 法律推理: {'启用' if USE_RAG_LEGAL else '关闭'}")
    logger.info("=" * 60)

    success, failed = 0, 0
    for tpl in templates:
        logger.info(f"\n📄 生成【{os.path.basename(tpl)}】...")
        try:
            out_path, meta = generate_one(tpl, CASE)
            if out_path is None:
                logger.warning(f"   ⏭️ 跳过")
                failed += 1
                continue

            logger.info(f"   ✅ docx: {os.path.basename(out_path)}")
            success += 1

            if ALSO_PDF:
                pdf_path = docx_to_pdf(out_path)
                if pdf_path:
                    logger.info(f"   📕 pdf : {os.path.basename(pdf_path)}")
        except FileNotFoundError as e:
            logger.warning(f"   ⏭️ 跳过（文件不存在）: {e}")
            failed += 1
        except Exception as e:
            logger.error(f"   ❌ 生成失败: {e}")
            traceback.print_exc()
            failed += 1

    logger.info("\n" + "=" * 60)
    logger.error(f"完成！成功 {success} 份，失败 {failed} 份")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()