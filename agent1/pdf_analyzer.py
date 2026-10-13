"""
logger = logging.getLogger("fire_agents.agent1")

PDF 分析模块
- 从 PDF 提取文本（支持文本型 + 扫描件 OCR 可选）
- 用 LLM 从 PDF 文本中抽取案情描述 + 结构化字段
"""
import os
import json
import tempfile
from pypdf import PdfReader
from config import ds_client
import logging


# ==================== PDF → 文本 ====================

def pdf_to_text(file_bytes: bytes, file_name: str = "input.pdf") -> str:
    """把 PDF 转成纯文本（文本型 PDF）"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        reader = PdfReader(tmp_path)
        parts = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                parts.append(f"--- 第 {i+1} 页 ---\n{text}")
        return "\n\n".join(parts)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def pdf_to_text_ocr(file_bytes: bytes) -> str:
    """
    扫描件 PDF 的 OCR（可选）
    需要额外安装：pip install pytesseract pdf2image pillow
    以及系统依赖 Tesseract-OCR 和 poppler
    """
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        images = convert_from_bytes(file_bytes, dpi=200)
        parts = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang="chi_sim")
            if text.strip():
                parts.append(f"--- 第 {i+1} 页 ---\n{text}")
        return "\n\n".join(parts)
    except ImportError:
        return ""  # OCR 依赖未装，返回空


# ==================== 自动选择提取方式 ====================

def extract_pdf_text(file_bytes: bytes, file_name: str = "input.pdf") -> str:
    """自动判断 PDF 类型：文本型直接提取，扫描型走 OCR"""
    text = pdf_to_text(file_bytes, file_name)
    # 少于 100 字符通常意味着这是扫描件
    if len(text.strip()) < 100:
        logger.info(f"    [info] PDF 文本过少（{len(text)} 字符），尝试 OCR...")
        ocr_text = pdf_to_text_ocr(file_bytes)
        if ocr_text:
            return ocr_text
    return text


# ==================== LLM 抽取案情 ====================

CASE_EXTRACT_PROMPT = """你是一位消防行政执法人员。请从以下文档内容中提取案情信息。

文档内容：
{pdf_text}

请返回 JSON（缺失字段填 null，不要编造）：
{{
  "案件概述": "一句话概括案件（时间、地点、当事人、违法行为）",
  "违法事实": "详细描述违法行为",
  "涉案单位": "单位全称",
  "涉案个人": "当事人姓名（如有）",
  "违法时间": "YYYY年M月D日",
  "违法地点": "具体地点",
  "涉及法条": "文档中提到的法律条款编号"
}}
"""


def extract_case_from_pdf(file_bytes: bytes, file_name: str = "input.pdf") -> dict:
    """从 PDF 中抽取案情（返回 case_context 的补充字段）"""
    text = extract_pdf_text(file_bytes, file_name)
    if not text.strip():
        return {"_error": "PDF 无法提取文本"}

    # 截断到 8000 字以内，避免超 token
    text = text[:8000]

    resp = ds_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": CASE_EXTRACT_PROMPT.format(pdf_text=text)}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = json.loads(resp.choices[0].message.content)

    # 转成 case_context 格式
    result = {}
    if raw.get("涉案单位"):
        result["unit_name"] = raw["涉案单位"]
    if raw.get("涉案个人"):
        result["interviewee"] = raw["涉案个人"]
    if raw.get("违法时间"):
        result["time"] = raw["违法时间"]
    if raw.get("违法地点"):
        result["location"] = raw["违法地点"]

    # 案情描述拼成完整的一段
    desc_parts = []
    if raw.get("案件概述"):
        desc_parts.append(raw["案件概述"])
    if raw.get("违法事实") and raw["违法事实"] != raw.get("案件概述"):
        desc_parts.append(raw["违法事实"])
    result["case_context"] = "。".join(desc_parts)

    return result


# ==================== LLM 抽取结构化字段 ====================

FIELD_FROM_PDF_PROMPT = """你是一位消防行政执法人员。请从以下文档中提取结构化字段。

文档内容：
{pdf_text}

请返回 JSON（缺失字段填 null）：
{{
  "unit_name": "单位全称",
  "unit_legal_person": "法定代表人姓名",
  "unit_address": "单位地址",
  "unit_contact": "联系电话",
  "unit_credit_code": "统一社会信用代码",
  "contact_person": "联系人姓名",
  "contact_phone": "联系人电话",
  "case_number": "案件编号"
}}
"""


def extract_fields_from_pdf(file_bytes: bytes, file_name: str = "input.pdf") -> dict:
    """从 PDF（比如已经填好的监督检查记录）中抽取结构化字段"""
    text = extract_pdf_text(file_bytes, file_name)
    if not text.strip():
        return {}

    text = text[:8000]

    resp = ds_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": FIELD_FROM_PDF_PROMPT.format(pdf_text=text)}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = json.loads(resp.choices[0].message.content)

    # 去掉 None 值
    return {k: v for k, v in raw.items() if v}