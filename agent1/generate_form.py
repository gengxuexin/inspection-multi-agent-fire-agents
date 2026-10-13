"""
logger = logging.getLogger("fire_agents.agent1")

通用表单文书生成器
"""
import json
import re
import datetime
from docx import Document
from config import ds_client
from rag_module import rag_ask
from document_io import fill_table_cells, fill_paragraph_placeholders
import logging


# 支持的所有复选框符号（含汉字"口"）
BOX_SYMBOLS = ["□", "☐", "口", "◻", "▢"]

def _reset_all_boxes(doc):
    """把所有 ☑ 重置为 □（在勾选前调用，防止模板原有误勾）"""
    def _do(para):
        for run in para.runs:
            run.text = run.text.replace("☑", "□")
    for para in doc.paragraphs:
        _do(para)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    _do(para)


def _replace_boxes_in_paragraph(para):
    for run in para.runs:
        for box in BOX_SYMBOLS:
            if box in run.text:
                run.text = run.text.replace(box, "☑")


def _check_box_in_doc(doc, keyword: str) -> bool:
    """在整个文档（段落 + 所有表格）里找到含关键词的位置，把复选框替换成 ☑"""
    for para in doc.paragraphs:
        if keyword in para.text:
            _replace_boxes_in_paragraph(para)

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if keyword not in cell.text:
                    continue
                for para in cell.paragraphs:
                    _replace_boxes_in_paragraph(para)
    return True


def _check_specific_item(doc, item_num: str):
    """精确勾选'第N项'：匹配 '□11.'、'口11.' 或 '□11、'"""
    pattern = re.compile(rf"([□☐口◻▢])\s*{re.escape(item_num)}[\.、]")

    def _do(para):
        for run in para.runs:
            run.text = pattern.sub(
                lambda m: m.group(0).replace(m.group(1), "☑"), run.text
            )

    for para in doc.paragraphs:
        _do(para)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    _do(para)


def _build_extract_prompt(field_schema: dict, case_context: str) -> str:
    lines = []
    for key, (desc, _default) in field_schema.items():
        if desc:
            lines.append(f'  "{key}": "{desc}"')
        else:
            lines.append(f'  "{key}": null')
    fields_json = "{\n" + ",\n".join(lines) + "\n}"

    return f"""你是一位消防行政执法人员。请从以下案情中提取结构化字段。

案情：
{case_context}

请严格返回以下 JSON（字段缺失填 null，不要编造）：
{fields_json}
"""


def _apply_defaults(fields: dict, field_schema: dict, case_context: dict) -> dict:
    from schemas import UNIT_FIELD_MAP
    result = {}

    case_id = case_context.get("case_id", "")
    year_match = re.search(r"(20\d{2})", case_id)
    num_match = re.search(r"(\d{6})$", case_id)

    default_fix_date = ""
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", case_context.get("time", ""))
    if m:
        try:
            dt = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            dt2 = dt + datetime.timedelta(days=15)
            default_fix_date = f"{dt2.year}年{dt2.month}月{dt2.day}日"
        except Exception:
            pass

    auto_defaults = {
        "年份": year_match.group(1) if year_match else "2026",
        "文号": num_match.group(1) if num_match else "000001",
        "编号续": num_match.group(1) if num_match else "000001",
        "日期": case_context.get("time", ""),
        "决定日期": case_context.get("time", ""),
        "检查时间": case_context.get("time", ""),
        "检查日期": case_context.get("time", ""),
        "改正期限": default_fix_date,
        "检查人": case_context.get("interviewer", ""),
        "消防监督检查员签名": case_context.get("interviewer", ""),
        "被告知人": case_context.get("interviewee", ""),
        "被处罚人": case_context.get("interviewee", ""),
        "承办人意见": "经审查，该案件符合立案条件，建议立案。",
        "负责人": case_context.get("负责人", "王大队长"),
    }

    for key, (desc, default) in field_schema.items():
        v = fields.get(key)
        if v is None or v == "":
            if default is not None:
                v = default
            elif key in auto_defaults:
                v = auto_defaults[key]
        result[key] = v

    for field_name, case_key in UNIT_FIELD_MAP.items():
        if result.get(field_name) in (None, "") and case_context.get(case_key):
            result[field_name] = case_context[case_key]

    for k, v in case_context.items():
        if k not in result or result[k] in (None, ""):
            result[k] = v

    return result


def generate_form_docx(case_context, template_path, output_path, field_schema):
    """通用表单文书生成"""
    case_desc = case_context.get("case_context", "")

    # 1. LLM 抽字段
    logger.info(f"    [1/4] LLM 抽取 {len(field_schema)} 个字段...")
    resp = ds_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": _build_extract_prompt(
            field_schema, json.dumps(case_context, ensure_ascii=False)
        )}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    fields = json.loads(resp.choices[0].message.content)

    # 2. 默认值 + 自动补全
    fields = _apply_defaults(fields, field_schema, case_context)

    # 3. RAG 补法律依据
    if "法律依据" in field_schema and not fields.get("法律依据"):
        logger.info(f"    [2/4] RAG 检索法律依据...")
        try:
            raw = rag_ask(f"针对以下违法行为应当依据哪条法规？只需给出完整的'《法规名》第X条'格式：{case_desc}")
            first_line = raw.split("\n")[0].strip()
            m = re.search(
                r"《[^》]+》第[一二三四五六七八九十百千零\d]+条"
                r"(?:第[一二三四五六七八九十\d]+款)?"
                r"(?:第[一二三四五六七八九十\d]+项)?",
                first_line
            )
            fields["法律依据"] = m.group(0) if m else first_line[:80]
        except Exception as e:
            logger.error(f"        ⚠️ RAG 失败：{e}")
    else:
        logger.warning(f"    [2/4] 跳过 RAG")

    # 3.5 处罚决定为空时二次生成
    if "处罚决定" in field_schema and not fields.get("处罚决定"):
        logger.info(f"    [3/4] 生成处罚决定建议...")
        try:
            suggest = ds_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content":
                    f"根据以下法律依据，为一个消防违法案件给出建议的处罚决定（一句话）：\n"
                    f"法律依据：{fields.get('法律依据', '')}\n"
                    f"违法事实：{case_desc}\n"
                    f"只返回一句话，如：'罚款壹万元整' 或 '责令停产停业并处罚款伍仟元整'。"
                }],
                temperature=0.3,
            )
            fields["处罚决定"] = suggest.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"        ⚠️ 处罚建议生成失败：{e}")
    else:
        logger.warning(f"    [3/4] 跳过处罚建议")

    # 4. 回填模板
    logger.info(f"    [4/4] 回填模板...")
    doc = Document(template_path)

    mapping = {k: ("" if v is None else str(v)) for k, v in fields.items()}
    fill_paragraph_placeholders(doc, mapping)
    for table in doc.tables:
        fill_table_cells(table, mapping)

    # 复选框勾选
    source = str(fields.get("案件来源") or "")
    if "1" in source or "监督检查" in source or "发现" in source:
        _check_box_in_doc(doc, "消防监督管理工作中发现")
    if "2" in source or "举报" in source:
        _check_box_in_doc(doc, "举报经核实")
    if "3" in source or "移送" in source:
        _check_box_in_doc(doc, "移送")

    opinion = str(fields.get("立案意见") or "")
    if "属于我单位管辖" in opinion or "及时调查处理" in opinion:
        _check_box_in_doc(doc, "属于我单位管辖")

    # 精确勾选违法行为项数
    item_num = re.sub(r"[第项\s]", "", str(fields.get("违法行为项数") or ""))
    if item_num.isdigit():
        _check_specific_item(doc, item_num)

    if "告知" in template_path or "告知笔录" in template_path:
        _check_box_in_doc(doc, "处罚前告知")
        _check_box_in_doc(doc, "听证告知")

    doc.save(output_path)
    return output_path