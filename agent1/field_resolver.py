"""
logger = logging.getLogger("fire_agents.agent1")

字段三级分类 + 派生值规则。
- FACT    事实类：从案情明文抽（时间/地点/当事人/违法事实）
- LEGAL   法律类：RAG 检索本地知识库（法律依据/触犯条款/定性）
- DERIVED 派生类：规则/默认值/其他字段映射（文号/日期/复议机关/送达日期）
"""
import re
import datetime
from enum import Enum
import logging


class FieldKind(str, Enum):
    FACT = "fact"
    LEGAL = "legal"
    DERIVED = "derived"


# ==================== 派生类规则表 ====================
# key 是模板字段关键词（匹配字段名或 label 的子串）
# value 是取值函数：输入 case_dict，输出字符串（或 None）
#
# 匹配顺序：从上往下，第一个命中的生效。

def _year(case):
    m = re.search(r"(20\d{2})", case.get("time", "") or case.get("case_id", ""))
    return m.group(1) if m else ""


def _num(case):
    cid = case.get("case_id", "") or ""
    m = re.search(r"(\d{6})$", cid)
    return m.group(1) if m else ""


def _plus_days(case, days=7):
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", case.get("time", ""))
    if not m:
        return ""
    try:
        dt = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        d2 = dt + datetime.timedelta(days=days)
        return f"{d2.year}年{d2.month}月{d2.day}日"
    except Exception:
        return ""


DERIVED_RULES = [
    # ---- 文号/编号类（程序性） ----
    (["文号"],               lambda c: _num(c) or "000111"),
    (["编号续"],             lambda c: _num(c) or "000111"),
    (["编号"],               lambda c: "X消监字"),
    (["年份"],               _year),

    # ---- 日期类（从 case["time"] 推导） ----
    (["改正期限"],           lambda c: _plus_days(c, 7)),
    (["检查日期", "检查时间", "询问时间"], lambda c: c.get("time", "")),
    (["决定日期", "送达日期", "日期"], lambda c: c.get("time", "")),

    # ---- 固定机构/机关（本地固定） ----
    (["消防救援机构名称", "执行告知单位", "听证组织单位"],
     lambda c: "某市高新区消防救援大队"),
    (["复议机关"],           lambda c: "某市人民政府"),
    (["管辖法院"],           lambda c: "某市人民"),

    # ---- 固定条款（标准文本） ----
    (["执行方式和期限", "缴纳罚款"],
     lambda c: "自收到本决定书之日起十五日内到指定银行缴纳罚款；"
               "逾期不缴纳的，每日按罚款数额的百分之三加处罚款，"
               "加处罚款的数额不超过罚款的数额。"),
    (["证据", "证据清单"],
     lambda c: "现场检查记录、询问笔录、现场照片、火灾事故认定书、"
               "证人证言、物证、鉴定意见、视频资料"),
    (["附件清单"],
     lambda c: "1.行政处罚决定书；2.现场检查记录；3.询问笔录；"
               "4.现场照片；5.火灾事故认定书；6.证据清单；7.法律文书送达回证。"),

    # ---- 从 case_dict 映射（人名/单位/联系人） ----
    (["被告知单位法定代表人", "单位法定代表人", "法定代表人"],
     lambda c: c.get("unit_legal_person", "")),
    (["被告知人签名", "被告知人", "被询问人"],
     lambda c: c.get("interviewee", "")),
    (["被处罚人"],
     lambda c: c.get("unit_name", "") or c.get("interviewee", "")),
    (["被检查单位签收人", "被检查单位随同检查人员签名"],
     lambda c: c.get("unit_legal_person", "")),
    (["消防监督检查员签名", "告知人", "询问人", "检查人", "承办人"],
     lambda c: c.get("interviewer", "")),
    (["记录人"],             lambda c: c.get("recorder", "小月亮")),
    (["负责人"],             lambda c: c.get("负责人", "王大队长")),

    # ---- 通用程序性套话（不含具体案件信息） ----
    (["立案审批"],           lambda c: "同意立案。"),
    (["复核结论"],
     lambda c: "被告知人未提出陈述和申辩，依法作出行政处罚决定。"),
    (["听证处理结果"],
     lambda c: "被告知人放弃听证，依法作出行政处罚决定。"),
    (["陈述申辩内容"],
     lambda c: "对上述告知事项，你（单位）是否提出陈述和申辩？"),
    (["陈述申辩回答"],       lambda c: "不提出陈述和申辩。"),
    (["听证询问"],           lambda c: "对上述告知事项，你是否要求听证？"),
    (["听证回答"],           lambda c: "不要求听证。"),
]


# ==================== 法律类关键词 ====================
LEGAL_FIELD_KEYWORDS = [
    "法律依据", "法条", "法律条款", "违反", "触犯", "条款",
    "法律责任", "处罚依据", "定性", "罪名", "涉嫌",
    "适用法律", "所触犯", "法律后果",
]

# 处罚建议类（走 RAG + LLM 生成）
PENALTY_KEYWORDS = ["处罚决定", "处罚建议", "拟处罚", "处理决定"]


# ==================== 事实类关键词（兜底） ====================
FACT_FIELD_KEYWORDS = [
    "姓名", "性别", "年龄", "出生", "身份证", "住址", "联系方式",
    "单位名称", "地址", "电话", "信用代码", "统一社会",
    "时间", "地点", "日期", "询问", "检查",
    "案情", "简要案情", "违法事实", "事实",
    "伤亡", "损失", "火灾",
    "法定代表人", "联系人",
    "被检查单位", "被询问人", "被告知人", "被处罚人",
]


# ==================== 分类函数 ====================
def classify_field(key: str, label: str = "") -> FieldKind:
    """判断单个字段属于哪一类。优先级：LEGAL > DERIVED > FACT。"""
    text = f"{key} {label}"

    # 1. 法律类（最高优先级）
    for kw in LEGAL_FIELD_KEYWORDS:
        if kw in text:
            return FieldKind.LEGAL

    # 2. 处罚建议类 → 归为 LEGAL（由 RAG + LLM 推导）
    for kw in PENALTY_KEYWORDS:
        if kw in text:
            return FieldKind.LEGAL

    # 3. 派生类（有规则能生成）
    for keys, _fn in DERIVED_RULES:
        for k in keys:
            if k in text:
                return FieldKind.DERIVED

    # 4. 兜底：事实类
    return FieldKind.FACT


def split_fields(fields: list) -> dict:
    """按三类分组。返回 {kind: [fields]}"""
    result = {FieldKind.FACT: [], FieldKind.LEGAL: [], FieldKind.DERIVED: []}
    for f in fields:
        k = classify_field(f["key"], f.get("label", ""))
        result[k].append(f)
    return result


# ==================== 派生类值生成 ====================
def resolve_derived(key: str, label: str, case: dict):
    """按规则生成派生字段的值。返回 (value, source)。"""
    text = f"{key} {label}"
    for keys, fn in DERIVED_RULES:
        for k in keys:
            if k in text:
                try:
                    v = fn(case)
                except Exception as e:
                    v = None
                if v:
                    return v, f"规则推导（匹配关键词：{k}）"
    return None, ""


def resolve_all_derived(fields: list, case: dict) -> dict:
    """批量生成派生字段。返回 {key: {value, source, confidence}}"""
    result = {}
    for f in fields:
        v, s = resolve_derived(f["key"], f.get("label", ""), case)
        if v:
            result[f["key"]] = {
                "value": v,
                "source": s,
                "confidence": "high",
            }
    return result


# ==================== 处罚建议生成 ====================
PENALTY_PROMPT = """你是消防行政执法人员。请根据以下案情、违法事实和法律依据，
给出建议的处罚决定。

【案情】
{case_text}

【法律依据】
{legal_basis}

要求：
1. 分别列出对每个责任主体的处罚，格式为：
   "对XX公司罚款XX万元并责令停产停业整顿；对XX处行政拘留X日，因涉嫌犯罪移送司法机关。"
2. 罚款金额参考《消防法》相关条款的法定幅度，取中位数
3. 涉及人命的，从重处罚
4. 只返回一句话，不要任何解释
"""


def generate_penalty_advice(case_text: str, legal_basis: str) -> str:
    """根据案情 + 法律依据生成处罚决定建议"""
    from config import ds_client
    try:
        resp = ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": PENALTY_PROMPT.format(
                case_text=case_text, legal_basis=legal_basis,
            )}],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[field_resolver] 处罚建议生成失败：{e}")
        return ""