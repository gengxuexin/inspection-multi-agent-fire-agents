"""
logger = logging.getLogger("fire_agents.agent1")

案情 × 模板字段 匹配。三级分流：
- FACT    从案情明文抽
- LEGAL   RAG 检索本地知识库
- DERIVED 规则/默认值推导
"""
import json
from config import ds_client
from rag_legal_advisor import (
    extract_legal_fields_from_case, retrieve_legal_context,
)
from field_resolver import (
    FieldKind, split_fields, resolve_all_derived, generate_penalty_advice,
)


EXTRACT_PROMPT = """你是消防行政执法人员。请从案情中为每个字段抽取值。

【案情】
{case_text}

【待抽取字段】
{fields_json}

要求：
1. 严格返回 JSON，key 是字段名（与清单完全一致）
2. 每个字段返回 {{"value": ..., "source": ..., "confidence": ...}}
   - value: 从案情中抽取或合理归纳的值；找不到填 null
   - source: 案情中支持该值的原文片段（10-50 字）
   - confidence: high / medium / low
3. **案情里明确的直接抄；需要归纳的做合理归纳**
   - "简要案情" → 把案情概要成一段
   - "违法事实" → 时间+地点+当事人+行为+后果完整叙述
   - "具体问题" → 列举具体违法违规行为
4. 找不到就填 null，不要编造不存在的数字/号码/日期

返回示例：
{{
  "简要案情": {{"value": "...", "source": "...", "confidence": "high"}}
}}
"""

VERIFY_PROMPT = """请核对下面每个字段的抽取是否与案情原文一致。

【案情原文】
{case_text}

【待核对字段】
{extracted_json}

对每个字段返回：
{{"字段名": {{"consistent": true/false, "reason": "一句话", "corrected": "若不一致给出正确值，否则null"}}}}

只返回 JSON。
"""


def _call_llm(prompt_template: str, **kwargs) -> dict:
    content = prompt_template.format(**kwargs)
    resp = ds_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def extract_fields_from_case(
    case_text: str,
    fields: list,
    case_dict: dict = None,
    use_rag_for_legal: bool = True,
) -> dict:
    """
    三级分流：
    - FACT    → LLM 从案情抽
    - LEGAL   → RAG 检索 + LLM 归纳
    - DERIVED → 规则推导（不调 LLM）
    """
    case_dict = case_dict or {}
    groups = split_fields(fields)

    result = {}

    # 1. 事实类：LLM 抽取
    fact_fields = groups[FieldKind.FACT]
    if fact_fields:
        payload = json.dumps(
            {f["key"]: f.get("label", f["key"]) for f in fact_fields},
            ensure_ascii=False, indent=2,
        )
        try:
            result.update(_call_llm(
                EXTRACT_PROMPT, case_text=case_text, fields_json=payload,
            ))
        except Exception as e:
            logger.error(f"[case_field_matcher] 事实类抽取失败：{e}")

    # 2. 法律类：RAG
    legal_fields = groups[FieldKind.LEGAL]
    if legal_fields and use_rag_for_legal:
        # 分离"法律依据/触犯条款" 与 "处罚决定"
        legal_basis_fields = [
            f for f in legal_fields
            if any(k in f["key"] for k in
                   ["法律依据", "法条", "条款", "违反", "触犯", "定性", "涉嫌"])
        ]
        penalty_fields = [
            f for f in legal_fields
            if any(k in f["key"] for k in ["处罚决定", "处罚建议", "拟处罚", "处理决定"])
        ]

        # 2a. 法律依据类
        if legal_basis_fields:
            try:
                legal_result = extract_legal_fields_from_case(case_text, legal_basis_fields)
                result.update(legal_result)
            except Exception as e:
                logger.error(f"[case_field_matcher] 法律依据抽取失败：{e}")

        # 2b. 处罚决定类：依赖法律依据生成
        if penalty_fields:
            legal_basis = result.get("法律依据", {}).get("value", "") if isinstance(
                result.get("法律依据"), dict) else result.get("法律依据", "")
            if not legal_basis:
                # 兜底：从 RAG 拿
                snippets = retrieve_legal_context(case_text, top_k=5)
                legal_basis = "；".join(s["content"][:150] for s in snippets[:3])
            advice = generate_penalty_advice(case_text, legal_basis)
            for pf in penalty_fields:
                result[pf["key"]] = {
                    "value": advice,
                    "source": f"基于法律依据 {legal_basis[:80]} 生成",
                    "confidence": "medium",
                }

    elif legal_fields and not use_rag_for_legal:
        # RAG 关闭时，法律字段降级走 LLM
        payload = json.dumps(
            {f["key"]: f.get("label", f["key"]) for f in legal_fields},
            ensure_ascii=False, indent=2,
        )
        try:
            result.update(_call_llm(
                EXTRACT_PROMPT, case_text=case_text, fields_json=payload,
            ))
        except Exception:
            pass

    # 3. 派生类：规则推导（不调 LLM）
    derived_fields = groups[FieldKind.DERIVED]
    if derived_fields and case_dict:
        result.update(resolve_all_derived(derived_fields, case_dict))

    return result


def verify_extraction(case_text: str, extracted: dict) -> dict:
    payload = json.dumps(
        {k: (v.get("value") if isinstance(v, dict) else v)
         for k, v in extracted.items() if v},
        ensure_ascii=False, indent=2,
    )
    return _call_llm(VERIFY_PROMPT, case_text=case_text, extracted_json=payload)


def build_review_table(fields: list, extracted: dict, verified: dict,
                       legal_fields: list = None) -> list:
    """兼容调用方传 legal_fields（用于标记来源为 RAG法条）"""
    legal_keys = set()
    if legal_fields:
        for lf in legal_fields:
            legal_keys.add(lf["key"])

    rows = []
    for f in fields:
        key = f["key"]
        ext = extracted.get(key) or {}
        ver = verified.get(key) or {}
        if not isinstance(ext, dict):
            ext = {"value": ext}

        value = ext.get("value")
        if ver.get("consistent") is False and ver.get("corrected"):
            value = ver["corrected"]

        if value is None or value == "":
            status = "未检索到"
        elif ver.get("consistent") is False:
            status = "需人工核对"
        elif ext.get("confidence") == "low":
            status = "低置信"
        else:
            status = "待确认"

        # ★ 优先用传入的 legal_fields 判定来源
        if key in legal_keys:
            kind_label = "RAG法条"
        else:
            kind = classify_field(key, f.get("label", ""))
            kind_label = {
                FieldKind.FACT: "案情",
                FieldKind.LEGAL: "RAG法条",
                FieldKind.DERIVED: "规则推导",
            }[kind]

        rows.append({
            "字段": key,
            "来源": kind_label,
            "抽取值": value or "",
            "原文出处": (ext.get("source") or "")[:100],
            "置信度": ext.get("confidence", ""),
            "校验": "一致" if ver.get("consistent") is True
                    else ("不一致" if ver.get("consistent") is False else "未校验"),
            "状态": status,
            "确认": status in ("待确认", "低置信"),
        })
    return rows


# 从 field_resolver 导入，供 app.py 使用
from field_resolver import classify_field
import logging