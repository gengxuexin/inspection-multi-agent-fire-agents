"""
logger = logging.getLogger("fire_agents.agent1")

法律推理模块：用本地 RAG 知识库为"法律依据 / 违法事实 / 触犯条款 / 定性"类字段生成答案。

流程：
1. 识别哪些字段需要法律推理（关键词白名单 + 黑名单）
2. 直接调 retriever 拿原始条文（不经过 LLM 归纳，保留原文）
3. 用 LLM 从案情 + 检索条文中归纳出字段答案，并返回引用的条款原文作为 source
"""
import json
import re
from config import ds_client
import logging


# ==================== 字段分类 ====================

# 触发 RAG 的关键词
LEGAL_FIELD_KEYWORDS = [
    "法条", "法律依据", "依据", "违反", "触犯", "条款",
    "法律责任", "处罚依据", "定性", "罪名", "违规", "违法",
    "适用法律", "所触犯", "涉嫌",
]

# 非法律字段黑名单（避免误伤）
NON_LEGAL_FIELDS = [
    "姓名", "性别", "年龄", "出生", "身份证", "住址", "联系方式",
    "单位名称", "法定代表人", "地址", "电话", "信用代码",
    "时间", "地点", "日期", "编号", "文号", "年份",
    "签收人", "签名", "承办人", "告知人", "被询问人", "被检查人",
    "检查人", "记录人", "询问人", "执法人员",
]


def is_legal_field(key: str, label: str = "") -> bool:
    """判断字段是否需要法律推理"""
    text = f"{key} {label}"

    # 黑名单优先：只要命中就排除
    for kw in NON_LEGAL_FIELDS:
        if kw in text:
            return False

    # 白名单：命中任一即需要 RAG
    for kw in LEGAL_FIELD_KEYWORDS:
        if kw in text:
            return True

    return False


def split_fields(fields: list) -> tuple:
    """把字段分成 (普通字段, 法律字段)"""
    normal, legal = [], []
    for f in fields:
        if is_legal_field(f["key"], f.get("label", "")):
            legal.append(f)
        else:
            normal.append(f)
    return normal, legal


# ==================== RAG 检索 ====================

def retrieve_legal_context(case_text: str, top_k: int = 8) -> list:
    """
    直接调 retriever 拿原始条文片段，返回 [{content, source, score?}, ...]
    失败时返回空列表。
    """
    try:
        from rag_module import _build_chain
        _, retriever = _build_chain(top_k=top_k)
        docs = retriever.invoke(case_text)
        result = []
        for d in docs:
            result.append({
                "content": d.page_content,
                "source": d.metadata.get("source", "?"),
            })
        return result
    except FileNotFoundError as e:
        logger.info(f"[rag_legal_advisor] 向量库不存在：{e}")
        return []
    except Exception as e:
        logger.error(f"[rag_legal_advisor] RAG 检索失败：{e}")
        return []


def format_legal_context(snippets: list, max_len: int = 400) -> str:
    """把检索到的片段拼接成 LLM 可读的上下文"""
    if not snippets:
        return "（未检索到相关法条，请基于常识谨慎判断）"

    parts = []
    for i, s in enumerate(snippets, 1):
        content = s["content"][:max_len]
        parts.append(f"--- 片段 {i}（来源：{s['source']}）---\n{content}")
    return "\n\n".join(parts)


# ==================== LLM 归纳 ====================

LEGAL_FIELD_PROMPT = """你是消防行政执法人员。请根据以下案情和本地知识库检索到的法律条文，
为每个"法律类字段"给出准确的答案。

【案情】
{case_text}

【本地知识库检索到的法律条文】
{legal_context}

【待填写字段】
{fields_json}

要求：
1. 严格返回 JSON，key 是字段名，与清单完全一致
2. 每个字段返回 {{"value": ..., "source": ..., "confidence": ...}}
   - value: 根据案情和法条归纳的答案
   - source: 引用的法条编号和原文片段（30-150 字），必须来自上面检索到的条文
   - confidence: high / medium / low
3. 若法条中找不到明确依据，value 填 null，confidence 填 low，source 留空
4. 字段语义按以下规则填写：
   - 字段名含"法律依据/法条/依据/条款/适用法律" → 只列条款编号，
     格式如"《中华人民共和国消防法》第六十条第一款第七项"，
     多条用顿号分隔
   - 字段名含"违法事实/违法行为" → 写成一段完整叙述（时间+地点+当事人+行为+后果）
   - 字段名含"违反/触犯/涉嫌/定性" → 写出被违反的条款或可能的违法行为定性
   - 字段名含"处罚依据" → 列出处罚条款
5. 如果案情中已明确提到条款编号（如"《XX法》第X条"），优先使用案情中的条款，
   但必须在 source 中给出对应条文原文（从检索结果里找）
6. 不要编造未在检索结果中出现的条款编号
7. 不要输出与条款无关的评论

返回示例：
{{
  "法律依据": {{"value": "《中华人民共和国消防法》第六十条、第六十一条", "source": "第六十条 单位违反本法规定，有下列行为之一的...", "confidence": "high"}},
  "违法事实": {{"value": "2025年11月14日...", "source": "第六十条...", "confidence": "high"}}
}}
"""


def extract_legal_fields_from_case(
    case_text: str,
    legal_fields: list,
    top_k: int = 8,
) -> dict:
    """
    对法律字段做 RAG 检索 + LLM 归纳。
    返回结构与 case_field_matcher.extract_fields_from_case 一致。
    """
    if not legal_fields:
        return {}

    # 1. RAG 检索
    snippets = retrieve_legal_context(case_text, top_k=top_k)
    if not snippets:
        logger.info("[rag_legal_advisor] 无检索结果，法律字段将返回空值")
        return {
            f["key"]: {
                "value": None,
                "source": "",
                "confidence": "low",
            }
            for f in legal_fields
        }

    legal_context = format_legal_context(snippets)

    # 2. LLM 归纳
    fields_json = json.dumps(
        {f["key"]: f.get("label", f["key"]) for f in legal_fields},
        ensure_ascii=False, indent=2,
    )

    resp = ds_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": LEGAL_FIELD_PROMPT.format(
                case_text=case_text,
                legal_context=legal_context,
                fields_json=fields_json,
            ),
        }],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# ==================== 二次校验：条款号反查原文 ====================

ARTICLE_PATTERN = re.compile(
    r"《([^》]+)》第([一二三四五六七八九十百千零\d]+)条"
    r"(?:第([一二三四五六七八九十\d]+)款)?"
    r"(?:第([一二三四五六七八九十\d]+)项)?"
)


def enrich_source_from_retrieval(value: str, snippets: list) -> str:
    """
    从 LLM 归纳出的 value 中提取条款号，在检索片段中找对应原文作为 source。
    用作二次校验时的 source 补强。
    """
    if not value or not snippets:
        return ""

    articles = ARTICLE_PATTERN.findall(value)
    if not articles:
        return ""

    sources = []
    for law, art, *_ in articles:
        key = f"第{art}条"
        for s in snippets:
            if key in s["content"] and law in s["content"]:
                # 截取该条附近的一段
                idx = s["content"].find(key)
                start = max(0, idx - 20)
                end = min(len(s["content"]), idx + 200)
                sources.append(s["content"][start:end].strip())
                break
    return "；".join(sources[:3])