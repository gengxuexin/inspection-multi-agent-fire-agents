# chunking.py
"""法条感知清洗 + 分块（RAG 检索质量的根因治理层）

为什么需要这个模块
------------------
RAGAS 实测（top_k=20）context_precision=0.165、context_recall=0.6。
根因不在 prompt，而在入库前的文本质量与切分粒度：

1. **PDF 噪声**：`--- 第 14 页 ---`、页脚页码、`\\x14\\x00 0 0 3 2 \\x14`
   （PDF 内部标记符）等被原样切进 chunk，向量与 BM25 都被污染，
   语义检索把"含噪声"的chunk 排到前面。
2. **法条被拦腰截断**：法条是"第X条"为单位的**语义原子**。
   通用 RecursiveCharacterTextSplitter 按 500 字硬切，
   第六十条第一款可能被切到两个 chunk 里，任一片段都不完整。
3. **元数据缺失**：只有 source/doc_id，检索时无法按"法律名+条款号"
   做结构化过滤，只能靠向量硬碰。

本模块提供：
    clean_pdf_text()    —— 去页眉页脚/PDF 私有标记符/空白塌缩
    cn_to_int()         —— 中文数字转int（"两百零五" -> 205）
    article_of()        —— 从chunk 文本提取条款号（写入元数据）
    article_matches()   —— 条款号布尔匹配（等价于 article_match_level >= 1）
    article_match_level() —— 条款匹配**精确度分级** 0~3（3=到项/2=到款/1=到条）
    LegalChunker        —— 法条边界优先切分，chunk 带 law/article/chunk_id 元数据
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable


class _Doc:
    """langchain_core 不可用时的轻量 Document 替身（字段完全一致）。

    清洗 / 条款解析是纯函数逻辑，不应被 langchain 依赖绑死；
    只有真正需要产出 Document 时才要求 langchain 存在。
    """

    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"_Doc(chars={len(self.page_content)}, meta={self.metadata})"


def _document_class() -> Any:
    try:
        from langchain_core.documents import Document
        return Document
    except ImportError:
        return _Doc

# ==================== 条款号识别 ====================

# 中文数字/阿拉伯数字统一字符集（法条全部写法，含"两百""一百一十"）
_NUM = r"[一二三四五六七八九十百千零〇两壹贰叁肆伍陆柒捌玖拾佰仟\d]+"

# ==================== 清洗 ====================

# 页码标记：--- 第 14 页 --- / —第 14 页— / 第 14 页 / - 14 -
_RE_PAGE_MARK = re.compile(
    r"^[ \t]*(?:[-—–_=~]{0,4}\s*(?:第\s*)?\d{1,4}\s*页(?:\s*[-—–_=~]{0,4})?)[ \t]*$",
    re.MULTILINE,
)
# PDF 内部标记符：\x14 ... \x14（Adobe 私有），中间可含任意非换行字符
_RE_PDF_MARK = re.compile(r"\x14[^\n]{0,32}?\x14|\x14[^\n]{0,24}")
# 单行页脚：整行几乎只有数字/符号/英文（页码、版权行）
_RE_FOOTER = re.compile(
    r"^[ \t]*[.\-—–_=~|/\\©®\s]*[\d\w\.\-]{0,24}[.\-—–_=~|/\\©®\s]*$"
)
# 章节标题（第X章/第X节/第X编）—— 页眉剔除时需豁免
_RE_CHAPTER = re.compile(rf"^第\s*{_NUM}\s*[编章节]")
# 连续 3 个以上空行 -> 2 个换行
_RE_BLANK = re.compile(r"\n{3,}")
# 行尾空白 + 全角空格
_RE_TRAIL = re.compile(r"[ \t　]+(?=\n)")


def _drop_repeated_short_lines(text: str, min_count: int = 2) -> str:
    """删除在文中**重复出现**的短行（页眉/页脚/版权行的典型特征）。

    PDF 每页都有同一句页眉（如"版权所有""XX市消防支队"），它们对检索是纯噪声。
    单靠"短+无中文"会漏掉"版权所有"这类中文页脚，因此改用**跨页重复**判定。
    章节标题（第X章/第X节）显式豁免。
    """
    lines = text.split("\n")
    cand: dict[str, int] = {}
    for ln in lines:
        s = ln.strip()
        if not s or len(s) > 40:
            continue
        if _RE_CHAPTER.match(s):
            continue
        if s.endswith(("。", "；", "：", "，")):
            continue
        cand[s] = cand.get(s, 0) + 1

    repeated = {s for s, c in cand.items() if c >= min_count}
    if not repeated:
        return text

    kept = []
    for ln in lines:
        if ln.strip() in repeated:
            # 该行全部由重复行构成时直接丢；否则把重复片段挖掉
            if ln.strip() == ln:
                continue
            for s in repeated:
                ln = ln.replace(s, "")
            if not ln.strip():
                continue
        kept.append(ln)
    return "\n".join(kept)


def clean_pdf_text(text: str, *, drop_footers: bool = True) -> str:
    """清洗 PDF/OCR 抽取文本。

    :param text:          原始抽取文本
    :param drop_footers:  是否丢弃页眉页脚（重复短行 + 纯符号短行）
    :return: 清洗后文本
    """
    if not text:
        return ""

    # 1) PDF 私有标记符（必须在控制字符清洗之前，否则 \x14 已被删掉）
    text = _RE_PDF_MARK.sub(" ", text)

    # 2) 页码标记行
    text = _RE_PAGE_MARK.sub("", text)

    # 3) 行尾空白
    text = _RE_TRAIL.sub("", text)

    # 4) 跨页重复的页眉/页脚
    if drop_footers:
        text = _drop_repeated_short_lines(text)

    # 5) 页脚行：逐行判断，短行且无中文 → 丢弃
    if drop_footers:
        kept: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and len(stripped) <= 40:
                has_cjk = re.search(r"[一-鿿]", stripped)
                if not has_cjk and _RE_FOOTER.match(stripped):
                    continue
            kept.append(line)
        text = "\n".join(kept)

    # 6) 其余控制字符（保留 \n \t）
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)

    # 7) 行内多余空格塌缩（中文文本里多空格都是噪声）
    text = re.sub(r"[ 　]{2,}", " ", text)

    # 8) 空行塌缩
    text = _RE_BLANK.sub("\n\n", text)

    return text.strip()


# ==================== 中文数字转int ====================

# 中文数字 -> 阿拉伯数字（覆盖法条全部写法，含"两百""一百一十"）
_CN_DIGIT = {
    "零": 0, "〇": 0, "一": 1, "壹": 1, "二": 2, "贰": 2, "两": 2,
    "三": 3, "叁": 3, "四": 4, "肆": 4, "五": 5, "伍": 5,
    "六": 6, "陆": 6, "七": 7, "柒": 7, "八": 8, "捌": 8,
    "九": 9, "玖": 9,
}
_CN_UNIT = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}


def cn_to_int(s: str) -> int | None:
    """中文数字转int。'六十'->60, '一百一十'->110, '二十三'->23。

    :return: 解析失败返回 None
    """
    if not s:
        return None
    if s.isdigit():
        return int(s)

    total, section = 0, 0
    for ch in s:
        if ch in _CN_DIGIT:
            section = _CN_DIGIT[ch]
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if section == 0:
                section = 1  # "十五" -> 15
            total += section * unit
            section = 0
        else:
            return None
    return total + section


# 条款号：第六十条 / 第60条 / 第一百一十条第三款第四项
RE_ARTICLE = re.compile(rf"第\s*({_NUM})\s*条")
# 扁平三段式分组：条 / 款 / 项 各自独立捕获，便于逐段归一
RE_ITEM = re.compile(
    rf"(第\s*{_NUM}\s*条)(\s*第\s*{_NUM}\s*款)?(\s*第\s*{_NUM}\s*项)?"
)


def _strip_num(raw: str | None) -> str | None:
    """'第一款' -> 1；'第 2 款' -> 2；无法解析返回 None。"""
    if not raw:
        return None
    digits = re.sub(r"[第\s]+", "", raw)
    digits = re.sub(r"[条款项]", "", digits)
    val = cn_to_int(digits)
    return str(val) if val is not None else None


def article_of(text: str) -> str | None:
    """从 chunk 文本中提取**首个**条款号，归一为"第X条"。

    "第六十条" -> "第60条"；"第60条" -> "第60条"；无则 None。
    """
    if not text:
        return None
    m = RE_ARTICLE.search(text)
    if not m:
        return None
    n = cn_to_int(m.group(1))
    return f"第{n}条" if n is not None else None


def all_articles_of(text: str) -> list[str]:
    """提取 chunk 中出现的**全部**条款号（法条常引用他条）。"""
    if not text:
        return []
    out: list[str] = []
    for m in RE_ARTICLE.finditer(text):
        n = cn_to_int(m.group(1))
        if n is not None:
            out.append(f"第{n}条")
    return out


def normalize_article(text: str) -> str | None:
    """把任意写法的条款引用归一为"第N条[第M款][第K项]"（全部阿拉伯数字）。

    "第六十条第一款第七项" -> "第60条第1款第7项"
    "第60条第1款"          -> "第60条第1款"
    "本法第15条"          -> "第15条"

    :return: 解析失败返回 None
    """
    if not text:
        return None
    m = RE_ITEM.search(text)
    if not m:
        m2 = RE_ARTICLE.search(text)
        if not m2:
            return None
        n = cn_to_int(m2.group(1))
        return f"第{n}条" if n is not None else None

    n = _strip_num(m.group(1))
    if n is None:
        return None
    out = f"第{n}条"
    k = _strip_num(m.group(2))
    if k is not None:
        out += f"第{k}款"
    j = _strip_num(m.group(3))
    if j is not None:
        out += f"第{j}项"
    return out


def article_matches(text: str, target: str) -> bool:
    """chunk 文本是否包含目标条款（数字归一后比较，避免"六十" vs "60" 漏配）。

    层级降级：目标带款/项时，先尝试全等，再退到仅比"条"，最后退到数字子串。

    :param text:   chunk 内容
    :param target: 目标条款，如 "第60条" 或 "第六十条第一款"
    """
    if not text or not target:
        return False
    tgt = normalize_article(target) or target
    if tgt in text:
        return True

    # 降级1：只比"条"级
    m = RE_ARTICLE.search(tgt)
    base = None
    if m:
        base = f"第{cn_to_int(m.group(1))}条"
        if base and base in text:
            return True

    # 降级2：文本可能全用中文数字，反向归一文本中的条款号再比
    for art in all_articles_of(text):
        if art == tgt or (base and art == base):
            return True
        # 层级包含关系：目标"第60条第一款" vs 文本"第60条"
        if tgt.startswith(art) or art.startswith(tgt):
            return True
    return False


# ==================== 条款匹配分级 ====================
# 为什么要分级（由真实召回失败案例驱动，非理论设计）：
#
# 评估问题"《消防法》第六十条第一款第七项规定了什么？"在 top_k=5 下
# precision=0 / recall=0，召回的5 条分别是"第六十条第一款第一项""第四十七条"
# "第六十条第一款第二项"…… 全是"第六十条的邻居"，唯独缺第七项本身。
#
# 原因：旧 article_matches 的层级降级只比对到"条"级，把款/项的差异全丢了。
# 于是"第六十条第一款第一项"与"第六十条第一款第七项"同权，
# 解读性片段把真正答案挤出 top-k。
#
# 另一个反直觉点：法条原文的项写作"（七）"而不是"第7项"，所以纯字面匹配
# 对**真正的答案**返回 False，对噪声片段返回 True —— 区分度完全反了。
# 必须识别中文括号枚举标记。

_CN_D_FOR_INT = {0: "零", 1: "一", 2: "二", 3: "三", 4: "四",
                 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}


def _int_to_cn(n: int) -> str:
    """int -> 中文数字（0~99）。7 -> "七"；15 -> "十五"；20 -> "二十"。"""
    if n < 0 or n > 99:
        return str(n)
    if n < 10:
        return _CN_D_FOR_INT[n]
    tens, ones = divmod(n, 10)
    head = "十" if tens == 1 else _CN_D_FOR_INT[tens] + "十"
    return head + (_CN_D_FOR_INT[ones] if ones else "")


def _strip_clauses(normalized: str, depth: int) -> str:
    """从归一条款串取前 depth 级。

    "第60条第1款第7项" depth=2 -> "第60条第1款"；depth=1 -> "第60条"
    """
    parts = normalized.split("第")
    if depth <= 1:
        return f"第{parts[1]}"
    if len(parts) < 2 + depth:
        return f"第{parts[1]}"
    return "第" + "第".join(parts[1:1 + depth])


def _clause_no(normalized: str, depth: int) -> int | None:
    """取归一条款串第 depth 级编号。"第60条第1款第7项" depth=3 -> 7。

    注意 split("第") 后每段仍带"条/款/项"后缀（"1款"/"7项"），
    必须先剥离，否则 cn_to_int 解析失败返回 None。
    """
    parts = normalized.split("第")
    if len(parts) < 1 + depth:
        return None
    return cn_to_int(re.sub(r"[条款项]", "", parts[depth]))


def _all_item_markers(text: str) -> set:
    """提取文本中所有中文括号枚举项的编号。"（一）…（七）" -> {1, ..., 7}。"""
    out = set()
    for m in re.finditer(r"[（(]\s*([一二三四五六七八九十]+)\s*[）)]", text):
        v = cn_to_int(m.group(1))
        if v is not None:
            out.add(v)
    return out


def _has_item_marker(text: str, item_no: int) -> bool:
    """文本中是否出现第 item_no 个括号枚举项标记，如"（七）"。"""
    if item_no < 1:
        return False
    cn = _int_to_cn(item_no)
    # 兼容半角"(七)"与全角"（七）"，允许中间空白
    return re.search(rf"[（(]\s*{re.escape(cn)}\s*[）)]", text) is not None


def _mentions_other_item(text: str, tgt: str) -> bool:
    """文本是否显式提到了与目标**不同**的项号。

    识别两类写法：归一条款串"第60条第1款第6项"，以及枚举标记"（六）"。
    """
    item_no = _clause_no(tgt, 3)
    if item_no is None:
        return False

    for m in RE_ITEM.finditer(text):
        j = _strip_num(m.group(3))
        if j is not None and int(j) != item_no:
            return True

    # 只在文本出现多个枚举标记时才做"存在其他项"判断，避免单标记误报
    markers = _all_item_markers(text)
    return len(markers) > 1 and item_no not in markers


def _mentions_clause(text: str, tgt: str) -> bool:
    """文本是否显式提到了目标款号。

    两种证据：归一条款串"第60条第1款"，或中文括号"（第一款）"。
    """
    clause_no = _clause_no(tgt, 2)
    if clause_no is None:
        return False

    for m in RE_ITEM.finditer(text):
        k = _strip_num(m.group(2))
        if k is not None and int(k) == clause_no:
            return True

    cn = _int_to_cn(clause_no)
    return re.search(rf"[（(]\s*第\s*{re.escape(cn)}\s*款\s*[）)]", text) is not None


def article_match_level(text: str, target: str) -> int:
    """条款匹配**精确度分级**：3=完全命中目标，2=到款，1=到条，0=不匹配。

    分级结果直接决定精排权重，"精确到项"必须远重于"只到条"，
    否则解读性片段会与法条原文同权，把答案挤出 top-k。

    级别是**相对目标粒度**的：目标只到"条"时，命中该条即满级 3；
    目标到"项"时，命中项才是 3，只到条只能算 1。

    :param text:   chunk 内容
    :param target: 目标条款，如"第六十条第一款第七项"
    :return: 0~3
    """
    if not text or not target:
        return 0

    tgt = normalize_article(target)
    if not tgt:
        return 0

    item_no = _clause_no(tgt, 3)
    clause_no = _clause_no(tgt, 2)

    # 目标只到"条"：命中即满级，没有更细层级可区分
    if item_no is None and clause_no is None:
        target_art = _strip_clauses(tgt, 1)
        if target_art in text or target_art in all_articles_of(text):
            return 3
        return 0

    # 情形 A：文本直接写出归一条款串（解读性片段多为此类）
    for level, cand in ((3, tgt),
                        (2, _strip_clauses(tgt, 2)),
                        (1, _strip_clauses(tgt, 1))):
        if cand and cand in text:
            return level

    # 情形 B：法条原文的枚举项写作"（七）"而非"第7项" —— 真正答案的典型形态
    target_art = _strip_clauses(tgt, 1)
    if target_art not in all_articles_of(text):
        return 0

    if item_no is not None and _has_item_marker(text, item_no):
        return 3

    # 情形 B-：文本明确提到了**别的项**，说明它讲的不是目标项。
    # 这类片段是主要噪声源，只能按"到条"计权，不能因为顺带提到
    # "第一款"就升到"到款"。
    if item_no is not None and _mentions_other_item(text, tgt):
        return 1

    # 情形 C：只确认到"款"。必须文本里确实出现了目标款号，否则
    # "第六十条违反本法规定的"这种只提条号的片段会被误升到到款级。
    if clause_no is not None and _mentions_clause(text, tgt):
        return 2
    return 1


# ==================== 裁量基准表格重组 ====================
# precision 低的头号噪声源（实测：RAGAS 报告里的多数噪声 chunk 属此类）。
#
# 成因：裁量基准附件是**三列表格**，PDF 抽出来是这样：
#     第二十三条
#     定》第五十条
#     严重 1.申请人以欺骗、贿赂等不正当手段取得…的。
#     对未经准予变更注册而继续执业的行政处罚
#     定》第五十二条
#     严重 1.以注册消防工程师名义在5 份以上…的。
#     一般 1.以注册消防工程师名义在2 至 4 份…的。
#     较轻 1.以注册消防工程师名义在1 份…的。
#     对执业活动质量不符合标准的行政处罚
#     严重 1.有5 处（项）以上不符合标准的。
#     …
#
# 三列被拍平成一维文本，左列"违法行为名"还被 PDF 折行打散成
# "对执业活动质/量不符合国家/标准、行业标/准的行政处罚" 四行。
#
# 法条边界切分在这里完全失效（表格行没有"第X条"），
# 500 字硬切的结果是一块里塞了 3~5 个互不相干的违法行为主题
# —— 查电动自行车会召回"有5 处（项）以上不符合标准的"。
#
# 解法：按「档位行（严重/一般/较轻）」切分组，每组 = 一个违法行为
# + 它的全部档位，这才是一个语义原子。

# 档位行：严重/一般/较轻/不予处罚/从重处罚 后跟序号
_RE_TIER_ROW = re.compile(
    r"^\s*(?:不予处罚|从重处罚|从轻处罚|减轻处罚|严重|一般|较轻|轻微)"
    r"\s*[0-9０-９]*\s*[.．、。]?"
)


def _is_tier_row(line: str) -> bool:
    """是否是裁量基准的档位行（"严重 1.xxx" / "较轻 1.xxx"）。"""
    return bool(_RE_TIER_ROW.match(line))


def _join_wrapped(name_lines: list[str]) -> str:
    """把 PDF 折行打散的左列还原成完整名称。

    "对执业活动质" + "量不符合国家" + "标准、行业标" + "准的行政处罚"
    -> "对执业活动质量不符合国家标准、行业标准的行政处罚"
    """
    return "".join(l.strip() for l in name_lines)


def regroup_tier_table(text: str) -> list[str] | None:
    """把裁量基准表格重组为「一个违法行为 + 其全部档位」的语义单元。

    真实行序（已用生产索引逐行核对）：
        对未经准予变          <- 名称行（PDF 折行打散）
        更注册而继续
        执业的行政处
        师管理规定》 第二
        十二条
        定》第五十二
        严重 1.以注册消防工程师名义在5 份以上…的。   <- 档位行
        一般 1.…2 至 4 份…的。
        较轻 1.…1 份…的。
        对消防安全技              <- 下一个名称行
        …
    即「名称行(多行) → 档位行(1~3行)」循环。算法：逐行扫描，遇到档位行
    就回溯吃掉它前面连续的名称行，构成一个语义单元。

    :param text: 含档位行的文本
    :return: 重组后的语义单元列表；不含表格结构时返回 None
    """
    lines = text.split("\n")
    tier_idx = [i for i, l in enumerate(lines) if _is_tier_row(l)]

    # 至少两档、且总量合理，否则不是裁量基准表
    # （实测单块最多 10 个档位行，阈值 14 留余量）
    if len(tier_idx) < 2 or len(tier_idx) > 14:
        return None

    # 把档位行按"是否属于同一组"聚合：两个档位行之间若只有空行，
    # 就是同一违法行为的不同档位；夹着实质内容行则是不同违法行为。
    # 不能简单用 tier_idx 相邻性判断 —— PDF 抽出的档位行之间常有空行。
    groups_tier: list[list[str]] = []
    cur: list[str] = [lines[tier_idx[0]].strip()]
    for a, b in zip(tier_idx, tier_idx[1:]):
        gap = lines[a + 1:b]
        if all(not g.strip() for g in gap):
            cur.append(lines[b].strip())
        else:
            groups_tier.append(cur)
            cur = [lines[b].strip()]
    groups_tier.append(cur)

    # 前导残片（表头/法条号，如"第二十三条" "定》第五十条"）不属于任何名称行
    head_end = tier_idx[0]
    head = _join_wrapped([l for l in lines[:head_end] if l.strip()])

    out: list[str] = []
    # 名称行收集：pos 标记名称块起点，逐组向前推进。
    #
    # pos 初值用 head_end（而非 0）：表头行已在 head 里收集过，从 0 开始会
    # 把它当成第一组的名称行重复拼一次（实测踩过）。
    pos = head_end  # 下一个尚未消费的行号（跳过表头）
    for gi, tier_lines in enumerate(groups_tier):
        first_tier_line = next(i for i in range(pos, len(lines))
                               if _is_tier_row(lines[i]))
        # 名称块 = [pos, first_tier_line)
        name_lines = [l for l in lines[pos:first_tier_line] if l.strip()]
        name = _join_wrapped(name_lines)

        # 兜底：某些表格首组的名称行被PDF 抽进了表头区，导致名称为空。
        # 此时用表头顶上，否则整个首组会被 if len(name) >= 6 丢掉 ——
        # 丢掉的是"第一个违法行为"，代价太大。
        if len(name) < 6 and gi == 0 and head:
            name = head

        if len(name) >= 6:
            body = name + "。\n" + "\n".join(tier_lines)
            out.append(body)

        # 消费掉本组的名称行 + 档位行（含其间空行）
        last = first_tier_line
        while last < len(lines) and (
                _is_tier_row(lines[last]) or not lines[last].strip()):
            last += 1
        pos = last

    if not out:
        return None

    # 表头只在第一组里出现一次（它是全表的编号，不属于某个违法行为）。
    # 若首组已用 head 兜底当名称，则不重复前置。
    if head and not out[0].startswith(head):
        out[0] = f"{head}\n{out[0]}"
    return out


# ==================== 目录条目切分（P4） ====================
#
# 47 条评测打出来的第四个补丁。现象与排查过程记录在此，因为它和P1
# 是同一类错误的两个实例：
#
# 查"消防产品目录中的电线电缆类产品包括哪些？"，答案就在一个 chunk 里：
#     "一、电线电缆（3 种）
#      1.不包括阻燃电线电缆、耐火电线电缆…"
# 但该 chunk 完整形态是：
#     [自我声明程序 A/B 的说明]
#     一、电线电缆（3 种）
#      1.不包括阻燃电线电缆…
#     1.电线组件（0101） 适用于家用和类似一般设备所用的…   <- 下一个条目
#
# 即：用户要的是"电线电缆"这一个条目，而它被夹在"总体说明"和"下一个条目"
# 之间，**没有任何结构性标记告诉检索系统"到这里是一个条目的结束"**。
# Bag-of-words 对齐失效 —— 命中"电线电缆"的 chunk 混了另外两个主题，
# 命中"产品/目录"的 chunk 又不含答案，无论怎么调权重都排不准。
#
# 这和 P1（裁量基准表格）本质相同：**PDF 的视觉结构在纯文本里丢失了，
# 必须在入库前把它恢复出来**，否则检索层再聪明也无从判断边界。
#
# 目录条目的结构特征（消防产品目录实测）：
#     一、电线电缆（3 种）            <- 中文序数 + 类目名 + 计数
#     1.电线组件（0101） 适用于…    <- 阿拉伯序数 + 编码 + 名词释义
#     4.插头插座（0201） 1.适用于…  <- 编号 + 类目名 + 适用说明
# 切点 = 中文序数行，或"数字. 名词（数字编码）"行。

# ★ 目录条目边界识别（P4）
#
# 背景：PDF 的视觉结构在纯文本里丢失了。《消防产品目录》里
# "一、电线电缆（3 种）/ 1.不包括阻燃电线电缆…" 是两行独立内容，
# 但入库时会被切在一个大chunk 里和自我声明说明粘在一起，检索时排名极低。
#
# 实测两份真实目录文档，边界信号完全不同，因此需要两套判据：
#
#   A. 序号型——《关于优化强制性产品认证目录的公告（2020年第18号）》
#      4056 个非空行里：
#        中文序数+类目名+计数   16 处
#        数字.名词(4位编码)      19 处
#      典型："一、电线电缆（3种）" / "1.电线组件（0101） 适用于…"
#
#   B. 标准号锚型——《消防产品目录（2022年修订本）》
#      2264 个非空行里，序号型切点 **0 处**，通篇是三行拍平的
#      "产品名+说明 / 说明续行 / GB 标准号"：
#
#        点型感温火灾探测器 对温度和/或升温速率…的火灾
#        探测器。 GB 4716
#        独立式感烟火灾探测报警器 包括感烟探测、…的报警器
#
#      唯一稳定信号是**标准号独占整行**。注意必须是"独占"：
#      导则类正文中标准号是句内引用（"GB 50354 等技术标准检查验收合格
#      基础上进行…"），行首匹配会把整篇切碎。
#
# 全库 103 个文件实测（判据零误伤，见下表）：
#
#   文件                                        非空行  序数切点  独占标准号  占比
#   关于优化强制性产品认证目录的公告              4056    35       365      9.00%
#   （应急消〔2022〕136号）消防产品目录2022        2264     0       127      5.61%
#   浙江省建筑工程消防验收操作技术导则（试行）    8561     0        13      0.15%
#   浙江省电动自行车充停、充换场所建设技术导则     345     0         0      0.00%
#   其余 99 个文件                                —       0         0      ~0
#
# 导则的 0.15% 与目录的 5.61% 差37 倍，阈值取 2% 留足安全边际。

# ---- 体系 A：序号型切点 ----
_RE_CATALOG_HEAD = re.compile(
    r"^\s*[一二三四五六七八九十]+\s*[、.．]\s*\S+?（\s*\d+\s*种\s*）"
)
_RE_CATALOG_ITEM = re.compile(
    r"^\s*\d+\s*[.．]\s*\S+?[（(]\s*\d{3,4}\s*[)）]"
)

# ---- 体系 B：标准号**独占整行** ----
# 允许 GB 15631 / GB/T15934 / GB 15322.1 / DB33/T 1021 及尾部"等""、"。
# 行尾必须干净（可选一个"等/、"），确保整行除了标准号没有正文。
_RE_SOLO_STD = re.compile(
    r"^\s*(?:GB|GB/T|DB|DB\d{2}|JGJ|CJJ)\s*[A-Z]?\s*\d{2,5}[\d.\-–—]*"
    r"\s*(?:[、,，]?\s*(?:等)?)?\s*$"
)

# 体系 A 启用阈值：至少这么多切点才认定是目录（正文偶然出现编号不算）
_CATALOG_MIN_CUTS = 8
# 体系 B 启用阈值：独占标准号行占非空行的比例
_CATALOG_STD_RATIO = 0.02
_CATALOG_MIN_STD = 8


def _is_catalog_head(line: str) -> bool:
    return bool(_RE_CATALOG_HEAD.match(line))


def _is_catalog_item(line: str) -> bool:
    return bool(_RE_CATALOG_ITEM.match(line))


def _is_solo_std(line: str) -> bool:
    """是否是独占整行的标准号（产品目录条目的结尾锚点）。"""
    return bool(_RE_SOLO_STD.match(line))


def _slice_by_cuts(
    lines: list[str],
    cuts: list[int],
    std_cut: bool = False,
) -> list[str]:
    """按切点序列把行切成块。

    :param std_cut: True 表示切点是"标准号行"，此时标准号行归**前一块**
        （它是条目的收尾）；False 表示切点行作为**后一块**的开头
        （序号行本身就是条目头）。
    """
    out: list[str] = []
    lead = [l for l in lines[:cuts[0]] if l.strip()]
    if lead:
        out.append("\n".join(lead))
    bounds = list(zip(cuts, cuts[1:] + [len(lines)]))
    for a, b in bounds:
        seg = lines[a:b] if std_cut else lines[a:b]
        body = [l for l in seg if l.strip()]
        if body:
            out.append("\n".join(body))
    return out


def regroup_catalog_entries(text: str) -> list[str] | None:
    """把目录类文本按**条目边界**切开，每个条目 = 一个语义单元。

    两套判据（按真实数据，先A 后 B）：

      A. 序号型——"一、电线电缆（3 种）" / "1.电线组件（0101）"
         启用条件：切点数 >= ``_CATALOG_MIN_CUTS``。
         **不再用中位间隔判定**：实测该文档 35 个切点的中位间隔是 95 行
         （每个大类下有一整张几十行的产品表），任何"间隔要小"的阈值都会
         把它误判为非目录——这正是之前 `is_catalog=0` 的直接原因。

      B. 标准号锚型——《消防产品目录（2022年修订本）》通篇是
         "产品名+说明+GB 标准号"的拍平三行结构，没有任何序号。
         启用条件：**独占整行**的标准号占比 >= ``_CATALOG_STD_RATIO``
         且数量 >= ``_CATALOG_MIN_STD``。必须要求"独占整行"，否则
         《浙江省建筑工程消防验收操作技术导则》里句内引用的
         "GB 50354 等技术标准…" 会被当成切点，把 8561 行正文切碎。

    :param text: 候选文本
    :return: 条目列表；不含目录结构时返回 ``None``（走普通分块）
    """
    lines = text.splitlines()
    nonblank = [l for l in lines if l.strip()]
    if len(nonblank) < 20:
        return None

    # ---- 体系 A：序号型切点 ----
    cuts = [i for i, l in enumerate(lines)
            if _is_catalog_head(l) or _is_catalog_item(l)]
    if len(cuts) >= _CATALOG_MIN_CUTS:
        out = _slice_by_cuts(lines, cuts, std_cut=False)
        if len(out) >= 2:
            return out

    # ---- 体系 B：独占标准号锚型 ----
    std_cuts = [i for i, l in enumerate(lines) if _is_solo_std(l)]
    if len(std_cuts) < _CATALOG_MIN_STD:
        return None
    if len(std_cuts) / len(nonblank) < _CATALOG_STD_RATIO:
        return None

    # 首个标准号之前的内容（含前言）归入第一个条目。
    # 早期实现把"第一个标准号之前的全部行"当作前言单独成块，结果
    # 第一个产品条目的说明行被卷进前言块，条目失去自己的标准号。
    # PDF 纯文本里无法可靠区分前言与首条目描述，因此不剥离——
    # 宁可前言粘在首条目上，也不让条目边界丢失。
    out: list[str] = []
    start = 0
    for i in std_cuts:
        body = [l for l in lines[start:i + 1] if l.strip()]
        if body:
            out.append("\n".join(body))
        start = i + 1

    # 最后一个标准号之后的尾部（可能是下一个大类的标题）
    tail = [l for l in lines[std_cuts[-1] + 1:] if l.strip()]
    if tail:
        joined = "\n".join(tail)
        if not out or out[-1] != joined:
            out.append(joined)

    return out if len(out) >= 2 else None


# ==================== 法条感知分块 ====================

# 法条起始：行首或空白后的"第X条"
_RE_LAW_START = re.compile(rf"(?:^|(?<=[\s，。；：、]))第\s*{_NUM}\s*条")
# 章/节/编标题：第X章 / 第X节 / 第X编 —— 块边界上要剥离，否则会粘到上一块尾部
# 造成"第二条的chunk 末尾挂着 第三章火灾预防"这类跨界污染。
# 单独出现时（前后无正文）作为独立段保留，不丢字。
_RE_CHAPTER_LINE = re.compile(
    rf"^第\s*{_NUM}\s*[编章节][^\n]*$", re.MULTILINE)


class LegalChunker:
    """按法条边界优先切分，chunk 带law / article / chunk_id 元数据。

    切分策略（三级回退）：
        1. 先按法条边界切：一条 = 一个语义原子
        2. 单条过长 -> 在句号边界二次切分（保留 overlap）
        3. 仍过长 -> 硬切（最后兜底）

    元数据：
        law      法律名（从文件名推断，或由 add_law 注入）
        article  该 chunk 首条条款号，如 "第60条"；无则 None
        chunk_id 内容哈希，用于精确去重
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk: int = 40,
        law: str | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk = min_chunk
        self.law = law

    # ---- 法条边界切分 ----
    def _split_by_article(self, text: str) -> list[str]:
        """在"第X条"出现处切开，返回原文段落列表（保留原文，不丢字）。

        同时把章/节/编标题从所属法条段尾部剥离 —— 否则会出现
        "第二条的chunk 末尾挂着 第三章火灾预防" 的跨界污染。
        被剥离的标题并入下一段的开头（下一段即该章首条），语义归属正确。
        """
        starts = [m.start() for m in _RE_LAW_START.finditer(text)]
        if not starts:
            return [text]
        if starts[0] > 0:
            # 首个法条之前的内容（法律名、第一章标题等）作为前言段
            starts.insert(0, 0)

        segments: list[str] = []
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else len(text)
            seg = text[s:e].strip()
            if not seg:
                continue
            # 剥离段末的章/节标题行，移到下一段
            chapters: list[str] = []
            last_nl = seg.rfind("\n")
            if last_nl > 0:
                tail = seg[last_nl + 1:].strip()
                if tail and _RE_CHAPTER.match(tail) and \
                        RE_ARTICLE.search(seg[:last_nl]):
                    chapters = [tail]
                    seg = seg[:last_nl].rstrip()
            if not seg:
                # 该段正文全被标题占满，退回原样，交由 min_chunk 过滤
                seg = text[s:e].strip()
                chapters = []
            if seg:
                segments.append(seg)
            if chapters:
                segments.append(chapters[0])  # 标题作为独立短段
        return segments

    # ---- 句级二次切分 ----
    def _split_long(self, seg: str) -> list[str]:
        """对超长法条段，在句末标点处带 overlap 切分。"""
        if len(seg) <= self.chunk_size:
            return [seg]

        sentences = re.split(r"(?<=[。！？；])", seg)
        sentences = [s for s in sentences if s]
        if not sentences:
            return [seg[i:i + self.chunk_size]
                    for i in range(0, len(seg), self.chunk_size)]

        chunks: list[str] = []
        buf = ""
        for sent in sentences:
            # 单句就超长（极少见）：硬切
            if len(sent) > self.chunk_size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                for i in range(0, len(sent), self.chunk_size):
                    chunks.append(sent[i:i + self.chunk_size])
                continue
            if len(buf) + len(sent) <= self.chunk_size:
                buf += sent
            else:
                if buf:
                    chunks.append(buf)
                # 上一句尾部作为 overlap
                tail = buf[-self.chunk_overlap:] if self.chunk_overlap else ""
                buf = tail + sent
        if buf.strip():
            chunks.append(buf)
        return chunks

    # ---- 主入口 ----
    def split_text(self, text: str, source: str = "") -> list:
        """清洗 + 切分 + 打元数据。

        四级切分策略：
            1. 先按法条边界切：一条 = 一个语义原子
            2. 裁量基准表格 -> 按「违法行为 + 全部档位」重组
               （法条切分对表格无效，见 regroup_tier_table 的说明）
            3. 单条过长 -> 在句号边界二次切分（保留 overlap）
            4. 仍过长 -> 硬切（最后兜底）

        :param text:   原始全文（会先 clean_pdf_text）
        :param source: 来源文件名，用于元数据与法名推断
        :return:       Document 列表
        """
        cleaned = clean_pdf_text(text)
        if not cleaned:
            return []

        law = self.law or self._guess_law(source)
        Doc = _document_class()
        docs: list = []
        for seg in self._split_by_article(cleaned):
            for body, kind in self._split_segment(seg):
                # min_chunk 只过滤纯噪声残片；含法条号或章标题的一律保留
                # （"第十五条"这类短法条、以及"第三章 火灾预防"章标题
                #   本身都是完整语义单元，丢了就是 recall 损失）
                has_article = bool(RE_ARTICLE.search(body))
                has_chapter = bool(_RE_CHAPTER_LINE.search(body))
                if len(body) < self.min_chunk and not (has_article or has_chapter):
                    continue
                art = article_of(body)
                cid = hashlib.md5(
                    f"{source}|{art}|{body}".encode("utf-8")
                ).hexdigest()[:16]
                docs.append(Doc(
                    page_content=body,
                    metadata={
                        "law": law,
                        "article": art,
                        "source": source,
                        "chunk_id": cid,
                        "chars": len(body),
                        # 检索层据此差异化降权：
                        #   tier_table — 档位组并列多档位，通用查询下噪声大
                        #   catalog    — 目录条目，问目录时反而是答案（豁免降权）
                        "is_tier_table": kind == "tier_table",
                        "is_catalog": kind == "catalog",
                    },
                ))
        return docs

    def _split_segment(self, seg: str) -> list[tuple[str, str]]:
        """切分单个法条段 -> [(文本, 类型标记), ...]。

        类型标记：``"tier_table"``（裁量基准档位单元）/ ``"catalog"``（目录条目）
        / ``""``（普通）。检索层据此做差异化降权。

        切分优先级：目录条目切分（P4）→ 裁量表格重组（P1）→ 长度切分。
        两种结构重组都优先于长度切分：先按语义单元拆，再对仍然过长的
        单元做句级二次切分。

        **P4 必须排在 P1 之前**：实测《关于优化强制性产品认证目录的公告》
        里每个产品条目都是"若干行说明 + 一个处罚档位"的排版，
        ``regroup_tier_table`` 会把它整篇判成档位表（45 块全部标成
        tier_table），于是电线电缆这类**目录条目被当成档位组**，
        既丢了 ``is_catalog`` 标记，又让 P2 降权判据用错。
        目录特征（大量"数字.名词(4位编码)"切点、独占标准号密度）比
        档位特征更结构化、更难偶然命中，所以先判目录。
        """
        entries = regroup_catalog_entries(seg)
        if entries:
            out = []
            for e in entries:
                for piece in self._split_long(e):
                    body = piece.strip()
                    if body:
                        out.append((body, "catalog"))
            return out

        groups = regroup_tier_table(seg)
        if groups:
            out = []
            for g in groups:
                for piece in self._split_long(g):
                    body = piece.strip()
                    if body:
                        out.append((body, "tier_table"))
            return out

        return [(p.strip(), "") for p in self._split_long(seg) if p.strip()]

    @staticmethod
    def _guess_law(source: str) -> str | None:
        """从文件名猜法律名（去掉扩展名与"中华人民共和国"之外的修饰）。"""
        if not source:
            return None
        import os
        stem = os.path.splitext(os.path.basename(source))[0]
        return stem or None

    def split_documents(self, docs: Iterable) -> list:
        """兼容 langchain splitter 接口。"""
        out: list = []
        for d in docs:
            src = d.metadata.get("source", "")
            chunks = self.split_text(d.page_content, source=src)
            # 继承原文档的权限元数据（dept/secret_level/...不能丢）
            for c in chunks:
                c.metadata.update({
                    k: v for k, v in d.metadata.items()
                    if k not in ("article", "chunk_id", "chars")
                })
            out.extend(chunks)
        return out


# ==================== 目录/清单型 chunk 识别（P2：名录降权） ====================
#
# 生产索引实测（3990 chunk）后的结论 —— 与最初假设不同，记录在此备查：
#   最初以为噪声源是"标准号引用清单"（含 >=2 个 GB/T 号的 chunk，136 个）。
#   实测抽样发现其中 37 个是《浙江省建筑工程消防验收操作技术导则》的
#   **实质检查细则**（"检查内容和方法 / 查验抽样数量 / 误差允许范围"），
#   正是用户会问的答案，**绝不能降权**。真正 100% 是噪声的是另一类：
#
#     《消防产品目录（2022年修订本）》                48 chunk
#     《关于优化强制性产品认证目录的公告》            43 chunk
#     —— "一、电线电缆（3 种）1.不包括阻燃电线电缆…" 纯型号罗列，
#        不含任何义务性规定或处罚后果。
#
# 因此判据从"标准号数量"改为"**是否含义务/后果性表述**"，
# 标准号只作辅助特征（目录类文档几乎必然带标准号）。

# GB/T、GB、JGJ 等标准号引用
_RE_STD_REF = re.compile(
    r"(?:GB|DB|JGJ|CJJ|CECS|TB|SL|DL|QC)\s*/?\s*[A-Z]?\s*"
    r"\d{2,5}(?:[.\-]\d{1,4})*"
)

# 目录/清单型文档的标题特征
_RE_CATALOG_TITLE = re.compile(r"(目录|清单|名录|一览表|描述与界定表)")

# 义务性/后果性表述：出现即说明这段在"规定应当怎么做"或"违反会怎样"
_RE_NORMATIVE = re.compile(
    r"(?:应当|必须|不得|禁止|由.{0,8}处罚|处以|罚款|警告|责令|吊销|没收|"
    r"违法行为|法律责任|违规)"
)

# 标准号达到该数量即视为"型号罗列"辅证
STD_REF_THRESHOLD = 2

# 查询意图豁免词：**用户明确在问目录本身时，目录 chunk 就是答案**。
#
# 这是 47 条评测打出来的补丁：4/6 陷阱题 recall=0，根因全部是 P2 把
# 《消防产品目录》的 chunk 降权挤出 top-k —— 而用户问的恰恰是
# "消防产品目录涵盖哪些大类"，这些 chunk 里就写着"一、电线电缆（3 种）"。
#
# 教训：**降权判据只看 chunk 自身不够，必须叠一层查询意图。**
# 同一个 chunk 对"电动自行车怎么处罚"是噪声，对"产品目录有哪些"是唯一答案。
_CATALOG_INTENT = re.compile(r"(目录|名录|清单|一览|哪些类|大类|认证程序)")


def is_standard_catalog(text, metadata=None, query=None):
    """判断 chunk 是否为**目录/清单型条目**（型号罗列，非规范性内容）。

    命中条件：
      1. 文档标题含"目录/清单/名录/一览表/描述与界定表"
      2. 正文**不含**任何义务性或后果性表述（应当/不得/处罚/罚款/违法…）
      3. 正文有型号罗列特征：>=2 个标准号，或"（N 种）"式计数

    :param query: 用户查询原文。命中 _CATALOG_INTENT 时**豁免降权**——
        用户在问目录本身时，目录 chunk 是正确答案，不能压掉。
    """
    if not text:
        return False

    # P4 兜底：分块阶段已按目录条目切分的chunk（is_catalog=True）
    # 本身就是"一个条目 = 一个语义单元"，不再混淆多主题，
    # 这种不需要再降权 —— 它恰恰是目录类问题最该召回的东西。
    if (metadata or {}).get("is_catalog"):
        return False

    md = metadata or {}
    title = "%s%s" % (md.get("law", ""), md.get("source", ""))
    if not _RE_CATALOG_TITLE.search(title):
        return False

    # 条件 2：含义务/后果表述 -> 不是纯目录，放行
    if _RE_NORMATIVE.search(text):
        return False

    # ★ 查询意图豁免。放在标题/义务性两道闸门**之后**，
    # 否则标题不含"目录"的内容也会被误豁免，豁免范围就过宽了。
    if query and _CATALOG_INTENT.search(query):
        return False

    # 条件 3：型号罗列特征
    has_std = len(set(_RE_STD_REF.findall(text))) >= STD_REF_THRESHOLD
    has_count = bool(re.search(r"[（(]\s*\d+\s*种\s*[)）]", text))
    return has_std or has_count
def is_tier_table_chunk(doc) -> bool:
    """读取分块阶段写入的表格标记。

    表格行（裁量基准三列表被拍平）即使重组后，一段内仍并列多个
    "档位 + 后果"，对通用查询噪声大；但**不能直接过滤** ——
    用户问"电动自行车怎么处罚"时，裁量基准第五节的表格正是答案。
    因此检索层只做轻度降权，保留召回机会。
    """
    return bool((getattr(doc, "metadata", None) or {}).get("is_tier_table"))


__all__ = [
    "clean_pdf_text",
    "cn_to_int",
    "article_of",
    "all_articles_of",
    "normalize_article",
    "article_matches",
    "article_match_level",
    "regroup_tier_table",
    "regroup_catalog_entries",
    "is_standard_catalog",
    "is_tier_table_chunk",
    "LegalChunker",
]