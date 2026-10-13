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
    article_matches()   —— 条款号模糊匹配（数字归一 + 层级降级）
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

        :param text:   原始全文（会先clean_pdf_text）
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
            for piece in self._split_long(seg):
                body = piece.strip()
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
                    },
                ))
        return docs

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


__all__ = [
    "clean_pdf_text",
    "cn_to_int",
    "article_of",
    "all_articles_of",
    "normalize_article",
    "article_matches",
    "LegalChunker",
]