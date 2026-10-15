# change_detector.py
"""
人工修改检测Agent —— 对比"机器生成稿"与"人工修改后稿"，输出改动分析报告。

业务场景
---------
现有审批流是：generate → output/pending/ → 人工编辑 → approve → output/official/
但审批只记录"通过/驳回"，不记录**人工改了什么**。这导致：

  1. 模板质量缺陷无法闭环 —— 人工每次都要手改同样的地方，
     却没人统计过"到底哪几句被反复改"，改完也不会反哺模板。
  2. LLM 抽取错误无法定位 —— 分不清是"模型抽错了"还是"人理解不同"，
     两种问题解法完全相反。
  3. 风险改动不可见 —— 人工若把处罚金额从"五万"改成"五百万"，
     系统无感知；这类越权改动必须能被拦下。
  4. 幻觉无法追溯 —— 若终稿出现知识库里没有的条款，要能立刻报警。

本模块做四件事
--------------
1. **结构化 diff**：文本差异算法（difflib.SequenceMatcher），
   输出增/删/改的片段对，而不是一坨diff文本。
2. **LLM 语义归因**：对每个改动，调用 judge 模型判断
   "人为什么改" —— 属于抽取错误 / 事实修正 / 格式偏好 / 越权改动 / 其他。
3. **风险分级**：金额、条款号、法律名称等关键要素被改动 → 高风险，
   报告里单独标出，供管理员优先复核。
4. **报告落盘**：JSON（机器可读）+ Markdown（人可读）双份，
   存到 output/change_reports/，按 review_id 归档。

用法
----
    from change_detector import analyze_changes
    report = analyze_changes(original_path, modified_path, review_id="RV-xxx")

    # 命令行
    python change_detector.py --original a.docx --modified b.docx --review-id RV-001
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ==================== 报告输出目录 ====================
REPORT_DIR = os.path.join(_SCRIPT_DIR, "output", "change_reports")


# ==================== 风险要素识别 ====================
# 人工改动里最需要人工复核的几类要素：动它们等于改了法律适用或额度。
# 命中任一即标记为高风险，报告里单列出来。

# 金额（中文数字或阿拉伯数字 + 元/万元）
_RE_AMOUNT = re.compile(
    r"(?:[零一二三四五六七八九十百千万亿两壹贰叁肆伍陆柒捌玖拾佰仟\d.]+\s*"
    r"(?:元|万元|万|亿元|亿|罚款|处)"
    r"|\d+(?:\.\d+)?\s*(?:元|万元|亿元))"
)
# 条款引用
_RE_ARTICLE = re.compile(r"第\s*[一二三四五六七八九十百千零〇两\d]+\s*"
                         r"(?:条|款|项)")
# 法律名称（书名号）
_RE_LAW_NAME = re.compile(r"《[^》]{2,40}》")
# 关键程序性动词 —— 这些变了可能改变法律后果
_RE_PROCEDURE = re.compile(
    r"(责令改正|行政处罚|吊销|暂扣|停产停业|拘留|罚款|警告|没收|"
    r"从重|从轻|减轻|免予|不予|撤回|撤销|确认|受理|立案|终止)"
)


@dataclass
class ChangeItem:
    """一处改动。"""
    change_id: str
    change_type: str          # insert / delete / replace
    original_text: str
    modified_text: str
    similarity: float         # 0~1，越低改动越大
    risk_level: str           # high / medium / low
    risk_reasons: list[str] = field(default_factory=list)
    llm_reason: str = ""      # LLM 归因
    llm_category: str = ""    # 抽取错误/事实修正/格式偏好/越权改动/其他
    llm_suggestion: str = ""  # 建议动作


@dataclass
class ChangeReport:
    """一份完整的改动分析报告。"""
    report_id: str
    review_id: str
    created_at: str
    original_file: str
    modified_file: str
    total_changes: int
    high_risk_count: int
    summary: str
    changes: list[dict] = field(default_factory=list)
    llm_analysis: str = ""
    template_suggestions: list[str] = field(default_factory=list)
    # 落盘路径（save_report=True 时填充）
    json_path: str = ""
    markdown_path: str = ""


# ==================== 文本提取 ====================
def extract_text(path: str) -> str:
    """从 docx / txt / md 提取纯文本。docx 走python-docx。

    :raises ValueError: 不支持的文件类型
    """
    ext = os.path.splitext(path)[1].lower().lstrip(".")

    if ext in ("txt", "md", "json"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    if ext == "docx":
        try:
            from docx import Document
        except ImportError as e:
            raise ValueError(
                "解析 docx 需要 python-docx，请 pip install python-docx"
            ) from e
        doc = Document(path)
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        # 表格内容也要进diff —— 处罚金额常在表里
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts)

    raise ValueError(
        f"暂不支持 .{ext} 文件。请先转为 .docx 或 .txt 后重试。"
    )


# ==================== 风险判定 ====================
def assess_risk(original: str, modified: str) -> tuple[str, list[str]]:
    """判断一处改动是否触及关键法律要素。

    :return: (risk_level, reasons)
    """
    reasons: list[str] = []

    # 纯新增时"旧文本为空"，要素集合天然不同（[] → [x]），
    # 这不是"要素被改动"，而是"新增内容本身带要素"。
    # 只在两侧都有内容时才做"变更"判定，纯新增走"包含"判定。
    is_insert = not original.strip()
    is_delete = not modified.strip()

    # 金额变化 —— 处罚额度直接关系到行政强制措施合法性
    amt_old = set(_RE_AMOUNT.findall(original))
    amt_new = set(_RE_AMOUNT.findall(modified))
    if is_insert:
        if amt_new:
            reasons.append(f"新增内容含金额：{sorted(amt_new)}")
    elif is_delete:
        if amt_old:
            reasons.append(f"删除内容含金额：{sorted(amt_old)}")
    elif amt_old != amt_new:
        reasons.append(
            f"金额变更：{sorted(amt_old)} → {sorted(amt_new)}"
        )

    # 条款引用变化 —— 适用条款变了等于法律适用变了
    art_old = set(_RE_ARTICLE.findall(original))
    art_new = set(_RE_ARTICLE.findall(modified))
    if is_insert:
        if art_new:
            reasons.append(f"新增内容引用条款：{sorted(art_new)}")
    elif is_delete:
        if art_old:
            reasons.append(f"删除内容涉及条款：{sorted(art_old)}")
    elif art_old != art_new:
        reasons.append(f"条款变更：{sorted(art_old)} → {sorted(art_new)}")

    # 法律名称变化 —— 可能张冠李戴
    law_old = set(_RE_LAW_NAME.findall(original))
    law_new = set(_RE_LAW_NAME.findall(modified))
    if not is_insert and not is_delete and law_old != law_new:
        reasons.append(f"法律名称变更：{sorted(law_old)} → {sorted(law_new)}")
    elif is_insert and law_new:
        reasons.append(f"新增内容引用法律：{sorted(law_new)}")

    # 程序性动词变化 —— 处罚种类变了
    proc_old = set(_RE_PROCEDURE.findall(original))
    proc_new = set(_RE_PROCEDURE.findall(modified))
    if is_insert:
        if proc_new:
            reasons.append(f"新增内容含处置方式：{sorted(proc_new)}")
    elif is_delete:
        if proc_old:
            reasons.append(f"删除内容含处置方式：{sorted(proc_old)}")
    elif proc_old != proc_new:
        reasons.append(
            f"处置方式变更：{sorted(proc_old)} → {sorted(proc_new)}"
        )

    if reasons:
        # 纯新增即使含要素也只算中风险 —— 不该把补充说明与越权改罚同等对待
        return ("medium", reasons) if is_insert else ("high", reasons)

    # 改动幅度过大也算中风险：可能整段重写导致脱稿
    ratio = difflib.SequenceMatcher(None, original, modified).ratio()
    if ratio < 0.3:
        return "medium", ["整段重写，与原文相似度低于 30%"]
    return "low", []


# ==================== 结构化 diff ====================
def _split_units(text: str) -> list[str]:
    """把文本切成可比对的语义单元。

    中文按标点切、英文按单词切，保留标点便于还原。
    这样 diff 出来是"句子级"而不是"字符级"，LLM 才好归因。
    """
    if not text:
        return []
    # 先按换行与句末标点切，粒度太粗时再按逗号切
    parts = re.split(r"(?<=[。！？；\n])", text)
    units: list[str] = []
    for p in parts:
        if not p.strip():
            continue
        if len(p.strip()) <= 60:
            units.append(p)
            continue
        # 长段落细分到分句
        sub = re.split(r"(?<=[，、：])", p)
        units.extend(s for s in sub if s.strip())
    return units


def diff_units(old: str, new: str) -> list[dict]:
    """产出结构化改动列表。

    :return: [{"type", "original", "modified"}]
    """
    a = _split_units(old)
    b = _split_units(new)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)

    out: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_part = "".join(a[i1:i2]).strip()
        new_part = "".join(b[j1:j2]).strip()

        if tag == "insert":
            out.append({"type": "insert", "original": "", "modified": new_part})
        elif tag == "delete":
            out.append({"type": "delete", "original": old_part, "modified": ""})
        else:  # replace
            # replace 可能是一对多/多对一，按相似度配对，剩余归为增删
            pairs, used_new = _pair_similar(a[i1:i2], b[j1:j2])
            paired_old = {p[0] for p in pairs}
            for o, n in pairs:
                out.append({"type": "replace", "original": o, "modified": n})
            # 未被配对的新文本 -> 纯新增
            for idx, n in enumerate(b[j1:j2]):
                if idx not in used_new and n.strip():
                    out.append({"type": "insert", "original": "",
                                "modified": n.strip()})
            # 未被配对的旧文本 -> 纯删除
            for o in a[i1:i2]:
                if o not in paired_old and o.strip():
                    out.append({"type": "delete", "original": o.strip(),
                                "modified": ""})
    return out


def _pair_similar(old_units: list[str],
                  new_units: list[str]) -> tuple[list[tuple[str, str]], set]:
    """在 replace 块内做相似度配对，让改动以"改前→改后"呈现。

    :return: (配对列表, 被消耗掉的新文本下标集合)
    """
    pairs: list[tuple[str, str]] = []
    used_new: set[int] = set()

    for o in old_units:
        best_idx, best_r = -1, 0.0
        for idx, n in enumerate(new_units):
            if idx in used_new:
                continue
            r = difflib.SequenceMatcher(None, o, n).ratio()
            if r > best_r:
                best_idx, best_r = idx, r
        # 0.3 阈值：低于此视为"删一段+加一段"而非"修改"
        if best_idx >= 0 and best_r >= 0.3:
            used_new.add(best_idx)
            pairs.append((o, new_units[best_idx]))

    return pairs, used_new


# ==================== LLM 归因 ====================
_LLM_PROMPT = """你是消防文书审核专家。下面是同一份文书在人工修改前后的差异。

【改动前】{old}
【改动后】{new}

判断人工为什么要改这一处，从以下类型中选一个：
1. 抽取错误 —— 模型从案情抽取字段时抽错了，人在纠正
2. 事实修正 —— 人工补充或更正了案件事实
3. 格式偏好 —— 仅排版/措辞风格调整，不涉及内容正确性
4. 越权改动 —— 改变了处罚金额、条款、法律适用等，需管理员复核
5. 其他 —— 以上都不符合

严格依据两段文本的实际差异判断，不要推测文本之外的信息。
以 JSON 返回：{{"category": "<类型>", "reason": "<一句话说明依据>", "suggestion": "<给管理员的建议动作>"}}"""


def _llm_analyze(old: str, new: str, max_chars: int = 300) -> dict:
    """调用 LLM 判断改动原因。失败返回空 dict（不阻塞主流程）。"""
    try:
        from config import chat_json
        prompt = _LLM_PROMPT.format(
            old=old[:max_chars] or "（空）",
            new=new[:max_chars] or "（空）",
        )
        result = chat_json([{"role": "user", "content": prompt}],
                           temperature=0.0, timeout=60)
        return result if isinstance(result, dict) else {}
    except Exception:
        # LLM 不可用不该让整个检测挂掉 —— 报告仍要出，只是没有归因
        return {}


def _overall_analysis(changes: list[dict], qa_text: str) -> str:
    """让 LLM 汇总整体改动画像与根因。"""
    if not changes:
        return ""
    try:
        from config import chat_text
        summary = []
        for c in changes[:20]:
            summary.append(
                f"[{c['change_type']}] "
                f"改动前: {c.get('original', '')[:80]}\n"
                f"改动后: {c.get('modified', '')[:80]}"
            )
        prompt = (
            "你是文书生成系统的质量负责人。以下是一份文书的人工修改记录，"
            "共 {n} 处改动，其中高风险{h} 处。\n\n{body}\n\n"
            "请分析：\n"
            "1. 这些改动反映的是模型抽取问题、模板问题，还是人工个性化需求\n"
            "2. 哪些属于系统性问题（应从代码/模板层面修），"
            "哪些属于一次性修正（不必改）\n"
            "3. 按优先级给出改进建议\n"
            "用中文回答，控制在 400 字内，直接给结论不要客套。"
        ).format(n=len(changes), h=sum(
            1 for c in changes if c.get("risk_level") == "high"),
            body="\n".join(summary))

        return chat_text([{"role": "user", "content": prompt}],
                         temperature=0.2, timeout=120)
    except Exception:
        return ""


# ==================== 报告渲染 ====================
def _render_markdown(rep: ChangeReport) -> str:
    """渲染 Markdown 报告（给人看）。"""
    lines = [
        f"# 文书人工改动分析报告",
        "",
        f"- **报告编号**：{rep.report_id}",
        f"- **审批单号**：{rep.review_id}",
        f"- **生成时间**：{rep.created_at}",
        f"- **改动总数**：{rep.total_changes} 处（高风险 {rep.high_risk_count} 处）",
        "",
        "## 一、结论摘要",
        "",
        rep.summary or "（无）",
        "",
        "## 二、高风险改动（需管理员优先复核）",
        "",
    ]

    high = [c for c in rep.changes if c["risk_level"] == "high"]
    if not high:
        lines.append("无高风险改动。")
    else:
        lines.append("| # | 改动前 | 改动后 | 风险原因 |")
        lines.append("|---|---|---|---|")
        for i, c in enumerate(high, 1):
            lines.append(
                f"| {i} | {_trunc(c['original_text'])} | "
                f"{_trunc(c['modified_text'])} "
                f"| {'；'.join(c['risk_reasons'])} |"
            )

    lines += ["", "## 三、全部改动明细", ""]
    if not rep.changes:
        lines.append("未检测到改动（人工直接批准了原稿）。")
    else:
        for i, c in enumerate(rep.changes, 1):
            lines.append(
                f"### {i}. [{c['change_type']}] "
                f"风险：{ {'high': '高', 'medium': '中', 'low': '低'}[c['risk_level']] }"
            )
            lines.append("")
            lines.append(
                f"- 改动前：{_trunc(c['original_text'], 200) or '（无）'}")
            lines.append(
                f"- 改动后：{_trunc(c['modified_text'], 200) or '（无）'}")
            if c["risk_reasons"]:
                lines.append(f"- 风险原因：{'；'.join(c['risk_reasons'])}")
            if c.get("llm_category"):
                lines.append(
                    f"- 归因：{c['llm_category']} —— {c.get('llm_reason', '')}"
                )
            if c.get("llm_suggestion"):
                lines.append(f"- 建议：{c['llm_suggestion']}")
            lines.append("")

    if rep.llm_analysis:
        lines += ["## 四、根因分析与改进建议", "", rep.llm_analysis, ""]

    if rep.template_suggestions:
        # 章节号随前文是否有内容顺延，避免出现「四」之后直接跳到「五」
        nxt = "五" if rep.llm_analysis else "四"
        lines += [f"## {nxt}、模板改进建议", ""]
        for s in rep.template_suggestions:
            lines.append(f"- {s}")
        lines.append("")

    return "\n".join(lines)


def _trunc(s: str, n: int = 120) -> str:
    """截断长文本用于表格显示。"""
    s = (s or "").replace("\n", " ").replace("|", "/").strip()
    return s if len(s) <= n else s[:n] + "…"


# ==================== 主入口 ====================
def analyze_changes(
    original_path: str,
    modified_path: str,
    review_id: str = "",
    use_llm: bool = True,
    save_report: bool = True,
) -> ChangeReport:
    """对比两份文书的改动，产出分析报告。

    :param original_path: 机器生成稿（output/pending/xxx.docx）
    :param modified_path: 人工修改后的稿
    :param review_id:     审批单号，用于归档关联
    :param use_llm:是否调用 LLM 做语义归因（关闭则只做规则分析）
    :param save_report:   是否落盘
    :return: 改动报告
    :raises FileNotFoundError: 任一文件不存在
    """
    for p in (original_path, modified_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"文件不存在：{p}")

    old_text = extract_text(original_path)
    new_text = extract_text(modified_path)

    raw_changes = diff_units(old_text, new_text)

    changes: list[dict] = []
    for idx, rc in enumerate(raw_changes, 1):
        o, n = rc["original"], rc["modified"]
        level, reasons = assess_risk(o, n)

        item = ChangeItem(
            change_id=f"C{idx:03d}",
            change_type=rc["type"],
            original_text=o,
            modified_text=n,
            similarity=round(
                difflib.SequenceMatcher(None, o, n).ratio(), 3),
            risk_level=level,
            risk_reasons=reasons,
        )

        # 只对高风险和中等风险调用 LLM —— 纯格式类改动不值得花 token
        if use_llm and level in ("high", "medium"):
            j = _llm_analyze(o, n)
            item.llm_category = j.get("category", "")
            item.llm_reason = j.get("reason", "")
            item.llm_suggestion = j.get("suggestion", "")

        changes.append(asdict(item))

    high_n = sum(1 for c in changes if c["risk_level"] == "high")

    # 摘要
    if not changes:
        summary = "未检测到人工改动，原稿直接进入审批。"
    else:
        types: dict[str, int] = {}
        for c in changes:
            types[c["change_type"]] = types.get(c["change_type"], 0) + 1
        type_txt = "、".join(
            {"insert": "新增", "delete": "删除", "replace": "修改"}[k]
            for k in types
        )
        summary = f"共 {len(changes)} 处改动（{type_txt}），其中高风险 {high_n} 处。"
        if high_n:
            summary += "高风险改动涉及金额/条款/法律适用，必须人工确认后方可归档。"
        else:
            summary += "未触及关键法律要素，风险可控。"

    # 模板改进建议（从改动里聚合"反复出现的位置"）
    suggestions: list[str] = []
    if len(changes) >= 3:
        suggestions.append(
            f"本次改动 {len(changes)} 处且集中在同类问题，"
            f"建议核查对应模板是否存在字段缺失或表述歧义"
        )
    if any(c["risk_level"] == "high" for c in changes):
        suggestions.append(
            "存在关键要素改动，应在模板中显式标注该字段来源，"
            "并考虑对高风险字段做二次确认"
        )

    rep = ChangeReport(
        report_id=f"CR-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
        review_id=review_id or "N/A",
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        original_file=original_path,
        modified_file=modified_path,
        total_changes=len(changes),
        high_risk_count=high_n,
        summary=summary,
        changes=changes,
        llm_analysis=_overall_analysis(changes, new_text) if use_llm else "",
        template_suggestions=suggestions,
    )

    if save_report:
        rep.json_path, rep.markdown_path = save_report_files(rep)

    return rep


# 文件名中非法字符（review_id 来自外部输入，不能直接拼进路径）
_UNSAFE_FS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename(raw: str, fallback: str = "unknown") -> str:
    """把外部输入清洗成安全的单个文件名片段。

    过滤三类问题：
      1. 路径分隔符与 Windows 保留字符
      2. 路径穿越残留 ".."（本身不含分隔符，需单独处理）
      3. 首尾空白与结尾的点/空格（Windows 下会被静默截断）

    供落盘与 glob 两侧共用，保证清洗规则一致。
    """
    cleaned = _UNSAFE_FS.sub("_", (raw or "").strip())
    cleaned = cleaned.replace("..", "_")
    cleaned = cleaned.rstrip(". ")
    return cleaned or fallback


def _safe_stem(review_id: str, report_id: str) -> str:
    """生成安全的报告文件名 stem（review_id + report_id）。"""
    return f"{safe_filename(review_id)}_{safe_filename(report_id)}"


def save_report_files(rep: ChangeReport) -> tuple[str, str]:
    """把报告落盘，返回 (json 路径, markdown 路径)。

    JSON 供程序汇总统计，Markdown 供管理员直接阅读。
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    stem = _safe_stem(rep.review_id, rep.report_id)

    json_path = os.path.join(REPORT_DIR, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(rep), f, ensure_ascii=False, indent=2)

    md_path = os.path.join(REPORT_DIR, f"{stem}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(rep))

    return json_path, md_path


# ==================== CLI ====================
def main():
    p = argparse.ArgumentParser(
        description="文书人工改动检测：对比改动前后并输出分析报告")
    p.add_argument("--original", required=True, help="机器生成稿路径")
    p.add_argument("--modified", required=True, help="人工修改后稿路径")
    p.add_argument("--review-id", default="", help="审批单号")
    p.add_argument("--no-llm", action="store_true",
                   help="跳过 LLM 归因（只做规则分析，不消耗 token）")
    args = p.parse_args()

    try:
        rep = analyze_changes(
            args.original, args.modified,
            review_id=args.review_id,
            use_llm=not args.no_llm,
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1

    print(f"\n{'=' * 60}")
    print(f"改动 {rep.total_changes} 处，高风险 {rep.high_risk_count} 处")
    print(f"摘要：{rep.summary}")
    if getattr(rep, "json_path", None):
        print(f"\nJSON 报告：{rep.json_path}")
        print(f"Markdown 报告：{rep.markdown_path}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())