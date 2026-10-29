# local_eval.py
"""
轻量 RAG 评估：不依赖 ragas，复用 ragas_eval 的检索/生成逻辑，
用 DeepSeek 自身作为 judge 计算四个指标：
    faithfulness      答案是否忠于检索到的上下文
    answer_relevancy  答案是否切题
    context_precision 检索到的上下文有多少条真正相关
    context_recall    ground_truth 的事实点是否被上下文覆盖

用法：
    cd agent1
    python eval/local_eval.py                          # 默认 5 条样例，输出 stdout
    python eval/local_eval.py --output eval/report_local.json
    python eval/local_eval.py --top-k 8 --samples custom.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT1_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT = os.path.dirname(_AGENT1_DIR)
for p in (_ROOT, _AGENT1_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# 复用已有逻辑，不重复造轮子
from ragas_eval import DEFAULT_SAMPLES, retrieve_contexts, generate_answer


# ==================== 弃答识别 ====================
#
# 弃答（模型回答"检索内容不足"）与四个 RAGAS 指标的解读强相关：
# report_v6 里34/47 条弃答，直接把 answer_relevancy 压到 0.579。
# 必须能可靠判定一条答案是否弃答，否则无法区分"检索差"与"模型不肯答"。
#
# ★★ 这个检测器换过 5 版实现才收敛，过程全部保留在下面，
#    因为它记录了一个反直觉的事实：**词法匹配做不好这件事**。
#
#   v1 只匹配"检索内容不足"     -> 漏 4 条（模型换了措辞）
#   v2 扩成 12 类否定模式        -> **误报 29 条**
#      原因：v7 的新 prompt 要求"作答后说明哪些情形未覆盖"，于是正常
#      答案长这样："…规定为：责令改正。[片段1] 检索内容未覆盖该条款的
#      适用细节。" 答案主体是完整回答，末尾一句补充说明。
#   v3 加"必须完全没有 [片段N]"   -> 误报 0，漏 9
#      原因：弃答答案里 [片段N] 满天飞（"片段1仅提到…[片段1]，但未…"）
#   v4 看 [片段N] 后 30 字有无否定 -> v6/v7 准确率来回震荡 0.81 / 0.85
#      原因：窗口怎么调都会在"正常答案恰好先说否定"和"弃答先给正面
#      铺垫"两边翻车。
#   v5 改成只看开头 60 字        -> 漏判 0，但 v7 误报 10
#      原因：正常答案开头 60 字内也会出现"应当不予注册""未覆盖"等
#      法定表述或补充说明。
#
# 结论：**弃答是语义判断，不是字符串匹配**。同一个意思能被写成
# "检索内容不足"、"无法完整回答"、"没有列出"、"未能涵盖"…十几种说法，
# 而正常答案里也随时会出现否定词。继续在词法上加规则只会无限震荡。
#
# 所以最终用 LLM judge（judge_abstain）—— 项目里本来就有四个指标的
# judge 基础设施，复用同一套调用与容错，判定口径与其余指标完全一致。
# 词法版降级为 lexical_abstain，只用于无 API Key 时兜底与人工抽查。
#
# ★ 弃答的真正标志：**答案的结论句是否是对问题的否定式主张**。
#
# ★★ 这个检测器换过 5 版实现才收敛，过程全部保留在下面，
#    因为它记录了一个反直觉的事实：**词法匹配做不好这件事**。
#
#   v1 只匹配"检索内容不足"     -> 漏 4 条（模型换了措辞）
#   v2 扩成12 类否定模式         -> **误报 29 条**
#      原因：v7 的新 prompt 要求"作答后说明哪些情形未覆盖"，于是正常
#      答案长这样："…规定为：责令改正。[片段1] 检索内容未覆盖该条款的
#      适用细节。" 答案主体是完整回答，末尾一句补充说明。
#   v3 加"必须完全没有 [片段N]"   -> 误报 0，漏 9
#      原因：弃答答案里 [片段N] 满天飞（"片段1仅提到…[片段1]，但未…"）
#   v4 看 [片段N] 后 30 字有无否定 -> v6/v7 准确率来回震荡 0.81 / 0.85
#      原因：窗口大小怎么调都会在"正常答案恰好先说否定"和"弃答先给正面
#      铺垫"两边翻车。
#   v5 改成只看开头 60 字        -> 漏判 0，但 v7 误报 10
#      原因：正常答案开头 60 字内也可能出现"应当不予注册""未覆盖"等
#      法定表述或补充说明。
#
# 结论：**弃答是语义判断，不是字符串匹配**。同一个意思能被写成
# "检索内容不足"、"无法完整回答"、"没有列出"、"未能涵盖"…十几种说法，
# 而正常答案里也随时会出现否定词。继续在词法上加规则只会无限震荡。
#
# 所以改为用 LLM judge —— 项目里本来就有四个指标的 judge 基础设施，
# 复用同一套调用与容错，判定口径与其余指标完全一致，没有额外成本。
#
# 词法版本保留为 `_LEXICAL_ABSTAIN`，用于：
#   1. 报告里快速筛出"大概率弃答"供人工抽查
#   2. LLM judge 不可用时（无 API Key）的降级路径
_LEXICAL_ABSTAIN = re.compile(
    r"(检索内容不足"
    r"|无法(?:完整)?(?:回答|确定|明确)"
    r"|未能回答"
    r"|没有(?:列出|给出|明确|说明|提供|涵盖|覆盖)"
    r"|未(?:列出|给出|明确|说明|提供|涵盖|覆盖)"
    r"|(?:均|都)?未(?:涉及|对应|写明|载明|提及|表明|包含|针对)"
    r"|不(?:涉及|包含|明确|写明|载明|说明|列出|给出|提供)"
    r"|与[^。]{0,8}(?:无关|不符)"
    r"|不足以(?:回答|支持)"
    r"|不能(?:回答|确定|得出)"
    r"|无法据此"
    r"|无从(?:判断|确定)"
    r")"
)


def lexical_abstain(answer: str) -> bool:
    """词法快筛：答案里是否出现"答不出来"的表述。

    单独用它判弃答会有约 20% 误报（见上方v2/v5 的实测记录），
    只用于人工抽查线索与 LLM 不可用时的降级。
    """
    return bool(answer) and bool(_LEXICAL_ABSTAIN.search(answer))


def lexical_abstain(answer: str) -> bool:
    """词法快筛：答案里是否出现"答不出来"的表述。

    单独用它判弃答会有约 20% 误报（见上方 v2/v5 的实测记录），
    只用于人工抽查线索与 LLM 不可用时的降级。
    """
    return bool(answer) and bool(_LEXICAL_ABSTAIN.search(answer))


def judge_abstain(question: str, answer: str) -> bool:
    """LLM judge：这条答案实质上是否放弃了回答问题。

    与其余三个指标共用 ``_call_llm_json``，口径一致、失败可降级。

    判定标准（刻意与 answer_relevancy 区分开）：
      - 只回答了问题的一部分，但确实给出了与问题相关的实质结论
        -> **不算弃答**（哪怕不完整）
      - 通篇在解释为什么不能回答、检索内容与问题无关
        ->算弃答
    这条边界正是 v7 修prompt 要达成的目标，所以必须用语义判定。

    :return: True 表示弃答
    """
    if not answer or not answer.strip():
        return True

    prompt = f"""你是 RAG 评估员。判断下面这条回答是否**放弃了回答问题**。

【问题】
{question}

【回答】
{answer}

判定标准：
1. 如果回答给出了任何针对该问题的实质结论（哪怕不完整、只覆盖部分情形、
   只回答了某个侧面），判 false —— 不算弃答。
2. 如果回答通篇只在解释"检索内容不足/不相关/没有覆盖/无法确定"，
   没有给出任何实质结论，判 true。
3. 注意区分：正常回答末尾补一句"检索内容未覆盖XX情形"属于**已作答**，
   不算弃答；只有当否定式说明是答案的全部内容时才算弃答。
4. 回答中引用法条原文里的"不予处罚""不得"等法定表述，不算弃答。

只输出 JSON：{{"abstained": true 或 false}}"""
    result = _call_llm_json(prompt)
    if "error" in result or "abstained" not in result:
        # LLM 不可用时降级到词法判断（会误报约 20%，但不会漏）
        return lexical_abstain(answer)
    return bool(result["abstained"])


# ==================== LLM judge 基础 ====================
def _llm_chain_snapshot() -> list:
    """记录本轮用的 LLM，写进报告，便于跨轮对比时对齐变量。

    为什么需要：换模型（deepseek-chat ↔ 别的）会让 judge 打分波动。
    不记录"这一轮到底谁在答"，分数差异就无法归因 —— 到底是
    策略变了还是模型变了。这是 2026-10-15 换LLM provider 时踩到的。

    单provider 部署下同样有用：至少能确认"这一轮用的是哪个模型"。
    """
    try:
        import config as _cfg
        # 若配置了多provider 链就用它，否则退回当前 DeepSeek 配置
        if hasattr(_cfg, "llm_status"):
            return [
                {"name": c["name"], "model": c["model"],
                 "available": c["available"]}
                for c in _cfg.llm_status()["chain"]
            ]
        cfg = _cfg.get_deepseek_config()
        return [{"name": "deepseek", "model": cfg["model"], "available": True}]
    except Exception:
        return []


def _call_llm_json(prompt: str, max_retries: int = 2) -> dict:
    """调 DeepSeek，强制 JSON 输出。失败重试，最终失败返回 {'error': ...}。"""
    from config import ds_client

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = ds_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=120,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content.strip()
            return json.loads(text)
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
    return {"error": last_err or "unknown"}


def _split_sentences(text: str, min_len: int = 8) -> list[str]:
    """按中文标点切句，过滤过短的碎片。"""
    if not text:
        return []
    parts = re.split(r"[。！？；\n]", text)
    return [p.strip() for p in parts if len(p.strip()) >= min_len]


# ==================== 四指标 judge ====================
def judge_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    把答案拆成陈述句，逐句判断是否能从 contexts 推出。
    得分 = 有据句数 / 总句数。答案为空或拒答 → 1.0（拒答本身不构成幻觉）。
    """
    if not answer or not contexts:
        return 1.0 if answer and len(answer) < 30 else 0.0
    sentences = _split_sentences(answer)
    if not sentences:
        return 1.0

    ctx_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = f"""你是 RAG 评估员。判断答案中的每个陈述句是否能从给定的检索片段中推出。
严格标准：必须能在片段中找到明确依据，不能靠常识补全。
如果答案在诚实拒答（如"未出现相关内容"），该句视为有据。

【检索片段】
{ctx_text}

【待判断的陈述句】
{numbered}

以 JSON 返回：{{"verdicts": [true, false, ...]}}，数组长度必须等于 {len(sentences)}。"""
    result = _call_llm_json(prompt)
    verdicts = result.get("verdicts", [])
    if not isinstance(verdicts, list) or not verdicts:
        return 0.0
    verdicts = verdicts[: len(sentences)]
    return round(sum(1 for v in verdicts if v) / len(sentences), 3)


def judge_answer_relevancy(question: str, answer: str) -> float:
    """答案是否回应了问题。1-5 分归一化到 0-1。"""
    if not answer:
        return 0.0
    prompt = f"""你是 RAG 评估员。给下面这个"答案对问题的回应程度"打 1-5 分。
评分标准：
5 = 直接、完整地回答了问题
4 = 回答了大方向但遗漏次要信息
3 = 部分相关，主要回答了别的问题
2 = 仅仅沾边
1 = 完全跑题或空答
如果答案在诚实拒答（明确说"检索结果未出现相关内容"），给 2 分（诚实但不解决问题）。

【问题】{question}
【答案】{answer}

以 JSON 返回：{{"score": <1-5>, "reason": "<一句话>"}}"""
    result = _call_llm_json(prompt)
    score = result.get("score", 0)
    if not isinstance(score, (int, float)) or score < 1:
        return 0.0
    return round(min(max(score, 1), 5) / 5.0, 3)


def judge_context_precision(question: str, contexts: list[str]) -> float:
    """逐条判断 context 是否与问题相关。得分 = 相关条数 / 总条数。"""
    if not contexts:
        return 0.0
    ctx_list = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = f"""你是 RAG 评估员。判断每条检索片段是否与问题直接相关。
"相关"的定义：片段内容能用于回答这个问题。
仅仅是同一份法规文档但不同条款，不算相关。

【问题】{question}
【检索片段】
{ctx_list}

以 JSON 返回：{{"relevant": [true, false, ...]}}，数组长度必须等于 {len(contexts)}。"""
    result = _call_llm_json(prompt)
    relevant = result.get("relevant", [])
    if not isinstance(relevant, list) or not relevant:
        return 0.0
    relevant = relevant[: len(contexts)]
    return round(sum(1 for v in relevant if v) / len(contexts), 3)


def judge_context_recall(ground_truth: str, contexts: list[str]) -> float:
    """ground_truth 拆成事实点，逐点检查是否被 contexts 覆盖。

    判定口径（本项目实测校准）：
    法条原文与参考答案常是两套话术—— 原文写"综合考量违法的事实、性质、
    手段、后果、情节和改正措施等因素"，参考答案写"根据违法行为的事实、
    性质、情节及社会危害程度依法确定处罚种类和幅度"。二者语义等价但
    无字面重合。若judge 要求字面明文，会把大量正确召回误判为0，
    使 context_recall 严重失真（实测 5条样例里 2 条属此类假阴性）。
    因此这里采用**语义等价**口径：措辞、结构、用词不同不构成扣分理由。
    """
    if not ground_truth:
        return 0.0
    facts = _split_sentences(ground_truth, min_len=4) or [ground_truth]
    ctx_text = "\n\n".join(contexts)
    numbered = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))
    prompt = f"""你是 RAG 评估员。判断参考答案中的每个事实点是否被检索片段覆盖。

判定口径（重要）：
1. 覆盖 = 片段表达了该事实点的**实质内容**。允许同义改写、术语转换、
   句式重组、抽象与具体的互译；**措辞不同不构成扣分理由**。
2. 法律原文与参考答案用词不同属正常。例如原文"综合考量违法的事实、性质、
   手段、后果、情节和改正措施等因素，确定处罚种类和幅度"，
   参考答案"根据违法行为的事实、性质、情节及社会危害程度依法确定处罚
   种类和幅度" —— 二者应当判为 covered=true。
3. 不要求逐字对应，也不要求片段中出现参考答案的全部修饰语；
   只要核心语义（约束对象 + 认定要素 + 结论）一致即算覆盖。
4. 仅当片段与该事实点**主题无关**，或表述的是另一回事，才判false。
5. 不确定时倾向判 true —— 本指标衡量的是"检索有没有把答案捞回来"，
   宁可放过边缘情况，也不要把正确召回误杀。

【检索片段】
{ctx_text}

【参考答案事实点】
{numbered}

以 JSON 返回：{{"covered": [true, false, ...]}}，数组长度必须等于 {len(facts)}。"""
    result = _call_llm_json(prompt)
    covered = result.get("covered", [])
    if not isinstance(covered, list) or not covered:
        return 0.0
    covered = covered[: len(facts)]
    return round(sum(1 for v in covered if v) / len(facts), 3)


# ==================== 基线对比 ====================
#
# ★ 这套机制的全部理由（2026-10-15 实测踩坑）：
#   v6/v7/v8 三轮跑完，总分"有涨有跌"，很自然会解读成调优时灵时不灵。
#   但逐条比对 contexts 发现三轮 47 条上下文**逐字完全相同**——
#   索引没动过，v8 是次空跑。而即便变量隔离了，LLM-as-judge 本身也会抖：
#   同输入两次跑，context_recall 能差 0.021。
#
#   v8 还有另一种误读：cp 从 0.520 掉到 0.325，
#   看起来像检索退化，实际是 **top_k 从 5 改成 8**，
#   precision 的分母变了。所以可比性必须查 top_k。
#
#   要锁住的重点是**别再把空跑/噪声误读成优化效果**。

KEYS = ("faithfulness", "answer_relevancy",
        "context_precision", "context_recall")

# 抖动幅度未知时的保守阈值。
#
# ★ 为什么必须是 0.10 而不是实测的 0.021：
#   实测抖动是在特定数据集 + 特定 judge 上测的，换数据集就不一定成立。
#   更重要的是：**单次 A/B 差值只是抖动幅度的下界，不是幅度本身。**
#   曾拿 v7/v8 的差值（recall 0.021）当阈值设成 0.02，
#   结果 v7→v8 的 recall 差值恰好 0.021，卡在边界上，
#   `abs(0.021) > 0.02` 判真，测试直接失败。
#
#   差值分布有方差，两次恰好接近不代表常态如此。
#   tests 里有一条专门锁定这个决策：若有人调回 0.02，
#   必须先解释为什么单次观测可以当阈值用。
_NOISE_FLOOR_UNKNOWN = 0.10


def _contexts_fingerprint(details: list) -> str:
    """上下文指纹：只哈希 question + contexts。

    ★ 绝不能包含答案与分数。
      含了的话分数一变指纹就变，空跑检测立刻失效——
      而空跑检测是这三项机制里最有用的一项。

    实测：v7/v8/v9 三份报告指纹完全相同（cb941a0116de7cef），
    正是空跑的铁证。
    """
    h = hashlib.sha256()
    for d in details:
        h.update((d.get("question") or "").encode("utf-8"))
        for c in (d.get("contexts") or []):
            h.update(c.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _fingerprint_of(rep: dict, details: list) -> str:
    """优先用报告里存的指纹，没有才算一遍。

    ★ 不能只信报告字段：万一是旧版本报告（没这字段）或被手改过，
      那就该自己算。存指纹的意义是省一次全量哈希，不是免于校验。
    """
    fp = rep.get("_contexts_fingerprint")
    if isinstance(fp, str) and fp:
        return fp
    return _contexts_fingerprint(details)


def _read_scores(rep: dict) -> dict:
    """取四指标均值。

    ★ 报告的真实结构：
        scores.<metric>       均值在这里
        scores._per_sample[]  逐条分数
        details[]             只有问答与检索结果，不含指标

    读错位置不会崩，只会让所有差值变成 0，
    于是"未超出噪声底"看起来像正常结论——比直接崩溃危险得多。
    """
    sc = rep.get("scores") or {}
    return {k: (float(sc[k]) if isinstance(sc.get(k), (int, float)) else None)
            for k in KEYS}


def _noise_from_baseline(base: dict) -> dict:
    """取基线里的实测噪声底。

    :return: {"floors": {指标: 值}, "measured": bool}
    ★ 用 dict 而不是 tuple —— tuple 的解包顺序写错不会报错，
      只会让「measured」拿到空 dict（falsy），
      机制静默降级成"几乎什么都不报"，与设计目的正好相反。
    """
    for src in (base.get("_noise_floor"),
                (base.get("scores") or {}).get("_noise_floor")):
        if isinstance(src, dict):
            vals = {k: float(src[k]) for k in KEYS
                    if isinstance(src.get(k), (int, float))}
            if vals:
                return {"floors": vals, "measured": True}
    return {"floors": {}, "measured": False}


def compare_with_baseline(cur_report: dict, baseline_path: str) -> dict:
    """与基线报告对比，返回结构化 verdict（供打印与断言）。"""
    out = {
        "error": None,
        "baseline_file": baseline_path,
        "checks": {},
        "deltas": {},
        "verdict": {},
    }

    # ★ 局部 import 而非依赖模块级：
    #   tests/test_eval_baseline.py 用 exec 摘出这段执行，
    #   只注入 json/os/hashlib —— 模块级的 pathlib 不会被带进来。
    import pathlib as _pathlib
    bp = _pathlib.Path(baseline_path)
    if not bp.is_file():
        out["error"] = f"基线文件不存在：{baseline_path}"
        return out
    try:
        base = json.loads(bp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        out["error"] = f"基线读取失败：{e}"
        return out

    cur_details = cur_report.get("details") or []
    base_details = base.get("details") or []
    if not cur_details or not base_details:
        out["error"] = "报告缺少 details，无法对比"
        return out

    # ---------- Q1 可比性 ----------
    cur_n = cur_report.get("num_samples") or len(cur_details)
    base_n = base.get("num_samples") or len(base_details)
    cur_k = cur_report.get("top_k")
    base_k = base.get("top_k")
    k_differs = (cur_k is not None and base_k is not None and cur_k != base_k)

    note = ""
    if cur_n != base_n:
        note = (f"样本数不同（基线 {base_n} vs 本轮 {cur_n}），"
                "分数不可比")
    elif k_differs:
        note = (f"top_k 不同（基线 {base_k} vs 本轮 {cur_k}），"
                "precision 分母变了，分数不可比"
                "（v8 的 cp 0.520→0.325 就是纯 top_k 5→8 造成的）")

    out["checks"]["comparable"] = {
        "passed": (cur_n == base_n) and not k_differs,
        "baseline_num_samples": base_n,
        "current_num_samples": cur_n,
        "baseline_top_k": base_k,
        "current_top_k": cur_k,
        "note": note,
    }
    if note:
        return out          # 不可比时不给 verdict——比什么都不报更危险

    # ---------- 差值 ----------
    cur_sc = _read_scores(cur_report)
    base_sc = _read_scores(base)
    deltas = {}
    for k in KEYS:
        a, b = cur_sc.get(k), base_sc.get(k)
        if a is None or b is None:
            continue
        deltas[k] = a - b
    out["deltas"] = deltas

    # ---------- Q2 空跑检测 ----------
    fp_cur = _fingerprint_of(cur_report, cur_details)
    fp_base = _fingerprint_of(base, base_details)
    same_fp = (fp_cur == fp_base)
    score_changed = any(abs(v) > 1e-9 for v in deltas.values())

    out["checks"]["variable_isolation"] = {
        "passed": not same_fp,
        "contexts_identical": same_fp,
        "current_fingerprint": fp_cur,
        "baseline_fingerprint": fp_base,
    }

    # 空跑判定见下方（放在噪声判定之后，但要区分两件事，见那处注释）

    # ---------- Q3 噪声判定 ----------
    _nz = _noise_from_baseline(base)
    floors = _nz["floors"]
    measured = _nz["measured"]
    if measured:
        floor_map = {k: floors.get(k, _NOISE_FLOOR_UNKNOWN) for k in KEYS}
        source = "基线实测"
        nz_note = "使用基线报告里的实测噪声底"
        passed = True
    else:
        floor_map = {k: _NOISE_FLOOR_UNKNOWN for k in KEYS}
        source = f"经验保守值 {_NOISE_FLOOR_UNKNOWN}"
        nz_note = ("基线未实测噪声底，用经验保守值。"
                   "建议在基线上跑两次同输入评测，实测抖动后再对比，"
                   "否则只能得到保守结论")
        passed = False

    out["checks"]["noise"] = {
        "passed": passed,
        "measured": measured,
        "source": source,
        "noise_floor": floor_map,
        "note": nz_note,
    }

    # ---------- 逐指标 verdict ----------
    sig_count = 0
    for k in KEYS:
        if k not in deltas:
            continue
        d = deltas[k]
        sig = abs(d) > floor_map[k]
        if sig:
            sig_count += 1
        out["verdict"][k] = {
            "delta": d,
            "exceeds_noise": sig,
            "verdict": "真实变化" if sig else "在噪声内，不可归因",
        }

    # ---------- Q2 空跑 ----------
    # ★ 语义澄清（这里我一度改错，值得记下来）：
    #   空跑的判据是「指纹一致 **且** 分数有任何变化」，
    #   **不是**「指纹一致且变化超出噪声底」。
    #
    #   为什么不用噪声底当条件：指纹一致已经意味着检索层零变化，
    #   此时分数的任何变化都不可能来自策略改进 —— 只能是 LLM 抖动。
    #   阈值是给"检索层确实变了"的情况做二级过滤用的；
    #   拿它来判空跑，v7/v8 那种 rec +0.021 就会被放过，
    #   而那正是最初被骗的那一轮。
    #
    #   而且指纹一致 + prec 0.221→0.222 也该报空跑：
    #   它说明"什么都没改却动了分数"，这本身就是需要知道的事实。
    if same_fp:
        if score_changed:
            out["checks"]["null_run"] = {
                "detected": True,
                "passed": False,
                "note": ("上下文指纹完全一致（" + fp_cur + "），"
                         "检索层零变化，但分数变了 → "
                         "差异 100% 来自 LLM 判分随机性，"
                         "**不是策略改进**。本轮数据不能证明任何优化有效。"),
            }
        else:
            out["checks"]["null_run"] = {
                "detected": False,
                "passed": True,
                "note": "上下文与分数均未变，结果可复现",
            }

    return out


def _as_dict(v) -> dict:
    """把任意值安全地变成 dict。

    ★ 打印函数的输入来自外部（测试、调用方、上一版实现），
      字段类型不受控。崩了等于整轮评测白跑——
      分数已经算完，只是没打出来。
      所以这里做类型校验，而不是用 `or {}` 兜底
      （后者遇到 bool 会短路返回 bool，随后 .get 就崩）。
    """
    return v if isinstance(v, dict) else {}


def print_baseline_report(out: dict) -> None:
    """把 verdict 打成人类可读的表格。

    ★ 必须能扛各种残缺输入 —— 它崩了等于整轮评测白跑，
      所以全程用 .get() 链而不是下标访问。
    """
    print("\n" + "=" * 66)
    print("基线对比")
    print("=" * 66)
    print(f"  基线文件：{out.get('baseline_file', '(未提供)')}")

    if out.get("error"):
        print(f"  [FAIL] {out['error']}")
        return

    checks = _as_dict(out.get("checks"))
    cmp_ = _as_dict(checks.get("comparable"))
    print(f"  样本数 基线={cmp_.get('baseline_num_samples')} "
          f"本轮={cmp_.get('current_num_samples')}   "
          f"top_k 基线={cmp_.get('baseline_top_k')} "
          f"本轮={cmp_.get('current_top_k')}")

    if not cmp_.get("passed"):
        print(f"  [不可比] {cmp_.get('note')}")
        return

    vi = _as_dict(checks.get("variable_isolation"))
    print(f"\n  指纹  本轮={vi.get('current_fingerprint')}  "
          f"基线={vi.get('baseline_fingerprint')}")

    nr = _as_dict(checks.get("null_run"))
    noise = _as_dict(checks.get("noise"))
    verdict = _as_dict(out.get("verdict"))

    if verdict:
        print(f"\n  {'指标':22s} {'差值':>9s} {'噪声底':>8s}")
        print("  " + "-" * 44)
        for k in KEYS:
            v = _as_dict(verdict.get(k))
            if not v:
                continue
            floor_map = _as_dict(noise.get("noise_floor"))
            f = floor_map.get(k, _NOISE_FLOOR_UNKNOWN)
            mark = "*" if v.get("exceeds_noise") else " "
            print(f"  {k:22s} {v.get('delta', 0.0):+9.4f} "
                  f"{f:8.3f}{mark}")

    total = len(verdict)
    sig = sum(1 for v in verdict.values()
              if _as_dict(v).get("exceeds_noise"))
    print()
    if nr.get("detected"):
        print("  [空跑] " + (nr.get("note") or ""))
        # ★ 空跑之后**继续**打印噪声统计，不要 return。
        #   否则看到空跑就以为没有别的信息了，
        #   而「哪几项本可以超噪声」恰恰是下一步该看的。
        print()
        if sig:
            names = [k for k, v in verdict.items()
                     if _as_dict(v).get("exceeds_noise")]
            print(f"  注：{sig}/{total} 项超出噪声底（{', '.join(names)}），"
                  f"但在上下文零变化的前提下，这些差异仍不可归因")
        else:
            print(f"  注：{sig}/{total} 项超出噪声底，"
                  f"全部在噪声内 —— 与'什么都没改'一致")

    print(f"  噪声底来源：{noise.get('source', '(未知)')}")
    if not noise.get("measured"):
        print(f"  ⚠ {noise.get('note', '')}")

    if sig:
        names = [k for k, v in verdict.items() if v.get("exceeds_noise")]
        print(f"  [可作结论] {sig}/{total} 项超出噪声底：{', '.join(names)}")
        rest = [k for k, v in verdict.items() if not v.get("exceeds_noise")]
        if rest:
            print(f"  [不可作结论] 未超出：{', '.join(rest)}")
    else:
        print(f"  [不可作结论] {sig}/{total} 项超出噪声底，"
              "全部在噪声内，不要写成'提升了'")


# ==================== 主流程 ====================
# ---------------------------------------------------------------- 噪声底
# 实测 LLM 判分抖动约 0.021（v7/v8 空跑轮的 recall 差值）。
# 取 0.05 ≈ 实测 2.4 倍余量。
#
# ★ 为什么不直接用实测值：
#   单次 A/B 差值只是抖动幅度的**下界**，不是幅度本身。
#   阈值取 0.021 时，恰好等于 0.021 的差值会被判成"超出噪声"，
#   而它其实没超 —— 曾因此让测试失败过一次。
_NOISE_FLOOR = 0.05

_METRIC_KEYS = ("faithfulness", "answer_relevancy",
                "context_precision", "context_recall")


def contexts_fingerprint(details: list) -> str:
    """上下文指纹：只看检索结果，不看答案和分数。

    ★ 指纹**绝不能包含答案或分数**。
      那样分数一变指纹就变，空跑检测立刻失效 ——
      而空跑检测恰恰是这套机制里最有用的一项。
    """
    h = hashlib.sha256()
    for d in details:
        h.update((d.get("question") or "").encode("utf-8"))
        for c in (d.get("contexts") or []):
            h.update(c.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _pick_scores(report: dict) -> dict:
    """从报告里取四个指标的均值。

    ★ 报告的真实结构（第一版读错过这里）：
        scores.faithfulness / answer_relevancy /
                context_precision / context_recall   <- 均值在这里
        scores._per_sample[]                       <- 逐条分数
        details[]                                   <- 只有问答与检索结果

    读错位置不会报错，只会让所有差值变成 0，
    于是"未超出噪声底"看起来像正常结论 —— 这比直接崩溃危险得多。
    """
    sc = report.get("scores") or {}
    out = {}
    for k in _METRIC_KEYS:
        v = sc.get(k)
        out[k] = float(v) if isinstance(v, (int, float)) else None
    return out


def _avg_metric(details: list, key: str) -> float:
    """兜底：若 scores 缺失，从 _per_sample 求均值。"""
    vals = [d.get(key) for d in details
            if isinstance(d.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def compare_with_baseline(details: list, baseline_path: str,
                          cur_scores: dict | None = None,
                          noise_floor: float = _NOISE_FLOOR) -> int:
    """与基线报告对比。返回 0=可作结论，1=不可作结论。"""
    bp = pathlib.Path(baseline_path)
    if not bp.is_file():
        print(f"\n[FAIL] 基线文件不存在：{baseline_path}")
        return 1
    try:
        base = json.loads(bp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"\n[FAIL] 基线文件读取失败：{e}")
        return 1

    base_details = base.get("details") or []
    if not base_details:
        print("\n[FAIL] 基线报告里没有 details，无法对比")
        return 1

    print("\n" + "=" * 66)
    print("基线对比")
    print("=" * 66)
    print(f"  基线: {baseline_path}")

    fp_new = contexts_fingerprint(details)
    fp_old = contexts_fingerprint(base_details)
    same_fp = (fp_new == fp_old)

    print(f"\n  上下文指纹  本轮={fp_new}  基线={fp_old}")
    if same_fp:
        print("    -> 检索层完全未变（空跑）")
    else:
        print("    -> 检索层已变化")

    print(f"\n  {'指标':22s} {'基线':>8s} {'本轮':>8s} {'差值':>9s}")
    print("  " + "-" * 52)
    # ★ 从 scores 读，不是从 details 读 —— 见 _pick_scores 的注释
    base_scores = _pick_scores(base)
    new_scores = (cur_scores if cur_scores is not None
                  else _pick_scores({"scores": {}}))

    deltas = {}
    for m in _METRIC_KEYS:
        a = base_scores.get(m)
        b = new_scores.get(m)
        if a is None or b is None:
            print(f"  {m:22s} {'(基线缺失)' if a is None else '(本轮缺失)'}")
            deltas[m] = None
            continue
        d = b - a
        deltas[m] = d
        mark = "*" if abs(d) > noise_floor else " "
        print(f"  {m:22s} {a:8.4f} {b:8.4f} {d:+9.4f}{mark}")

    usable = {m: v for m, v in deltas.items() if v is not None}
    if not usable:
        print("\n  [FAIL] 两份报告都取不到指标，无法对比")
        print("         请确认报告是 local_eval.py 生成的（含 scores 段）")
        return 1

    over = [m for m, d in usable.items() if abs(d) > noise_floor]
    under = [m for m, d in usable.items() if abs(d) <= noise_floor]

    print()
    if same_fp:
        print(f"  [空跑] 检索层零变化，但 {len(over)}/{len(_METRIC_KEYS)} "
              f"项超出噪声底")
        if over:
            print("     这些差异 100% 来自 LLM 判分随机性，不是策略改进")
        print("     结论：本轮数据不能证明任何优化有效")
        return 1

    print(f"  检索层已变化 -> 差异可归因到检索/生成策略（不是空跑）\n")
    if over:
        print(f"  [可作结论] 超出噪声底（+-{noise_floor}）：")
        for m in over:
            print(f"       {m:22s} {deltas[m]:+.4f}")
    if under:
        print(f"  [不可作结论] 未超出噪声底：")
        for m in under:
            print(f"       {m:22s} {deltas[m]:+.4f}")
        print("     要扩大样本量或重复多轮才能判断，不要写成'提升了'")

    blank = [d for d in details if not (d.get("answer") or "").strip()]
    if blank:
        pct = len(blank) / max(len(details), 1) * 100
        print(f"\n  [故障] {len(blank)}/{len(details)} 条（{pct:.0f}%）未生成答案")
        if pct > 5:
            print("     占比过高，本轮全部指标不可与其他轮对比")
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="轻量 RAG 评估（不依赖 ragas）")
    parser.add_argument("--samples", default=None, help="QA 样例 JSONL（默认内置样例）")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    parser.add_argument("--skip-judge", action="store_true", help="只跑检索+生成，不打分")
    parser.add_argument("--category", default=None,
                        help="只跑指定类别（直查法条/跨文档综合/陷阱题）")
    parser.add_argument("--difficulty", default=None,
                        help="只跑指定难度（easy/medium/hard）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    parser.add_argument(
        "--baseline", default=None,
        help="与基线报告对比（自动做空跑检测与噪声判定）")
    parser.add_argument(
        "--noise-floor", type=float, default=_NOISE_FLOOR,
        help=f"噪声底（默认 {_NOISE_FLOOR}，实测抖动约 0.021）")
    args = parser.parse_args()

    samples = DEFAULT_SAMPLES
    if args.samples and os.path.isfile(args.samples):
        samples = []
        with open(args.samples, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    samples.append(json.loads(line))

    # 按类别/难度筛选：优化时需要能定位"是哪类问题在拖后腿"，
    # 只看总分会掩盖"直查法条 0.95 但跨文档综合 0.4"这种关键差异。
    if args.category:
        samples = [s for s in samples if s.get("category") == args.category]
    if args.difficulty:
        samples = [s for s in samples if s.get("difficulty") == args.difficulty]
    if args.limit:
        samples = samples[:args.limit]

    print(f"📋 加载 {len(samples)} 条 QA 样例")

    # ★ 打印本轮用的 LLM 模型（2026-10-15 新增）
    #
    # 评测是长跑任务（47 条 × 5 次 LLM 调用），跑之前就该知道在用哪家。
    # 起因是一次 DeepSeek 返回 503，报告里分数普降却看不出原因 ——
    # 如果不记录"这一轮谁在答"，无法区分"策略退化"和"服务故障"。
    try:
        _chain = _llm_chain_snapshot()
        if _chain:
            print("🤖 LLM：" + " → ".join(
                f"{c['name']}({c['model']})" for c in _chain))
    except Exception:
        pass

    if not samples:
        print("⚠️ 筛选后没有样本，请检查 --category/--difficulty 取值")
        return

    # 1. 检索 + 生成
    eval_data = []
    for i, s in enumerate(samples, 1):
        q, gt = s["question"], s["ground_truth"]
        print(f"\n[{i}/{len(samples)}] {q}")
        contexts = retrieve_contexts(q, top_k=args.top_k)
        print(f"  检索到 {len(contexts)} 条上下文")
        answer = generate_answer(q, contexts)
        print(f"  答案长度: {len(answer)} 字")
        eval_data.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": gt,
            "category": s.get("category", "未分类"),
            "difficulty": s.get("difficulty", "unknown"),
        })

    # 2. 逐条打分
    scores: dict = {}
    if not args.skip_judge:
        print(f"\n📊 开始 LLM-as-judge 打分（{len(eval_data)} 条 × 4 指标）...")
        t0 = time.time()
        per_sample = []
        for i, d in enumerate(eval_data, 1):
            print(f"  [{i}/{len(eval_data)}] 打分中...", end=" ", flush=True)
            row = {
                "faithfulness":      judge_faithfulness(d["answer"], d["contexts"]),
                "answer_relevancy":  judge_answer_relevancy(d["question"], d["answer"]),
                "context_precision": judge_context_precision(d["question"], d["contexts"]),
                "context_recall":    judge_context_recall(d["ground_truth"], d["contexts"]),
            }
            # 弃答判定与四个指标同批进行：它决定 answer_relevancy
            # 的解读方式，必须用同一套 judge 口径，不能另起一套词法规则。
            row["_abstained"] = judge_abstain(d["question"], d["answer"])
            per_sample.append(row)
            print(json.dumps(row, ensure_ascii=False))

        # 聚合
        keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        scores = {
            k: round(sum(r[k] for r in per_sample) / len(per_sample), 3)
            for k in keys
        }
        scores["_elapsed_sec"] = round(time.time() - t0, 1)
        scores["_num_samples"] = len(eval_data)
        scores["_per_sample"] = per_sample

        # 分层统计：定位薄弱类别，避免总分掩盖结构性问题
        breakdown: dict = {}
        for dim in ("category", "difficulty"):
            groups: dict = {}
            for d, r in zip(eval_data, per_sample):
                g = d[dim]
                groups.setdefault(g, []).append(r)
            breakdown[dim] = {
                g: {
                    "n": len(rs),
                    **{
                        k: round(sum(x[k] for x in rs) / len(rs), 3)
                        for k in keys
                    },
                }
                for g, rs in sorted(groups.items())
            }
        scores["_breakdown"] = breakdown

        # ==================== 应答样本口径（★ 总分不变，只增加视角）========
        # 弃答样本的 context_precision 必然是 0（没有答案，
        # 「上下文是否相关」无从判定），但它们照样计入总分。
        # 实测 47 条里弃答 14 条 = 29.8% —— 总分 0.226 与应答口径 0.309
        # 的差距，主要来自这里。
        #
        # 分层看才能定位该优化哪个环节：
        #   召回到但掺噪声（recall 高、precision 低）→ rerank / top_k
        #   没召回到（recall 低）                  → 分块 / 查询改写
        _ans_rows = [r for r in per_sample if not r.get("_abstained")]
        _ab_rows = [r for r in per_sample if r.get("_abstained")]
        if _ans_rows:
            scores["_answered_only"] = {
                "n": len(_ans_rows),
                **{
                    k: round(sum(r[k] for r in _ans_rows) / len(_ans_rows), 3)
                    for k in keys
                },
            }
        if _ab_rows:
            scores["_abstained_only"] = {
                "n": len(_ab_rows),
                **{
                    k: round(sum(r[k] for r in _ab_rows) / len(_ab_rows), 3)
                    for k in keys
                },
            }

        # 按类别 × 是否应答 的交叉分层
        _cross: dict = {}
        for d, r in zip(eval_data, per_sample):
            bucket = "answered" if not r.get("_abstained") else "abstained"
            g = d.get("category", "未分类")
            _cross.setdefault(g, {}).setdefault(bucket, []).append(r)
        scores["_by_category_answered"] = {
            g: {
                bucket: {
                    "n": len(rows),
                    **{k: round(sum(x[k] for x in rows) / len(rows), 3)
                       for k in keys},
                }
                for bucket, rows in buckets.items()
            }
            for g, buckets in sorted(_cross.items())
        }

        # ★ 弃答诊断（report_v6 打出来的补丁，v7 加固，v8 改用 LLM judge）
        #
        # answer_relevancy 长期停在 0.58，根因不是检索而是**弃答**：
        #   弃答组 rel 均值 0.429 / 正常组 0.969
        # v6：34/47 条弃答，其中 14 条 recall>=0.75 却拒答（过度保守）。
        # v7 修 prompt（"只判断有没有相关片段，不判断是否完整"）后：
        #   弃答降到 11、过度保守降到 2、rel 0.579 -> 0.706，
        #   **faithfulness 一条没掉**（0.985 -> 0.985）
        #   —— 证明"少弃答"不会换来幻觉。
        #
        # ★ v8：弃答判定改用 LLM judge（见 judge_abstain 的注释）。
        #   v6 用字面匹配"检索内容不足"，v7 漏掉 4 条换了措辞的弃答，
        #   把 n_overcautious 低估成 2（真实 6）；改成词法模式组后又误报
        #   29 条，反复调窗口在v6/v7 上来回震荡。**词法做不了这件事**，
        #   因为弃答是语义判断。现在与四个指标共用同一套 judge 口径。
        #
        # 单看总分无法区分合理弃答与过度保守，所以把弃答按 recall 拆开。
        _abstain, _over, _fair = [], [], []
        for d, r in zip(eval_data, per_sample):
            ab = r.get("_abstained")
            if ab is None:            # 兼容旧报告（无该字段）
                ab = lexical_abstain(d.get("answer") or "")
            if not ab:
                _fair.append(r["answer_relevancy"])
            elif r["context_recall"] >= 0.75:
                _over.append({"q": d["question"], "rec": r["context_recall"]})
            else:
                _abstain.append(r["answer_relevancy"])
        scores["_abstain"] = {
            "n_abstain": len(_abstain) + len(_over),
            "n_overcautious": len(_over),      # 该答却没答（需修）
            "n_reasonable": len(_abstain),     # 检索确实没捞到（正常）
            "rel_when_answering": round(
                sum(_fair) / len(_fair), 3) if _fair else None,
            "rel_when_abstaining": round(
                sum(_abstain) / len(_abstain), 3) if _abstain else None,
            "overcautious_samples": _over,
        }

        print(f"\n{'='*60}\n🩺 弃答诊断")
        ab = scores["_abstain"]
        print(f"  弃答总数：{ab['n_abstain']}/{len(eval_data)}")
        print(f"    ├ 合理弃答（recall<0.75）：{ab['n_reasonable']}")
        print(f"    └ ⚠️ 过度保守（recall>=0.75）：{ab['n_overcautious']}")
        print(f"  正常作答 rel 均值：{ab['rel_when_answering']}")
        print(f"  弃答     rel 均值：{ab['rel_when_abstaining']}")
        for o in _over[:12]:
            print(f"    ! rec={o['rec']:.2f} {o['q'][:56]}")

        print(f"\n{'='*60}\n📐 口径对比（总分不变，只增加视角）")
        _ao = scores.get("_answered_only")
        _ao_all = scores.get("_abstained_only")
        if _ao:
            print(f"  应答样本 n={_ao['n']:<3} prec={_ao['context_precision']:<6} "
                  f"rec={_ao['context_recall']}")
        if _ao_all:
            print(f"  弃答样本 n={_ao_all['n']:<3} prec={_ao_all['context_precision']:<6} "
                  f"rec={_ao_all['context_recall']}")
        if _ao:
            print(f"  → 弃答样本的 precision 必然为 0（没答案就无从判断相关性），"
                  f"却占 {_ao_all['n']/len(per_sample)*100:.0f}% 计入总分")
            print(f"  → 优化前先看分层：recall 低=没召回到（改分块/查询改写）；"
                  f"recall 高 precision 低=掺了噪声（改 rerank/top_k）")

        print(f"\n{'='*60}\n📊 分层统计")
        for dim, groups in breakdown.items():
            print(f"\n  按 {dim}:")
            for g, m in groups.items():
                print(f"    {g:<12} n={m['n']:<3} "
                      f"faith={m['faithfulness']:<6} "
                      f"rel={m['answer_relevancy']:<6} "
                      f"prec={m['context_precision']:<6} "
                      f"rec={m['context_recall']}")

    # 3. 输出
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(eval_data),
        "top_k": args.top_k,
        # ★ 记录 LLM 故障条数与 provider：分数下降时要能立刻区分
        # "策略退化"和"服务故障"，否则会白排查半天。
        "_llm_failed": sum(
            1 for d in eval_data if not (d.get("answer") or "").strip()),
        "_llm_chain": _llm_chain_snapshot(),
        "scores": scores,
        "details": eval_data,
    }
    # 上下文指纹：只哈希 question + contexts，绝不含答案与分数。
    # 指纹一致 = 检索层完全没变 = 差异只来自生成与打分（空跑）。
    report["_contexts_fingerprint"] = contexts_fingerprint(eval_data)
    report["_noise_floor"] = args.noise_floor
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n✅ 报告已保存: {args.output}")
    else:
        print(f"\n{'='*60}")
        print(output)
    print(f"\n{'='*60}\n评估完成")

    # ★ 故障条数核查（2026-10-15 新增）
    #
    # DeepSeek 挂掉时 generate_answer 会返回空串，该条四个指标全记 0。
    # 混在总分里就是"分数变差"，但真实原因是**服务故障**，不是策略退化。
    # 所以必须单独数出来：占比超过阈值时这一轮结果不可用于对比。
    _blank = [d["question"] for d in eval_data if not (d.get("answer") or "").strip()]
    if _blank:
        print(f"\n🚨 有 {len(_blank)}/{len(eval_data)} 条因 LLM 故障未生成答案：")
        for _q in _blank[:5]:
            print(f"    - {_q[:56]}")
        if len(_blank) > 5:
            print(f"    ...另有 {len(_blank)-5} 条")
        print("   ⚠️ 这些条目计 0 分，会拉低总分。"
              "**本轮结果不可与其他轮直接对比**，请先修好 LLM 再重跑。")

    # ---- 基线对比 / 空跑检测 ----
    if getattr(args, "baseline", None):
        rc = compare_with_baseline(
            eval_data, args.baseline,
            cur_scores=report.get("scores"),
            noise_floor=args.noise_floor)
        if rc != 0:
            sys.exit(rc)
    else:
        print("\n💡 想确认这轮改动到底有没有效果，加 --baseline 指向上一轮报告：")
        print("   --baseline ../report_v10.json")
        print("   不加的话，看到的分数变化可能只是 LLM 判分抖动。")


if __name__ == "__main__":
    main()