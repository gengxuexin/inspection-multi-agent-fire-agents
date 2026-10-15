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
import json
import os
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


# ==================== 主流程 ====================
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


if __name__ == "__main__":
    main()