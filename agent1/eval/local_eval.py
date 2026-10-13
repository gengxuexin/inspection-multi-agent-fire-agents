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


# ==================== LLM judge 基础 ====================
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
    """ground_truth 拆成事实点，逐点检查是否被 contexts 覆盖。"""
    if not ground_truth:
        return 0.0
    facts = _split_sentences(ground_truth, min_len=4) or [ground_truth]
    ctx_text = "\n\n".join(contexts)
    numbered = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))
    prompt = f"""你是 RAG 评估员。判断参考答案中的每个事实点是否被检索片段覆盖。
覆盖 = 片段中有对应的明文表述（同义改写也算），而不是靠推理。

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
    parser.add_argument("--samples", default=None, help="QA 样例 JSONL（默认内置 5 条）")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    parser.add_argument("--skip-judge", action="store_true", help="只跑检索+生成，不打分")
    args = parser.parse_args()

    samples = DEFAULT_SAMPLES
    if args.samples and os.path.isfile(args.samples):
        samples = []
        with open(args.samples, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    samples.append(json.loads(line))
    print(f"📋 加载 {len(samples)} 条 QA 样例")

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

    # 3. 输出
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(eval_data),
        "top_k": args.top_k,
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


if __name__ == "__main__":
    main()