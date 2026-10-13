# ragas_eval.py
"""
RAGAS 评估脚本：对样例 QA 集计算 faithfulness / answer_relevancy，输出 JSON 报告。

用法：
    cd agent1
    python eval/ragas_eval.py                       # 默认 eval/qa_samples.jsonl
    python eval/ragas_eval.py --samples custom.jsonl --top-k 8 --output eval/report.json

依赖：
    pip install -r ../../requirements-eval.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ---- sys.path ----
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT1_DIR = os.path.dirname(_SCRIPT_DIR)
_ROOT = os.path.dirname(_AGENT1_DIR)
for p in (_ROOT, _AGENT1_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import httpx

# ==================== 默认 QA 样例 ====================
DEFAULT_SAMPLES = [
    {
        "question": "《消防法》第六十条第一款第七项规定了什么？",
        "ground_truth": "其他消防安全违法行为依照《中华人民共和国消防法》第六十条第一款第七项的规定处罚",
    },
    {
        "question": "公众聚集场所投入使用营业前消防安全检查实行什么制度？",
        "ground_truth": "实行告知承诺管理，申请人可自主选择采用告知承诺方式办理",
    },
    {
        "question": "消防行政处罚的裁量基准是什么？",
        "ground_truth": "根据违法行为的事实、性质、情节及社会危害程度依法确定处罚种类和幅度",
    },
    {
        "question": "浙江省消防行政处罚裁量规定对轻微违法行为如何处理？",
        "ground_truth": "对轻微违法行为不予处罚，实行包容审慎监管",
    },
    {
        "question": "高层民用建筑消防安全管理规定对电动自行车有什么要求？",
        "ground_truth": "禁止在高层民用建筑的公共门厅、疏散走道、楼梯间、安全出口停放电动自行车或充电",
    },
]


# ==================== 检索 ====================
def retrieve_contexts(question: str, top_k: int = 8) -> list[str]:
    """调 retrieval_backend.search 拿法条上下文。"""
    try:
        from retrieval_backend import search
        results = search(question, top_k=top_k)
        return [r["content"][:500] for r in results if r.get("content")]
    except Exception as e:
        print(f"  ⚠️ 检索失败: {e}")
        return []


# ==================== LLM 生成答案 ====================
def generate_answer(question: str, contexts: list[str]) -> str:
    """用 DeepSeek 基于检索到的法条回答问题。"""
    from config import ds_client

    context_text = "\n\n".join(f"--- 片段 {i+1} ---\n{c}" for i, c in enumerate(contexts))
    prompt = (
        "你是消防法律助手。请基于下面检索到的法律条文回答问题。\n"
        "只输出检索结果中真实出现的内容，不要编造条款。\n\n"
        f"【检索到的法条】\n{context_text}\n\n"
        f"【问题】\n{question}"
    )
    try:
        resp = ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            timeout=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️ LLM 生成失败: {e}")
        return ""


# ==================== RAGAS 评估 ====================
def run_ragas(eval_data: list[dict]) -> dict:
    """用 RAGAS 计算 faithfulness / answer_relevancy。"""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
    except ImportError:
        print("⚠️ ragas 未安装，跳过评估。请 pip install -r ../../requirements-eval.txt")
        return {"error": "ragas not installed"}

    ds = Dataset.from_list(eval_data)
    print(f"\n📊 开始 RAGAS 评估（{len(ds)} 条）...")
    t0 = time.time()

    try:
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy],
        )
        elapsed = time.time() - t0
        scores = dict(result)
        scores["_elapsed_sec"] = round(elapsed, 1)
        scores["_num_samples"] = len(eval_data)
        return scores
    except Exception as e:
        print(f"⚠️ RAGAS 评估失败: {e}")
        return {"error": str(e)}


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description="RAGAS 评估")
    parser.add_argument("--samples", default=None,
                        help="QA 样例 JSONL（默认内置 5 条）")
    parser.add_argument("--top-k", type=int, default=8,
                        help="检索 top_k")
    parser.add_argument("--output", default=None,
                        help="输出 JSON 路径（默认 stdout）")
    args = parser.parse_args()

    # 1. 加载样例
    samples = DEFAULT_SAMPLES
    if args.samples and os.path.isfile(args.samples):
        samples = []
        with open(args.samples, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    samples.append(json.loads(line))
    print(f"📋 加载 {len(samples)} 条 QA 样例")

    # 2. 逐条检索 + 生成
    eval_data = []
    for i, s in enumerate(samples, 1):
        q = s["question"]
        gt = s["ground_truth"]
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

    # 3. RAGAS 评估
    scores = run_ragas(eval_data)

    # 4. 输出
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

    print(f"\n{'='*60}")
    print("评估完成")


if __name__ == "__main__":
    main()
