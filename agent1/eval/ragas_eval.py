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
# 内置兜底样例。**完整评测集在 agent1/eval/qa_samples.jsonl（47 条）**，
# 带category/difficulty/verified 字段以支持分层统计。
# 这里的 5 条只是无--samples 参数时的最小可运行集合。
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
def retrieve_contexts(question: str, top_k: int = 5) -> list[str]:
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
    """用 DeepSeek 基于检索到的法条回答问题。

    为什么加"强制标注来源"（P3）
    ----------------------------
    上一轮样本 3 的 faithfulness 只有 0.75：答案 904 字，把检索到的
    4 个片段揉成一段叙述，其中夹了片段里没有的概括性表述。根因是
    prompt 只说"只输出检索结果中真实出现的内容"，**没有要求可追溯**——
    模型把多个片段平均融合后，就分不清哪句话来自哪一段了。

    加了三条硬约束：
      1. 每条结论后必须标[片段N]，把陈述与来源显式绑定
      2. **禁止跨片段合并改写**，逐片回答
      3. 片段不足以回答时明确说"检索内容不足"，而不是硬答

    第3 条尤其重要：允许"答不知道"能显著降低幻觉，因为 faithfulness
    惩罚的是"检索内容里没有的东西"，而不是"答得少"。

    ★ 第3 条的措辞必须给"不足"下可操作的定义（47 条评测打出来的补丁）
    ------------------------------------------------------------------
    report_v6：34/47 条弃答，answer_relevancy 被压到 0.579。逐条核对后
    发现其中 **14 条是 context_recall=1.0 却仍然弃答**，属于过度保守，
    答案其实就在片段里。模型的原话暴露了错误的判据：

      · [24] 消防技术文书应当使用什么式样？
        ctx[0] 开头就是"消防行政法律文书式样" -> 模型仍答"检索内容不足"
      · [30] 消防法关于消防设施维护保养的要求是什么？
        ctx[3] 写着"应当每月进行维护保养，确保其完好有效和处于正常运行状态"
        -> 模型因"该片段未标明是《消防法》第几条"而弃答
      · [2] 裁量基准是什么？
        模型承认片段里有严重/一般/较轻档位，但因"未完整、明确地写出
        整体性内容"而弃答 —— 它在要**穷尽列举**，而问题只需要定义

    共同的错误：把"**完整**"当成了回答的前提。而 RAGAS 的
    answer_relevancy 判的是"答案是否切题"，答不出时一律给 0.4，
    于是 34 条弃答直接把该指标按比例拉低（弃答组均值 0.429，
    正常组 0.969）。

    所以规则 3 改为：**只判断"有没有相关片段"，不判断"是否完整"**。
    """
    from config import ds_client

    context_text = "\n\n".join(f"--- 片段 {i+1} ---\n{c}" for i, c in enumerate(contexts))
    prompt = (
        "你是消防法律助手。请基于下面检索到的法律条文回答问题。\n\n"
        "严格遵守以下规则：\n"
        "1. 每一条结论后面必须用 [片段N] 标注来源，N 是片段编号。\n"
        "2. 只能使用某个片段中明确写到的内容。禁止把多个片段的信息合并、"
        "概括或推断后当成一条结论。\n"
        "3. **判断标准：只要有任意一个片段与问题相关，就必须基于它作答。**\n"
        "   提问者要的通常不是穷尽列举，而是规则、定义或主要要求。\n"
        "   - 片段已给出答案，即使答案不完整、只覆盖部分情形，也要作答，"
        "并在末尾用一句话说明还有哪些情形检索内容未覆盖。\n"
        "   - 片段没有标明出处的（例如只写了条文内容、未写条款号），"
        "照样直接引用，不要因为无法确认条款号就拒答。\n"
        "   - 只有当**所有片段都与问题无关**时，才回答"
        "「检索内容不足，无法回答该问题」。\n"
        "4. 引用条款号、罚款金额等具体数字时必须原文照抄，不得改写。\n\n"
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
        # ★ LLM 失败时返回空串会让这一条的四个指标全部记0，
        # 在 47 条里就是 1/47 的假低分，而且报告里看不出是"答不出"
        # 还是"服务挂了"。这里显式标记，让排障时能立刻区分。
        print(f"  ⚠️ LLM 生成失败（该条将计 0 分，请检查服务是否可用）：{e}")
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
    parser.add_argument("--top-k", type=int, default=5,
                        help="检索 top_k（默认 5；实测越大噪声越多，"
                             "context_precision 反而下降。上限由 retrieval_backend 收敛到 10）")
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
