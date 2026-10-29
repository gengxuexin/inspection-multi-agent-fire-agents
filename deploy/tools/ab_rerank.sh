#!/usr/bin/env bash
# 精排改动的 A/B 对照实验。
#
# ★ 为什么必须跑两次而不是只跑一次：
#   RAGAS 的四个指标都由 LLM 判分，实测单次抖动约 0.021。
#   只跑一轮新代码、跟历史报告比，历史报告是**旧精排 + 别的改动**混在一起，
#   变量不隔离 —— 这正是当初 v7/v8/v9 三轮空跑却被当成有效结论的根源。
#
#   正确做法：同一份索引、同一批样本、同一模型，只切精排这一个变量，跑两轮。
#
# 流程：
#   1. 当前代码（新精排）-> report_rerank_new.json
#   2. 切回旧精排 -> report_rerank_old.json
#   3. 切回新精排（务必切回来，否则下次跑评测用到的是旧逻辑）
#   4. 用 --baseline 做对比，含空跑检测与噪声判定
set -uo pipefail

cd /home/geng/fire_agent_v2/fire_agents
PY=/home/geng/.venv/bin/python
HYBRID=agent1/hybrid.py
BAK=/tmp/hybrid_ab.bak

echo "=========================================="
echo " 精排 A/B 对照实验"
echo "=========================================="

if [ ! -f "$HYBRID" ]; then
  echo "✗ 找不到 $HYBRID"
  exit 1
fi

# ---- 备份当前（应为新精排）----
cp "$HYBRID" "$BAK"
echo "已备份当前精排实现"

# ---- 确认当前是新精排 ----
if grep -q '_IDF_MAX = 8.0' "$HYBRID"; then
  echo "✓ 当前是【新精排】(_IDF_MAX=8.0 + idf^2 + 疑问句折扣)"
else
  echo "✗ 当前不是新精排，先确认代码状态"
  exit 1
fi

# ---- 第 1 轮：新精排 ----
echo
echo ">>> 第 1 轮：新精排"
"$PY" agent1/eval/local_eval.py \
  --samples agent1/eval/qa_samples.jsonl \
  --top-k 5 \
  --output report_rerank_new.json 2>&1 | tail -14

# ---- 切回旧精排 ----
echo
echo ">>> 切换到【旧精排】(_IDF_MAX=3.0 + bool 命中)"
"$PY" - <<'PYEOF'
import pathlib
p = pathlib.Path("agent1/hybrid.py")
s = p.read_text(encoding="utf-8")

# 1) IDF 上限 3.0
s = s.replace("_IDF_MAX = 8.0", "_IDF_MAX = 3.0")

# 2) 覆盖率：idf^2 + 疑问句折扣 -> 线性 idf + bool 命中
old_block = """            if token_set:
                got = sum(w * w for t, w in weights.items() if t in text)
                cov = got / total_w
                if cue_ratio > 0:
                    cov *= (1.0 - self._CUE_DISCOUNT * cue_ratio)
                score += self.W_OVERLAP * cov"""
new_block = """            if token_set:
                got = sum(w for t, w in weights.items() if t in text)
                score += self.W_OVERLAP * (got / total_w)"""
if old_block in s:
    s = s.replace(old_block, new_block, 1)
    print("  覆盖率已切回 bool 命中")
else:
    print("  ! 未找到覆盖率块（结构可能已变）")

# 3) 归一化基准随之改回线性
s = s.replace(
    "            total_w = sum(\n"
    "                w * w for w in weights.values()) or 1.0",
    "            total_w = sum(weights.values()) or 1.0",
)

p.write_text(s, encoding="utf-8")
print("  已切回旧精排")
PYEOF

if ! grep -q '_IDF_MAX = 3.0' "$HYBRID"; then
  echo "✗ 切换失败，恢复备份"
  cp "$BAK" "$HYBRID"
  exit 1
fi

# ---- 第 2 轮：旧精排 ----
echo
echo ">>> 第 2 轮：旧精排"
"$PY" agent1/eval/local_eval.py \
  --samples agent1/eval/qa_samples.jsonl \
  --top-k 5 \
  --output report_rerank_old.json 2>&1 | tail -14

# ---- 务必切回新精排 ----
echo
echo ">>> 恢复【新精排】"
cp "$BAK" "$HYBRID"
if grep -q '_IDF_MAX = 8.0' "$HYBRID"; then
  echo "  ✓ 已恢复"
else
  echo "  ✗ 恢复失败！手动检查 $HYBRID"
fi

# ---- 对比 ----
echo
echo "=========================================="
echo " 对比结论（新精排 vs 旧精排）"
echo "=========================================="
"$PY" agent1/eval/local_eval.py \
  --samples agent1/eval/qa_samples.jsonl \
  --top-k 5 \
  --limit 0 \
  --skip-judge \
  --output /tmp/throwaway.json 2>&1 | tail -2 || true

"$PY" - <<'PYEOF'
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(".")
spec = importlib.util.spec_from_file_location(
    "le", ROOT / "agent1" / "eval" / "local_eval.py")
le = importlib.util.module_from_spec(spec)
spec.loader.exec_module(le)

old = json.loads((ROOT / "report_rerank_old.json").read_text(encoding="utf-8"))
new = json.loads((ROOT / "report_rerank_new.json").read_text(encoding="utf-8"))

rc = le.compare_with_baseline(
    new.get("details") or [],
    "report_rerank_old.json",
    cur_scores=new.get("scores"),
)

print()
print("新精排四项指标：")
for k in ("faithfulness", "answer_relevancy",
          "context_precision", "context_recall"):
    o = (old.get("scores") or {}).get(k)
    n = (new.get("scores") or {}).get(k)
    print(f"  {k:22s} {o if o is not None else '?':>8} -> {n if n is not None else '?':>8}")

raise SystemExit(rc)
PYEOF