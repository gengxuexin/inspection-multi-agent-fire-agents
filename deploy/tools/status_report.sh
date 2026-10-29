#!/usr/bin/env bash
# 项目状态盘点：把"做到了什么程度"用可验证的数字说清楚。
set -uo pipefail

PROJ=/home/geng/fire_agent_v2/fire_agents
DESK=/mnt/c/Users/Lenovo/Desktop/fire_agent_v2/fire_agents
PY=/home/geng/.venv/bin/python

hr() { printf '%s\n' "------------------------------------------------------------"; }

echo "============================================================"
echo " 1. 代码规模"
echo "============================================================"
cd "$PROJ"
code=$(find . -name '*.py' -not -path './.git/*' -not -path '*/__pycache__/*' \
       -not -path './deploy/tools/*' -not -path './agent1/demo/*' 2>/dev/null | wc -l)
loc=$(find . -name '*.py' -not -path './.git/*' -not -path '*/__pycache__/*' \
      -not -path './deploy/tools/*' -not -path './agent1/demo/*' 2>/dev/null \
      | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
tools=$(ls deploy/tools/*.py 2>/dev/null | wc -l)
toolsloc=$(cat deploy/tools/*.py 2>/dev/null | wc -l)
demo=$(ls agent1/demo/*.py agent1/demo/*.html agent1/demo/*.sh 2>/dev/null | wc -l)
demoloc=$(cat agent1/demo/*.py agent1/demo/*.html 2>/dev/null | wc -l)
docs=$(ls docs/*.md 2>/dev/null | wc -l)
docsloc=$(cat docs/*.md 2>/dev/null | wc -l)

printf '  业务代码      %4d 文件 %6d 行\n' "$code" "${loc:-0}"
printf '  验证脚本      %4d 文件 %6d 行\n' "$tools" "$toolsloc"
printf '  演示模块      %4d 文件 %6d 行\n' "$demo" "$demoloc"
printf '  文档          %4d 文件 %6d 行\n' "$docs" "$docsloc"
hr

echo "============================================================"
echo " 2. 测试"
echo "============================================================"
out=$("$PY" -m pytest tests/ -q 2>&1 | tail -1)
echo "  $out"
per=$(for f in tests/test_*.py; do
  n=$(grep -c 'def test_' "$f" 2>/dev/null || echo 0)
  printf '%s %s\n' "$(basename "$f" .py | sed 's/test_//')" "$n"
done)
echo "$per" | awk '{printf "    %-24s %3d\n", $1, $2}'
hr

echo "============================================================"
echo " 3. 服务状态"
echo "============================================================"
docker ps --filter name=fire-agents --format '{{.Names}}\t{{.Status}}' 2>/dev/null \
  | sort | awk -F'\t' '{printf "    %-32s %s\n", $1, $2}'
if ss -ltn 2>/dev/null | grep -q ':8800'; then
  echo "    演示服务                       RUNNING (http://127.0.0.1:8800)"
fi
hr

echo "============================================================"
echo " 4. 镜像"
echo "============================================================"
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' 2>/dev/null \
  | grep -E '^fire-' | sort | awk -F'\t' '{printf "    %-28s %s\n", $1, $2}'
hr

echo "============================================================"
echo " 5. 桌面 / WSL 一致性"
echo "============================================================"
diff_count=0
for f in agent1/hybrid.py agent1/eval/local_eval.py agent1/demo/index.html \
         agent1/demo/trace_pipeline.py agent1/demo/demo_server.py \
         tests/test_hybrid.py tests/test_eval_baseline.py \
         deploy/docker/docker-compose.yml requirements.txt; do
  if [ -f "$PROJ/$f" ] && [ -f "$DESK/$f" ]; then
    cmp -s "$PROJ/$f" "$DESK/$f" || { echo "    不一致: $f"; diff_count=$((diff_count+1)); }
  else
    echo "    缺失: $f"; diff_count=$((diff_count+1))
  fi
done
[ "$diff_count" = "0" ] && echo "    9 个关键文件两侧完全一致"
hr

echo "============================================================"
echo " 6. 评测可信度机制"
echo "============================================================"
if [ -f report_v7.json ] && [ -f report_v8.json ]; then
  "$PY" - <<'PYEOF'
import json, pathlib
def fp(rep):
    import hashlib
    h = hashlib.sha256()
    for d in (rep.get("details") or []):
        h.update((d.get("question") or "").encode())
        for c in (d.get("contexts") or []):
            h.update(c.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]

for v in ("v7", "v8", "v9"):
    p = pathlib.Path(f"report_{v}.json")
    if p.is_file():
        rep = json.loads(p.read_text(encoding="utf-8"))
        sc = rep.get("scores") or {}
        print(f"    report_{v}  指纹={fp(rep)}  "
              f"rel={sc.get('answer_relevancy')}  prec={sc.get('context_precision')}")
print()
print("    v7/v8/v9 指纹相同 -> 那三轮是空跑，结论无效")
PYEOF
fi
hr