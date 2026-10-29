#!/usr/bin/env bash
# 改动四个服务的 server 之后，先做语法体检再起服务。
# 这不是多余的：挂载 metrics 的脚本是按锚点行插入的，
# 位置错了（比如插到 if __name__ 块里、或插断了某个函数体）
# 语法可能仍然合法，但服务启动会出问题。
set -uo pipefail
cd /home/geng/fire_agent_v2/fire_agents

PY=/home/geng/.venv/bin/python
FAIL=0

echo "=== 语法检查 ==="
for f in agent1/server.py gateway.py agent2/server.py agent3/server.py \
         common/metrics.py common/metrics_middleware.py; do
  if [ ! -f "$f" ]; then
    echo "  ✗ $f 不存在"
    FAIL=1
    continue
  fi
  if out="$($PY -c "import ast,pathlib,sys; ast.parse(pathlib.Path(sys.argv[1]).read_text())" "$f" 2>&1)"; then
    echo "  ✓ $f"
  else
    echo "  ✗ $f"
    echo "$out" | tail -4 | sed 's/^/      /'
    FAIL=1
  fi
done

echo
echo "=== metrics 挂载位置核对 ==="
for f in agent1/server.py gateway.py agent2/server.py agent3/server.py; do
  # 计数时要排除 import 块里的名字（那行带缩进且以逗号结尾）
  n_install="$(grep -c '^install_metrics_endpoint(app' "$f" 2>/dev/null || echo 0)"
  n_mw="$(grep -c '^install_http_middleware(app' "$f" 2>/dev/null || echo 0)"
  n_imp="$(grep -c '^from common.metrics_middleware import (' "$f" 2>/dev/null || echo 0)"

  # ★ 判断"被错误缩进"要精确：调用语句必须在行首（缩进 0）。
  #   第一版这里写成 '^    install_metrics_endpoint'，
  #   结果匹配到了 import 块里那行 "    install_metrics_endpoint,"（带逗号），
  #   四个明明插对了的服务全被判成"塞进了函数体"——
  #   检查脚本自己错了，报出的问题会让人去改本来正确的代码。
  bad="$(grep -c '^[[:space:]]\+install_metrics_endpoint(app' "$f" 2>/dev/null || echo 0)"

  status="✓"
  [ "$n_install" != "1" ] && status="✗ 端点调用=$n_install"
  [ "$n_mw" != "1" ] && status="✗ 中间件调用=$n_mw"
  [ "$n_imp" != "1" ] && status="✗ import=$n_imp"
  [ "$bad" != "0" ] && status="✗ 调用被缩进（塞进了函数体）"
  echo "  $status  $f  (端点=$n_install 中间件=$n_mw import=$n_imp 误缩进=$bad)"
  [ "$status" != "✓" ] && FAIL=1
done

echo
echo "=== import 冒烟（metrics 模块能否独立加载）==="
if out="$(timeout 60 $PY -c "
import sys
sys.path.insert(0, '.')
from common import metrics
print('metrics 可用:', metrics.available())
print('渲染长度:', len(metrics.render_latest()[0]))
from common.metrics_middleware import normalize_path
for p in ['/api/doc/change-reports/CR-20261016-48aeda',
          '/tasks/12345',
          '/api/doc/change-reports/8f14e45f-ceea-167a-5a36-dedd4bea2543',
          '/health']:
    print(f'  {p[:44]:46s} -> {normalize_path(p)}')
" 2>&1 | grep -v Warning)"; then
  echo "$out" | sed 's/^/  /'
else
  echo "$out" | tail -6 | sed 's/^/  /'
  FAIL=1
fi

echo
if [ "$FAIL" = "0" ]; then
  echo "  ✓ 全部检查通过"
else
  echo "  ✗ 有问题，先修再起服务"
fi
exit $FAIL