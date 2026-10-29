#!/usr/bin/env bash
# ==============================================================================
# 消防智能执法系统 · 一键启动
#
#   ./up.sh              启动核心服务（redis + gateway + agent1/2/3）
#   ./up.sh milvus       额外启动 Milvus 向量库
#   ./up.sh obs          额外启动可观测栈（OTel + Langfuse）
#   ./up.sh all          全部启动
#   ./up.sh down         停止并移除容器
#   ./up.sh logs [svc]   查看日志（不指定服务则全部）
#   ./up.sh status       查看状态
#   ./up.sh preflight    只跑构建前体检
#   ./up.sh verify       只跑体检（服务已起时），不启动任何东西
#
# ★ 与「docker compose up -d」的区别：
#   compose up 成功只代表容器被创建了，不代表服务能用。
#   镜像缺依赖、挂载路径不存在、端口被占这些问题，up 全都不会报错，
#   只会在你打开 Swagger 时才发现页面是 502。
#   所以这个脚本在 up 之后强制跑一遍 verify —— 起不来要当场喊出来。
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

COMPOSE="docker compose"
SERVICES_CORE=(redis agent1 agent2 agent3 gateway)

# ---------- 终端着色 ----------
if [ -t 1 ]; then
  R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; B=$'\033[34m'; N=$'\033[0m'
else
  R=""; G=""; Y=""; B=""; N=""
fi

step() { echo "${B}==>${N} $*"; }
ok()   { echo "  ${G}✓${N} $*"; }
warn() { echo "  ${Y}!${N} $*"; }
err()  { echo "  ${R}✗${N} $*" >&2; }

# ---------- 前置检查 ----------
preflight() {
  step "前置检查"

  command -v docker >/dev/null 2>&1 || { err "未找到 docker，请先安装 Docker"; exit 1; }

  if ! docker info >/dev/null 2>&1; then
    err "Docker daemon 未运行。请启动 Docker Desktop 后重试。"
    exit 1
  fi
  ok "Docker daemon 正常"

  # .env 是硬依赖：compose 里所有服务都 env_file: ../../.env，
  # 文件不存在时 compose 会报一个和真实原因无关的错误，很难查。
  if [ ! -f "$PROJECT_ROOT/.env" ]; then
    err ".env 不存在：$PROJECT_ROOT/.env"
    echo "     复制模板并填写 API Key：cp .env.example .env"
    exit 1
  fi
  ok ".env 存在"

  # 载入 .env 里的端口配置，让端口检查与 compose 用同一套来源
  if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$PROJECT_ROOT/.env"
    set +a
  fi

  # 端口占用是阻塞项，必须在构建之前查。
  # 端口冲突的报错出现在网络编程阶段 —— 那时镜像已经构建完，
  # 几十分钟的等待全部白费，而且报错信息不会告诉你是谁占的端口。
  check_ports || exit 1

  #向量库是运行时挂载的，缺了 agent1 起得来但一检索就报错。
  if [ ! -f "$PROJECT_ROOT/agent1/faiss_full/index.faiss" ]; then
    warn "向量库索引缺失：agent1/faiss_full/index.faiss"
    warn "  agent1 会启动但检索不可用。先构建索引："
    warn "    cd $PROJECT_ROOT && python agent1/build_index.py --rebuild"
  else
    ok "向量库索引就位"
  fi

  check_ports
}

# ---------- 端口占用检查 ----------
# 为什么要单独查：compose 报端口冲突时的错误信息极难定位 ——
#   "failed to bind host port 0.0.0.0:8003/tcp: address already in use"
# 它不会告诉你是谁占的。而最常见的成因恰恰很明显：
# 本地已经用 python 直接起了服务，端口全被占着。
# 这种情况下报错发生在网络编程阶段，日志里前面的构建信息全都白打印了。
check_ports() {
  local busy=()
  # 与 compose 保持同一套端口来源（.env 可覆盖）。
  # ★ Redis 的 6379 也要查：本机常常已经有别的 Redis 在跑
  #   （比如另一套项目的容器），而 compose 的报错只会说
  #   "failed to bind host port 0.0.0.0:6379/tcp: address already in use"，
  #   完全看不出是别的项目在占。
  #   这个坑很隐蔽：应用四个端口都换掉了以为万事大吉，
  #   结果在 redis 这里才炸。
  for p in "${GATEWAY_PORT:-8000}" "${AGENT1_PORT:-8001}" \
           "${AGENT2_PORT:-8002}" "${AGENT3_PORT:-8003}" \
           "${REDIS_PORT:-6379}"; do
    if port_in_use "$p"; then busy+=("$p"); fi
  done

  if [ "${#busy[@]}" -eq 0 ]; then
    ok "服务端口均空闲"
    return 0
  fi

  err "端口被占用：${busy[*]}"
  echo
  echo "  占用者："
  if command -v ss >/dev/null 2>&1; then
    local pattern
    pattern="$(IFS='|'; echo "${busy[*]}")"
    ss -tlnp 2>/dev/null | grep -E ":(${pattern})[[:space:]]" | sed 's/^/    /'
  elif command -v lsof >/dev/null 2>&1; then
    for p in "${busy[@]}"; do
      lsof -i ":$p" -sTCP:LISTEN 2>/dev/null | tail -n +2 | sed 's/^/    /'
    done
  else
    echo "    （装 lsof 或 ss 可以看到具体进程）"
  fi
  echo
  echo "  最常见的原因：本地已经用 python 直接起了服务，或另一套项目占着端口。"
  echo "  处理方式二选一："
  echo "    1) 停掉占用者，让容器接管（容器化跑的意义就在这里）"
  echo "    2) 给容器换端口，在 .env 里设："
  echo "         GATEWAY_PORT=18000"
  echo "         AGENT1_PORT=18001"
  echo "         AGENT2_PORT=18002"
  echo "         AGENT3_PORT=18003"
  echo "         REDIS_PORT=16379"
  echo
  return 1
}

port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -tln 2>/dev/null | grep -qE "[:.]${port}[[:space:]]"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -i ":$port" -sTCP:LISTEN >/dev/null 2>&1
  else
    # 没有工具就假设空闲，让docker 去报错 —— 误报占用比漏报更烦人
    return 1
  fi
}

# ---------- 体检 ----------
#逐个服务查真实 HTTP 端点。compose 的 healthcheck 只说明容器活着，
# 这里额外确认「业务端点真的返回 200」。
verify() {
  step "服务体检"
  local fails=0

  check_http() {
    local name="$1" url="$2"
    local code
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" 2>/dev/null || echo "000")
    if [ "$code" = "200" ]; then
      ok "$name  $url  → $code"
    else
      err "$name  $url  → $code"
      fails=$((fails + 1))
    fi
  }

  # 等待服务起来：容器 started 不等于 uvicorn 已监听端口。
  # 这里主动轮询，最多等 60 秒。
  step "等待服务就绪（最多 60s）"
  local waited=0
  while [ "$waited" -lt 60 ]; do
    local up
    up=$(docker compose ps --services --filter status=running 2>/dev/null | wc -l | tr -d ' ')
    if [ "$up" -ge 5 ]; then break; fi
    sleep 2; waited=$((waited + 2))
    printf '.'
  done
  echo
  ok "容器已全部运行（耗时 ${waited}s）"

  echo
  step "端点检查"
  # 端口从环境变量读，与 compose 的端口映射保持同一套来源。
  # 写死 8000-8003 的话，一旦用 .env 换了端口，
  # 这里会去打别人机器上的服务，体检结果全是200 —— 绿灯假象比红灯更危险。
  local p_gw="${GATEWAY_PORT:-8000}"
  local p_a1="${AGENT1_PORT:-8001}"
  local p_a2="${AGENT2_PORT:-8002}"
  local p_a3="${AGENT3_PORT:-8003}"
  check_http "gateway " "http://localhost:${p_gw}/"
  check_http "agent1   " "http://localhost:${p_a1}/health"
  check_http "agent2   " "http://localhost:${p_a2}/health"
  check_http "agent3   " "http://localhost:${p_a3}/health"

  echo
  if [ "$fails" -gt 0 ]; then
    err "$fails 个端点未通过"
    echo
    echo "  排查顺序（别急着看全部日志，先看崩溃的那个）："
    echo "    docker compose ps                # 谁没起来"
    echo "    docker compose logs --tail=80 <服务名>"
    echo
    echo "  最常见的三类原因："
    echo "    1. 依赖缺失 → 日志里是 ModuleNotFoundError"
    echo "    2. 挂载路径不存在 → 日志里是 FileNotFoundError，且多为向量库/模板"
    echo "    3. .env 里的 Key 没填 → 日志里是 401/403 或连接超时"
    return 1
  fi
  ok "全部端点通过"
  return 0
}

print_summary() {
  cat <<EOF

${G}════════════════════════════════════════════════════════${N}
  核心服务已就绪
${G}════════════════════════════════════════════════════════${N}
    网关gateway    http://localhost:8000/docs
    知识库 agent1   http://localhost:8001/docs
    语音   agent2   http://localhost:8002/docs
    视觉   agent3   http://localhost:8003/docs

${B}常用命令${N}
    ./up.sh logs agent1     查看某服务日志
    ./up.sh status          查看状态
    ./up.sh down            停止并清理

EOF
}

case "${1:-up}" in
  up)
    shift || true
    case "${1:-}" in
      milvus) PROFILES=(--profile milvus) ;;
      obs)    PROFILES=(--profile obs) ;;
      all)    PROFILES=(--profile milvus --profile obs) ;;
      *)      PROFILES=() ;;
    esac
    preflight
    step "构建镜像"
    # 构建前再跑一次部署预检。镜像构建要好几分钟（agent2/3 装 torch），
    # 预检能在几秒内挑出包名写错、COPY 漏目录这类"构建成功但运行即崩"的问题。
    if [ -x "$SCRIPT_DIR/preflight.sh" ]; then
      if ! bash "$SCRIPT_DIR/preflight.sh"; then
        echo
        err "部署预检未通过。仍要继续构建吗？（输入 n 中止）"
        read -r ans
        [ "$ans" = "n" ] && exit 1
      fi
      echo
    fi
    $COMPOSE "${PROFILES[@]}" build
    step "启动服务"
    $COMPOSE "${PROFILES[@]}" up -d "${SERVICES_CORE[@]}"
    if verify; then
      print_summary
    else
      exit 1
    fi
    ;;

  down)
    step "停止并移除容器"
    $COMPOSE --profile milvus --profile obs down --remove-orphans
    ok "已停止（数据卷保留，需要彻底清理加 --volumes）"
    ;;

  logs)
    if [ $# -ge 2 ]; then $COMPOSE logs -f --tail=100 "$2"; else $COMPOSE logs -f --tail=50; fi
    ;;

  status)
    $COMPOSE --profile milvus --profile obs ps
    ;;

  verify)
    verify
    ;;

  preflight)
    bash "$SCRIPT_DIR/preflight.sh"
    ;;

  *)
    sed -n '2,22p' "$0"
    exit 1
    ;;
esac