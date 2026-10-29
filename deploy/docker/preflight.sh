#!/usr/bin/env bash
# ==============================================================================
# 构建前体检：在花十几分钟构建之前，先把能在几秒内发现的问题挑出来
#
#   ./preflight.sh
#
# ★ 为什么需要这个脚本：
#   1. Dockerfile 里写错系统包名（libsnd1 vs libasound2、libglib2.0-0 vs
#      libglib2.0-0t64），要等到构建推进到 apt 那一步才失败 —— 前面几分钟全白等。
#      而包名不能凭印象写，Debian 各版本差别很大（trixie 做了 64-bit time_t 迁移）。
#   2. Python 依赖缺失（像 agent1 缺 pdfplumber）**构建完全成功**，
#      docker build 一路绿灯，只有 run 起来才 ModuleNotFoundError。
#   3. 镜像 COPY 用 glob 漏掉子目录（COPY agent1/*.py 不含 agent1/eval/），
#      同样构建成功、运行报错。
#
#   这三类问题有一个共同点：构建成功 ≠ 能跑。
#   这个脚本把它们全部提前到构建之前。
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ -t 1 ]; then R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; B=$'\033[34m'; N=$'\033[0m'
else R=""; G=""; Y=""; B=""; N=""; fi

step() { echo "${B}==>${N} $*"; }
ok()   { echo "  ${G}✓${N} $*"; }
warn() { echo "  ${Y}!${N} $*"; }
err()  { echo "  ${R}✗${N} $*"; }

PROBLEMS=0
BASE_IMAGE="${BASE_IMAGE:-python:3.10-slim}"

DOCKER_OK=0
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  DOCKER_OK=1
fi

# ---------- 1. Dockerfile 里的系统包是否真实存在 ----------
step "检查 Dockerfile 系统包（$BASE_IMAGE）"

if [ "$DOCKER_OK" -eq 1 ]; then
  # 交给 Python 做：这段逻辑需要读 Dockerfile、多行解析、在容器里查包，
  # 全塞进 bash 会遇到转义地狱 —— 而且踩过一次：
  # awk '{print $2}' 的 $2 被外层 shell 吃掉，输出无法解析，
  # 结果是"有问题的包"被判成"没问题"，漏报比误报危险得多。
  py=""
  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then py="$cand"; break; fi
  done

  if [ -n "$py" ] && [ -f "$PROJECT_ROOT/deploy/tools/check_syspkgs.py" ]; then
    # 这一步要起容器跑 apt-get update，最慢。给 180s 上限 ——
    # 预检的意义是"快速发现问题"，本身卡住就本末倒置了。
    if out="$(timeout 180 "$py" "$PROJECT_ROOT/deploy/tools/check_syspkgs.py" \
               --base "$BASE_IMAGE" \
               --dockerfiles "$PROJECT_ROOT/deploy/docker" 2>&1)"; then
      echo "$out" | sed 's/^/  /'
      n="$(echo "$out" | grep -cE '^\s*✗' || true)"
      PROBLEMS=$((PROBLEMS + n))
    else
      rc=$?
      if [ "$rc" -eq 124 ]; then
        warn "系统包检查超时（镜像拉取或 apt 索引慢），跳过"
      else
        echo "$out" | sed 's/^/  /'
        warn "系统包检查执行失败"
      fi
    fi
  else
    warn "找不到 python 或 check_syspkgs.py，跳过系统包检查"
  fi
else
  warn "Docker daemon 未运行，跳过系统包检查"
fi

# ---------- 2. COPY 是否用 glob 漏掉子目录 ----------
step "检查 COPY 是否覆盖子目录"

for df in "$PROJECT_ROOT"/deploy/docker/*.Dockerfile; do
  [ -f "$df" ] || continue
  name="$(basename "$df")"

  # 形如 COPY agent1/*.py agent1/ 的 glob 写法
  if grep -qE '^COPY +[a-z0-9_]+/\*' "$df"; then
    err "$name  用了 COPY <dir>/*.py —— 不含子目录，eval/ ui/ modules/ 会被漏掉"
    grep -nE '^COPY +[a-z0-9_]+/\*' "$df" | sed 's/^/        /'
    PROBLEMS=$((PROBLEMS + 1))
  else
    ok "$name  COPY 覆盖完整"
  fi
done

# ---------- 3. Python 依赖是否齐备 ----------
step "检查 requirements 语法 + 启动导入链"

if [ -z "$py" ]; then
  warn "找不到 python，跳过依赖检查"
else
  # 3a. requirements 语法。实测一个 `vosk>=0.3.45# 注释`（少一个空格）
  #     让 docker build 白跑了 14 分钟才失败 —— 这种纯语法错误
  #     用 pip 的解析器几秒就能查出来。
  if [ -f "$PROJECT_ROOT/deploy/tools/check_requirements.py" ]; then
    if out="$(timeout 60 "$py" "$PROJECT_ROOT/deploy/tools/check_requirements.py" 2>&1)"; then
      echo "$out" | sed 's/^/  /'
    else
      echo "$out" | sed 's/^/  /'
      PROBLEMS=$((PROBLEMS + 1))
    fi
    echo
  fi

  # 3b. 包是否真的存在于 PyPI。真实的失败案例是 `baiduocr>=2.0` ——
  #     包名是照着 import 名猜的，PyPI 上根本没有，pip 报
  #     "No matching distribution found"，同样是白等几分钟才失败。
  #
  #     加超时保护：网络查询不该让"构建前检查"本身变成卡点。
  #     超时按跳过处理，不阻塞构建 —— 依赖装不上时 pip 会明确报错，
  #     这里只是提前一步。
  if [ -f "$PROJECT_ROOT/deploy/tools/check_pypi.py" ]; then
    if out="$(timeout 90 "$py" "$PROJECT_ROOT/deploy/tools/check_pypi.py" \
                --root "$PROJECT_ROOT" --timeout 5 2>&1)"; then
      echo "$out" | sed 's/^/  /'
    else
      rc=$?
      if [ "$rc" -eq 124 ]; then
        warn "PyPI 校验超时（网络问题），跳过"
      else
        echo "$out" | sed 's/^/  /'
        PROBLEMS=$((PROBLEMS + 1))
      fi
    fi
    echo
  fi

  # 3c. 启动导入链。只查落在启动链上的缺失 —— eval/ 或 agent2/3 的按需依赖
  #     缺失不影响启动，全报出来只会淹没真问题。
  AUDIT="$PROJECT_ROOT/deploy/tools/audit_import_chain.py"
  if [ -f "$AUDIT" ]; then
    if out="$("$py" "$AUDIT" 2>&1)"; then
      line="$(echo "$out" | grep -E '启动链缺失依赖' | head -1)"
      n="$(echo "$line" | grep -oE '[0-9]+' | head -1)"
      if [ "${n:-0}" -gt 0 ]; then
        err "启动链上缺 $n 个依赖 —— 这些缺失会让容器启动即崩"
        echo "$out" | sed -n '/启动链缺失依赖/,$p' | tail -n +2 | sed 's/^/        /'
        PROBLEMS=$((PROBLEMS + 1))
      else
        ok "启动链依赖齐备"
      fi
    else
      warn "导入链分析执行失败，跳过"
    fi
  else
    warn "audit_import_chain.py 不存在，跳过导入链检查"
  fi
fi

# ---------- 4b. 容器内的 import 路径 ----------
step "检查容器内 import 路径"

CIMP="$PROJECT_ROOT/deploy/tools/check_container_imports.py"
if [ -n "$py" ] && [ -f "$CIMP" ]; then
  # ★ 这一项抓的是「本地正常、容器里 ModuleNotFoundError」这类问题。
  #   原因：本地 CWD 是项目根，Python 自动把 CWD 放进 sys.path；
  #   而 agent3 的容器 WORKDIR 是 /app/agent3，common 包却在 /app/common。
  #   静态检查发现不了 —— 文件都在、语法都对，只有按容器方式跑才暴露。
  if out="$("$py" "$CIMP" "$PROJECT_ROOT" 2>&1)"; then
    echo "$out" | sed 's/^/  /'
  else
    echo "$out" | sed 's/^/  /'
    PROBLEMS=$((PROBLEMS + 1))
  fi
  echo
fi

# ---------- 5. 运行时挂载是否存在 ----------
step "检查运行时挂载"

if [ -f "$PROJECT_ROOT/agent1/faiss_full/index.faiss" ]; then
  ok "向量库索引就位"
else
  warn "向量库索引缺失（agent1 会启动但检索不可用）"
  echo "        python agent1/build_index.py --rebuild"
fi

if [ -d "$PROJECT_ROOT/agent1/templates" ] && [ -n "$(ls -A "$PROJECT_ROOT/agent1/templates" 2>/dev/null)" ]; then
  ok "文书模板就位"
else
  warn "模板目录为空或不存在，文书生成会找不到模板"
fi

if [ -f "$PROJECT_ROOT/.env" ]; then
  ok ".env 存在"
else
  warn ".env 缺失，服务起来也调不通 LLM"
  echo "        cp .env.example .env"
fi

# ---------- 汇总 ----------
echo
if [ "$PROBLEMS" -gt 0 ]; then
  err "发现 $PROBLEMS 个会在构建期或运行期致命的问题"
  echo "        修完再构建，否则会白等好几分钟"
  exit 1
fi

ok "预检通过，可以构建"
exit 0