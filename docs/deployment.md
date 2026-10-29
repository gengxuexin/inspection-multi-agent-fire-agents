# 部署指南

面向**第一次接手这个项目的人**。目标是让你在 10 分钟内把整套服务跑起来，并且出问题时知道往哪看。

---

## 1. 前置条件

| 组件 | 版本要求 | 用途 |
|------|---------|------|
| Docker Desktop / Docker Engine | 24+ | 跑全部服务 |
| Docker Compose | v2.20+ | 编排（`docker compose`，不是 `docker-compose`） |
| 磁盘 | ≥ 10 GB | 镜像 + Milvus + Langfuse 数据卷 |
| 内存 | ≥ 8 GB | 四个Python 服务 + Milvus 同时跑 |

不需要本地装 Python，不需要本地装 FAISS/Milvus。

---

## 2. 三步启动

```bash
cd deploy/docker

# 1) 准备配置（只需一次）
cp ../../.env.example ../../.env
# 编辑 .env，填入 DEEPSEEK_API_KEY / EMBEDDING_API_KEY 等

# 2) 启动
./up.sh              # 核心服务：redis + gateway + agent1/2/3
./up.sh milvus# 额外启动 Milvus 向量库
./up.sh all          # 全部启动（含可观测栈）

# 3) 打开
#    http://localhost:8000/docs
```

启动成功后会打印各服务的 Swagger 地址和常用命令。

---

## 3. `up.sh` 与 `docker compose up -d` 的区别

`docker compose up -d` 成功**只代表容器被创建了**，不代表服务能用。

镜像缺依赖、挂载路径不存在、端口被占——这些问题 `up` 全都不会报错，
只会在你打开 Swagger 时看到一片502，或者检索时忽然报「向量库不完整」。

所以 `up.sh` 在 `up` 之后强制跑一遍 `verify`：逐个打真实 HTTP 端点，
失败就当场打印排查方向，而不是让你自己去猜。

```bash
./up.shverify       # 只体检，不启动
```

> ★ **`./up.sh` 里用的是 `build` 而不是 `up -d --build`。**
> 但如果你**手工改过代码**，必须显式加 `--build`：
>
> ```bash
> docker compose up -d --build redis gateway agent1 agent2 agent3
> ```
>
> 不加就会遇到一个很难察觉的情况：**源码改了，服务行为没变。**
> 因为镜像名字（`fire-agent1:latest`）没变，compose 认为镜像是最新的，
> 直接复用了旧镜像里的旧代码。表现是：
> `curl /metrics` 返回 404，但你刚明明在代码里挂了这个端点。
>
> 判断方法（一行）：
> ```bash
> docker exec fire-agents-agent1-1 grep -c 'install_metrics_endpoint' /app/agent1/server.py
> ```
> 源码里是 2、容器里是 0 → 镜像没重建。

### 更早一步：`preflight.sh`

本项目吃过两次亏，都是同一类问题——**构建成功，但构建期浪费了几分钟到十几分钟**：

| 真实案例 | 报错 | 白等时长 |
|---------|------|---------|
| `dashscope>=1.14# 阿里云`（注释前少空格） | `Invalid requirement` | 14 分钟 |
| `baiduocr>=2.0`（包名靠 import 名反推，PyPI 上不存在） | `No matching distribution found` | 7 分钟 |
| `libsnd1`（Ubuntu 包名，Debian 13 已改名） | `Unable to locate package` | 数分钟 |

而这三个检查加起来只要 **18 秒**。

`preflight.sh` 在构建前检查五类问题：

| 检查项 | 抓什么问题 | 依据 |
|--------|-----------|------|
| Dockerfile 系统包 | 包名在基础镜像里不存在 | 起容器跑 `apt-cache policy` |
| COPY 是否用 glob | `COPY agent1/*.py` 漏掉 `eval/` `ui/` | 正则扫 Dockerfile |
| requirements 语法 | 注释紧贴版本号、格式错误 | **pip 自己的解析器** |
| 包是否存在于 PyPI | 靠 import 名反推出的假依赖 | 查 PyPI JSON API |
| 启动导入链 | 代码 import 了但没声明的依赖 | AST 递归展开 import |

```bash
./up.sh preflight     # 或直接 ./preflight.sh
```

它会自动调用 `deploy/tools/` 下的审计脚本：

| 脚本 | 作用 |
|------|------|
| `check_syspkgs.py` | 查包名在基础镜像中是否真实存在（**带缓存**，命中后0.2s） |
| `check_requirements.py` | 用 pip 自己的解析器校验 requirements 语法 |
| `check_pypi.py` | 校验包在 PyPI 上真实存在（**8线程并发**） |
| `audit_import_chain.py` | 递归展开 import 链，只报**会导致启动失败**的缺失依赖 |
| `audit_deps.py` | 全量对比代码 import 与 requirements 声明 |
| `audit_services.py` | 对 gateway/agent2/agent3 分别做启动链审计 |

> **这些工具本身都经过反向验证。**
> 最初的 `check_syspkgs.py` 因shell 转义问题（`awk '{print $2}'` 的 `$2` 被外层吃掉）
> 而**静默漏报**——报告「系统包齐备」，但 `libsnd1` 根本不存在。
> 第二版又因多行 RUN 解析错误，把 4 个 Dockerfile 全报成「无 apt 安装」。
> 一个永远说「通过」的检查工具比没有检查更危险，因为它会让人以为已经查过了。
>
> 现在每加一个检查，都会**先注入一个真实缺陷验证它能报出来**，再确认修复后恢复全绿。

### 依赖的一个原则

`requirements` 里的每个包都应该能追溯到代码里真实的 `import`。

> 反例：`baiduocr` 是照着 `agent3/modules/baiduOCR.py` 这个文件名猜出来的包名，
> 但那个文件其实是**用 `requests` 直连百度 HTTP API**，根本不 import 任何百度 SDK。
>
> `audit_deps.py` 的对比方向是「代码 import → 包」，这个方向是安全的；
> 反过来「包 → 代码」会把想象出来的依赖也带进来。

---

## 4. 服务拓扑

```
                    ┌──────────────┐
                    │   gateway    │  :8000
                    │  (LangGraph) │
                    └──┬───┬───┬───┘
          ┌────────────┘   │   └────────────┐
          ▼                ▼                ▼
   ┌────────────┐   ┌────────────┐   ┌────────────┐
   │  agent1    │   │  agent2│   │  agent3    │
   │ 法规检索    │   │ 语音/ASR  │   │ 视觉/YOLO   │
   │  文书生成   │   │  :8002    │   │   :8003     │
   │   :8001    │   └────────────┘   └────────────┘
   └─────┬──────┘
         │
    ┌────┴─────┬──────────┐
    ▼          ▼          ▼
┌────────┐┌────────┐ ┌──────────┐
│ redis  ││ FAISS  │ │  Milvus  │  ← --profile milvus
│ :6379  ││ (挂载) │ │  :19530  │
└────────┘└────────┘ └──────────┘
```

### 端口一览

| 服务 | 端口 | 说明 |
|------|------|------|
| gateway | 8000 | 统一入口，外部只应暴露这一个 |
| agent1 | 8001 | 法规检索 + 文书生成 |
| agent2 | 8002 | 语音识别 / 合成 |
| agent3 | 8003 | 图像 / 视频分析 |
| redis | 6379 | 熔断器状态、异步任务队列 |
| Milvus | 19530 | 向量库（可选，默认走 FAISS） |
| **Prometheus** | **9090** | 指标存储（`--profile obs`） |
| **Grafana** | **3001** | 看板（`--profile obs`） |
| Langfuse | 3000 | LLM 观测（`--profile obs`） |
| OTel | 4317/4318 | Trace 采集（`--profile obs`） |

> **生产部署不要把 8001/8002/8003 暴露出去。**
> 这三个端口直接暴露等于绕过 gateway 的鉴权与限流。
> compose 里的端口映射是为了本地调试，生产应由防火墙或反向代理拦截。
>
> Grafana 用 3001 而不是 3000，因为 Langfuse 占了 3000。
> 两者都是 `obs` profile，互不冲突。

### 端口被占用怎么办

本地已经用 `python` 直接起了服务是很常见的情况——容器化跑的意义恰恰在于
**用容器接管这些服务**，所以这时应该停掉本地进程：

```bash
# 找出占用者（up.sh 会自动做这一步并打印占用进程）
ss -tlnp | grep -E ':800[0-3]'
```

如果确实需要两者并存（比如对比"本地跑的"和"容器里跑的"），给容器换一组端口。
在 `.env` 里加：

```bash
GATEWAY_PORT=18000
AGENT1_PORT=18001
AGENT2_PORT=18002
AGENT3_PORT=18003
REDIS_PORT=16379
```

或临时指定：

```bash
GATEWAY_PORT=18000 REDIS_PORT=16379 ./up.sh
```

> **`REDIS_PORT` 也要改，这一点很容易漏。**
> 本机常常已经有别的 Redis 在跑（另一套项目的容器、或本机装的 redis-server），
> 而 6379 是默认值。我第一次换端口时就只改了应用四个端口，
> 结果应用全起来了，却在 redis 这里炸：
> `failed to bind host port 0.0.0.0:6379/tcp: address already in use`。
> 这个坑很隐蔽，因为前四个端口确实通了，会让人以为已经搞定。
> 所以 `up.sh` 的端口检查覆盖**全部五个端口**。

> **端口检查为什么必须放在构建之前。**
> compose 报端口冲突的时机是网络编程阶段，那时镜像已经构建完了——
> 几十分钟的等待全部白费。而它的报错是
> `failed to bind host port 0.0.0.0:8003/tcp: address already in use`，
> **不会告诉你是谁占的**。所以 `up.sh` 在 preflight 阶段就查端口并打印占用进程。

---

## 5. 运行时挂载（重要）

以下内容**不在镜像里**，靠 volume 挂载：

| 宿主路径 | 容器路径 | 缺失后果 |
|---------|---------|---------|
| `agent1/faiss_full/` | `/app/agent1/faiss_full` | 服务能起，检索报「向量库不完整」 |
| `agent1/knowledge_base/` | `/app/agent1/knowledge_base` | 同上 |
| `agent1/templates/` | `/app/agent1/templates` | 文书生成找不到模板 |
| `.env` | （运行时注入，不进镜像） | 鉴权失败 / LLM 401 |

**首次使用需要先构建索引**：

```bash
cd ../..
python agent1/build_index.py --rebuild
```

`./up.sh` 在启动前会检查 `index.faiss` 是否存在，缺失时给出警告但**不阻断**
—— 因为构建索引耗时较长，是否要构建是使用者的决定，不该由脚本替他做。

---

## 6. 健康检查：两个端点，两种职责

| 端点 | 含义 | 失败后果 |
|------|------|---------|
| `/health` | 存活：进程还在不在 | **重启容器** |
| `/ready` | 就绪：能不能接流量 | **摘流量**，等恢复 |

**这个区分不是形式主义。** 混用是最常见的误配：

> 如果 `/health` 里去查 Redis/Milvus，那么外部依赖抖一下 → 存活探针失败 →
> 容器被重启 → 但重启根本修不好依赖问题 → 抖动被放大成全部服务反复重启。

所以：
- `agent1 /health`只回答「进程还在」，不碰任何外部依赖
- `agent1 /ready` 检查向量库索引、模板目录、输出目录可写性（全是本地静态检查，不做网络调用）
- `agent2 /ready` 只查 `ffmpeg` 在不在 PATH——**不加载 Whisper 模型**，否则每次探针都要重载几百 MB 的模型
- `agent3 /ready` 只查权重文件存在——**不跑 YOLO 推理**
- `gateway` 的存活探针打根路径 `/`，**不打** `/agents/health`（后者会连带探测下游，agent1 挂了会把 gateway 也拖下水）

---

## 7. 镜像构建

```bash
./build.sh                      # 本地构建全部
REGISTRY=registry.local/fire ./build.sh   # 构建并推送
```

依赖分层做了两处刻意的安排：

1. **依赖层在业务代码之前** —— 改一行业务代码重建只需几秒，不用重装 torch
2. **agent2/agent3 的重依赖单独一层** —— ASR/TTS/YOLO 体积大，
   全塞进 `requirements.txt` 会让 gateway 和 agent1 的镜像无谓地膨胀几个 GB

```bash
pip install -r requirements.txt -r requirements-agent2.txt   # agent2
pip install -r requirements.txt -r requirements-agent3.txt   # agent3
```

### 7.1 构建上下文与镜像体积

> ★ **构建上下文必须是项目根**，不是 `deploy/docker`：
> ```bash
> cd /path/to/fire_agents
> docker build -f deploy/docker/agent3.Dockerfile .
> ```
> 因为 `.dockerignore` 是按构建上下文根目录生效的。
> 在 `deploy/docker` 里构建会报
> `failed to compute cache key: "/agent3": not found`。

**镜像体积**：agent3 原本 11.3 GB，其中 **nvidia CUDA 运行时占 3.2 GB**——
而本项目是 CPU 推理，这 28% 是纯浪费（拉取慢、占磁盘、
容器冷启动还要解压几 GB 的 .so）。

现在先装 CPU 版 torch：

```dockerfile
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

**这里必须用 `--extra-index-url` 而不是 `--index-url`。**
`--index-url` 会**完全替换**索引源，于是 torch 的构建依赖 `flit_core`
在 PyTorch 索引里找不到，直接报：

```
ERROR: Could not find a version that satisfies the requirement flit_core<4,>=3.11
```

**安装顺序也很关键**：先装 CPU 版 torch，再装 `requirements-agent3.txt`。
反过来先装 `ultralytics`，它会把 CUDA 版 torch 一起拖进来。

### 7.2 安全设计

- **非 root 运行，且 uid 与宿主用户对齐**：四个服务都以 `appuser` 启动，
  uid 通过 `ARG HOST_UID/HOST_GID` 传入（默认 1000）。
  **对齐是必须的**：compose 把 `agent1/output/` 从宿主机挂载进来时，
  挂载点属主是宿主用户的 uid，若容器内是别的 uid，写入一律
  `PermissionError`。
  这个坑的隐蔽之处：**单独 `docker run`（不挂 output/）一切正常，
  镜像「验证通过」，一上 compose 就崩** —— 很容易误判成镜像问题。：四个服务都以 `appuser`（uid 10000+）启动。
  容器被突破时攻击者拿到的不是宿主root。
- **`.env` 被 `.dockerignore` 排除**：即使当前 Dockerfile 没有宽泛的 `COPY . .`，
  留着这条规则是为了给未来加宽泛 COPY 的人留护栏——
  镜像层一旦写入密钥，即使后续删除也仍留在历史里，能被 push、被 pull。

---

## 8. 排查手册

### 容器起不来 / 反复重启

```bash
docker compose ps                # 谁没起来、什么状态
docker compose logs --tail=100 agent1
```

按日志里的关键字直接定位：

| 日志关键字 | 原因 | 处理 |
|-----------|------|------|
| `ModuleNotFoundError` | 镜像缺依赖 | 检查 `requirements*.txt` 与 Dockerfile 的 `COPY` |
| `FileNotFoundError` + 向量库 | 未挂载或未构建索引 | `python agent1/build_index.py --rebuild` |
| `FileNotFoundError` + templates | 模板目录未挂载 | 确认 `agent1/templates/` 存在且非空 |
| `401` / `403` | `.env` 里的 token 不一致 | 四个服务必须用同一个 `AGENT_SHARED_TOKEN` |
| `Connection refused` → LLM | `.env` 里 API Key 没填或网络不通 | 检查 Key 与代理配置 |

### 服务 healthy 但检索报错

这是最容易误判的一种：**进程活着 ≠ 依赖就绪**。

```bash
curl http://localhost:8001/ready
```

`/ready` 会直接告诉你哪一项 degraded——是索引缺失、模板缺失，还是输出目录只读。

### 端口被占用

```bash
docker compose ps                # 看是不是自己起的
lsof -i :8001                    # macOS/Linux
netstat -ano | findstr :8001     # Windows
```

### 彻底清理

```bash
./up.sh down            # 停容器，数据卷保留
docker compose --profile milvus --profile obs down --volumes   # 连数据卷一起删
```

> `--volumes` 会**永久删除** Milvus 向量库和 Langfuse 里的历史 trace，不可恢复。
> 除非明确要重置，否则不要加这个参数。

---

## 9. 常用命令

```bash
./up.sh                      # 启动
./up.sh milvus               # 启动 + Milvus
./up.sh all                  # 全量（含 Milvus + 可观测栈）
./up.sh status               # 查看状态
./up.sh logs agent1          # 跟踪某服务日志
./up.sh preflight            # 构建前体检（不启动）
./up.sh verify               # 服务起后体检
./up.sh down                 # 停止

./verify-business.sh         # 带鉴权调真实业务接口（检索 + 生成）
```

`verify-business.sh` 存在的理由：`/health` 返回 200 只能证明进程活着，
证明不了挂载的向量库能用、embedding 配置正确。这个脚本会真的发一条检索请求，
把返回的法条打出来。

---

## 10. 可观测栈

```bash
./up.sh all                  # 或 docker compose --profile obs up -d
```

| 组件 | 地址 | 作用 |
|------|------|------|
| Prometheus | http://localhost:9090 | 指标存储与查询 |
| Grafana | http://localhost:3001/d/fire-agents-overview | 看板 |

### 验证

```bash
./deploy/tools/verify_monitoring.sh
```

逐层验证：应用 `/metrics` → Prometheus targets → 查询 →
Grafana 数据源与看板。

> **为什么要逐层查而不只看容器状态。**
> Prometheus 与 Grafana 都能"容器 running 但没数据"：
> 抓取目标写错、路径不对、数据源没配、看板查询语法有误 ——
> 这些都不会让容器退出，只会让图上什么都没有。
>
> 特别提醒：`/metrics` 返回 200 **不等于有指标**。
> 本项目踩过这个坑 —— `prometheus-client` 漏在 requirements 之外，
> 而 `common/metrics.py` 做了 ImportError 降级，所以服务正常启动、
> `/metrics` 也返回 200，只是响应体里一句
> `# prometheus_client not installed`。
> **优雅降级会把依赖缺失藏起来。**
> 该靠 `preflight.sh` 在构建期抓，而不是等 Prometheus targets 全是 up 却查不到数据。

### 指标清单

| 指标 | 看什么 |
|------|--------|
| `fire_http_requests_total` | 请求量与错误率 |
| `fire_http_request_duration_seconds` | 端到端延迟（P50/P95/P99） |
| `fire_rag_retrieve_duration_seconds` | 检索延迟 |
| `fire_rag_retrieved_chunks` | 单次检索返回几个片段 |
| `fire_llm_calls_total` / `fire_llm_tokens_total` | LLM 调用与 token 消耗 |
| `fire_circuit_breaker_state` | 熔断器状态（0/1/2） |
| `fire_budget_exhausted_total` | 预算耗尽次数 |

### 高基数防护

指标标签不能是用户输入。本项目的路径标签会做归一化：

```
/api/doc/change-reports/CR-20261016-48aeda  →  /api/doc/change-reports/{id}
/tasks/12345                                →  /tasks/{id}
/api/x/8f14e45f-ceea-167a-5a36-dedd4bea2543 →  /api/x/{uuid}
```

> **这是 Prometheus 最容易踩、后果又最严重的坑。**
> 每个 `review_id` 都会生成一条新的时间序列，
> 一千次请求就是一千条 series，内存和时间都被拖垮，
> 而且**没有任何报错**，只表现为"查询变慢"。
> Prometheus 官方建议单条指标的 series 数控制在 1000 以内。

---


---

## 11. 这套部署是怎么验证的

交付前实际跑过，不是"看着对"：

```bash
# 1. 四个镜像全部构建通过
docker compose build

# 2. 逐个启动并确认 healthy
docker compose up -d
docker compose ps        # 四个服务均 (healthy)

# 3. 打真实端点
curl localhost:8001/health    # {"service":"Agent1","status":"ok"}
curl localhost:8001/ready     # 三项检查全 ok

# 4. 带鉴权调业务接口，确认检索与生成链路真的通
./verify-business.sh
```

第4 步的实际输出（节选）：

```json
// /api/doc/templates
{"status":"success","templates":["告知笔录.docx","处罚决定书.docx", ...],"count":7}

// /api/rag/retrieve
{"status":"success","count":3,"snippets":[
  {"content":"第十一条 对消防安全重点单位履行法定消防安全职责情况的监督抽查...",
   "source":"消防监督检查规定（公安部第120号）..."}]}

// /api/rag/ask
{"status":"success","answer":"根据《消防监督检查规定》：**第十一条** 对消防安全重点单位..."}
```

> **为什么必须做到第 4 步。**
> 修复前的 agent1 镜像 `docker build` 是**全绿的**，但 `docker run` 立刻
> `ModuleNotFoundError: No module named 'pdfplumber'`。
> 只跑到"构建通过"就宣布完成，等于把一个起不来的服务当成交付物。

---

## 13. 已知限制

诚实列出来，免得你在生产才发现：

1. **Milvus 是单机standalone**，没有高可用、没有备份。生产需要自行接入集群方案。
2. **Langfuse / Grafana 的默认口令仍在配置里**
   （`changeme` / `admin`），这是本地演示配置，上线前必须替换：
   ```bash
   GRAFANA_PASSWORD=<强密码>
   # Langfuse 的 NEXTAUTH_SECRET / SALT 也要改
   ```
3. **异步任务队列基于 Redis 内存**，进程重启会丢任务，没有持久化。
4. **agent2/agent3 的模型权重不在镜像里**，首次调用会下载，需要预留时间与磁盘。
   权重没到位时 `/ready` 会报 `degraded`——这是**正确行为**，
   因为"能返回 200"不等于"能干活"。
5. **未配置资源 limits**：CPU/内存无上限，一个异常请求可能拖垮整台机器。
   生产应加 `deploy.resources.limits`。
6. **Prometheus 只保留 15 天数据**，超出后看板查不到更早的历史。
   按需调整 `--storage.tsdb.retention.time`。
7. **没有告警规则**：现在只有看板，没有"错误率超阈值就通知"。
   生产要接 Alertmanager，否则看板只能等人去看——**而没人会一直看**。
8. **服务间调用没有超时兜底的全局配置**：
   熔断器保护了下游 LLM，但内部服务互调（gateway → agent1）
   缺统一的超时预算，长尾请求会累积占满连接池。