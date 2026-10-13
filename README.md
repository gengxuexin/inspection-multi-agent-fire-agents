# 消防智能执法多模态 Agent 系统

> 面向消防执法场景的多模态 AI 系统：法规 RAG 问答 · 案件材料分析 · 法律文书自动生成
>
> 网关统一鉴权审计 · 三级级联路由 · 三态熔断降级 · LangGraph 状态图编排 · 人机协同审批流

![CI](https://github.com/your-name/fire-agents/actions/workflows/ci.yml/badge.svg)

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 目录

- [核心能力](#核心能力)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [接口一览](#接口一览)
- [工程设计](#工程设计)
- [目录结构](#目录结构)
- [开发与测试](#开发与测试)
- [部署](#部署)
- [安全设计](#安全设计)

---

## 核心能力

| 能力            | 说明                                     |
| ------------- | -------------------------------------- |
| **法规 RAG 问答** | 消防法规知识库混合检索（BM25 + 向量），支持文档级权限过滤       |
| **多模态案件分析**   | 图片 / 视频 / 音频 / 文档自动分类路由，分模态并行处理后综合研判   |
| **文书自动生成**    | 模板驱动 + LLM 字段抽取，三级字段分流（事实 / 法律 / 待补充）  |
| **人机协同审批**    | 文书先入待审区，人工审批通过后才转正式目录                  |
| **视觉隐患识别**    | YOLO 检测 + 百度 OCR + 多模态大模型综合判断，输出分级告警   |
| **语音交互**      | 讯飞在线听写 / 录音转写 / Whisper 离线识别 + 多引擎 TTS |

---

## 系统架构

```
用户入口（飞书 / Dify / HTTP API）
    │  Authorization: Bearer <JWT>
    ▼
Gateway :8000 ── 统一鉴权 · 审计 · 级联路由 · 熔断 · 异步任务
    │  X-Internal-Token
    ├──────────────┬──────────────┬──────────────┐
    ▼              ▼              ▼
Agent1 :8001    Agent2 :8002    Agent3 :8003
RAG + 文书      语音 ASR/TTS    视觉 VLM
草稿审批流
    │              │              │
    └──────────────┴──────────────┘
                   ▼
         common/  共享能力层
    config · auth · audit · resilience
    budget · tasks · router · tracing
                   ▼
         orchestrator_langgraph.py
         LangGraph 状态图 + SqliteSaver checkpoint
                   ▼
    Redis（任务状态）  Milvus / FAISS（向量检索 + 权限过滤）
```

### LangGraph 状态图

```
START → classify ─┬─→ vision     ─┐
                  ├─→ voice      ─┼→ synthesize → extract_fields
                  ├─→ doc_extract ─┘                   │
                  └─（无文件）─────────────────────────┘
                                                       ↓
                             report ← review_submit ← doc_draft
```

- **条件边**：`classify` 按文件类型 fan-out；`synthesize` / `doc_draft` 失败时降级直达 `report`
- **Checkpoint**：`SqliteSaver` 按 `thread_id = conversation_id` 持久化，支持会话级断点续传
- **预算守卫**：每个节点入口校验 `BudgetTracker`，超限干净终止

---

## 快速开始

### 1. 配置环境

```bash
cp .env.example .env
```

编辑 `.env` 填入密钥。**最少需要 `DEEPSEEK_API_KEY`**，其余按需启用：

```env
DEEPSEEK_API_KEY=sk-xxx          # 必填：LLM 主链路
JWT_SECRET=<至少 32 字节随机串>   # 必填：签发 JWT
DASHSCOPE_API_KEY=sk-xxx         # Agent3 VLM / Agent2 TTS
BAIDU_API_KEY / BAIDU_SECRET_KEY # Agent3 OCR
```

生成安全密钥：

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### 2. 安装依赖

```bash
pip install -r requirements.txt          # 运行时
pip install -r requirements-dev.txt      # 开发（测试 + lint）
```

Agent2 / Agent3 的重型依赖按需安装：

```bash
pip install faster-whisper edge-tts ultralytics opencv-python
```

### 3. 构建向量索引

知识库文档放入 `agent1/knowledge_base/` 后：

```bash
cd agent1
python build_index.py --backend faiss --mode full     # FAISS（本地，无需额外组件）
python build_index.py --backend milvus --mode full    # Milvus（需配置 MILVUS_URI）
```

### 4. 启动

```bash
./start_all.sh                    # 启动全部服务
./start_all.sh --with-milvus      # 预检 Redis + Milvus
./start_all.sh status             # 查看状态
./start_all.sh stop               # 停止
```

### 5. 三分钟走通一个完整链路

```bash
# ① 签发 Token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"sub":"张三","dept":"消防支队","role":"officer","clearance":2}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['token'])")

# ② 法规问答
curl -s http://localhost:8000/dispatch \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"消防法第六十条的处罚标准","conversation_id":"demo-1"}'

# ③ 法规 RAG（直连 Agent1）
curl -s http://localhost:8001/api/rag/ask \
  -H "X-Internal-Token: $AGENT_SHARED_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question":"公众聚集场所营业前需要哪些消防安全检查？"}'

# ④ 查看待审批文书
curl -s http://localhost:8001/api/doc/reviews -H "X-Internal-Token: $AGENT_SHARED_TOKEN"
```

---

## 接口一览

| 接口                                  | 方法   | 说明         |
| ----------------------------------- | ---- | ---------- |
| `:8000/`                            | GET  | 健康检查       |
| `:8000/agents/health`               | GET  | Agent 健康状态 |
| `:8000/auth/token`                  | POST | 签发 JWT     |
| `:8000/dispatch`                    | POST | 文本意图路由     |
| `:8000/dispatch_file`               | POST | 单文件分析      |
| `:8000/dispatch_multimodal`         | POST | 多文件综合分析    |
| `:8000/dispatch_async`              | POST | 提交异步任务     |
| `:8000/tasks/{id}`                  | GET  | 任务状态轮询     |
| `:8000/tasks/{id}/events`           | GET  | SSE 事件流    |
| `:8001/api/rag/ask`                 | POST | 法规问答       |
| `:8001/api/rag/retrieve`            | POST | 仅检索（不生成回答） |
| `:8001/api/rag/legal_extract`       | POST | 法律条款抽取     |
| `:8001/api/doc/templates`           | GET  | 文书模板列表     |
| `:8001/api/doc/generate`            | POST | 生成文书草稿     |
| `:8001/api/doc/generate_from_dict`  | POST | 依据结构化事实生成  |
| `:8001/api/doc/reviews`             | GET  | 待审批列表      |
| `:8001/api/doc/review/{id}/approve` | POST | 审批通过       |
| `:8001/api/doc/review/{id}/reject`  | POST | 驳回         |
| `:8001/api/case/extract`            | POST | 案件要素抽取     |
| `:8002/api/voice/transcribe`        | POST | 语音转写       |
| `:8002/api/voice/tts`               | POST | 语音合成       |
| `:8002/api/voice/qa`                | POST | 语音问答       |
| `:8003/api/vision/analyze_image`    | POST | 图片分析       |
| `:8003/api/vision/analyze_video`    | POST | 视频分析       |

### 权限等级

JWT 中的 `clearance` 字段映射到文档密级，检索时自动过滤：

| clearance | 含义 | 可检索文档               |
| --------- | -- | ------------------- |
| 0         | 公开 | `secret_level = 0`  |
| 1         | 内部 | `secret_level <= 1` |
| 2         | 秘密 | `secret_level <= 2` |
| 3         | 机密 | `secret_level <= 3` |
| 4         | 绝密 | 全部                  |

### 文书审批流

```
generate → output/pending/  → approve → output/official/
                          → reject  → output/reviews/
```

LLM 生成的文书**不会直接落到正式目录**，必须经人工审批，体现执法场景的人机协同要求。

---

## 工程设计

### 三级级联路由

按成本与延迟递增，逐级升级，避免为简单请求调用大模型：

```
L1  规则匹配        文件后缀 + 关键词打分      置信度 ≥ 0.6   ~0ms
L2  本地分类器      sklearn Pipeline          置信度 ≥ 阈值   ~5ms
L3  LLM 判定        DeepSeek 结构化输出       置信度 ≥ 0.5   ~800ms
     ↓ 全部未命中
   默认走法规 RAG

跳数超限（ROUTE_MAX_HOPS）→ 转人工
```

每一级命中即短路返回，`RouteDecision` 始终携带 `fallback_path` 供下游降级。

### 三态熔断器

```
     失败达阈值
closed ──────────→ open
  ↑                 │ 等待 reset_timeout
  │                 ↓
  └── 成功 ──── half_open
                   │ 试探失败
                   └──→ open（刷新冷却）
```

- 失败计数**在重试全部耗尽后才 +1**，避免单次调用击穿阈值
- 熔断打开时若有 `fallback` 则直接降级，否则抛 `CircuitOpenError`
- 重试采用指数退避：`base_delay * 2^attempt`

### 预算控制

`BudgetTracker` 在每个编排节点入口校验 Token 与成本上限，超限抛 `BudgetExceededError` 干净终止，避免失控循环。

### 双检索后端

| 后端           | 优势                           | 适用         |
| ------------ | ---------------------------- | ---------- |
| Milvus       | 原生 `expr` 表达式过滤，适合大规模 + 权限过滤 | 生产         |
| FAISS + BM25 | 零外部依赖，混合检索                   | 本地开发 / 小规模 |

未配置 `MILVUS_URI` 时自动回退 FAISS，无需改代码。

### 可观测性

- **OpenTelemetry**：Trace 贯穿网关 → 编排 → Agent 调用
- **Langfuse**：LLM 调用链路的输入输出、耗时、Token 成本
- **结构化审计**：`logs/audit.jsonl`，请求摘要经正则脱敏（手机号 / 身份证 / 密钥）

---

## 目录结构

```
fire_agents/
├── .env.example              # 配置模板（复制为 .env 使用）
├── .github/workflows/        # CI：lint / test / docker
├── pyproject.toml            # pytest / ruff / mypy 配置
├── requirements.txt          # 运行时依赖
├── requirements-dev.txt      # 开发依赖
├── start_all.sh              # 一键启停
│
├── gateway.py                # 统一网关：鉴权 / 审计 / 路由 / 熔断 / 异步
├── orchestrator_langgraph.py # LangGraph 状态图编排（生产）
├── multimodal_orchestrator.py# 早期编排实现（legacy 回退，可选）
│
├── common/                   # 共享能力层
│   ├── config.py             # 单一配置源（pydantic-settings，56 字段）
│   ├── auth.py               # JWT 鉴权（HS256 / OIDC RS256）+ 内部 Token
│   ├── audit.py              # 审计日志 + 脱敏
│   ├── resilience.py         # 三态熔断器 + 重试 + 降级
│   ├── budget.py             # Token / 成本预算
│   ├── tasks.py              # 异步任务状态（Redis）
│   ├── router.py             # 三级级联路由
│   └── tracing.py            # OpenTelemetry + Langfuse
│
├── agent1/                   # 法规 RAG + 文书生成
│   ├── server.py             # FastAPI 服务
│   ├── retrieval_backend.py  # 统一检索（Milvus / FAISS 回退）
│   ├── build_index.py        # 向量库构建
│   ├── case_pipeline.py      # 案件材料 → 文书的处理流水线
│   ├── case_field_matcher.py # 字段抽取与校验
│   ├── field_resolver.py     # 三级字段分类
│   ├── kb_meta.yaml          # 文档元数据（dept / role / secret_level）
│   ├── ui/app.py             # Streamlit 交互界面
│   └── eval/ragas_eval.py    # RAG 质量评估
│
├── agent2/                   # 语音服务
│   ├── asr/                  # 讯飞在线 / 讯飞转写 / Whisper / Vosk
│   ├── tts/                  # CosyVoice / Edge / pyttsx3
│   └── pipelines/            # 语音问答 / 视频转写
│
├── agent3/                   # 视觉服务
│   ├── agent/                # 综合判断（VLM 融合）
│   └── modules/              # YOLO / OCR / 抽帧 / 视频切分
│
├── scripts/                  # 诊断与调试工具
├── tests/                    # 单元测试（74 用例）
└── deploy/docker/            # Dockerfile + compose + 可观测栈
```

### 不入库的大文件

以下内容已在 `.gitignore` 排除，需本地生成或另行获取：

| 内容                | 生成方式                                                |
| ----------------- | --------------------------------------------------- |
| 消防法规知识库           | 放入 `agent1/knowledge_base/` 后运行 `build_index.py`    |
| FAISS / Milvus 索引 | `python build_index.py --backend faiss --mode full` |
| Whisper / YOLO 权重 | 由 faster-whisper、ultralytics 自动下载或手动放置              |
| 意图分类器             | `python train_classifier.py`                        |
| 文书输出与日志           | 运行时生成                                               |

---

## 开发与测试

```bash
# 全部测试
pytest tests/ -v

# 覆盖率
pytest tests/ --cov=common --cov-report=term-missing

# 静态检查
ruff check .
ruff format --check .

# 类型检查（共享包与编排层为严格模式）
mypy common gateway.py orchestrator_langgraph.py
```

### 测试覆盖

| 文件                         | 用例数 | 覆盖内容                |
| -------------------------- | --- | ------------------- |
| `tests/test_auth_audit.py` | 19  | JWT 签发校验、越界收敛、审计脱敏  |
| `tests/test_resilience.py` | 20  | 三态熔断流转、重试、降级回退      |
| `tests/test_router.py`     | 23  | 三级级联、非法 kind 过滤、转人工 |
| `tests/test_budget.py`     | 12  | 预算累计、超限终止、合并汇总      |

`tests/test_router.py` 通过 monkeypatch 隔离 sklearn 模型与 LLM 调用，**测试无需模型文件和网络**。

---

## 部署

```bash
cd deploy/docker

# 基础（网关 + 三 Agent + Redis）
docker compose up -d redis gateway agent1 agent2 agent3

# 含 Milvus 向量库
docker compose --profile milvus up -d

# 含可观测栈（OTel Collector + Langfuse）
docker compose --profile obs up -d

# 全量
docker compose --profile milvus --profile obs up -d
```

构建并推送镜像：

```bash
REGISTRY=registry.example.com/fire TAG=v2.0 ./build.sh
```

### 环境要求

| 组件     | 版本                      |
| ------ | ----------------------- |
| Python | 3.10+                   |
| Redis  | 6+（可选，缺省则异步任务接口不可用）     |
| Milvus | 2.4+（可选，缺省回退 FAISS）     |
| 内存     | ≥ 4GB（含本地 Embedding 模型） |

---

## 安全设计

| 面向    | 措施                                                         |
| ----- | ---------------------------------------------------------- |
| 身份认证  | JWT HS256，本地密钥与 OIDC RS256 双模；HS256 强制密钥 ≥ 32 字节（RFC 7518） |
| 权限控制  | `clearance` 四级权限映射文档密级，检索层强制过滤；越界值自动收敛                     |
| 服务间调用 | `X-Internal-Token` + `hmac.compare_digest` 常数时间比较，防时序侧信道   |


| 审计留痕 | 全接口结构化审计，敏感字段正则脱敏后落盘 |
| 密钥管理 | `.env` 不入库，提供 `.env.example` 模板 |
| 成本失控 | Token / 成本双维度预算守卫，超限干净终止 |

---

## 开发记录

| 版本 | 内容 |
|---|---|
| v1.0 | 基础三 Agent 搭建（RAG / 语音 / 视觉） |
| v1.1 | RAG 混合检索优化（BM25 + 向量） |
| v1.2 | VLM 融合判断 |
| v1.3 | 多模态综合编排 |
| v1.4 | LLM 迁移至 DeepSeek |
| v2.0 | 生产化：鉴权审计、熔断降级、预算控制、级联路由、LangGraph 编排、Milvus、Docker |
| v2.1 | 工程化：配置单一来源、print → logging、74 单测、CI 流水线、安全加固 |

---

## License

MIT
### RAG 质量评估

评估脚本：`agent1/eval/local_eval.py`（LLM-as-judge，四维度，对齐 RAGAS 口径）
样本：5 条消防法规典型问答，含标准答案

| 指标 | top_k=8 | top_k=20 | 说明 |
|---|---|---|---|
| Faithfulness | 1.00 | 0.986 | 检索不足时明确拒答，不产生幻觉 |
| Answer Relevancy | 0.60 | 0.88 | 拒答率下降，直接命中率上升 |
| Context Precision | 0.150 | 0.165 | 噪声比例未改善——排序能力是瓶颈 |
| Context Recall | 0.200 | 0.600 | 扩大召回窗口使 2/5 样本翻盘 |

**关键观察**：`top_k` 从 8 提到 20 后，recall 翻三倍但 precision 几乎不动。
这定位了真正的瓶颈：**不是召回窗口太小，是排序不够精准**。
下一步的正确方向是引入 rerank，而非继续加大 top_k。

**已定位的三类失败模式**（未修复）：

1. **知识库缺口**：部分问题（如"公众聚集场所告知承诺"）在早期版本中无相关文档，向量检索退化为邻近召回
2. **长文档条款定位**：法规末尾的具体条款（如电动自行车在《高层规定》第 47 条）容易被文档开头的总则挤出 top-8
3. **语义粒度不匹配**：ground_truth 是"总纲"性表述，而分块后检索到的是"细则表格"

**设计取舍**：执法场景下，系统优先保证 Faithfulness（不生成无据内容），
接受检索不足时的明确拒答。这一策略由 `generate_answer` 的 system prompt 强制约束。