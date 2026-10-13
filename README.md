# 消防智能执法系统

> 多模态感知 · 法律知识增强 · 智能分析研判 · 文书自动生成 · 辅助执法决策

## 一、系统架构

```
用户入口（飞书 / Dify / API）
    ↓ Authorization: Bearer <JWT>
Gateway :8000（统一调度 + 鉴权 + 审计 + 级联路由 + 熔断降级）
    ├─ /dispatch            → 文本意图路由
    ├─ /dispatch_file       → 单文件分析
    ├─ /dispatch_multimodal → 多文件综合
    ├─ /dispatch_async      → 异步任务（长任务）
    ├─ /tasks/{id}          → 任务状态轮询
    ├─ /tasks/{id}/events   → SSE 事件流
    └─ /auth/token          → 签发 JWT
    ↓ X-Internal-Token
┌──────────────┬──────────────┬──────────────┐
│ Agent1 :8001 │ Agent2 :8002 │ Agent3 :8003 │
│ RAG+文书     │ 语音+TTS     │ 视觉+VLM     │
│ 草稿审批流   │              │              │
└──────────────┴──────────────┴──────────────┘
    ↓                              ↓
  common/（共享：config/auth/audit/resilience/budget/tasks/router/tracing）
    ↓
  orchestrator_langgraph.py（StateGraph 编排 + checkpoint）
    ↓
  Redis（任务状态）  Milvus/FAISS（向量检索 + 权限过滤）
```

## 二、目录结构

```
fire_agents/
├── .env                    # 全局环境变量
├── start_all.sh            # 一键启动（--with-redis --with-milvus）
├── requirements.txt        # 核心依赖
├── requirements-eval.txt   # RAGAS 评估依赖
├── gateway.py              # 调度网关（鉴权+审计+路由+熔断+异步）
├── multimodal_orchestrator.py  # 多模态编排（legacy）
├── orchestrator_langgraph.py  # LangGraph 编排（生产版）
├── server.py               # [DEPRECATED] 旧入口，308 反代到 gateway
├── common/                # 共享包
│   ├── config.py           # pydantic-settings 统一配置
│   ├── auth.py             # JWT 鉴权 + 内部 Token
│   ├── audit.py            # 审计日志 + 脱敏
│   ├── resilience.py       # 熔断器 + 保护调用
│   ├── budget.py           # Token/成本预算
│   ├── tasks.py            # Redis 任务状态管理
│   ├── router.py           # 级联路由（规则→分类器→LLM）
│   └── tracing.py          # OpenTelemetry + Langfuse
├── agent1/                 # RAG + 文书生成
│   ├── server.py           # FastAPI 服务（鉴权+审批流）
│   ├── retrieval_backend.py # 统一检索（Milvus expr 过滤 + FAISS 回退）
│   ├── build_index.py      # 向量库构建（--backend milvus|faiss）
│   ├── kb_meta.yaml        # 知识库文档元数据（dept/role/secret_level）
│   ├── rag_module.py / hybrid.py / rag_legal_advisor.py
│   ├── case_field_matcher.py / field_resolver.py
│   ├── knowledge_base/full # 法规知识库
│   ├── faiss_full/         # FAISS 向量索引
│   ├── templates/          # 文书模板
│   ├── output/pending      # 草稿（待审批）
│   ├── output/official     # 正式文书（审批后）
│   ├── output/reviews      # 审批记录
│   └── eval/ragas_eval.py  # RAGAS 评估
├── agent2/                 # 语音服务
├── agent3/                 # 视觉服务
├── data/intent_train.jsonl # 意图分类器训练数据
├── train_classifier.py     # 训练脚本
├── models/                 # 训练产物（intent_clf.joblib）
├── deploy/docker/          # Docker 部署
│   ├── *.Dockerfile
│   ├── docker-compose.yml
│   └── build.sh
└── logs/                   # 日志（audit.log + checkpoints.sqlite）
```

## 三、快速开始

### 1. 配置环境变量

编辑 `.env`：

```env
# LLM
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# 鉴权
JWT_SECRET=你的随机密钥
AGENT_SHARED_TOKEN=内部调用Token
AUTH_DISABLED=0          # 1=本地开发跳过鉴权

# 基础设施
REDIS_URL=redis://127.0.0.1:6379/0
MILVUS_URI=              # 空则回退 FAISS
MILVUS_COLLECTION=fire_laws

# 编排
ORCHESTRATOR_MODE=legacy  # legacy | langgraph
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 构建向量库

```bash
cd agent1
python build_index.py --backend faiss --mode full    # FAISS
python build_index.py --backend milvus --mode full   # Milvus（需先配 MILVUS_URI）
```

### 4. 训练意图分类器（可选）

```bash
python train_classifier.py
```

### 5. 一键启动

```bash
./start_all.sh                      # 基础启动
./start_all.sh --with-redis         # 预检 Redis
./start_all.sh --with-milvus        # 预检 Redis + Milvus
./start_all.sh status               # 查看状态
./start_all.sh stop                 # 停止
```

## 四、鉴权使用

### 签发 Token

```bash
# 通过 gateway 的 /auth/token 接口
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"sub":"张三","dept":"消防支队","role":"officer","clearance":2}'
# → {"token":"eyJ...", "expires_in": 28800}
```

或用 Python：

```python
from common.auth import create_token
token = create_token(sub="张三", dept="消防支队", role="officer", clearance=2)
```

### 携带 Token

所有 `/dispatch*` 接口需要 `Authorization: Bearer <token>`：

```bash
curl http://localhost:8000/dispatch \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{"query":"消防法第六十条","conversation_id":"c1"}'
```

### clearance 等级

| 值 | 含义 | 可检索文档 |
|---|---|---|
| 0 | 公开 | secret_level=0 |
| 1 | 内部 | secret_level<=1 |
| 2 | 秘密 | secret_level<=2 |
| 3 | 机密 | secret_level<=3 |
| 4 | 绝密 | 全部 |

## 五、异步任务

长任务（视频分析、多模态综合）走异步接口：

```bash
# 提交异步任务
TASK_ID=$(curl -s -X POST http://localhost:8000/dispatch_async \
  -H "Authorization: Bearer eyJ..." \
  -F "query=综合分析" \
  -F "files=@photo.jpg" \
  -F "files=@audio.wav" | jq -r .task_id)

# 轮询状态
curl http://localhost:8000/tasks/$TASK_ID \
  -H "Authorization: Bearer eyJ..."

# SSE 事件流
curl -N http://localhost:8000/tasks/$TASK_ID/events \
  -H "Authorization: Bearer eyJ..."
```

## 六、文书审批流

```
generate → output/pending/ (draft) → approve → output/official/ (approved)
                                  → reject   → reviews/ (rejected)
```

```bash
# 生成草稿（返回 review_id）
curl -X POST http://localhost:8001/api/doc/generate_from_dict \
  -H "X-Internal-Token: xxx" \
  -F "template_file=@template.docx" \
  -F 'case_dict_json={"case_context":"...","case_id":"auto_123"}'
# → {"status":"success","review_id":"rev_20260101_abc123"}

# 查看待审批列表
curl http://localhost:8001/api/doc/reviews -H "X-Internal-Token: xxx"

# 审批通过（移入 official/）
curl -X POST http://localhost:8001/api/doc/review/rev_20260101_abc123/approve \
  -H "X-Internal-Token: xxx"

# 驳回
curl -X POST http://localhost:8001/api/doc/review/rev_20260101_abc123/reject \
  -H "X-Internal-Token: xxx" -d '{"reason":"事实描述需补充"}'
```

## 七、Docker 部署

```bash
cd deploy/docker

# 基础启动
docker compose up -d redis gateway agent1 agent2 agent3

# 含 Milvus
docker compose --profile milvus up -d

# 含可观测（OTEL + Langfuse）
docker compose --profile obs up -d

# 全量
docker compose --profile milvus --profile obs up -d

# 构建推送私有仓
REGISTRY=registry.local/fire TAG=v2 ./build.sh
```

## 八、核心接口

| 接口 | 方法 | 用途 |
|---|---|---|
| `:8000/auth/token` | POST | 签发 JWT |
| `:8000/dispatch` | POST | 文本路由 |
| `:8000/dispatch_file` | POST | 单文件分析 |
| `:8000/dispatch_multimodal` | POST | 多文件综合 |
| `:8000/dispatch_async` | POST | 异步任务 |
| `:8000/tasks/{id}` | GET | 任务状态 |
| `:8000/tasks/{id}/events` | GET | SSE 事件流 |
| `:8001/api/rag/ask` | POST | 法规问答 |
| `:8001/api/doc/generate` | POST | 文书草稿 |
| `:8001/api/doc/reviews` | GET | 待审批列表 |
| `:8001/api/doc/review/{id}/approve` | POST | 审批通过 |
| `:8002/api/voice/transcribe` | POST | 语音转写 |
| `:8003/api/vision/analyze_image` | POST | 图片分析 |

## 九、RAGAS 评估

```bash
cd agent1
pip install -r ../../requirements-eval.txt
python eval/ragas_eval.py --top-k 8 --output eval/report.json
```

## 十、关键技术

- **鉴权审计**：JWT（HS256/OIDC-ready）+ 结构化审计日志 + 脱敏
- **级联路由**：规则（L1）→ sklearn 分类器（L2）→ DeepSeek LLM（L3），超 hop 升级人工
- **韧性**：三态熔断器 + 指数退避重试 + 降级回退
- **LangGraph 编排**：StateGraph 状态机 + 条件边 + SqliteSaver checkpoint
- **向量检索**：Milvus（expr 权限过滤）+ FAISS/BM25 混合回退
- **预算控制**：Token/成本预算追踪，超限干净终止
- **可观测**：OpenTelemetry Tracing + Langfuse LLM 调用日志
- **RAG 检索**：BM25 + 向量混合检索 + 查询解析精准定位
- **文书生成**：模板驱动 + LLM 字段抽取 + 三级字段分流 + 草稿审批流

## 十一、开发记录

- v1.0 基础三 Agent 搭建
- v1.1 RAG 混合检索优化
- v1.2 VLM 融合判断
- v1.3 多模态综合编排
- v1.4 LLM 换 DeepSeek
- v2.0 生产化改造：鉴权+审计+熔断+预算+级联路由+LangGraph+Milvus+Docker
