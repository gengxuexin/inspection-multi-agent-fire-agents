# Gateway · 统一入口 Skill

**角色**：所有请求的唯一入口。负责意图路由、熔断降级、审计留痕。

**代码位置**：`gateway.py`（根目录）
**端口**：8000

---

## 1. 通过什么形式调用

### 主入口：`POST /dispatch`

这是**首选入口**。调用方不需要自己判断该用哪个 Agent —— 交给网关路由。

```bash
curl -X POST http://gateway:8000/dispatch \
  -H "Authorization: Bearer <jwt>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "消防通道被堵塞应该怎么处罚？",
    "top_k": 8
  }'
```

**请求字段**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 用户问题或指令 |
| `file_path` | string | 否 | 文件路径（图片/音频/文档） |
| `top_k` | int | 否 | 检索条数，默认 8 |
| `force_kind` | string | 否 | 强制路由：`rag`/`vision`/`voice`/`doc`，跳过意图识别 |
| `async_mode` | bool | 否 | 见下 |

### 异步模式：`POST /dispatch_async`

长任务（文书生成、视频分析）用这个，立即返回 `task_id`：

```bash
curl -X POST http://gateway:8000/dispatch_async \
  -H "Authorization: Bearer <jwt>" \
  -H "Content-Type: application/json" \
  -d '{"query": "生成一份立案通知书", "file_path": "/data/case.docx"}'
```

然后轮询 `GET /tasks/{task_id}`。

### 文件上传：`POST /dispatch_file`

multipart 表单，适合浏览器直传：

```bash
curl -X POST http://gateway:8000/dispatch_file \
  -H "Authorization: Bearer <jwt>" \
  -F "file=@photo.jpg" \
  -F "query=这张图有什么隐患"
```

---

## 2. 网关怎样识别调用

### 三级级联路由

```
L1 规则（关键词）  → 命中且 confidence ≥ 0.6 → 直接路由，零 LLM 开销
L2 分类器（ML）    → confidence ≥ 0.6 → 路由
L3 LLM 分类        → 兜底，prompt 明确给出四类定义
都不中 → kind="human"，转人工
```

**L1 关键词表**（`common/router.py`）：

| 意图 | 触发词 |
|------|--------|
| `vision` | 图片、照片、图像、视频、画面、看图、检查图、隐患、消防设施、灭火器、消火栓、通道、堵塞、消防水带、消防柜、安全出口 |
| `voice` | 语音、录音、音频、听一下、转写、识别这段 |
| `doc` | 生成文书、生成一份、生成个、写一份、写个、通知书、审批表、立案表、笔录、决定书、告知书 |
| `rag` | 兜底（其余全部归此类） |

**L3 的 LLM 分类 prompt 里四类的定义**（这是路由语义最权威的表述）：

- `rag` = 法规咨询 / 知识问答 / 处罚标准咨询
- `vision` = 图片或视频隐患识别
- `voice` = 语音交互（转写、问答）
- `doc` = 文书生成

### 路由信封

每次 dispatch 都返回 `route` 字段，说明**为什么这么路由**：

```json
{
  "status": "success",
  "routed_to": "rag",
  "route": {
    "kind": "rag",
    "confidence": 0.9,
    "source": "rule",
    "hops": 1,
    "fallback_path": "direct_llm",
    "detail": "L1 规则命中（rag）"
  },
  "result": { }
}
```

| 字段 | 含义 |
|------|------|
| `source` | `rule`（L1 命中）/ `classifier`（L2）/ `llm`（L3）/ `escalate`（转人工） |
| `hops` | 级联了几层才定下来 |
| `fallback_path` | 该意图的降级路径，见下表 |
| `detail` | 人可读的判定说明 |

**降级路径**（`FALLBACK_MAP`）：

| 意图 | 降级路径 | 含义 |
|------|----------|------|
| `vision` | `ocr_only` | YOLO 不可用，只做 OCR |
| `voice` | `transcribe_only` | 问答不可用，只做转写 |
| `doc` | `rag` | 文书生成不可用，退回法规检索 |
| `rag` | `direct_llm` | 检索不可用，直接问 LLM |
| `human` | `manual_review` | 转人工 |

---

## 3. 输出格式契约

### 成功

```json
{
  "status": "success",
  "routed_to": "rag",
  "route": { },
  "result": { }
}
```

`result` 的结构由被路由到的 Agent 决定（见各自文档）。

### 熔断降级

```json
{
  "status": "degraded",
  "routed_to": "vision",
  "route": { },
  "result": {
    "note": "视觉服务暂不可用，已降级为 OCR 识别"
  }
}
```

**调用方必须把 degraded 提示给用户** —— 结果虽然能用，但完整性下降。

### 熔断完全打开（无可用降级）

```json
{
  "status": "error",
  "message": "Circuit breaker [vision] is open"
}
```

### 干跑模式

`DRY_RUN=1` 时只做路由判断不执行：

```json
{
  "status": "dry_run",
  "route": { },
  "routed_to": "rag"
}
```

**用途**：调路由规则时用它验证，不产生实际调用与费用。

---

## 4. 辅助端点

| 端点 | 方法 | 用途 |
|------|------|------|
| `/` | GET | 服务信息 + 各 Agent 地址 |
| `/agents/health` | GET | 三个 Agent 的存活状态 |
| `/auth/token` | POST | 签发 JWT（**仅联调/测试**，生产应走 OIDC IdP） |

### `POST /auth/token`

```json
{
  "sub": "officer_001",
  "dept": "消防支队",
  "role": "officer",
  "clearance": 2
}
```

返回 `{"access_token": "...", "token_type": "Bearer"}`。

**注意**：这个端点不校验任何凭据就能签发 token —— 它是为了联调方便，
生产环境必须关闭或加保护。

---

## 5. 熔断降级机制

网关层用 `common/resilience.py` 的 `protected_call`（异步版）：

- 三态熔断器：`closed → open → half_open`
- 失败阈值 5 次转 OPEN，30 秒后试探恢复
- 指数退避重试（base_delay 0.5s，最多 8s）

| 熔断器名 | 保护的调用 |
|----------|-----------|
| `rag` | agent1 法规检索 |
| `voice` | agent2 语音 |
| `vision` | agent3 视觉 |
| `doc` | agent1 文书生成 |

查询当前状态：`GET /agents/health` 查各 Agent 存活；
**LLM 层**的熔断器状态在 agent1 的 `GET /api/ops/llm/breakers`（见 agent1 文档）。

---

## 6. 调用方检查清单

- [ ] 带了 `Authorization: Bearer <jwt>`？
- [ ] JWT 的 `clearance` 符合用户实际权限？（越界会被收敛，但会拿到错误的可见范围）
- [ ] 分清 `success` / `degraded` / `error`？
- [ ] `degraded` 时提示了用户？
- [ ] 长任务用了 `dispatch_async` 而不是 `dispatch`？
- [ ] 「查无结果」和「系统故障」区分开了？
