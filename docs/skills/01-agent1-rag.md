# Agent1 · 消防法规检索与文书生成 Skill

**角色**：消防法规 RAG 问答、法律字段抽取、执法文书生成（草稿→审批→终稿）。
**代码位置**：`agent1/server.py`
**端口**：8001

---

## 1. 通过什么形式调用

### 何时该用 agent1

| 场景 | 端点 |
|------|------|
| 「消防通道被堵塞怎么处罚？」 | `POST /api/rag/ask` |
| 「找出所有涉及电动自行车的规定」 | `POST /api/rag/retrieve` |
| 从案情描述抽取法律适用字段 | `POST /api/rag/legal_extract` |
| 从上传文档抽取案情要素 | `POST /api/case/extract` |
| 生成执法文书草稿 | `POST /api/doc/generate` |
| 审批 / 驳回文书 | `POST /api/doc/review/{id}/approve` / `reject` |
| 查看待审文书列表 | `GET /api/doc/reviews` |
| 查看文书改动报告 | `GET /api/doc/change-reports/{id}` |

### 网关路由关键词

命中 `rag` 或 `doc` 意图时会路由到 agent1：

- `rag`：所有法规咨询类问题（兜底）
- `doc`：生成文书、生成一份、写一份、通知书、审批表、立案表、笔录、决定书、告知书

---

## 2. Agent1 怎样识别调用

### 权限过滤（最重要的机制）

检索层按 **JWT claims** 过滤可见范围：

```
dept in [用户部门, "通用"]
AND secret_level <= 用户密级
AND (valid_until == 0 OR valid_until > 当前时间)
```

| JWT claim | 默认值 | 含义 |
|-----------|--------|------|
| `dept` | `消防支队` | 只能看到本部门 + 通用文档 |
| `clearance` | `2` | 密级上限：0=公开 1=内部 2=秘密 3=机密 4=绝密 |
| `role` | `officer` | 角色（当前主要用于元数据标注） |

**越界处理**：`clearance` 超出 0-4 会被收敛到合法区间，
防止伪造超权 clearance 绕过文档级权限过滤。

★★★ 这套过滤是「空转」的，除非语料里有对应数据
--------------------------------------------------
实测索引元数据分布（4019 块）：

```
secret_level: {0: 3953, 1: 66}   ← 等级 > 2 的：0 条
valid_until:  全部为 0（永不过期）
dept:         {通用: 3971, 消防支队: 48}
```

也就是说**当前语料里没有任何涉密文档，也没有一条法规带失效日期**，
过滤逻辑虽正确但从未被真实数据触发。

这带来两个后果：
1. 已废止的法规照样能被检索出来 → 模型可能引用作废条款 → **执法风险**
2. 涉密文档接入时，过滤规则第一次在真实数据上运行 → 可能暴露漏洞

**所以接真实数据前，必须先补齐元数据流程**（见 `kb_meta.yaml`）。

### 检索质量现状

`context_precision` 是当前瓶颈。分层数据（47 条评测集）：

| 类别 | n | 应答 precision | 应答 recall | 判断 |
|------|---|---------------|------------|------|
| 直查法条 | 20 | 0.343 | 0.79 | 召回到了但**掺了噪声** |
| 跨文档综合 | 21 | 0.294 | 0.53 | 同上 |
| 陷阱题 | 6 | 0.200 | 0.00 | **该查的没查到** |

总分 0.226 里，约 30% 的样本是弃答（precision 恒为 0，无从判断相关性）。
只看应答样本，precision = 0.309 —— 这才是检索质量的真实水平。

**调用方须知**：`answer` 里可能出现**弃答文案**（「未检索到相关规定」）。
这不是错误，是系统在「查不到」时的诚实表达 —— 在执法场景里，
明确说「不知道」远好过编一个条款号。

---

## 3. 输入格式

### `POST /api/rag/ask`

```json
{
  "question": "消防通道被堵塞应该怎么处罚？",
  "top_k": 8
}
```

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `question` | string | 是 | — | 法规咨询问题 |
| `top_k` | int | 否 | 8 | 检索条数，上限 10 |

### `POST /api/rag/retrieve`

```json
{
  "query": "电动自行车",
  "top_k": 8
}
```

只返回检索片段，不生成答案。**做二次加工时用这个**。

### `POST /api/rag/legal_extract`

```json
{
  "case_text": "案情描述文本……",
  "legal_fields": [
    {"key": "法律依据", "label": "法律依据"},
    {"key": "处罚种类", "label": "处罚种类"}
  ],
  "top_k": 8
}
```

`legal_fields` 是要抽取的字段清单（`key` + `label`）。

### `POST /api/doc/generate`

```json
{
  "case": { "案情字段": "值" },
  "templates": ["立案通知书"]
}
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `case` | 否 | 案情字典，缺省用 `DEFAULT_CASE` |
| `templates` | 否 | 模板名列表（不带 `.docx`），空 = 全部模板 |

**产出是草稿**，写入 `output/pending/`，返回 `review_id`。
必须经人工 `approve` 才移入 `output/official/` 成为终稿。

### `POST /api/case/extract`

multipart：`file`（案情文档）+ 可选 `query`。

---

## 4. 输出格式契约

### `POST /api/rag/ask`

```json
{
  "status": "success",
  "answer": "根据《中华人民共和国消防法》第六十条……"
}
```

**`answer` 可能包含弃答文案**（检索不到时的诚实表达）。

### `POST /api/rag/retrieve`

```json
{
  "status": "success",
  "count": 3,
  "snippets": [
    {
      "content": "条文原文……",
      "source": "中华人民共和国消防法.pdf",
      "score": 0.95,
      "metadata": {
        "doc_id": "…", "law": "中华人民共和国消防法",
        "article": "第六十条", "dept": "通用",
        "role": "officer", "secret_level": 0, "valid_until": 0,
        "chunk_id": "…"
      }
    }
  ]
}
```

`metadata.source` + `metadata.article` 可直接拼成
`《法规名》第X条` 形式的引用 —— 这正是引用溯源检测的依据。

### `POST /api/doc/generate`

```json
{
  "status": "success",
  "review_id": "rev_a1b2c3d4e5",
  "generated": [
    {"template": "立案通知书", "path": "output/pending/…", "filled": 12, "missing": 3}
  ],
  "errors": []
}
```

| 字段 | 说明 |
|------|------|
| `review_id` | 审批单 ID，approve/reject/change-reports 都要用 |
| `filled` / `missing` | 填充成功 / 缺失的字段数（**缺失字段要人工补**） |
| `errors` | 失败的模板及原因 |

### 审批流

```
generate → output/pending/ + review_id
    ↓ approve → output/official/（终稿）
    ↓ reject  → 记录驳回原因
```

`GET /api/doc/change-reports/{review_id}` 返回人工修改前后的 diff ——
**审批时靠它判断人工改了什么**。

---

## 5. 运维端点

| 端点 | 用途 |
|------|------|
| `GET /api/ops/model/probe` | 检测本地模型环境（Ollama 是否连通、装了哪些模型） |
| `GET /api/ops/model/check` | 实测检索精度并给出建议 |
| `POST /api/ops/model/download` | 让 Agent 自动下载模型（需本机有 ollama CLI） |
| `GET /api/ops/model/guidance` | 手动下载指引（在线 / 离线包两种路径） |
| `GET /api/ops/llm/breakers` | LLM 层熔断器状态快照 |

### `GET /api/ops/model/check` 的返回

```json
{
  "environment": {
    "provider_mode": "cloud",
    "ollama_connected": true,
    "installed_models": ["qwen2.5:7b"],
    "target_model": "qwen2.5:7b",
    "has_target_model": true,
    "needs_model": false
  },
  "retrieval": {
    "ok": true,
    "precision": 0.31,
    "recall": 0.61,
    "n": 20,
    "method": "lexical_overlap"
  },
  "threshold": 0.6,
  "below_threshold": true,
  "advice": [
    {
      "level": "warn",
      "code": "low_retrieval_precision",
      "title": "检索精度偏低（0.310 < 0.6）",
      "detail": "检索层是当前瓶颈。注意：**换更大的生成模型解决不了这个问题**……",
      "actions": [{"kind": "doc", "label": "看检索调优记录", "ref": "docs/rag-tuning-journey.md"}]
    }
  ]
}
```

**关键提示**：`advice` 里 precision 偏低时明确写了
「换更大的 LLM 无济于事」—— 因为检索不准时，模型再强也只是在错的材料上做推理。

---

## 6. 调用方检查清单

- [ ] 带了 `X-Internal-Token`？
- [ ] 文书生成后走了审批流（草稿 ≠ 终稿）？
- [ ] 检查了 `filled` / `missing`，缺失字段人工补了？
- [ ] 区分了「弃答文案」和「正常答案」？
- [ ] 引用溯源时用了 `metadata.source` + `metadata.article`？
- [ ] 知道当前 context_precision 只有 0.309（应答口径）？
