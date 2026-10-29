# MCP Server · 文书改动检测 Skill

**角色**：把「人工修改与 AI 原文的差异检测」暴露为 MCP 工具，
让任意 MCP 客户端（Claude Desktop / Cursor / 自建 Agent）直接调用，
**不必起本项目服务、走 HTTP、发鉴权头**。

**代码位置**：`mcp/change_detector_server.py`
**安装与配置**：见 [`docs/mcp-server.md`](../mcp-server.md)（含 SDK 2.x 改名坑）

> 本文档只讲**调用契约**。安装、配置、客户端接入方式看上面那份。

---

## 1. 提供的工具（Tools）

### `compare_documents` — 对比两份文书

**用途**：检测「人工修改后的文书」与「AI 生成的原稿」之间的差异。
执法场景里，审批人常要确认人工改了哪些字段、改动是否影响法律适用。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `original_path` | string | 是 | AI 生成的原稿路径 |
| `modified_path` | string | 是 | 人工修改后的文书路径 |
| `report_dir` | string | 否 | 报告输出目录（默认 agent1 的 change-reports 目录） |

**返回**：报告 ID + 摘要。**完整 diff 要再调 `get_report` 拿。**

### `list_reports` — 列出已有报告

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `limit` | int | 否 | 最多返回几条 |

**用途**：查历史改动报告（审批追溯、复盘）。

### `get_report` — 取单份报告详情

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `report_id` | string | 是 | 报告 ID |

---

## 2. 提供的资源（Resources）

| URI | 名称 | 用途 |
|-----|------|------|
| `reports://index` | 报告索引 | 所有报告的清单 |
| `reports://{report_id}` | 单份报告 | 指定报告的完整内容 |

**用途**：MCP 客户端可以把资源挂进上下文，让模型直接「看到」改动详情，
而不必逐个调工具。

---

## 3. 提供的提示词（Prompts）

### `review_changes`

**用途**：生成「审查文书改动」的提示词模板。

**用途场景**：审批人在 IDE / 客户端里问「帮我看看这份文书改了什么」，
客户端拉起这个 prompt，把改动报告填进去，模型给出审查意见。

---

## 4. 调用方怎么识别

MCP 客户端通过**工具名**识别，不需要关键词匹配 —— 这是 MCP 与
本项目内部 HTTP 路由（L1 关键词表）的根本区别：

| | 内部 HTTP 路由 | MCP |
|---|---|---|
| 识别方式 | 关键词 + 分类器 + LLM 三级级联 | **工具名精确匹配** |
| 确定性 | 关键词命中可能误判 | 确定性调用 |
| 参数 | JSON body | 工具 schema 声明 |

所以「通过什么关键词调用」这个问题在 MCP 语境下**不成立** ——
调用方直接点名 `compare_documents`。

---

## 5. 输出格式契约

MCP 工具返回的是 **JSON 文本**（content 里的 text 字段）。

### `compare_documents`

```json
{
  "report_id": "chg_20261029_a1b2c3",
  "summary": {
    "total_changes": 5,
    "by_type": {"字段值修改": 3, "新增段落": 1, "删除段落": 1},
    "critical": ["法律依据"]
  },
  "message": "已生成改动报告 chg_20261029_a1b2c3"
}
```

### `get_report`

```json
{
  "report_id": "chg_20261029_a1b2c3",
  "changes": [
    {
      "type": "字段值修改",
      "location": "处罚决定",
      "original": "罚款壹万元整",
      "modified": "罚款贰万元整",
      "impact": "high"
    }
  ]
}
```

| 字段 | 说明 |
|------|------|
| `type` | 改动类型：字段值修改 / 新增段落 / 删除段落 / 格式变更 |
| `location` | 位置（字段名或段落锚点） |
| `original` / `modified` | 改前 / 改后内容 |
| `impact` | 影响程度：`high` / `medium` / `low` |

---

## 6. 与 HTTP 链路的关系

文书改动检测有**两条路径**，用途不同：

| 路径 | 入口 | 适合 |
|------|------|------|
| MCP | `mcp/change_detector_server.py` | IDE 内的交互式审查 |
| HTTP | `GET /api/doc/change-reports/{review_id}` | 审批流集成 |

**两者调用的是同一个 `agent1/change_detector.py` 核心逻辑**，
所以结果一致。区别只在调用方式与鉴权（MCP 靠客户端配置，HTTP 靠 Token）。

---

## 7. 调用方检查清单

- [ ] MCP SDK 版本 ≥ 2.0？（1.x 的 import 路径不同）
- [ ] 路径参数是**客户端与服务端共享的文件系统**上的路径？
  （本地 MCP Server 可以；远程 Server 不行）
- [ ] 改动报告的 `impact: high` 项人工确认过了？
