# MCP Server · 改动检测工具

把 `agent1/change_detector.py` 的人工改动检测能力封装成 MCP 工具，
让任意 MCP 客户端（Claude Desktop / Cursor / 自建 Agent）都能直接调用，
而不必先起本项目的服务、走 HTTP、发鉴权头。

---

## 一、装依赖

```bash
pip install "mcp>=2.0,<3"
```

> ★ **版本坑：MCP SDK 2.x 把 `FastMCP` 重命名成了 `MCPServer`。**
> 照着网上 v1 的教程写会得到：
>
> ```
> ModuleNotFoundError: No module named 'mcp.server.fastmcp'
> This is mcp 2.x, where FastMCP was renamed to MCPServer
> ```
>
> 所以本项目用：
> ```python
> from mcp.server.mcpserver import MCPServer
> ```
> 要强行兼容 v1 就pin `mcp<2`，但新写的代码应该跟进新版本。

---

## 二、配置客户端

### Claude Desktop（Windows）

配置文件位置：

```
%APPDATA%\Claude\claude_desktop_config.json
```

把 `claude_desktop_config.json` 的内容并进去：

```json
{
  "mcpServers": {
    "fire-change-detector": {
      "command": "wsl",
      "args": [
        "-d", "Ubuntu-24.04",
        "-u", "geng",
        "--",
        "/home/geng/.venv/bin/python",
        "/home/geng/fire_agent_v2/fire_agents/mcp/change_detector_server.py"
      ],
      "env": { "MCP_TRANSPORT": "stdio" }
    }
  }
}
```

> 为什么要绕 `wsl`：
> 本项目跑在 WSL 里，而 Claude Desktop 是 Windows 程序，看不到
> WSL 的文件系统与 venv。**即使配了 `\\wsl$\...` 路径，Python 解释器
> 仍然找不到**，所以必须通过 `wsl.exe` 在 WSL 侧启动。
>
> 注意：`command` 用 `wsl`（不是完整路径）时需要 Git Bash/WSL 已加入 PATH；
> 若报「找不到 wsl」，改成完整路径 `"C:\\Windows\\System32\\wsl.exe"`。

### Claude Code（如果用）

```bash
claude mcp add fire-change-detector \
  -- /home/geng/.venv/bin/python /home/geng/fire_agent_v2/fire_agents/mcp/change_detector_server.py
```

---

## 三、暴露的能力

### 工具

| 工具 | 作用 | 关键参数 |
|------|------|---------|
| `compare_documents` | 对比机器稿与人工稿，产出改动分析报告 | `original_path`、`modified_path`、`review_id`、`use_llm`、`save_report` |
| `list_reports` | 列历史报告，按时间倒序 | `review_id`、`limit` |
| `get_report` | 取单份报告完整明细 | `report_id` |

### 资源

| URI | 内容 |
|-----|------|
| `reports://index` | 历史报告索引（JSON） |
| `reports://{report_id}` | 单份报告详情（JSON） |

### 提示

| 名称 | 作用 |
|------|------|
| `review_changes` | 生成审查文书改动的提示词；给了 `report_id` 就把报告内容嵌进去 |

---

## 四、典型用法

```
用户：帮我看看这份处罚决定书人工改了什么，风险在哪
      /home/geng/fire_agent_v2/fire_agents/agent1/output/pending/xxx.docx
      /home/geng/fire_agent_v2/fire_agents/agent1/output/pending/xxx_revised.docx

Claude 调用 compare_documents →拿到报告 → 再用 review_changes 做深度分析
```

`use_llm=false` 时只跑规则分析：**快、零 API 成本、无网络依赖**，
适合批量跑或做回归测试。`use_llm=true` 会额外做语义归因
（区分"抽取错误 / 事实修正 / 格式偏好 / 越权改动"），更准但要调 API。

---

## 五、验证

```bash
# 1. 造一对测试文书（含罚款金额、法律条款等高风险改动）
python deploy/tools/make_test_docs.py /tmp/mcp_test

# 2. 检查工具注册与错误处理
python deploy/tools/check_mcp_server.py

# 3. 端到端跑一遍（真实调用 + 读回落盘报告）
python deploy/tools/e2e_mcp_test.py /tmp/mcp_test
```

`e2e_mcp_test.py` 实测输出：

```
找到金额改动：…罚款人民币 5000 元整。 → …罚款人民币 500 元整。
风险等级：high  依据：["金额变更：['5000 元'] → ['500 元']"]
✓ 金额改动被判为高风险

list_reports → status=success count=5
get_report  → 改动数=9 高危=4 建议数=2
```

---

## 六、踩过的坑（都留在了代码注释里）

### 1. 报告目录不一致 —— 「调用成功」但功能不可用

最初 `change_detector.py` 把报告写到 `output/change_reports/`，
而 MCP 工具去 `output/reviews/` 找。结果：

- `compare_documents` 返回 `success`，还给了 `json_path`
- `list_reports` 永远返回 0
- `get_report` 永远报「找不到报告」

**调用成功 ≠ 功能可用。** 只测"函数能调通"完全发现不了这类问题 ——
必须真的走一遍「生成 → 落盘 → 读回」的闭环。

修法：`REVIEWS_DIR = Path(cd.REPORT_DIR)`，直接复用被调模块的常量。
复制一份路径字符串会再次漂移，这正是原 bug 的成因。

### 2. stdout 被协议占用

stdio transport 下，stdout 是JSON-RPC 帧的通道。任何 `print()`
都会污染响应，客户端报「解析响应失败」但完全看不出原因。
所以启动日志一律走 `stderr`。

### 3. 路径安全策略要放开一半

`MCPServer` 默认 `ResourceSecurity(reject_absolute_paths=True)`，
而本工具必须读用户的绝对路径（文书可能在 `/home/...` 也可能在 `D:/...`），
全开会挡住正常用法。正确做法是**只放开这一条**：

```python
ResourceSecurity(
    reject_path_traversal=True,   # 保留 —— 真实安全边界
    reject_absolute_paths=False,  # 放开 —— 业务需要
    reject_null_bytes=True,       # 保留 —— 真实安全边界
)
```

`reject_path_traversal` 和 `reject_null_bytes` 一起关掉等于让工具
能被用来读任意文件，不能因为"省事"就一起开。

---

## 七、局限

1. **文档格式**：目前支持 `extract_text` 能处理的类型（docx/pdf/txt）。
   扫描件 PDF 需要先 OCR，本工具不做。
2. **`use_llm=true` 依赖外部 API**：断网或 Key 失效时会失败。
   生产部署建议加降级：LLM 调用失败时回落到规则分析，而不是整体报错。
3. **无鉴权**：MCP 协议层不做身份认证。stdio transport 下由本机权限兜底，
   但若改用 `sse` / `streamable-http` 对外暴露，**必须自行加鉴权**。
4. **改动单元切分较粗**：目前按行/段落切，
   "5000 改成 500"和"整段重写"会被算成不同粒度的改动，
   风险评级的粒度受此限制。