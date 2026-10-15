# 文书人工改动检测 Agent

> 生成草稿 → 人工编辑 → 审批。审批原本只记录"通过/驳回"，
> 不记录**人改了什么**。本Agent 自动对比改动前后，产出可归档的分析报告。

## 一、解决什么问题

没有这个环节时，人工审批存在四个盲区：

| 盲区 | 后果 |
|---|---|
| 模板缺陷不闭环 | 人工每次都手改同样的地方，却没人统计过改哪几处，改完也不反哺模板 |
| LLM 抽取错误无法定位 | 分不清是"模型抽错"还是"人理解不同"，两种问题解法完全相反 |
| 风险改动不可见 | 人工把罚款从"五万元"改成"五十万元"，系统无感知 |
| 幻觉无法追溯 | 终稿出现知识库里没有的条款，不能自动报警 |

## 二、怎么用

### 方式一：随审批流自动跑（推荐）

已接入 `POST /api/doc/review/{review_id}/approve`：

1. 生成草稿时自动在 `output/pending/.origin/` 存一份原稿快照
2. 人工改完点approve 时，自动逐份对比快照与改后稿
3. 报告落在 `output/change_reports/`，并写回审批记录的 `change_reports` 字段

审批响应会带上 `high_risk_count`，并在 `message` 里提示管理员复核。

查询报告：

```bash
curl http://localhost:8001/api/doc/change-reports/<review_id> \
  -H "X-Internal-Token: <token>"
```

### 方式二：命令行单独跑

```bash
cd agent1

# 规则分析（快，不消耗 token）
python3 change_detector.py --original pending/a.docx --modified pending/b.docx \
  --review-id RV-001 --no-llm

# 带 LLM 语义归因（慢，消耗 token，但能说清"人为什么改"）
python3 change_detector.py --original pending/a.docx --modified pending/b.docx \
  --review-id RV-001
```

## 三、报告存储位置

```
agent1/output/change_reports/
├── RV-20261014-001_CR-20261014-a1b2c3.json    ← 机器可读，供汇总统计
└── RV-20261014-001_CR-20261014-a1b2c3.md      ← 人可读，给管理员看
```

以审批单号开头命名，管理员按 `RV-*` 就能把所有相关报告串起来。
该目录已在 `.gitignore` 中（`agent1/output/`），不会污染仓库。

## 四、报告结构

Markdown 报告包含五部分：

1. **结论摘要** —— 改动总数、高风险数量、是否需要人工确认
2. **高风险改动表** —— 只列触及关键要素的改动，直接看"改前→改后"
3. **全部改动明细** —— 每处的原文、新文、风险等级、LLM 归因
4. **根因分析** —— LLM 判断是模型问题、模板问题还是人工个性化需求
5. **模板改进建议** —— 哪些该从模板层面修，哪些是一次性修正

## 五、风险分级规则

**高风险（必须人工确认）** —— 触及关键法律要素：

- 金额变更：`五千元` → `五十万元`
- 条款变更：`第六十条` → `第六十一条`
- 法律名称变更：`《消防法》` → `《安全生产法》`
- 处置方式变更：`警告` → `拘留`

**中风险** —— 纯新增内容含上述要素，或整段重写（相似度 < 30%）。
不与越权改罚同等对待。

**低风险** —— 措辞、排版、事实补充等常规调整。

## 六、LLM 归因分类

`analyze_changes(use_llm=True)` 时，每处高/中风险改动会做语义归因：

| 类别 | 含义 | 建议动作 |
|---|---|---|
| 抽取错误 | 模型从案情抽错字段，人在纠正 | 修抽取 prompt / schema |
| 事实修正 | 人工补充案件事实 | 扩充案情字段 |
| 格式偏好 | 仅排版措辞 | 修模板 |
| 越权改动 | 改了金额/条款/法律适用 | **管理员介入** |

为控制耗时，只有高/中风险改动才调 LLM，纯格式类不花 token。
审批接口默认 `use_llm=False`（同步接口不能被 LLM 拖慢）。

## 七、开发中修掉的真实缺陷

1. **改动重复计数** —— 同一段内容既报 `replace` 又报 `insert`。
   根因：`_pair_similar` 用 `remaining.index(best)` 记录下标，
   而比对时用的是原列表下标，两者错位。改用 `enumerate` 记录真实下标。
2. **渲染时字段名不一致** —— `_render_markdown` 读 `c['original']`，
   但 `asdict(ChangeItem)` 的字段是 `original_text`，直接 KeyError。
3. **文件名注入** —— `review_id` 默认值 `"N/A"` 被当路径分隔符导致写入失败；
   且 `review_id` 来自 HTTP 路径参数，存在路径穿越风险。
   新增 `safe_filename()` 统一过滤 `<>:"/\|?*`、控制字符与 `..`。
4. **插入类改动风险误判** —— 纯新增时 `[] → ['罚款']` 被判高风险。
   区分"要素被修改"与"新增内容带要素"，后者降为中风险。
5. **报告章节跳号** —— 无 LLM 分析时从"三"直接跳到"五"。改为动态编号。

## 八、测试

```bash
python3 -m pytest tests/test_change_detector.py -q
```

39个用例，覆盖 diff 三种类型、去重、配对阈值、七类风险分级、
报告落盘与文件名安全（含 `../../etc/passwd` 等路径穿越输入）。

全量：`python3 -m pytest tests/ -q` → 162 passed。

## 九、扩展方向

- **接入 audit.py** —— 把高风险改动计入审计事件，纳入 `common/tracing`
- **统计报表** —— 遍历 `change_reports/*.json` 聚合，
  找出被反复修改的模板位置，反哺模板迭代
- **实时拦截** —— 在 `/api/doc/review/approve` 里加参数，
  高风险时要求二次确认才允许归档