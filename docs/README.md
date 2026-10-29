# 文档索引

> 15 份文档，按用途分组。面试前重点看加粗的三份。

---

## 面试相关（最需要）

| 文档 | 行数 | 什么时候看 |
|------|------|-----------|
| **`docs/pitch-script.md`** | 253 | **口播稿**（90 秒 / 5 分钟 / 10 分钟三个版本） |
| **`docs/case-stories.md`** | 319 | **五个案例故事** + 数字速查 + 追问应答 |
| **`docs/resume-project.md`** | 153 | **简历描述**（一句 / 150 字 / 300 字） |
| `docs/interview-prep.md` | 223 | 预判问答 Q1-Q10 |

**四份的关系**：

```
resume-project   → 简历上写什么
pitch-script     → 面试时嘴上报什么
case-stories     → 背后的完整细节 + 追问准备
interview-prep   → 对方问时的应答
```

---

## 技术方案

| 文档 | 内容 |
|------|------|
| `docs/project-overview.md` | 架构总览、四服务职责、数据流 |
| `docs/rerank-ab-log.md` | **精排三轮 A/B 完整实录**（含失败与取舍） |
| `docs/eval-trust.md` | 评测可信度机制（指纹/空跑/噪声底） |
| `docs/embedding-selection.md` | embedding 模型选型对比 |
| `docs/change-detector.md` | 人工改动风险分级 |

## 部署与运维

| 文档 | 内容 |
|------|------|
| `docs/deployment.md` | compose 部署、排查手册、监控栈 |
| `docs/mcp-server.md` | MCP Server（3 工具 + 2 资源 + 1 提示） |
| `docs/demo-guide.md` | 演示页说明 + **可溯源率 20% 的实测** |

## FDE 方向（如果投 FDE / 交付岗）

| 文档 | 内容 |
|------|------|
| `docs/tco-analysis.md` | 成本测算：**人力占 73.7%，LLM 只占 0.2%** |
| `docs/discovery-template.md` | Discovery 调研模板 + 消防场景示例 |

---

## 现场演示

```bash
bash agent1/demo/run_demo.sh        # → http://127.0.0.1:8800
```

**演示页能展示**：检索各阶段耗时、命中片段与得分、弃答判定、
引用溯源三态。全部是服务端实跑，无 mock。

**⚠️ 别主动要求演示** —— 除非面试官说"给我看看"。
主动演示自己的失败需要底气，而且会打断对话节奏。

---

## 数字核对

面试材料里的数字由脚本校验，改完记得跑：

```bash
python deploy/tools/verify_interview_numbers.py   # 三份面试材料
python deploy/tools/verify_story_numbers.py       # case-stories 速查表
```

> ★ 这两个脚本的意义不是"检查格式"，
> 而是**确保文档里没有凭印象填的数字**。
>
> 我自己写速查表时就把指标填成了早期基线
> （`rel 0.88 / prec 0.165`，实际是 `0.698 / 0.226`）。
> 面试时说错数字比自己不记得更糟 —— 对方会顺着追问，
> 而和一个报告对不上的数字圆不回来。
>
> 精确的脚本数刻意写成 `50+` 量级：
> 每加一个工具数字就变一次，写死它本身是自找麻烦。

---

## 项目当前数据

```
语料      103 份法规文档 / 4019 个语义分块
代码      94 文件 / 19,643 行（业务）
验证      275 个单测/ 50+ 个验证脚本（6,500+ 行）
服务      4 个Python 服务 + Redis + Prometheus + Grafana，34 个 API 端点
镜像      agent1 1.79GB / agent2 5.02GB / agent3 3.84GB / gateway 1.33GB
检索质量  faithfulness 0.982 / answer_relevancy 0.698
          context_precision 0.226（瓶颈） / context_recall 0.548
噪声      LLM judge 抖动 0.021 → 噪声底设 0.10
溯源      10 个问题，可溯源率 20%（真实发现，非 bug）
```

---

## 待补

- [ ] **扩评测样本** —— 47 条里跨文档类 21 条，
      单类标准误约 0.10，比噪声底还大一倍。
      继续调参测的是噪声不是信号。
- [ ] 演示页加 top_k 3/5/8 对比（讲 v8 的 precision 误读）
- [ ] 三轮 A/B 的recall 变化做成可视化面板
