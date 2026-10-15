# Embedding 模型选型说明

> 背景：本地 Ollama 向量化 3841 个 chunk 耗时约 3 小时（实测 3.04s/chunk），
> 无法接受迭代，需改为云端 API。

## 一、结论

**采用阿里云百炼 `text-embedding-v4`（1024 维）**，密钥写入 `.env` 的 `EMBEDDING_API_KEY`。

## 二、需求约束

| 约束 | 原因 |
|---|---|
| **维度必须 1024** | 既有 FAISS 索引（`bge-large-zh-v1.5`）是 1024 维，换维度必须重建全部索引 |
| **中文语义检索强** | 知识库是消防法律法规，问句是自然语言（"公众聚集场所投入使用营业前检查实行什么制度"），不是关键词检索 |
| **批量接口** | 3841 chunk 要能并发批量提交，逐条调用会成为新瓶颈 |
| **OpenAI 兼容** | 现有代码用 `OpenAI(base_url=...)` 封装，换 API 不该改架构 |

## 三、候选对比

| 模型 | 维度 | 中文效果 | 速度 | 成本 | 兼容性 |
|---|---|---|---|---|---|
| **阿里云 `text-embedding-v4`** | 1024 ✅ | 强（中文专项优化） | 快 | 前 1000 万 token 免费 | ✅ 完全兼容 |
| 阿里云 `text-embedding-v3` | 1024 ✅ | 略逊v4 | 快 | 免费额度 | ✅ |
| 智谱 `embedding-3` | 1024 ✅ | 强 | 快 | 有免费额度 | ✅ |
| OpenAI `text-embedding-3-large` | 3072 ❌ | 中文偏弱 | 快 | 贵 | 维度要重建 |
| OpenAI `text-embedding-3-small` | 1536 ❌ | 中文一般 | 很快 | 便宜 | 维度要重建 |
| 硅基流动 `BAAI/bge-m3` | 1024✅ | 强 | 中 | 免费 | ✅ |
| 本地 Ollama `bge-large-zh` | 1024 ✅ | 强 | ❌ 3h | 免费 | ✅ |

### 选 `text-embedding-v4` 的四个理由

1. **维度恰好 1024** —— 这是硬约束。`text-embedding-3-large` 是 3072维，
   中文检索效果虽好，但必须重建索引且存储占用翻 3 倍，不划算。
2. **中文语义检索优化过** —— 阿里云在中文语料上做过针对性训练，
   而 OpenAI 系列主要面向英文训练，中文法律文本的长尾表述（"告知承诺"、
   "包容审慎监管"这类）召回表现不稳定。
3. **成本可忽略** —— 3841 chunk × 约 400 字 ≈ 150 万字，
   在免费额度内，**一次性重建索引零成本**，后续增量才产生费用。
4. **已有账号体系** —— 项目里Agent3 已在用 `DASHSCOPE_API_KEY`，
   可直接复用同一把密钥，不引入新的供应商和新的账单。

### 为什么保留本地 Ollama

`EMBEDDING_PROVIDER=ollama` 一行配置即可切回本地。保留它是因为：
内网/涉密环境下可能不允许调公网 API，此时需要能立刻降级。

## 四、需要你做的操作

### 1. 申请密钥

打开**阿里云百炼控制台** → API-KEY 管理 → 新建密钥 → 复制 `sk-` 开头的串。

### 2. 写入位置

编辑项目根目录的 `.env`，找到这一行：

```bash
EMBEDDING_API_KEY=
```

把密钥填在等号后面（**不要加引号**）：

```bash
EMBEDDING_API_KEY=sk-你的密钥粘在这里
```

> `.env` 已在 `.gitignore` 中，密钥不会被提交到仓库。
> 若你不想单独管理这把 key，也可以**留空** —— 程序会自动复用
> 已有的 `DASHSCOPE_API_KEY`。

### 3. 验证配置生效

```bash
python3 -c "from common.config import embedding_settings; \
c=embedding_settings(); print(c['model'], c['dim'], c['provider'])"
```

期望输出：`text-embedding-v4 1024 dashscope`

### 4. 重建索引（必须）

换了embedding 模型，**旧的 FAISS 向量全部作废**：

```bash
python3 agent1/build_index.py --mode full --rebuild
```

程序会先做**连通性 + 维度预检**，几秒内就能发现 key 填错或维度不匹配，
不会再出现"跑了3 小时才失败"的情况。

预计 3841 chunk 约需 **3～6 分钟**（云端批量，10 条/请求）。

### 5. 重跑评估

```bash
cd agent1
python3 eval/local_eval.py --top-k 5 --output eval/after_api_emb.json
```

> 注意用 `local_eval.py`（自研 LLM-as-judge），不要用 `ragas_eval.py` ——
> 后者需要 `pip install ragas`，你当前环境没装，跑起来会报 `ragas not installed`。

## 五、配置项速查

全部在项目根 `.env`：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `EMBEDDING_PROVIDER` | `dashscope` | `dashscope`（云端）\| `ollama`（本地） |
| `EMBEDDING_API_KEY` | 空 | ★ **你需要填这个**。留空则复用 `DASHSCOPE_API_KEY` |
| `EMBEDDING_MODEL` | `text-embedding-v4` | 模型名 |
| `EMBEDDING_BASE_URL` | 阿里云 compatible-mode | OpenAI 兼容端点 |
| `EMBEDDING_DIM` | `1024` | ⚠️ 必须与模型实际维度一致 |
| `EMBEDDING_BATCH_SIZE` | `10` | 单请求文本数（DashScope 上限 10） |
| `EMBEDDING_TIMEOUT` | `120` | 单请求超时（秒） |
| `OLLAMA_BASE_URL` | 空 | 本地回退用，自动探测 |
| `OLLAMA_EMBEDDING_MODEL` | `quentinz/bge-large-zh-v1.5` | 本地回退模型 |

## 六、常见问题

**Q：能不能不重建索引，直接换模型？**
不能。向量维度变了，新模型产生的向量与旧索引不在同一空间，检索结果完全无意义。
程序会在预检阶段拦截维度不符的情况并报错。

**Q：报 `401 Unauthorized`？**
密钥无效或已过期。回控制台确认，注意不要把 `sk-` 前缀漏掉。

**Q：报连接超时？**
云端 API 需要走系统代理。程序已对云端启用 `trust_env=True`，
确认 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量已设置。

**Q：想省钱，只在必要时用 API？**
保持 `EMBEDDING_PROVIDER=ollama` 即可，代码路径完全一致，随时切换。