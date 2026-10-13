# Agent3 - 消防语音处理器

> 消防视频/音频的完整语音处理流水线：
> **音视频剥离 → 语音识别 → 语义梳理 → 关键词提取 → 语音合成 → 语音问答**

---

## 一、功能概览

| 章节 | 功能 | 实现文件 |
|------|------|---------|
| 2.1 | 音视频剥离（ffmpeg） | `utils/audio_extract.py` |
| 2.2.1.1 | 讯飞在线识别 | `asr/xfyun_online.py` |
| 2.2.1.2 | 讯飞录音文件转写 | `asr/xfyun_lfasr.py` |
| 2.1.2 | Vosk 离线识别 | `asr/vosk_engine.py` |
| 2.1.3 | Whisper 离线识别（CUDA 加速） | `asr/whisper_engine.py` |
| 2.3 | 语义梳理（本地 Ollama） | `llm/ollama_client.py` |
| 2.4.1.1 ~ 2.4.1.4 | TTS 音色 / 克隆 / 合成 / 情感 | `tts/aliyun_cosyvoice.py`、`tts/edge_engine.py` |
| 2.5 | 语音问答机器人（Gradio） | `pipelines/voice_qa_bot.py` |

---

## 二、环境要求

- **Python**：3.10 或 3.11
- **ffmpeg**：需要预先安装并加入 PATH
- **Ollama**：本地运行 `qwen2.5:3b` 或 `qwen2.5:7b`
- **可选**：NVIDIA GPU（≥ 4GB 显存，用于 Whisper CUDA 加速）

---

## 三、安装

```bash
conda create -n media_env python=3.11 -y
conda activate media_env

# 核心依赖
pip install requests openai pyyaml gradio

# 离线 ASR
pip install faster-whisper vosk

# TTS
pip install edge-tts pyttsx3 dashscope

# 音频处理
pip install pydub
```

若使用 Whisper GPU 加速，额外安装 CUDA 版 PyTorch：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 四、环境变量配置

所有密钥通过环境变量注入，**不在代码里硬编码**。

### 4.1 Windows PowerShell（永久生效）

```powershell
# 讯飞（在线听写 + 录音文件转写）
[System.Environment]::SetEnvironmentVariable("XFYUN_APP_ID", "你的app_id", "User")
[System.Environment]::SetEnvironmentVariable("XFYUN_API_KEY", "你的api_key", "User")
[System.Environment]::SetEnvironmentVariable("XFYUN_API_SECRET", "你的api_secret", "User")
[System.Environment]::SetEnvironmentVariable("LFASR_APP_ID", "你的app_id", "User")
[System.Environment]::SetEnvironmentVariable("LFASR_SECRET_KEY", "你的secret", "User")

# 阿里云（CosyVoice TTS）
[System.Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "sk-xxx", "User")
[System.Environment]::SetEnvironmentVariable("DASHSCOPE_WORKSPACE_ID", "你的WorkspaceId", "User")

# 可选配置
[System.Environment]::SetEnvironmentVariable("OLLAMA_MODEL", "qwen2.5:3b", "User")
[System.Environment]::SetEnvironmentVariable("WHISPER_MODEL_SIZE", "medium", "User")
[System.Environment]::SetEnvironmentVariable("VOSK_MODEL_PATH", "D:\Docker\vosk\vosk-model-cn-0.22", "User")
```

**设置后关闭终端重新打开。**

### 4.2 Linux / macOS

```bash
export XFYUN_APP_ID="你的app_id"
export XFYUN_API_KEY="..."
export XFYUN_API_SECRET="..."
export LFASR_APP_ID="你的app_id"
export LFASR_SECRET_KEY="..."
export DASHSCOPE_API_KEY="sk-..."
export DASHSCOPE_WORKSPACE_ID="..."
```

建议写入 `~/.bashrc` 或 `~/.zshrc`。

### 4.3 验证环境变量

```powershell
python main.py check_env
```

---

## 五、目录结构

```text
agent3_语音机器人/
├── config.py                   # 全局配置（从环境变量读密钥）
├── main.py                     # 统一命令行入口
├── README.md
├── input/                      # 输入视频
├── output/                     # 输出结果
│   ├── whisper_llm/            # Whisper 识别结果
│   ├── xfyun_lfasr/            # 讯飞识别结果
│   ├── vosk/                   # Vosk 识别结果
│   ├── tts_demo/               # TTS 试听音频
│   └── voice_bot/              # 问答机器人语音
├── models/
│   └── whisper/                # Whisper 模型缓存
├── pipelines/
│   ├── video_to_text.py        # 视频→文本主流程
│   └── voice_qa_bot.py         # Gradio 问答界面
├── asr/                        # 语音识别引擎
│   ├── whisper_engine.py
│   ├── vosk_engine.py
│   ├── xfyun_lfasr.py
│   └── xfyun_online.py
├── llm/
│   └── ollama_client.py        # 本地大模型调用
├── tts/                        # 语音合成
│   ├── edge_engine.py
│   ├── aliyun_cosyvoice.py
│   └── pyttsx3_engine.py
└── utils/
    ├── audio_extract.py        # 音视频剥离
    └── fire_keywords.py        # 消防关键词提取
```

---

## 六、使用

### 6.1 检查环境

```bash
python main.py check_env
```

**预期输出：**

```text
✅ Whisper 设备：cuda (float16)     # 有 GPU
# 或
✅ Whisper 设备：cpu (int8)         # 无 GPU
```

### 6.2 视频 → 文本（Whisper）

```bash
python main.py video2text "input\1.mp4"
```

**输出（`output/whisper_llm/`）：**

| 文件 | 说明 |
|------|------|
| `1.wav` | 提取的音频 |
| `1_识别文本.txt` | 原始识别 |
| `1_梳理文本.txt` | 大模型润色后 |
| `1_消防关键信息.json` | 关键词提取 |

### 6.3 换用讯飞文件转写

```bash
python main.py video2text "input\1.mp4" xfyun_lfasr
```

### 6.4 换用 Vosk 离线

```bash
python main.py video2text "input\1.mp4" vosk
```

### 6.5 批量处理

```bash
python main.py batch whisper
```

### 6.6 启动语音问答机器人

```bash
# 先启动 Ollama
ollama serve

# 另开终端启动机器人
python main.py voicebot
```

浏览器打开 `http://127.0.0.1:7860`，即可用语音提问。

### 6.7 生成 TTS 试听样例

```bash
python main.py tts_demo
```

输出到 `output/tts_demo/`。

### 6.8 列出所有中文音色

```bash
python main.py list_voices
```

---

## 七、性能说明

| 引擎 | 环境 | 10 分钟音频耗时 |
|------|------|---------------|
| Whisper medium + CPU (int8) | i7-8650U | 约 8~12 分钟 |
| Whisper medium + GPU (float16) | RTX 3060 | 约 1~2 分钟 |
| 讯飞 LFASR 在线 | 无本地要求 | 约 30~60 秒（网络依赖） |
| Vosk | CPU | 约 2~4 分钟 |

**优先推荐：**

- **有 GPU** → Whisper（精度高 + 速度快）
- **无 GPU、允许联网** → 讯飞 LFASR
- **完全离线、无 GPU** → Vosk（精度略低）

---

## 八、常见问题

### Q1: 提示"未设置 DASHSCOPE_API_KEY"

重新打开终端（环境变量只在**新终端**生效），或验证：

```powershell
echo $env:DASHSCOPE_API_KEY
```

### Q2: Whisper 检测不到 GPU

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

若为 `False`，说明 PyTorch 是 CPU 版，需重装 CUDA 版：

```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Q3: edge-tts 报 asyncio 错误

已在 `tts/edge_engine.py` 修复（拆分为 `synthesize_async` 和 `synthesize`）。

### Q4: 讯飞在线识别返回错误码

常见错误码：

| 错误码 | 含义 | 解决 |
|--------|------|------|
| `10165` | 音频格式不对 | 需 16kHz / 16bit / 单声道 PCM |
| `10161` | 音频时长超限 | 缩短音频或分段 |
| `11200` | 授权失败 | 检查 Key 是否过期 |

### Q5: Vosk 模型路径错误

下载中文模型并解压到指定路径：

```bash
# 下载地址
https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip

# 解压后设置环境变量
[System.Environment]::SetEnvironmentVariable("VOSK_MODEL_PATH", "D:\path\to\vosk-model-cn-0.22", "User")
```

### Q6: ffmpeg 找不到

确认 ffmpeg 已安装并加入 PATH：

```bash
ffmpeg -version
```

如未安装，从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载，解压后把 `bin` 目录加入系统 PATH。

---