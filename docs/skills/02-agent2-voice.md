# Agent2 · 语音交互 Skill

**角色**：语音转写（ASR）、语音问答（ASR→LLM→TTS）、语音合成（TTS）。
**代码位置**：`agent2/server.py`
**端口**：8002

---

## 1. 通过什么形式调用

| 场景 | 端点 | 方法 |
|------|------|------|
| 上传录音/视频转文字 | `/api/voice/transcribe` | POST |
| 给定本地音频路径做语音问答 | `/api/voice/qa` | POST |
| 文字转语音 | `/api/voice/tts` | POST |

### 网关路由关键词

命中 `voice` 意图时路由到 agent2：

> 语音、录音、音频、听一下、转写、识别这段

---

## 2. Agent2 怎样识别调用

### ASR 引擎选择（`engine` 参数）

| engine | 说明 | 特点 |
|--------|------|------|
| `whisper` | faster-whisper | 精度高，支持 CUDA 加速，**需模型文件** |
| `vosk` | Vosk | **完全离线、无 GPU**，精度略低 |
| 讯飞 | 讯飞在线听写 | 云端，精度最高，需 `XFYUN_*` 配置 |

**降级链**：优先按 `engine` 指定；未指定时按配置顺序尝试；
云端不可达自动回落到本地 Vosk。

### 视频输入处理

`transcribe` 收到视频文件（`.mp4/.avi/.mkv/.mov/.flv/.wmv`）时，
先抽音频再转写。抽音频失败会返回 `status: "error"`（不是 degraded）。

---

## 3. 输入格式

### `POST /api/voice/transcribe`

multipart：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `file` | file | 是 | — | 音频或视频文件 |
| `engine` | string | 否 | `whisper` | `whisper` / `vosk` / 讯飞 |

### `POST /api/voice/qa`

```json
{
  "audio_path": "/data/record.wav",
  "voice": "zh-CN-XiaoxiaoNeural",
  "emotion": "正常"
}
```

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `audio_path` | string | 是 | — | **服务器本地路径**（不是上传） |
| `voice` | string | 否 | `zh-CN-XiaoxiaoNeural` | TTS 音色 |
| `emotion` | string | 否 | `正常` | TTS 情感 |

**注意**：`audio_path` 是服务端可访问的路径。这适合「现场设备已经落盘」
的场景；浏览器直传请用 `transcribe`。

### `POST /api/voice/tts`

```json
{
  "text": "请立即疏散人员并对占用消防通道的单位依法处罚。",
  "voice": "zh-CN-XiaoxiaoNeural",
  "emotion": "正常",
  "output_path": "/data/out.mp3"
}
```

---

## 4. 输出格式契约

### `POST /api/voice/transcribe`

```json
{
  "status": "success",
  "text": "这里是识别出来的文字内容",
  "engine": "whisper"
}
```

### `POST /api/voice/qa`

```json
{
  "status": "success",
  "user_text": "用户说的原话",
  "answer_text": "系统回答的文字",
  "audio_path": "output/voice_bot/qa_1764300000.mp3"
}
```

**三段式**：识别 → 回答 → 语音文件路径。
`user_text` 单独返回，方便调用方做「我说的是 A，系统理解成了 B」的核对。

### `POST /api/voice/tts`

```json
{
  "status": "success",
  "audio_path": "/data/out.mp3"
}
```

### 失败情形

```json
{
  "status": "error",
  "message": "未识别到语音内容"
}
```

| 场景 | message |
|------|---------|
| 音频文件不存在 | `音频不存在：{path}` |
| 语音为空 | `未识别到语音内容` |
| 视频抽音频失败 | `音频提取失败` |

---

## 5. 配置依赖

| 环境变量 | 用途 | 缺失后果 |
|----------|------|----------|
| `XFYUN_APP_ID` / `XFYUN_API_KEY` / `XFYUN_API_SECRET` | 讯飞在线听写 | 不可用，回落本地 |
| `WHISPER_MODEL_PATH` | Whisper 模型路径 | 空则用 faster-whisper 默认缓存 |
| `WHISPER_DOWNLOAD_ROOT` | 模型下载目录 | 空则 `agent2/models/whisper` |
| `VOSK_MODEL_PATH` | Vosk 模型路径 | 无 Vosk 时不可用 |
| `TTS_VOICE_EDGE` | 默认音色 | `zh-CN-XiaoxiaoNeural` |
| `TTS_EMOTION` | 默认情感 | `正常` |

**内网部署提示**：讯飞是云端服务，内网不可达。
配置 `WHISPER_MODEL_PATH` + `VOSK_MODEL_PATH` 后可完全离线运行。

---

## 6. 调用方检查清单

- [ ] `audio_path` 是服务端可访问的路径？
- [ ] 内网环境已配 Whisper 或 Vosk 的本地模型？
- [ ] 「未识别到语音内容」是当成业务结果（空输入）而不是系统故障？
- [ ] 需要核对识别质量时看了 `user_text`？
