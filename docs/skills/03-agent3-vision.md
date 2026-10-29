# Agent3 · 消防设施视觉隐患识别 Skill

**角色**：对消防设施照片/视频做隐患识别（YOLO 目标检测 + OCR 文字识别 + VLM 综合判断）。
**代码位置**：`agent3/server.py`、`agent3/agent/fire_agent.py`
**端口**：8003

---

## 1. 通过什么形式调用

| 场景 | 端点 | 方法 |
|------|------|------|
| 上传单张图片分析 | `/api/vision/analyze_image` | POST |
| 上传视频分析 | `/api/vision/analyze_video` | POST |

### 网关路由关键词

命中 `vision` 意图时路由到 agent3：

> 图片、照片、图像、视频、画面、看图、检查图、隐患、消防设施、灭火器、消火栓、通道、堵塞、消防水带、消防柜、安全出口

**注意**：文件扩展名也会参与路由判断 ——
`.jpg/.jpeg/.png/.bmp/.webp` → image，`.mp4/.avi/.mkv/.mov/.flv` → video。

---

## 2. Agent3 怎样识别调用

### 三段式分析流水线

```
图片/视频
   ↓
① OCR      读图上的文字（设备铭牌、标识牌、告示）
   ↓
② YOLO     检测消防设施目标（灭火器、消火栓、消防柜、安全出口…）
   ↓
③ VLM 综合判断  融合前两者的结果 + 原图，给出隐患结论
```

**关键设计**：VLM 判断时**同时拿到 YOLO 结果和原图路径** ——
只看结构化检测结果会丢失图像细节（如「灭火器箱被挪用但箱体还在」）。

### 能力开关

| 参数 | 作用 | 关掉的后果 |
|------|------|-----------|
| `do_yolo` | 是否跑目标检测 | 无 `yolo` 字段，只有 OCR + VLM 判断 |
| `do_ocr` | 是否跑文字识别 | 无 `ocr` 字段，只有 YOLO + VLM 判断 |

**降级**：任一子步骤失败，**不中断整体流程** ——
该字段变成 `{"error": "..."}`，其余照常。所以结果里可能只有 `ocr` 没有 `yolo`。

---

## 3. 输入格式

### `POST /api/vision/analyze_image`

multipart：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `file` | file | 是 | — | 图片文件 |
| `do_yolo` | bool | 否 | `True` | 是否跑 YOLO |
| `do_ocr` | bool | 否 | `True` | 是否跑 OCR |

### `POST /api/vision/analyze_video`

multipart：

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `file` | file | 是 | — | 视频文件 |
| `segment_seconds` | int | 否 | 4 | 分段长度（秒） |
| `fps` | int | 否 | 1 | 抽帧率 |
| `do_yolo` | bool | 否 | `False` | 视频默认**不跑** YOLO（耗时） |

**视频默认关 YOLO** 是有意的：逐帧检测开销大。
需要逐帧目标统计时显式传 `do_yolo=true`。

---

## 4. 输出格式契约

### `POST /api/vision/analyze_image`

```json
{
  "status": "success",
  "result": {
    "source": "/tmp/xxx/photo.jpg",
    "type": "image",
    "ocr": {
      "ocr_texts": ["灭火器箱", "MFZ/ABC4"],
      "hotel_name": ""
    },
    "yolo": {
      "total_objects": 3,
      "counts": {"灭火器": 2, "消火栓": 1}
    },
    "judgment": {
      "overall_risk": "低风险",
      "summary": "……",
      "items": []
    }
  }
}
```

| 字段 | 说明 |
|------|------|
| `type` | `image` / `video` |
| `ocr` | 识别到的文字。子步骤失败时为 `{"error": "..."}` |
| `yolo` | 检测到的目标与分类统计。同上 |
| `judgment` | **VLM 综合判断** —— 真正要用的结论 |

**结果同时落盘**：`agent3/output/image_result_{文件名}.json`。

### 视频

```json
{
  "status": "success",
  "result": {
    "source": "/tmp/xxx/video.mp4",
    "type": "video",
    "segments": [
      {
        "index": 0,
        "timestamp": "00:00:00",
        "ocr": { },
        "yolo": { },
        "judgment": { }
      }
    ],
    "summary": { }
  }
}
```

### 失败

```json
{
  "status": "error",
  "message": "文件不存在"
}
```

---

## 5. 配置依赖

| 环境变量 | 用途 | 缺失后果 |
|----------|------|----------|
| `DASHSCOPE_API_KEY` | VLM（综合判断） | **无法出 `judgment`**，只有 OCR/YOLO 原始结果 |
| `BAIDU_API_KEY` / `BAIDU_SECRET_KEY` | 百度 OCR | 无 `ocr` 字段 |
| YOLO 权重路径 | 目标检测 | 无 `yolo` 字段 |

**内网部署提示**：DASHSCOPE 是云端服务。
VLM 不可达时，`judgment` 会缺失 —— **这时只有原始检测数据，
没有隐患结论**，不能当成「未发现隐患」。

---

## 6. 调用方检查清单

- [ ] 区分了「无隐患」和「VLM 不可用导致没有 judgment」？
  （`judgment` 缺失 ≠ 没隐患）
- [ ] `ocr` / `yolo` 字段里的 `error` 检查了？
- [ ] 视频场景确认了 `do_yolo` 是否符合预期（默认关闭）？
- [ ] 内网环境确认了 VLM 的可用性？
