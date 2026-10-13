# -*- coding: utf-8 -*-
"""
统一配置 - 所有敏感信息从环境变量读取

设置环境变量（PowerShell，永久生效）：
  [System.Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "sk-xxx", "User")
  [System.Environment]::SetEnvironmentVariable("BAIDU_API_KEY", "xxx", "User")
  [System.Environment]::SetEnvironmentVariable("BAIDU_SECRET_KEY", "xxx", "User")
设置完成后重开终端。
"""
# -*- coding: utf-8 -*-
import os

# ★ 自动从 .env 加载环境变量（无论怎么启动都能读到）
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".env"
    )
    load_dotenv(_env_path)
except ImportError:
    # 没装 python-dotenv 就手动读
    _env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".env"
    )
    if os.path.exists(_env_path):
        with open(_env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
                    
# ========== API 密钥（从环境变量读，不硬编码）==========
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BAIDU_API_KEY = os.getenv("BAIDU_API_KEY", "")
BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY", "")

if not DASHSCOPE_API_KEY:
    raise RuntimeError(
        "未设置环境变量 DASHSCOPE_API_KEY。\n"
        "在 PowerShell 里运行：\n"
        "  [System.Environment]::SetEnvironmentVariable('DASHSCOPE_API_KEY', 'sk-xxx', 'User')\n"
        "然后重开终端。"
    )

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ========== 模型 ==========
VLM_MODEL = "qwen3-vl-flash"      # 视觉理解
JUDGE_MODEL = "qwen-plus"         # 综合判断（纯文本推理）

# ========== 路径 ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ========== YOLO 模型 ==========
YOLO_WEIGHTS = "/home/geng/fire_agents/agent3/input/fire_yolo/weights/best.pt"
# ========== 告警类型定义（3.2.5 用）==========
ALERT_CATEGORIES = {
    "fire":              "明火告警",
    "smoke":             "烟雾告警",
    "blocked_passage":   "消防通道堵塞告警",
    "missing_equipment": "缺少关键消防设备警示",
    "expired_equipment": "消防设备过期/失效",
    "other":             "其他消防安全隐患",
}

ALERT_LEVELS = {
    "critical": "紧急（需立即处置）",
    "high":     "高（当天整改）",
    "medium":   "中（限期整改）",
    "low":      "低（提示）",
    "normal":   "正常",
}