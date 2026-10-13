# agent3/agent/config.py
"""Agent3 配置（视觉：YOLO 检测 + 百度 OCR + VLM 综合判断）。

配置来源统一为 common.config.get_settings()（项目根 .env，单一配置源）。
密钥缺失时不在 import 阶段抛错，改为延迟到实际调用时报错，
保证健康检查、无密钥子功能（如仅 YOLO）仍可正常启动。
"""
import logging
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.config import get_settings

logger = logging.getLogger("fire_agents.agent3")

_s = get_settings()

# ========== API 密钥 ==========
DASHSCOPE_API_KEY = _s.dashscope_api_key
BAIDU_API_KEY = _s.baidu_api_key
BAIDU_SECRET_KEY = _s.baidu_secret_key

BASE_URL = _s.dashscope_base_url

# ========== 模型 ==========
VLM_MODEL = _s.vlm_model       # 视觉理解
JUDGE_MODEL = _s.judge_model   # 综合判断（纯文本推理）

# ========== 路径 ==========
BASE_DIR = os.path.join(_ROOT, "agent3")
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
YOLO_WEIGHTS_DIR = os.path.join(INPUT_DIR, "fire_yolo", "weights")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== YOLO 模型 ==========
# 留空则由 ultralytics 使用其默认预训练权重
YOLO_WEIGHTS = _s.yolo_weights or os.path.join(YOLO_WEIGHTS_DIR, "best.pt")


# ========== 告警类型定义（综合判断用）==========
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


# ========== 启动检查 ==========
def require_dashscope() -> str:
    """需要调用 VLM 时调用，密钥缺失则给出可操作的报错。"""
    if not DASHSCOPE_API_KEY:
        raise RuntimeError(
            "未配置 DASHSCOPE_API_KEY，VLM 功能不可用。\n"
            "请在项目根 .env 中添加：DASHSCOPE_API_KEY=sk-xxx"
        )
    return DASHSCOPE_API_KEY


def require_baidu() -> tuple[str, str]:
    """需要调用百度 OCR 时调用，密钥缺失则给出可操作的报错。"""
    if not (BAIDU_API_KEY and BAIDU_SECRET_KEY):
        raise RuntimeError(
            "未配置 BAIDU_API_KEY / BAIDU_SECRET_KEY，OCR 功能不可用。\n"
            "请在项目根 .env 中添加：BAIDU_API_KEY=xxx 与 BAIDU_SECRET_KEY=xxx"
        )
    return BAIDU_API_KEY, BAIDU_SECRET_KEY


def check_env(verbose: bool = False) -> bool:
    missing = []
    if not DASHSCOPE_API_KEY:
        missing.append("DASHSCOPE_API_KEY")
    if not (BAIDU_API_KEY and BAIDU_SECRET_KEY):
        missing.append("BAIDU_API_KEY/BAIDU_SECRET_KEY")
    if missing:
        logger.warning("以下配置未设置（相关功能不可用）：%s", ", ".join(missing))
    if verbose:
        logger.info("VLM 模型：%s，判断模型：%s", VLM_MODEL, JUDGE_MODEL)
    return len(missing) == 0
