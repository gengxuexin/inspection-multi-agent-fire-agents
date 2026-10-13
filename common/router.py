# common/router.py
"""级联意图路由：L1 规则 → L2 本地分类器 → L3 LLM 兜底 → 升级人工。

三级降级策略：
  L1 规则（文件后缀 + 关键词打分）      conf >= 0.6      → source="rule"
  L2 分类器（models/intent_clf.joblib） conf >= threshold → source="classifier"
  L3 LLM（DeepSeek 结构化输出）         conf >= 0.5      → source="llm"
  全部未命中                                              → 默认走法规 RAG

用法：
    from common.router import route, build_route_envelope
    decision = route("看看这张灭火器照片", file_path="a.jpg")
    envelope = build_route_envelope(decision, query="看看这张灭火器照片")
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

try:
    from common.config import get_settings, project_root
except ImportError:  # pragma: no cover - 以包内相对方式导入
    from .config import get_settings, project_root

VALID_KINDS = ("rag", "vision", "voice", "doc")

# 各意图的降级兜底路径（供下游 Agent 在主链路失败时使用）
FALLBACK_MAP = {
    "vision": "ocr_only",
    "voice": "transcribe_only",
    "doc": "rag",
    "rag": "direct_llm",
    "human": "manual_review",
}

# 关键词表（与 gateway.py 意图识别逻辑保持一致）
VISION_KW = ["图片", "照片", "图像", "视频", "画面", "看图", "检查图",
             "隐患", "消防设施", "灭火器", "消火栓", "通道", "堵塞",
             "消防水带", "消防柜", "安全出口"]
VOICE_KW = ["语音", "录音", "音频", "听一下", "转写", "识别这段"]
DOC_KW = ["生成文书", "生成一份", "生成个", "写一份", "写个",
          "通知书", "审批表", "立案表", "笔录", "决定书", "告知书"]

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".flv")
_AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac")


@dataclass
class RouteDecision:
    """一次路由决策的完整描述。"""
    kind: str                       # rag | vision | voice | doc | human
    confidence: float               # 0.0 ~ 1.0
    source: str                     # rule | classifier | llm | default | escalate
    hops: int = 0                   # 当前路由跳数（重路由时由调用方递增）
    task_summary: str = ""
    fallback_path: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ==================== L1 规则识别 ====================
def detect_kind(query: str, file_path: str = "") -> Tuple[str, float]:
    """L1 规则识别：文件后缀 + 关键词打分。

    返回 (kind, confidence)；无命中返回 ("", 0.0)，交给 L2/L3。
    """
    q = query or ""
    fp = (file_path or "").lower()

    # 1) 文件后缀最可信
    if fp.endswith(_IMAGE_EXTS) or fp.endswith(_VIDEO_EXTS):
        return "vision", 1.0
    if fp.endswith(_AUDIO_EXTS):
        return "voice", 1.0

    # 2) 关键词打分（与 gateway.py L46-81 同源逻辑）
    scores = {"vision": 0, "voice": 0, "doc": 0}
    for kw in VISION_KW:
        if kw in q:
            scores["vision"] += 1
    for kw in VOICE_KW:
        if kw in q:
            scores["voice"] += 1
    for kw in DOC_KW:
        if kw in q:
            scores["doc"] += 1
    best_kind, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score > 0:
        return best_kind, min(0.9, 0.6 + 0.1 * best_score)

    # 3) 无命中
    return "", 0.0


# ==================== L2 本地分类器 ====================
_clf_cache = None            # 已加载的 Pipeline 缓存
_clf_load_failed = False     # 加载过一次失败后不再反复尝试（文件缺失仍允许热加载）
_CLF_PATH = os.path.join(project_root(), "models", "intent_clf.joblib")


def _classify_ml(query: str) -> Optional[Tuple[str, float]]:
    """L2 意图分类器：models/intent_clf.joblib（train_classifier.py 产出）。

    返回 (label, proba)；模型缺失/加载失败/推理异常 → None。
    """
    global _clf_cache, _clf_load_failed
    q = (query or "").strip()
    if not q or _clf_load_failed:
        return None
    if _clf_cache is None:
        try:
            import joblib
            if not os.path.exists(_CLF_PATH):
                return None  # 尚未训练，允许后续热加载
            _clf_cache = joblib.load(_CLF_PATH)
        except Exception:
            _clf_load_failed = True
            _clf_cache = None
            return None
    try:
        proba = _clf_cache.predict_proba([q])[0]
        idx = int(proba.argmax())
        label = str(_clf_cache.classes_[idx])
        return label, float(proba[idx])
    except Exception:
        return None


# ==================== L3 LLM 兜底分类 ====================
_llm_client = None
_LLM_SYSTEM_PROMPT = (
    "你是消防执法系统的意图分类器。判断用户请求属于哪一类，只输出 JSON，"
    '格式：{"kind": "rag|vision|voice|doc", "confidence": 0.0到1.0之间的小数}。'
    "kind 说明：rag=法规咨询/知识问答/处罚标准咨询；vision=图片或视频隐患识别；"
    "voice=语音或录音转写；doc=生成执法文书。不要输出 JSON 以外的任何内容。"
)


def _extract_json(text: str) -> str:
    """从 LLM 输出中提取 JSON 串（容忍 ```json 围栏与前后闲话）。"""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if 0 <= start < end:
        return t[start:end + 1]
    return t


def _classify_llm(query: str) -> Optional[Tuple[str, float]]:
    """L3 LLM 兜底分类（DeepSeek，temperature=0 / timeout=10 / max_tokens=50）。

    openai 未安装、api_key 为空、超时或解析失败 → None。
    """
    global _llm_client
    q = (query or "").strip()
    if not q:
        return None
    try:
        s = get_settings()
    except Exception:
        return None
    api_key = getattr(s, "deepseek_api_key", "") or ""
    if not api_key:
        return None
    if _llm_client is None:
        try:
            from openai import OpenAI
        except ImportError:
            return None
        try:
            _llm_client = OpenAI(
                api_key=api_key,
                base_url=getattr(s, "deepseek_base_url", "") or None,
            )
        except Exception:
            _llm_client = None
            return None
    try:
        resp = _llm_client.chat.completions.create(
            model=getattr(s, "deepseek_model", "deepseek-chat"),
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": q[:200]},
            ],
            temperature=0,
            timeout=10,
            max_tokens=50,
        )
        text = (resp.choices[0].message.content or "").strip()
        data = json.loads(_extract_json(text))
        kind = str(data.get("kind", "")).strip().lower()
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        return None  # 超时 / JSON 解析失败等一律降级
    if kind not in VALID_KINDS:
        return None
    return kind, max(0.0, min(1.0, confidence))


# ==================== 级联路由主入口 ====================
def route(query: str, file_path: str = "", hops: int = 0) -> RouteDecision:
    """三级级联路由：L1 规则 → L2 分类器 → L3 LLM → 默认 RAG。"""
    try:
        s = get_settings()
        max_hops = int(getattr(s, "route_max_hops", 3))
        conf_threshold = float(getattr(s, "classifier_conf_threshold", 0.6))
    except Exception:
        max_hops, conf_threshold = 3, 0.6

    # 跳数超限 → 升级人工
    if hops >= max_hops:
        return RouteDecision(
            kind="human", confidence=0.0, source="escalate", hops=hops,
            fallback_path=FALLBACK_MAP["human"],
            detail="路由跳数超限，升级人工处理",
        )

    # ---- L1 规则 ----
    kind, conf = detect_kind(query, file_path)
    if kind and conf >= 0.6:
        return RouteDecision(
            kind=kind, confidence=round(conf, 2), source="rule", hops=hops,
            fallback_path=FALLBACK_MAP.get(kind, "direct_llm"),
            detail=f"L1 规则命中（{kind}）",
        )

    # ---- L2 本地分类器 ----
    ml = _classify_ml(query)
    if ml is not None:
        kind, conf = ml
        if kind in VALID_KINDS and conf >= conf_threshold:
            return RouteDecision(
                kind=kind, confidence=round(conf, 2), source="classifier", hops=hops,
                fallback_path=FALLBACK_MAP.get(kind, "direct_llm"),
                detail=f"L2 分类器命中（{kind}）",
            )

    # ---- L3 LLM 兜底 ----
    llm = _classify_llm(query)
    if llm is not None:
        kind, conf = llm
        if kind in VALID_KINDS and conf >= 0.5:
            return RouteDecision(
                kind=kind, confidence=round(conf, 2), source="llm", hops=hops,
                fallback_path=FALLBACK_MAP.get(kind, "direct_llm"),
                detail=f"L3 LLM 命中（{kind}）",
            )

    # ---- 全部未命中：默认走法规 RAG ----
    return RouteDecision(
        kind="rag", confidence=0.0, source="default", hops=hops,
        fallback_path=FALLBACK_MAP["rag"],
        detail="三级均未命中，默认走法规 RAG",
    )


def build_route_envelope(decision: RouteDecision, query: str, file_path: str = "") -> dict:
    """把路由决策打包成下游 Agent 可直接消费的任务信封。"""
    q = query or ""
    return {
        "task_summary": q[:120],
        "context_slice": q[-200:],
        "fallback_path": FALLBACK_MAP.get(decision.kind, "direct_llm"),
        "source": decision.source,
        "confidence": round(decision.confidence, 2),
        "hops": decision.hops,
        "file_path": file_path or "",
    }
