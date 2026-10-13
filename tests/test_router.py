# tests/test_router.py
"""common/router.py 单元测试：L1 规则 / L2 分类器 / L3 LLM 三级级联路由。

通过 monkeypatch 隔离 L2（sklearn 模型）与 L3（DeepSeek LLM），
使测试无需模型文件与网络即可完整验证级联与降级逻辑。

运行（项目根目录）：
    python -m pytest tests/test_router.py -v
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import pytest
except ImportError as exc:  # pragma: no cover
    raise ImportError("缺少 pytest，请先安装：pip install pytest") from exc

try:
    import common.router as router
    from common.router import (
        FALLBACK_MAP,
        RouteDecision,
        build_route_envelope,
        detect_kind,
        route,
    )
except ImportError:  # pragma: no cover
    pytest.skip("common.router 不可用", allow_module_level=True)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """默认屏蔽 L3 LLM 调用，避免测试触网。"""
    monkeypatch.setattr(router, "_classify_llm", lambda q: None)


# ---------------- L1 规则识别 ----------------


def test_image_extension_routes_to_vision():
    kind, conf = detect_kind("", "case_photo.jpg")
    assert kind == "vision"
    assert conf == 1.0, "文件后缀是最可信信号，应给满分"


def test_video_extension_routes_to_vision():
    assert detect_kind("", "clip.mp4")[0] == "vision"


def test_audio_extension_routes_to_voice():
    kind, conf = detect_kind("", "record.m4a")
    assert kind == "voice"
    assert conf == 1.0


def test_vision_keyword_scores_above_threshold():
    kind, conf = detect_kind("帮我看看这个灭火器照片", "")
    assert kind == "vision"
    assert conf >= 0.6


def test_doc_keyword_routes_to_doc():
    assert detect_kind("生成一份立案审批表", "")[0] == "doc"


def test_voice_keyword_routes_to_voice():
    assert detect_kind("识别这段录音", "")[0] == "voice"


def test_no_signal_returns_empty():
    kind, conf = detect_kind("今天天气不错", "")
    assert kind == ""
    assert conf == 0.0, "无信号应交由 L2/L3 处理"


def test_empty_query_is_safe():
    assert detect_kind("", "") == ("", 0.0)


def test_confidence_capped_below_one():
    """关键词再多，规则层置信度也不应达到 1.0（留给 L2/L3 空间）。"""
    q = "图片照片图像视频画面看图检查图隐患消防设施灭火器消火栓通道堵塞"
    _, conf = detect_kind(q, "")
    assert 0.6 <= conf < 1.0


# ---------------- 级联路由 ----------------


def test_l1_rule_short_circuits(monkeypatch):
    """L1 命中后不应再调用 L2/L3。"""
    called = {"ml": False, "llm": False}

    def fake_ml(q):
        called["ml"] = True
        return None

    def fake_llm(q):
        called["llm"] = True
        return None

    monkeypatch.setattr(router, "_classify_ml", fake_ml)
    monkeypatch.setattr(router, "_classify_llm", fake_llm)

    d = route("看看这张照片", "")
    assert d.kind == "vision"
    assert d.source == "rule"
    assert called == {"ml": False, "llm": False}, "L1 命中必须短路后续层级"


def test_l2_used_when_l1_misses(monkeypatch):
    monkeypatch.setattr(router, "_classify_ml", lambda q: ("doc", 0.85))

    d = route("帮我弄个材料", "")
    assert d.kind == "doc"
    assert d.source == "classifier"
    assert d.confidence == 0.85


def test_l2_low_confidence_escalates_to_l3(monkeypatch):
    """L2 置信度低于阈值时应升级到 L3。"""
    monkeypatch.setattr(router, "_classify_ml", lambda q: ("doc", 0.30))
    monkeypatch.setattr(router, "_classify_llm", lambda q: ("vision", 0.80))

    d = route("帮我弄个材料", "")
    assert d.source == "llm", "L2 低置信必须升级"
    assert d.kind == "vision"


def test_l3_used_when_l2_unavailable(monkeypatch):
    monkeypatch.setattr(router, "_classify_ml", lambda q: None)
    monkeypatch.setattr(router, "_classify_llm", lambda q: ("voice", 0.9))

    d = route("随便一句话", "")
    assert d.kind == "voice"
    assert d.source == "llm"


def test_invalid_kind_from_classifier_is_rejected(monkeypatch):
    """分类器返回非法 kind 时不应被采信。"""
    monkeypatch.setattr(router, "_classify_ml", lambda q: ("hack", 0.99))
    monkeypatch.setattr(router, "_classify_llm", lambda q: None)

    d = route("无关文本", "")
    assert d.source == "default", "非法 kind 必须被过滤"


def test_all_levels_miss_falls_back_to_rag(monkeypatch):
    monkeypatch.setattr(router, "_classify_ml", lambda q: None)
    monkeypatch.setattr(router, "_classify_llm", lambda q: None)

    d = route("完全无法识别的内容", "")
    assert d.kind == "rag"
    assert d.source == "default"
    assert d.fallback_path == FALLBACK_MAP["rag"]


def test_hops_exceeded_escalates_to_human(monkeypatch):
    """跳数超限应直接转人工，不再走任何识别层级。"""
    called = {"ml": False}

    def fake_ml(q):
        called["ml"] = True
        return ("vision", 0.99)

    monkeypatch.setattr(router, "_classify_ml", fake_ml)

    d = route("看看这张照片", "", hops=99)
    assert d.kind == "human"
    assert d.source == "escalate"
    assert d.fallback_path == FALLBACK_MAP["human"]
    assert called["ml"] is False, "转人工后不应再调用识别器"


def test_hops_boundary_uses_configured_max(monkeypatch):
    """hops 恰好等于上限即应转人工。"""
    monkeypatch.setenv("ROUTE_MAX_HOPS", "2")
    from common.config import get_settings
    get_settings.cache_clear()
    try:
        d = route("随便", "", hops=2)
        assert d.kind == "human"
    finally:
        get_settings.cache_clear()


def test_decision_always_has_fallback_path():
    """每个决策都必须带降级路径，供下游 Agent 兜底。"""
    for q, fp in [("看照片", "a.jpg"), ("生成文书", ""), ("无法识别", "")]:
        d = route(q, fp)
        assert d.fallback_path, f"query={q!r} 缺少 fallback_path"


def test_decision_confidence_rounded():
    d = route("看看这个灭火器照片", "")
    assert d.confidence == round(d.confidence, 2)


def test_route_decision_to_dict():
    d = RouteDecision(kind="rag", confidence=0.5, source="default")
    out = d.to_dict()
    assert out["kind"] == "rag"
    assert out["source"] == "default"
    assert "hops" in out and "fallback_path" in out


# ---------------- 任务信封 ----------------


def test_envelope_contains_downstream_fields():
    d = route("看看这张照片", "a.jpg")
    env = build_route_envelope(d, "看看这张照片", "a.jpg")
    for key in ("task_summary", "context_slice", "fallback_path",
                "source", "confidence", "hops", "file_path"):
        assert key in env, f"信封缺少字段 {key}"


def test_envelope_truncates_long_query():
    d = route("x" * 500, "")
    env = build_route_envelope(d, "x" * 500)
    assert len(env["task_summary"]) <= 120, "task_summary 应截断避免超长 prompt"


def test_envelope_handles_empty_query():
    d = route("", "")
    env = build_route_envelope(d, "")
    assert env["task_summary"] == ""
    assert env["file_path"] == ""
