# tests/test_model_provisioner.py
"""模型供给模块的测试。

★ 这组测试要证明什么
-------------------
1. 环境探测在 Ollama 不可用时**不崩**，而是返回明确状态
2. 下载指引同时覆盖「有网」和「内网离线包」两条路径
3. 检索精度低于阈值时，建议指向**检索层**而不是「换更大的模型」
4. provider_mode 切换真的改变 LLM client 的选择

★ 第 3 条是重点
---------------
「检索精度低 → 提示换更准的模型」这个建议是**错的**：
检索不准时，模型再强也只是在错的材料上做漂亮的推理。
我在调优时实测过 context_precision 0.165→0.226 期间
faithfulness 一直稳在 0.98 —— 说明 faithfulness 高不代表检索好。

所以建议文本里必须明确写「换更大的 LLM 无济于事」，
避免运维照着建议去换模型却发现没用。
"""
import os

import pytest

from common.model_provisioner import (
    build_download_guidance,
    check_and_advise,
    evaluate_retrieval,
    probe_environment,
)


# ==================== 环境探测 ====================
def test_probe_environment_returns_structure():
    """探测结果结构固定，方便前端直接渲染。"""
    env = probe_environment()
    for k in ("provider_mode", "ollama_url", "ollama_connected",
              "installed_models", "target_model", "has_target_model",
              "needs_model"):
        assert k in env, f"缺少字段 {k}"


def test_probe_does_not_crash_when_ollama_absent():
    """★ Ollama 没装时不能崩 —— 这是最常见的实际情况。"""
    env = probe_environment()
    assert isinstance(env["ollama_connected"], bool)
    assert isinstance(env["installed_models"], list)
    if not env["ollama_connected"]:
        assert env["error"], "连不上时应有错误说明"


def test_needs_model_requires_both_connected_and_missing():
    """needs_model 的语义：服务通了但模型没装。

    服务都不通时，缺的是「装服务」不是「下模型」——
    这两件事的处置方式不同，不能混为一谈。
    """
    env = probe_environment()
    if not env["ollama_connected"]:
        assert env["needs_model"] is False, \
            "服务都没通，不该说「缺模型」"


def test_probe_reads_provider_mode():
    """provider_mode 必须能被读出来。"""
    env = probe_environment()
    assert env["provider_mode"] in ("cloud", "local", "hybrid")


# ==================== 下载指引 ====================
def test_guidance_has_online_and_offline():
    """★ 内网部署必须给离线包指引 —— 光给 ollama pull 没用。"""
    g = build_download_guidance("qwen2.5:7b")
    assert "online" in g and "offline" in g
    assert len(g["online"]["steps"]) >= 2
    assert len(g["offline"]["steps"]) >= 4


def test_guidance_offline_includes_pack_and_restore():
    """离线指引必须包含「打包」和「解包」两端。"""
    g = build_download_guidance("qwen2.5:7b")
    cmds = " ".join(s["cmd"] for s in g["offline"]["steps"])
    assert "tar" in cmds, "离线包指引缺打包命令"
    assert "scp" in cmds or "拷贝" in g["offline"]["note"], \
        "离线包指引缺传输步骤"


def test_guidance_mentions_provider_mode():
    """离线部署后必须改 provider_mode —— 否则还是走云端。"""
    g = build_download_guidance("qwen2.5:7b")
    notes = " ".join(g["notes"])
    assert "provider_mode" in notes


def test_guidance_model_included():
    g = build_download_guidance("qwen2.5:14b")
    assert g["model"] == "qwen2.5:14b"
    cmds = " ".join(s["cmd"] for s in g["online"]["steps"])
    assert "qwen2.5:14b" in cmds


# ==================== 建议生成 ====================
def test_check_and_advise_structure():
    r = check_and_advise(sample_limit=2)
    for k in ("environment", "retrieval", "threshold", "below_threshold",
              "advice"):
        assert k in r, f"缺少字段 {k}"
    assert isinstance(r["advice"], list)


def test_low_precision_advice_points_to_retrieval_not_bigger_model():
    """★ 核心：精度低时建议指向检索层，不能建议「换更大的模型」。

    检索不准时换更大的 LLM 没用 —— 模型再强也是在错的材料上推理。
    """
    r = check_and_advise(sample_limit=2)
    for item in r["advice"]:
        if item.get("code") == "low_retrieval_precision":
            detail = item.get("detail", "")
            assert "检索" in detail, "建议应指向检索层"
            assert ("无济于事" in detail or "解决不了" in detail
                    or "瓶颈" in detail), \
                "建议应明确说明换大模型无济于事"
            # 不能把「换模型」作为主要动作
            actions = item.get("actions", [])
            for a in actions:
                assert a.get("kind") != "switch_model", \
                    "精度低时不该建议换模型"


def test_advice_items_have_level_and_title():
    """建议项结构固定，前端才能统一渲染。"""
    r = check_and_advise(sample_limit=2)
    for item in r["advice"]:
        assert "level" in item
        assert "title" in item
        assert "code" in item
        assert item["level"] in ("ok", "warn", "error")


# ==================== 检索精度探测 ====================
def test_evaluate_retrieval_returns_structure():
    r = evaluate_retrieval(sample_limit=3)
    assert "ok" in r
    assert "precision" in r
    if r["ok"]:
        assert 0.0 <= r["precision"] <= 1.0, \
            f"precision 应在 0~1，实际 {r['precision']}"


def test_evaluate_retrieval_degrades_gracefully():
    """评测失败不能抛异常 —— 在线探测不该把服务搞崩。"""
    r = evaluate_retrieval(sample_limit=1)
    assert isinstance(r, dict)
    if not r["ok"]:
        assert "error" in r


# ==================== provider_mode 切换 ====================
def test_provider_mode_switch_affects_client_selection(monkeypatch):
    """★ provider_mode=local 时应优先用本地模型。

    这是「切到内网模式不用改业务代码」的保证。
    """
    from agent1 import config as agent1_config

    calls = {"local": 0, "cloud": 0}

    class _FakeLocal:
        model_name = "local-test"

    def fake_get_local_llm():
        calls["local"] += 1
        return _FakeLocal()

    def fake_get_deepseek():
        calls["cloud"] += 1
        return object()

    monkeypatch.setattr(agent1_config, "get_local_llm", fake_get_local_llm)
    monkeypatch.setattr(agent1_config, "get_deepseek", fake_get_deepseek)
    monkeypatch.setattr(agent1_config, "get_llm_model", lambda: "cloud-test")

    # --- cloud 模式：只用云端 ---
    monkeypatch.setattr("common.llm_guard._cfg",
                        lambda: _FakeCfg("cloud"))
    import common.llm_guard as lg
    client, model = lg._resolve_client()
    assert calls["cloud"] == 1
    assert calls["local"] == 0, "cloud 模式不该探测本地"

    # --- local 模式：优先本地 ---
    calls["local"] = 0
    calls["cloud"] = 0
    monkeypatch.setattr("common.llm_guard._cfg",
                        lambda: _FakeCfg("local"))
    client, model = lg._resolve_client()
    assert calls["local"] == 1, "local 模式应优先探测本地"
    assert calls["cloud"] == 0, "本地可用时不该用云端"

    # --- local 模式但本地不可用：回退云端 ---
    calls["local"] = 0
    calls["cloud"] = 0
    monkeypatch.setattr(agent1_config, "get_local_llm", lambda: None)
    monkeypatch.setattr("common.llm_guard._cfg",
                        lambda: _FakeCfg("local"))
    client, model = lg._resolve_client()
    assert calls["cloud"] == 1, "本地不可用应回退云端"


class _FakeCfg:
    def __init__(self, mode):
        self.provider_mode = mode
        self.llm_request_timeout = 30.0
        self.ollama_model = "qwen2.5:7b"
