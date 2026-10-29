# tests/test_audit_reliability.py
"""审计日志可靠性测试。

★ 这组测试要证明什么
-------------------
1. 正常写入 → 文件有内容，且是合法 JSONL
2. 写失败（目录不可写）→ **业务不中断**（不抛异常），但记录进死信
3. 写失败 → 计数增加，fail_rate > 0
4. AUDIT_STRICT=1 → 写失败时上抛
5. 死信有上限，不会无限增长
6. 同步落盘：写入后立刻能读到（不依赖进程退出 flush）

★ 为什么这组测试重要
-------------------
原实现是 best-effort，写失败**静默吞掉**。
「静默」正是问题所在 —— 没人为看不见的失败负责。
所以必须用测试把「失败要可见」这个性质钉死，
否则以后有人为了「稳定性」把异常吞回去，测试会拦住。
"""
import json
import os

import pytest

from common import audit_reliability as ar


@pytest.fixture(autouse=True)
def _reset():
    ar._write_ok_count = 0
    ar._write_fail_count = 0
    with ar._dead_letter_lock:
        ar._dead_letters.clear()
    os.environ.pop("AUDIT_STRICT", None)
    yield
    os.environ.pop("AUDIT_STRICT", None)


# ==================== 正常写入 ====================
def test_write_success(tmp_path):
    p = str(tmp_path / "audit.jsonl")
    assert ar.write_audit_record(p, {"a": 1}) is True

    with open(p, encoding="utf-8") as f:
        lines = [l for l in f.read().split("\n") if l.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"a": 1}


def test_write_appends(tmp_path):
    p = str(tmp_path / "audit.jsonl")
    ar.write_audit_record(p, {"n": 1})
    ar.write_audit_record(p, {"n": 2})
    with open(p, encoding="utf-8") as f:
        lines = [l for l in f.read().split("\n") if l.strip()]
    assert len(lines) == 2
    assert [json.loads(l)["n"] for l in lines] == [1, 2]


def test_creates_parent_dir(tmp_path):
    p = str(tmp_path / "deep" / "nested" / "audit.jsonl")
    assert ar.write_audit_record(p, {"x": 1}) is True
    assert os.path.exists(p)


def test_non_serializable_does_not_crash(tmp_path):
    """不可序列化的对象不该让审计崩掉 —— default=str 兜住。"""
    p = str(tmp_path / "audit.jsonl")
    assert ar.write_audit_record(p, {"obj": object()}) is True
    with open(p, encoding="utf-8") as f:
        json.loads(f.read().strip())   # 能解析即通过


# ==================== 写失败：业务不中断 ====================
def test_write_failure_does_not_raise(tmp_path):
    """★ 核心：写失败不能抛异常中断业务。

    审计故障不该让执法文书生成不了 —— 那才是真正的执法事故。
    """
    # 用一个不可写的路径（父路径是文件，不是目录）
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    bad = str(blocker / "sub" / "audit.jsonl")

    assert ar.write_audit_record(bad, {"a": 1}) is False, "应返回 False 而非抛异常"


def test_write_failure_goes_to_dead_letters(tmp_path):
    """写失败必须留痕 —— 不能「静默」。"""
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    bad = str(blocker / "sub" / "audit.jsonl")

    ar.write_audit_record(bad, {"important": "执法记录"})

    dl = ar.dead_letters()
    assert len(dl) == 1, "写失败必须进死信队列"
    assert dl[0]["record"]["important"] == "执法记录"
    assert "error" in dl[0]


def test_stats_reflect_failure(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    bad = str(blocker / "sub" / "audit.jsonl")

    ar.write_audit_record(bad, {"a": 1})
    st = ar.audit_stats()
    assert st["failed"] == 1
    assert st["written"] == 0
    assert st["fail_rate"] == 1.0
    assert st["dead_letters"] == 1


def test_stats_mixed(tmp_path):
    good = str(tmp_path / "a.jsonl")
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    bad = str(blocker / "sub" / "b.jsonl")

    ar.write_audit_record(good, {"n": 1})
    ar.write_audit_record(bad, {"n": 2})

    st = ar.audit_stats()
    assert st["written"] == 1
    assert st["failed"] == 1
    assert st["total"] == 2
    assert st["fail_rate"] == 0.5


# ==================== 严格模式 ====================
def test_strict_mode_raises(tmp_path):
    """AUDIT_STRICT=1 → 写失败上抛，给「审计必须保证」的部署场景。"""
    os.environ["AUDIT_STRICT"] = "1"
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    bad = str(blocker / "sub" / "audit.jsonl")

    with pytest.raises(Exception):
        ar.write_audit_record(bad, {"a": 1})


def test_strict_mode_off_by_default(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    bad = str(blocker / "sub" / "audit.jsonl")
    # 不抛
    ar.write_audit_record(bad, {"a": 1})


def test_strict_mode_values(tmp_path):
    """严格模式开关的各种写法都要认。"""
    for v in ("1", "true", "TRUE", "yes", "on"):
        os.environ["AUDIT_STRICT"] = v
        assert ar._strict_mode() is True, v
    for v in ("0", "false", "no", "off", ""):
        os.environ["AUDIT_STRICT"] = v
        assert ar._strict_mode() is False, v


# ==================== 死信上限 ====================
def test_dead_letters_capped(tmp_path):
    """死信不能无限增长 —— 否则内存会被撑爆。"""
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    bad = str(blocker / "sub" / "audit.jsonl")

    for i in range(ar._MAX_DEAD_LETTERS + 50):
        ar.write_audit_record(bad, {"i": i})

    dl = ar.dead_letters()
    assert len(dl) == ar._MAX_DEAD_LETTERS, "死信应被截断到上限"
    # 保留的是最新的
    assert dl[-1]["record"]["i"] == ar._MAX_DEAD_LETTERS + 49


# ==================== 同步落盘 ====================
def test_flushed_immediately(tmp_path):
    """写入后立刻能读到 —— 不依赖进程退出时的 flush。

    审计留痕要经得起「进程被 kill」的追问。
    """
    p = str(tmp_path / "audit.jsonl")
    ar.write_audit_record(p, {"immediate": True})
    with open(p, encoding="utf-8") as f:
        content = f.read()
    assert "immediate" in content, "内容应已落盘（不依赖缓冲区 flush）"
