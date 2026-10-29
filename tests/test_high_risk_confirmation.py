# tests/test_high_risk_confirmation.py
"""高危改动二次确认的测试。

★★ 为什么必须有这组测试
------------------------
「高风险改动要不要人工确认」这条规则一旦失效，后果是：
**改了法律依据/处罚金额的文书直接变成终稿**。

而这条规则是「拦截」型逻辑 —— 它只有在**真的拦住了**的时候才有意义。
如果它变成「提示型」（检测到高风险但照常批准），
功能测试会全过（因为返回里确实有 high_risk_count），
但拦截能力已经没了。

所以测试必须验证**文件没被移动**、**状态还是 pending**，
而不只是验证返回值里有个数字。

★★ 验证的三个不变量
------------------
1. 有高风险 + 未确认 → 文件**仍在 pending/**，状态仍 pending
2. 有高风险 + 已确认 → 文件移到 official/，状态 approved
3. 无高风险 + 未确认 → 直接批准（不增加摩擦）
"""
import json
import os
import shutil
import tempfile
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_env(tmp_path):
    """搭一个最小可用的审批环境。"""
    pending = tmp_path / "pending"
    official = tmp_path / "official"
    reviews = tmp_path / "reviews"
    for d in (pending, official, reviews):
        d.mkdir(parents=True, exist_ok=True)
    return pending, official, reviews


def _write_review(reviews, review_id, files, status="pending"):
    rec = {
        "review_id": review_id,
        "status": status,
        "files": files,
        "created_at": time.time(),
    }
    p = os.path.join(reviews, f"{review_id}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return p


# ==================== 逻辑层测试（不依赖 FastAPI）====================
def _simulate_approve(pending_dir, official_dir, reviews_dir, review_id,
                      change_reports, confirm_high_risk):
    """复刻 approve_review 的核心决策逻辑。

    与真实实现保持一致 —— 如果实现改了这里也要改。
    """
    review_path = os.path.join(reviews_dir, f"{review_id}.json")
    with open(review_path, "r", encoding="utf-8") as f:
        rec = json.load(f)

    if rec.get("status") == "approved":
        return {"status": "success", "message": "已审批通过，无需重复操作"}

    high_risk = sum(r.get("high_risk_count", 0) for r in change_reports
                    if isinstance(r, dict))

    if high_risk > 0 and not confirm_high_risk:
        rec["status"] = "pending"
        rec["pending_confirmation"] = {
            "high_risk_count": high_risk,
            "detected_at": time.time(),
        }
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        return {"status": "confirmation_required",
                "high_risk_count": high_risk}

    moved = []
    for item in rec.get("files", []):
        src_p = item.get("draft_path", "")
        if src_p and os.path.isfile(src_p):
            dst = os.path.join(official_dir, os.path.basename(src_p))
            shutil.move(src_p, dst)
            item["official_path"] = dst
            moved.append(dst)

    rec["status"] = "approved"
    rec["approved_at"] = time.time()
    rec.pop("pending_confirmation", None)
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    return {"status": "success", "official_files": moved}


# ==================== 测试 ====================
def test_high_risk_without_confirmation_blocks(tmp_path):
    """★ 核心：有高风险 + 未确认 → 拦截，文件不动，状态仍 pending。"""
    pending, official, reviews = _make_env(tmp_path)
    draft = pending / "test.docx"
    draft.write_text("草稿内容", encoding="utf-8")
    review_id = "rev_test1"
    _write_review(reviews, review_id, [{"draft_path": str(draft)}])

    res = _simulate_approve(str(pending), str(official), str(reviews),
                             review_id,
                             [{"high_risk_count": 2}], confirm_high_risk=False)

    assert res["status"] == "confirmation_required", \
        f"应拦截，实际 {res['status']}"
    assert res["high_risk_count"] == 2
    # ★ 文件必须还在 pending/
    assert draft.exists(), "拦截时不该移动文件"
    assert not (official / "test.docx").exists(), "拦截时不该产生终稿"
    # 状态仍是 pending
    rec = json.load(open(os.path.join(reviews, f"{review_id}.json"),
                         encoding="utf-8"))
    assert rec["status"] == "pending"
    assert rec["pending_confirmation"]["high_risk_count"] == 2


def test_high_risk_with_confirmation_approves(tmp_path):
    """★ 有高风险 + 已确认 → 放行，文件移入 official/。"""
    pending, official, reviews = _make_env(tmp_path)
    draft = pending / "test.docx"
    draft.write_text("草稿内容", encoding="utf-8")
    review_id = "rev_test2"
    _write_review(reviews, review_id, [{"draft_path": str(draft)}])

    res = _simulate_approve(str(pending), str(official), str(reviews),
                             review_id,
                             [{"high_risk_count": 2}], confirm_high_risk=True)

    assert res["status"] == "success"
    assert (official / "test.docx").exists(), "确认后应产生终稿"
    assert not draft.exists(), "确认后草稿应被移走"
    rec = json.load(open(os.path.join(reviews, f"{review_id}.json"),
                         encoding="utf-8"))
    assert rec["status"] == "approved"
    assert "pending_confirmation" not in rec


def test_no_high_risk_approves_directly(tmp_path):
    """★ 无高风险 + 未确认 → 直接批准（不增加摩擦）。

    这是设计上的关键：只在有风险时才拦。
    否则运营会习惯性勾选确认，机制就白做了。
    """
    pending, official, reviews = _make_env(tmp_path)
    draft = pending / "test.docx"
    draft.write_text("草稿内容", encoding="utf-8")
    review_id = "rev_test3"
    _write_review(reviews, review_id, [{"draft_path": str(draft)}])

    res = _simulate_approve(str(pending), str(official), str(reviews),
                             review_id,
                             [{"high_risk_count": 0}], confirm_high_risk=False)

    assert res["status"] == "success", "无高风险不该被拦"
    assert (official / "test.docx").exists()


def test_already_approved_is_idempotent(tmp_path):
    """已审批的重复调用幂等 —— 不该重复移动文件。"""
    pending, official, reviews = _make_env(tmp_path)
    draft = pending / "test.docx"
    draft.write_text("草稿内容", encoding="utf-8")
    review_id = "rev_test4"
    _write_review(reviews, review_id, [{"draft_path": str(draft)}])

    _simulate_approve(str(pending), str(official), str(reviews), review_id,
                       [{"high_risk_count": 0}], confirm_high_risk=False)
    res = _simulate_approve(str(pending), str(official), str(reviews), review_id,
                             [{"high_risk_count": 0}], confirm_high_risk=False)
    assert res["status"] == "success"
    assert "无需重复操作" in res["message"]


def test_block_then_confirm_flow(tmp_path):
    """★ 完整流程：拦截 → 人工确认 → 放行。"""
    pending, official, reviews = _make_env(tmp_path)
    draft = pending / "test.docx"
    draft.write_text("草稿内容", encoding="utf-8")
    review_id = "rev_test5"
    _write_review(reviews, review_id, [{"draft_path": str(draft)}])

    # 第一次：被拦
    r1 = _simulate_approve(str(pending), str(official), str(reviews), review_id,
                           [{"high_risk_count": 3}], confirm_high_risk=False)
    assert r1["status"] == "confirmation_required"
    assert draft.exists()

    # 人工确认后重来
    r2 = _simulate_approve(str(pending), str(official), str(reviews), review_id,
                           [{"high_risk_count": 3}], confirm_high_risk=True)
    assert r2["status"] == "success"
    assert (official / "test.docx").exists()
    assert not draft.exists()


def test_change_report_error_does_not_block(tmp_path):
    """★ 改动检测失败不该阻断审批 —— 但也不该误判为高风险。

    报告生成失败时 change_reports 里是 {"error": ...}，
    没有 high_risk_count → 视为 0 → 正常审批。
    """
    pending, official, reviews = _make_env(tmp_path)
    draft = pending / "test.docx"
    draft.write_text("草稿内容", encoding="utf-8")
    review_id = "rev_test6"
    _write_review(reviews, review_id, [{"draft_path": str(draft)}])

    res = _simulate_approve(str(pending), str(official), str(reviews), review_id,
                             [{"error": "改动检测失败"}],
                             confirm_high_risk=False)
    assert res["status"] == "success", "报告失败不该阻断审批"


# ==================== 代码层断言 ====================
def test_server_has_confirm_param():
    """approve 端点必须接受 confirm_high_risk 参数。"""
    src = open(os.path.join(ROOT, "agent1", "server.py"),
               encoding="utf-8").read()
    assert "confirm_high_risk" in src, "approve 端点缺少 confirm_high_risk"
    assert "confirmation_required" in src, "缺少 confirmation_required 状态"


def test_check_before_move():
    """★ 拦截必须发生在文件移动**之前** —— 不能靠回滚。

    第一版是「移动 → 拦截 → 回滚」，回滚失败会导致
    文书停在 official/ 但状态是 pending 的不一致。
    """
    src = open(os.path.join(ROOT, "agent1", "server.py"),
               encoding="utf-8").read()
    idx_confirm = src.find("if high_risk > 0 and not confirm:")
    idx_move = src.find("shutil.move(src_p, dst)")
    assert idx_confirm > 0, "找不到确认判断"
    assert idx_move > 0, "找不到文件移动"
    assert idx_confirm < idx_move, \
        "确认判断必须在文件移动之前"


def test_thread_id_unique():
    """★ thread_id 必须每次唯一 —— 秒级精度会导致同秒任务串台。

    只看**实际代码行**，不看注释 ——
    注释里保留了旧写法作为「为什么改」的说明，
    那是给人看的，不是执行路径。
    """
    src = open(os.path.join(ROOT, "orchestrator_langgraph.py"),
               encoding="utf-8").read()
    # 取实际赋值语句（排除注释行）
    code_lines = [l.strip() for l in src.split("\n")
                  if not l.strip().startswith("#")]
    code = "\n".join(code_lines)

    assert "uuid.uuid4().hex[:8]" in code, \
        "thread_id 缺 uuid 后缀，同秒并发任务会串台"
    # 旧写法只能出现在注释里
    old_in_code = 'f"task_{int(time.time())}"'
    assert old_in_code not in code, \
        "实际代码里还存在秒级精度的 thread_id"
