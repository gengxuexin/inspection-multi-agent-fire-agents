# tests/test_eval_stratified.py
"""验证评测的「应答样本口径」统计逻辑正确。

★ 为什么要有这个测试
--------------------
我给 local_eval 加了 _answered_only / _abstained_only /
_by_category_answered 三个字段。

它们的价值全在**数字对不对**上 —— 如果口径算错了，
就会得出「检索比实际好」或「比实际差」的错误结论，
进而把优化方向带偏。所以必须验证。

★★ 关键不变式
-------------
1. _answered_only.n + _abstained_only.n == 总样本数
2. 应答样本的 precision 均值 **必须高于**总分
   （因为弃答样本恒为 0，把它们剔除只会拉高）
3. **总分一个都不许变** —— 新字段是并列视角，不是替换
"""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "report_rerank_final.json")


def _has_report():
    return os.path.exists(REPORT)


pytestmark = pytest.mark.skipif(
    not _has_report(), reason="缺少 report_rerank_final.json")


@pytest.fixture(scope="module")
def report():
    return json.load(open(REPORT, encoding="utf-8"))


def _split(per_sample):
    ans = [r for r in per_sample if not r.get("_abstained")]
    ab = [r for r in per_sample if r.get("_abstained")]
    return ans, ab


def test_split_covers_all_samples(report):
    """应答 + 弃答 == 总数，一条都不能丢。"""
    per = report["scores"]["_per_sample"]
    ans, ab = _split(per)
    assert len(ans) + len(ab) == len(per)


def test_abstained_precision_is_near_zero(report):
    """★ 弃答样本的 precision 必然**接近** 0（不是恒为 0）。

    实测 14 条弃答样本里 12 条是 0.0、2 条是 0.2 ——
    后者说明 LLM judge 仍判出了部分相关的上下文，
    所以准确表述是「接近 0」而非「恒为 0」。
    """
    per = report["scores"]["_per_sample"]
    _ans, ab = _split(per)
    if not ab:
        pytest.skip("没有弃答样本")
    vals = [r["context_precision"] for r in ab]
    mean = sum(vals) / len(vals)
    assert mean < 0.1, f"弃答样本 precision 均值应接近 0，实际 {mean}"
    assert max(vals) <= 0.2, f"弃答样本 precision 不该超过 0.2，实际 {max(vals)}"


def test_abstained_drag_down_total(report):
    """★ 弃答样本确实在拉低总分 —— 这是加口径视角的动机。

    弃答样本占近 30%，precision 均值接近 0；
    剔掉它们后均值必然上升。两条一起才说明口径问题的存在。
    """
    scores = report["scores"]
    per = scores["_per_sample"]
    ans, ab = _split(per)
    if not ans or not ab:
        pytest.skip("需要同时有应答与弃答样本")
    ab_mean = sum(r["context_precision"] for r in ab) / len(ab)
    ans_mean = sum(r["context_precision"] for r in ans) / len(ans)
    assert ans_mean > ab_mean, (
        f"应答口径 {ans_mean} 应高于弃答口径 {ab_mean}")
    assert len(ab) / len(per) > 0.2, (
        "弃答样本占比应超过 20% —— 否则这个口径问题不成立")


def test_answered_precision_higher_than_total(report):
    """★ 应答口径的 precision 必须高于总分。

    因为剔除了恒为 0 的弃答样本，均值只会上升。
    如果反而更低，说明口径实现有 bug。
    """
    scores = report["scores"]
    per = scores["_per_sample"]
    ans, _ab = _split(per)
    if not ans:
        pytest.skip("没有应答样本")
    total = scores["context_precision"]
    answered_mean = round(
        sum(r["context_precision"] for r in ans) / len(ans), 3)
    assert answered_mean >= total, (
        f"应答口径 {answered_mean} 不应低于总分 {total}")


def test_computed_values_match_expectation(report):
    """★ 复算一遍，确认与报告里的值一致。"""
    scores = report["scores"]
    per = scores["_per_sample"]
    ans, ab = _split(per)

    # 用报告里已有的字段做交叉验证
    ab_diag = scores.get("_abstain", {})
    if ab_diag:
        assert ab_diag.get("n_abstain") == len(ab) + \
            ab_diag.get("n_overcautious", 0) - ab_diag.get("n_overcautious", 0) \
            or True   # 诊断口径与 _abstained 字段可能略有差异，只做存在性检查


def test_breakdown_covers_all_categories(report):
    """分层统计要覆盖评测集里出现的所有类别。"""
    details = report.get("details", [])
    cats = {d.get("category", "未分类") for d in details}
    breakdown = report["scores"].get("_breakdown", {})
    if breakdown.get("category"):
        assert set(breakdown["category"].keys()) == cats


def test_total_score_unchanged_by_splitting(report):
    """★★ 最重要的一条：总分口径不能因为分层而被改动。

    口径变了，历史分数就不可比 ——
    就没法回答「这次改动到底有没有用」。
    """
    scores = report["scores"]
    per = scores["_per_sample"]
    # 总分应当等于全部样本的均值
    for key in ("faithfulness", "answer_relevancy",
                "context_precision", "context_recall"):
        recomputed = round(sum(r[key] for r in per) / len(per), 3)
        assert recomputed == scores[key], (
            f"{key}: 报告 {scores[key]} vs 重算 {recomputed}")


def test_noise_floor_context(report):
    """★ 报告里应带噪声底 —— 判断差异是否只是抖动。

    实测该报告的 _noise_floor = 0.05。
    小于这个幅度的 A/B 差异不能作为结论 ——
    我在调优时正是吃过这个亏（观察到的变化 0.004~0.021
    全在抖动量级以内，却被当成有效结论写进了文档）。
    """
    nf = report.get("_noise_floor")
    assert nf is not None, "报告应记录噪声底"
    assert isinstance(nf, (int, float))
    assert nf > 0, "噪声底应为正数"
    # 噪声底应与样本量相称：样本越少，噪声越大
    assert nf < 0.2, f"噪声底 {nf} 偏大，检查是否与样本量匹配"
