# test_eval_baseline.py
"""
基线对比机制的测试。

背景（2026-10-15 实际踩到的坑）
--------------------------------
v6/v7/v8 三轮跑完，总分看起来"有涨有跌"，很自然会解读成优化时灵时不灵。
但逐条比对 contexts 后发现三轮的 47 条上下文**逐字完全相同**——索引没动过，
v8 是次空跑。而即便变量隔离了，LLM-as-judge 本身也会抖：
同输入两次跑，context_recall 能差 0.021。

于是本模块提供三个检查，缺一不可：
    Q1 变量隔离了吗？   contexts 指纹是否与基线一致
    Q2 是空跑吗？       上下文一致却分数变了 → 差异 100% 来自 LLM 随机性
    Q3 变化超噪声吗？   差值是否大于该指标的抖动幅度

这里锁住的重点是**别再把空跑/噪声误读成优化效果**——
那会导致往完全错误的方向继续调优。

实现加载方式：``local_eval.py`` 在模块顶层 import 了 ragas / config 等重依赖，
桌面环境装不上faiss / openai / numpy。与 ``test_eval_abstain.py`` 一致，
这里从源码里摘出「基线对比」区块单独执行——该区块只依赖标准库
（json / os / hashlib），既能真正跑到逻辑，又不受重依赖影响。
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

_EVAL = pathlib.Path(__file__).resolve().parents[1] / "agent1" / "eval" / "local_eval.py"


def _load():
    """从源码里摘出基线对比区块，避开重依赖 import。"""
    src = _EVAL.read_text(encoding="utf-8")
    start = src.find("# ==================== 基线对比")
    end = src.find("# ==================== 主流程")
    assert start > 0 and end > start, "基线对比区块定位失败，代码结构变了？"
    ns: dict = {}
    exec(compile("import json\nimport os\nimport hashlib\n" + src[start:end],
                 "local_eval_slice", "exec"), ns)
    return ns


_NS = _load()
_contexts_fingerprint = _NS["_contexts_fingerprint"]
compare_with_baseline = _NS["compare_with_baseline"]
print_baseline_report = _NS["print_baseline_report"]
_NOISE_FLOOR_UNKNOWN = _NS["_NOISE_FLOOR_UNKNOWN"]

KEYS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ==================== fixtures ====================
def _make_report(scores, contexts_map=None, n=3, top_k=5, fingerprint=None):
    """构造一份结构完整的报告。"""
    if contexts_map is None:
        contexts_map = {f"q{i}": [f"ctx-{i}-a", f"ctx-{i}-b"] for i in range(n)}
    details = [
        {
            "question": q,
            "answer": "答案" + q,
            "contexts": ctx,
            "ground_truth": "gt",
            "category": "直查法条",
            "difficulty": "easy",
        }
        for q, ctx in contexts_map.items()
    ]
    rep = {
        "timestamp": "2026-10-15 12:00:00",
        "num_samples": len(details),
        "top_k": top_k,
        "scores": dict(scores),
        "details": details,
    }
    if fingerprint is not None:
        rep["_contexts_fingerprint"] = fingerprint
    return rep


@pytest.fixture
def base_scores():
    return {
        "faithfulness": 0.985,
        "answer_relevancy": 0.706,
        "context_precision": 0.221,
        "context_recall": 0.548,
        "_per_sample": [],
    }


def _write(tmp_path, name, report):
    p = tmp_path / name
    p.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return str(p)


# ==================== 指纹 ====================
class TestContextsFingerprint:
    def test_identical_inputs_give_identical_fingerprint(self):
        a = _make_report({}, n=2)
        b = _make_report({}, n=2)
        assert _contexts_fingerprint(a["details"]) == _contexts_fingerprint(b["details"])

    def test_fingerprint_is_stable_across_calls(self):
        """同一份 details 反复算必须一致 —— 否则指纹无法用于跨轮对比。"""
        rep = _make_report({}, n=2)
        fps = {_contexts_fingerprint(rep["details"]) for _ in range(5)}
        assert len(fps) == 1

    def test_context_change_changes_fingerprint(self):
        """上下文变了指纹必须变 —— 这是"检出空跑"的前提。"""
        a = _make_report({}, n=2)
        b = _make_report({}, n=2)
        b["details"][0]["contexts"][0] = "被改过的上下文"
        assert _contexts_fingerprint(a["details"]) != _contexts_fingerprint(b["details"])

    def test_fingerprint_ignores_answers_and_scores(self):
        """指纹只由检索结果决定：答案变了、分数变了，指纹不能变。

        这是设计要点 —— 因为 LLM 生成有随机性，若把答案算进指纹，
        任何一轮都会被判成"变量已隔离"，空跑就检不出来了。
        """
        a = _make_report({}, n=2)
        b = _make_report({}, n=2)
        b["details"][0]["answer"] = "完全不同的答案"
        b["scores"]["faithfulness"] = 0.1
        assert _contexts_fingerprint(a["details"]) == _contexts_fingerprint(b["details"])

    def test_fingerprint_is_short_and_hex(self):
        fp = _contexts_fingerprint(_make_report({}, n=2)["details"])
        assert len(fp) == 16
        int(fp, 16)          # 能被当十六进制解析

    def test_empty_details_does_not_crash(self):
        assert isinstance(_contexts_fingerprint([]), str)


# ==================== 对比：可比性 ====================
class TestComparability:
    def test_same_n_and_k_is_comparable(self, tmp_path, base_scores):
        b = _write(tmp_path, "b.json", _make_report(base_scores))
        cur = _make_report(base_scores)
        r = compare_with_baseline(cur, b)
        assert r["checks"]["comparable"]["passed"] is True

    def test_different_num_samples_is_not_comparable(self, tmp_path, base_scores):
        """条数不同 → 分数没有对比意义，必须报警。"""
        b = _write(tmp_path, "b.json", _make_report(base_scores, n=5))
        cur = _make_report(base_scores, n=3)
        r = compare_with_baseline(cur, b)
        c = r["checks"]["comparable"]
        assert c["passed"] is False
        assert c["baseline_num_samples"] == 5
        assert c["current_num_samples"] == 3

    def test_different_top_k_is_not_comparable(self, tmp_path, base_scores):
        """top_k 不同是 v8踩过的坑：上下文 5→8 条，precision 分母变了。

        实测 v8 的 cp 从 0.520 掉到 0.325，完全是 top_k 5→8 造成的，
        不是检索退化。这个检查就是为了挡住这种误读。
        """
        b = _write(tmp_path, "b.json", _make_report(base_scores, top_k=5))
        cur = _make_report(base_scores, top_k=8)
        r = compare_with_baseline(cur, b)
        assert r["checks"]["comparable"]["passed"] is False
        assert r["checks"]["comparable"]["current_top_k"] == 8

    def test_missing_baseline_file_returns_error(self):
        r = compare_with_baseline(_make_report({}), "/nonexistent/xx.json")
        assert "error" in r


# ==================== 对比：空跑检测（核心）===================
class TestNullRunDetection:
    def test_identical_contexts_with_score_change_flags_null_run(
            self, tmp_path, base_scores):
        """v8 的真实情形：索引没动，分数却变了 → 差异 100% 是噪声。"""
        b = _write(tmp_path, "b.json", _make_report(base_scores, n=3))
        cur_scores = dict(base_scores)
        cur_scores["context_recall"] = 0.569     # v8 实测值
        cur = _make_report(cur_scores, n=3)
        r = compare_with_baseline(cur, b)
        nr = r["checks"]["null_run"]
        assert nr["detected"] is True
        assert nr["passed"] is False
        assert "LLM" in nr["note"]

    def test_identical_contexts_with_identical_scores_is_not_null_run(
            self, tmp_path, base_scores):
        """上下文一致、分数也一致 —— 那是真正可复现的结果，不算空跑。"""
        b = _write(tmp_path, "b.json", _make_report(base_scores, n=3))
        cur = _make_report(dict(base_scores), n=3)
        r = compare_with_baseline(cur, b)
        assert r["checks"]["null_run"]["detected"] is False
        assert r["checks"]["null_run"]["passed"] is True

    def test_changed_contexts_is_not_null_run(self, tmp_path, base_scores):
        """检索层真变了 → 分数差异有归因，不是空跑。"""
        b = _write(tmp_path, "b.json", _make_report(base_scores, n=3))
        cur = _make_report(
            dict(base_scores, context_recall=0.7), n=3,
            contexts_map={f"q{i}": [f"NEW-{i}", "b"] for i in range(3)},
        )
        r = compare_with_baseline(cur, b)
        assert "null_run" not in r["checks"]

    def test_variable_isolation_reports_correct_relation(
            self, tmp_path, base_scores):
        b = _write(tmp_path, "b.json", _make_report(base_scores, n=3))
        cur = _make_report(dict(base_scores), n=3)
        r = compare_with_baseline(cur, b)
        vi = r["checks"]["variable_isolation"]
        assert vi["contexts_identical"] is True
        assert vi["baseline_fingerprint"] == vi["current_fingerprint"]

    def test_baseline_stored_fingerprint_is_preferred(self, tmp_path, base_scores):
        """基线报告里存了指纹就直接用，不必重算 —— 省一次全量哈希。"""
        b = _make_report(base_scores, n=3)
        b["_contexts_fingerprint"] = "deadbeefdeadbeef"
        bp = _write(tmp_path, "b.json", b)
        cur = _make_report(dict(base_scores), n=3)
        r = compare_with_baseline(cur, bp)
        assert r["checks"]["variable_isolation"]["baseline_fingerprint"] == "deadbeefdeadbeef"


# ==================== 对比：噪声判定 ====================
class TestNoiseVerdict:
    def test_large_delta_exceeds_noise(self, tmp_path, base_scores):
        """rel +0.127（prompt 修复的真实效果）应判定为真实变化。"""
        b = _write(tmp_path, "b.json", _make_report(base_scores, n=3))
        cur = _make_report(dict(base_scores, answer_relevancy=0.833), n=3)
        r = compare_with_baseline(cur, b)
        v = r["verdict"]["answer_relevancy"]
        assert v["exceeds_noise"] is True
        assert v["verdict"] == "真实变化"

    def test_small_delta_is_within_noise(self, tmp_path, base_scores):
        """prec +0.004 这类小数变化应被判为噪声 —— 实测它就是抖动。"""
        b = _write(tmp_path, "b.json", _make_report(base_scores, n=3))
        cur = _make_report(dict(base_scores, context_precision=0.225), n=3)
        r = compare_with_baseline(cur, b)
        v = r["verdict"]["context_precision"]
        assert v["exceeds_noise"] is False
        assert "不可归因" in v["verdict"]

    def test_uses_measured_noise_floor_when_available(
            self, tmp_path, base_scores):
        """基线含 _noise_floor 时用实测值，而不是经验保守值。"""
        b = _make_report(base_scores, n=3)
        b["_noise_floor"] = {k: 0.001 for k in KEYS}
        bp = _write(tmp_path, "b.json", b)
        cur = _make_report(dict(base_scores, context_precision=0.225), n=3)
        r = compare_with_baseline(cur, bp)
        nz = r["checks"]["noise"]
        assert nz["measured"] is True
        assert nz["source"] == "基线实测"
        # 噪声底 0.001 时，+0.004 就超出噪声了
        assert r["verdict"]["context_precision"]["exceeds_noise"] is True

    def test_falls_back_to_conservative_floor(self, tmp_path, base_scores):
        """基线没实测噪声底时用保守阈值，并明确提示未实测。"""
        b = _write(tmp_path, "b.json", _make_report(base_scores, n=3))
        cur = _make_report(dict(base_scores), n=3)
        r = compare_with_baseline(cur, b)
        nz = r["checks"]["noise"]
        assert nz["measured"] is False
        assert nz["source"].startswith("经验保守值")
        assert nz["noise_floor"]["faithfulness"] == _NOISE_FLOOR_UNKNOWN
        assert nz["passed"] is False
        assert "跑两次" in nz["note"]

    def test_default_floor_exceeds_observed_jitter(self):
        """★ 锁定一条设计决策：默认阈值必须大于实测抖动幅度。

        起因是真踩过这个坑——最初拿 v7/v8 的差值（recall 0.021）当阈值设成
        0.02，结果 v7->v8 的 recall 差值恰好 0.021，卡在边界上，
        `abs(0.021) > 0.02` 判真，测试直接失败。

        根因是**单次 A/B 差值只是抖动幅度的下界，不是幅度本身**。
        差值分布有方差，两次恰好接近不代表常态如此。

        这条测试的作用是：若有人把阈值调回 0.02，必须先解释清楚
        为什么单次观测可以当阈值用。
        """
        # 实测抖动（v7 vs v8，同上下文纯 LLM 随机性）
        observed_jitter = {
            "faithfulness": 0.008,
            "answer_relevancy": 0.013,
            "context_precision": 0.004,
            "context_recall": 0.021,
        }
        for k, v in observed_jitter.items():
            assert _NOISE_FLOOR_UNKNOWN > v * 2, (
                f"{k}: 默认阈值 {_NOISE_FLOOR_UNKNOWN} 相对实测抖动 {v} "
                f"余量不足 2 倍，会像 v7/v8 那样卡在边界上"
            )

    def test_deltas_cover_all_four_metrics(self, tmp_path, base_scores):
        b = _write(tmp_path, "b.json", _make_report(base_scores, n=3))
        cur = _make_report(dict(base_scores), n=3)
        r = compare_with_baseline(cur, b)
        assert set(r["deltas"]) == set(KEYS)
        assert set(r["verdict"]) == set(KEYS)


# ==================== 打印不崩 ====================
class TestPrintBaselineReport:
    """打印函数是给人看的，但崩掉会毁掉整轮评测结果，必须能扛各种残缺输入。"""

    def _capture(self, capsys, cmp):
        print_baseline_report(cmp)
        return capsys.readouterr().out

    def test_prints_null_run_warning(self, capsys, tmp_path, base_scores):
        b = _write(tmp_path, "b.json", _make_report(base_scores, n=3))
        cur = _make_report(dict(base_scores, context_recall=0.569), n=3)
        out = self._capture(capsys, compare_with_baseline(cur, b))
        assert "空跑" in out

    def test_prints_incomparable_warning_with_both_values(
            self, capsys, tmp_path, base_scores):
        b = _write(tmp_path, "b.json", _make_report(base_scores, n=5, top_k=5))
        cur = _make_report(base_scores, n=3, top_k=8)
        out = self._capture(capsys, compare_with_baseline(cur, b))
        assert "不可比" in out
        assert "5" in out and "8" in out       # 两个 top_k 都要出现

    def test_prints_each_metric_row(self, capsys, tmp_path, base_scores):
        b = _write(tmp_path, "b.json", _make_report(base_scores, n=3))
        cur = _make_report(dict(base_scores, answer_relevancy=0.833), n=3)
        out = self._capture(capsys, compare_with_baseline(cur, b))
        for k in KEYS:
            assert k in out

    def test_warns_when_all_within_noise(self, capsys, tmp_path, base_scores):
        b = _write(tmp_path, "b.json", _make_report(base_scores, n=3))
        cur = _make_report(dict(base_scores, context_precision=0.222), n=3)
        out = self._capture(capsys, compare_with_baseline(cur, b))
        assert "0/4" in out

    def test_survives_missing_checks(self, capsys):
        """残缺 dict 也不能崩 —— 崩了就等于评测白跑。"""
        print_baseline_report({"baseline_file": "x.json"})
        assert "x.json" in capsys.readouterr().out

    def test_survives_empty_verdict(self, capsys, tmp_path, base_scores):
        b = _write(tmp_path, "b.json", _make_report(base_scores, n=3))
        cur = _make_report(dict(base_scores), n=3)
        r = compare_with_baseline(cur, b)
        r["verdict"] = {}
        self._capture(capsys, r)      # 不抛异常即可


# ==================== 真实报告的回归锁定 ====================
class TestAgainstRealReports:
    """用v7/v8 真实报告锁住"空跑能被检出"这个行为。

    这两份报告是 2026-10-15 实跑产生的，47 条上下文逐字相同，
    是最典型的空跑样本。"""
    REPO = pathlib.Path(__file__).resolve().parents[1]

    def _path(self, name):
        p = self.REPO / name
        if not p.is_file():
            pytest.skip(f"{name} 不存在")
        return p

    def test_v7_and_v8_contexts_are_identical(self):
        """前提断言：这两轮确实是空跑。若将来重跑导致上下文变化，
        本测试会失败并提醒同步更新下面的判定。"""
        with open(self._path("report_v7.json"), encoding="utf-8") as f:
            v7 = json.load(f)
        with open(self._path("report_v8.json"), encoding="utf-8") as f:
            v8 = json.load(f)
        assert _contexts_fingerprint(v7["details"]) == \
            _contexts_fingerprint(v8["details"]), \
            "v7/v8 上下文已不同，空跑检测的前提变了，需复核本文件"

    def test_real_null_run_is_flagged(self, tmp_path):
        with open(self._path("report_v7.json"), encoding="utf-8") as f:
            v7 = json.load(f)
        with open(self._path("report_v8.json"), encoding="utf-8") as f:
            v8 = json.load(f)
        b = _write(tmp_path, "v7.json", v7)
        r = compare_with_baseline(v8, b)
        assert r["checks"]["null_run"]["detected"] is True
        # rec +0.021 / faith +0.008 / prec +0.005 都在噪声内
        assert r["verdict"]["context_recall"]["exceeds_noise"] is False
        assert r["verdict"]["faithfulness"]["exceeds_noise"] is False

    def test_real_small_deltas_are_all_within_noise(self, tmp_path):
        """v7->v8 三个指标的小幅变化全部落在噪声内，不该被当成改进。"""
        with open(self._path("report_v7.json"), encoding="utf-8") as f:
            v7 = json.load(f)
        with open(self._path("report_v8.json"), encoding="utf-8") as f:
            v8 = json.load(f)
        b = _write(tmp_path, "v7.json", v7)
        r = compare_with_baseline(v8, b)
        exceeding = [k for k, v in r["verdict"].items() if v["exceeds_noise"]]
        assert exceeding == [], f"这些变化超出了经验噪声底，需复核: {exceeding}"
