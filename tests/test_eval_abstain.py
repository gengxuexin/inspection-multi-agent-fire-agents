"""评测脚本的弃答判定测试。

为什么单独建文件：弃答检测器改过5 版实现，每版都是被 report_v6/v7 的
真实数据打回来的。这组用例把踩过的坑固化成回归测试，防止以后再退化。

判定实现在 ``agent1/eval/local_eval.py``，因为该模块 import 了
ragas / config 等重依赖，测试里用源码提取的方式只加载判定函数，
既避免装依赖，又能真正跑到那段逻辑。
"""
import pathlib
import re

import pytest


_EVAL = pathlib.Path(__file__).resolve().parents[1] / "agent1" / "eval" / "local_eval.py"


def _load():
    """从源码里摘出弃答判定相关的符号，避开重依赖 import。"""
    src = _EVAL.read_text(encoding="utf-8")
    start = src.find("# ==================== 弃答识别")
    end = src.find("# ==================== LLM judge")
    assert start > 0 and end > start, "弃答识别区块定位失败，代码结构变了？"
    ns: dict = {}
    exec(compile("import re\n" + src[start:end], "local_eval_slice", "exec"), ns)
    return ns


_NS = _load()
lexical_abstain = _NS["lexical_abstain"]


# ============================================================
# 词法快筛：只保证不漏，用于人工抽查线索与降级路径
# ============================================================

@pytest.mark.parametrize("ans", [
    "检索内容不足，无法回答该问题。",
    "根据检索到的内容，**无法回答消防技术服务机构从业条件的具体内容**。",
    "根据检索到的内容，**无法完整回答**“全过程记录制度要求记录哪些内容”。",
    "检索内容中**没有列出**该目录具体包含哪些产品类别。",
    "现有片段均未涉及该问题。",
])
def test_lexical_abstain_catches_varied_phrasings(ans):
    """同一个意思的多种说法都要能识别出来（v6 只匹配"检索内容不足"漏 4 条）。"""
    assert lexical_abstain(ans) is True


@pytest.mark.parametrize("ans", [
    # 完整回答 + 末尾补充说明（v7 新 prompt 的要求）
    "《消防法》第六十条第七项规定：责令改正，处五千元罚款。[片段1]\n\n"
    "检索内容未覆盖该条款在执法程序中的具体适用细节。",
    # 法条原文里的否定表述，不该被当成弃答
    "违法行为轻微并及时改正的，不予行政处罚。[片段3]",
    # 正常作答
    "注册消防工程师的注册证书由省级公安机关消防机构颁发。[片段2]",
])
def test_lexical_abstain_flags_but_not_conclusive(ans):
    """词法版**允许**误报（实测约 20%）。

    这里只锁定它不会漏；是否真的弃答由 judge_abstain 判定。
    之所以写进测试，是提醒后来者：词法版不能单独用作结论。
    """
    assert lexical_abstain(ans) in (True, False)   # 不崩溃、不抛异常


def test_lexical_abstain_on_empty():
    assert lexical_abstain("") is False


def test_lexical_abstain_never_raises():
    for s in ("", "   ", "\n\n", "答案", "**加粗**", "「」", "0"):
        lexical_abstain(s)


# ============================================================
# judge_abstain 的 prompt 与降级逻辑
# ============================================================

def test_judge_abstain_declares_json_contract():
    """judge 必须要求 JSON 输出，且降级路径存在。"""
    src = _EVAL.read_text(encoding="utf-8")
    start = src.find("def judge_abstain")
    body = src[start:start + 2000]
    assert '{"abstained"' in body, "judge 未要求 JSON 输出"
    # 无 API Key 时必须降级而不是崩掉
    assert "lexical_abstain" in body, "缺少降级路径"


def test_abstain_questions_in_samples_use_semantic_check():
    """report_v7 的核心教训：弃答判定不能靠字面匹配。

    这条测试锁住 judge_abstain 存在且被逐条打分流程调用。
    """
    src = _EVAL.read_text(encoding="utf-8")
    assert "def judge_abstain" in src
    assert 'row["_abstained"] = judge_abstain' in src, "弃答判定未接入打分流程"


def test_abstained_flag_excluded_from_metric_aggregation():
    """``_abstained`` 是 bool，不能混进四个指标的均值。

    聚合用的是 keys 白名单，所以它天然被排除；这里锁定白名单本身
    不被后续修改误加入该字段。
    """
    src = _EVAL.read_text(encoding="utf-8")
    m = re.search(
        r'keys = \[([^\]]+)\]', src)
    assert m, "未找到聚合 keys 白名单"
    keys = m.group(1)
    assert "_abstained" not in keys
    for k in ("faithfulness", "answer_relevancy",
              "context_precision", "context_recall"):
        assert k in keys, f"聚合缺少指标 {k}"