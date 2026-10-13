# train_classifier.py
"""L2 意图分类器训练脚本（供 common/router.py 的级联路由 L2 层使用）。

用法：
    python train_classifier.py                       # 用现有 data/intent_train.jsonl 训练
    python train_classifier.py --regen-data          # 重新生成「内置扩样 + 种子」数据并训练
    python train_classifier.py --data data/intent_train.jsonl --out models/intent_clf.joblib

流程：
  1) 内置模板扩样 _expand_templates()：4 个意图类（rag/vision/voice/doc）
     各 25+ 个中文模板句式 × 7 个关键词槽位 + 固定句，扩到每类 ≥150 条；
  2) --regen-data 或数据文件不存在时，把「扩样数据 + 种子集」写出为 jsonl；
  3) 训练 TfidfVectorizer(char_wb, 1~2gram) + LogisticRegression，
     train_test_split(test_size=0.15, stratify=y)，打印 classification_report；
  4) joblib 保存 Pipeline 到 models/intent_clf.joblib（labels 由 Pipeline.classes_ 自带）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import List, Tuple

try:
    from common.config import project_root
except ImportError:  # 直接以脚本方式从其他目录运行时兜底
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common.config import project_root

VALID_LABELS = ("rag", "vision", "voice", "doc")

# ==================== 内置模板扩样库 ====================
# 每类 templates（均含 {kw} 槽位）× fillers 槽位填充，另加固定句 extras。
_TEMPLATE_BANK = {
    "vision": {
        "fillers": ["灭火器", "消火栓", "安全出口", "消防通道",
                    "消防水带", "消防柜", "应急照明灯"],
        "templates": [
            "看看这张图里的{kw}有没有问题",
            "检查一下{kw}是否合格",
            "帮我看看{kw}有没有隐患",
            "这张照片里的{kw}符合要求吗",
            "识别{kw}是否存在遮挡",
            "这个视频里{kw}被堵住了吗",
            "看图判断{kw}能不能正常使用",
            "检查{kw}的完好情况",
            "帮我分析这张{kw}的现场图片",
            "画面里{kw}的状态怎么样",
            "看看{kw}周围有没有堆放杂物",
            "帮我检查图片中{kw}的配置数量",
            "帮我看一下{kw}的照片",
            "视频里的{kw}有没有被占用",
            "识别这张图里的{kw}",
            "看一下现场画面中{kw}的设置情况",
            "这张图里{kw}的安装位置对吗",
            "帮我看看{kw}是不是过期了",
            "检查视频画面里的{kw}",
            "照片上{kw}的外观完好吗",
            "看看图中{kw}有没有被圈占",
            "帮我看看这段视频中的{kw}",
            "{kw}前堆了东西吗，帮我看下图",
            "帮我判断图里{kw}是否被损坏",
            "这张监控画面里{kw}正不正常",
            "给我看看{kw}的现场图",
        ],
    },
    "voice": {
        "fillers": ["录音", "音频", "执法记录仪录音", "现场录音",
                    "通话录音", "询问录音", "语音"],
        "templates": [
            "把这段{kw}转成文字",
            "听一下这段{kw}",
            "帮我转写这段{kw}",
            "识别这段{kw}里的对话",
            "这段{kw}里当事人说了什么",
            "把{kw}整理成文字材料",
            "帮我听{kw}里提到的关键信息",
            "转写{kw}，要带时间戳",
            "这段{kw}说了哪些内容",
            "把{kw}里的当事人陈述摘出来",
            "听一下{kw}里有没有程序性告知",
            "帮我转成文字，{kw}有点长",
            "识别{kw}，输出成书面记录",
            "把{kw}转写出来发给我",
            "这段{kw}帮我听听有没有矛盾之处",
            "把{kw}转成可供笔录引用的文字",
            "听一下{kw}里当事人承认违法事实了吗",
            "帮我把{kw}转写并总结要点",
            "识别这段{kw}里的身份证号和地址",
            "这段{kw}转文字后给我核对",
            "把{kw}里的承诺内容记下来",
            "转写{kw}最后两分钟",
            "帮我听听{kw}里的执法程序合不合规",
            "这段{kw}里的金额和期限是多少",
            "把{kw}内容转成文字版存档",
        ],
    },
    "doc": {
        "fillers": ["责令限期改正通知书", "立案审批表", "询问笔录",
                    "行政处罚决定书", "行政处罚告知书", "现场检查记录", "听证告知书"],
        "templates": [
            "帮我生成一份{kw}",
            "给我出一份{kw}",
            "写一份{kw}",
            "生成一份{kw}，今天要用",
            "帮我起草{kw}",
            "把检查情况写进{kw}",
            "出一份{kw}给当事人",
            "我要办{kw}",
            "按这次检查结果生成{kw}",
            "帮我把{kw}做出来",
            "写个{kw}，格式按规范来",
            "生成{kw}并发到我邮箱",
            "把当事人的情况填进{kw}",
            "出一份正式的{kw}",
            "帮我准备{kw}",
            "生成{kw}，附上法律依据",
            "写{kw}，罚款金额五千元",
            "把询问内容整理成{kw}",
            "今天必须出{kw}",
            "帮我出{kw}，一式两份",
            "生成{kw}的电子版和PDF",
            "把昨天的检查结果做成{kw}",
            "领导让我出一份{kw}",
            "根据案件事实生成{kw}",
            "写{kw}，把整改期限写进去",
        ],
    },
    "rag": {
        "fillers": ["堵塞消防通道", "灭火器过期没换", "楼道堆放杂物",
                    "电动车进楼充电", "消防栓没水", "无证动火作业", "安全出口上锁"],
        "templates": [
            "{kw}怎么处罚",
            "{kw}违反了哪条规定",
            "查一下{kw}相关的法律条文",
            "{kw}该怎么办",
            "{kw}的处罚标准是多少",
            "遇到{kw}该走什么程序",
            "{kw}归哪个部门管",
            "帮我查查{kw}的法律依据",
            "{kw}罚款金额一般是多少",
            "{kw}需要责令改正吗",
            "{kw}能不能临时查封",
            "咨询一下{kw}怎么处理",
            "{kw}的整改期限是多久",
            "{kw}会立案吗",
            "解释一下{kw}对应的法条",
            "{kw}的举报流程是什么",
            "单位存在{kw}会有什么后果",
            "{kw}适用的法律条款有哪些",
            "如果{kw}拒不改正怎么加重处罚",
            "{kw}的执法主体是谁",
            "帮我查{kw}的典型案例",
            "{kw}允许从轻处罚吗",
            "{kw}现场检查要注意什么",
            "法规里对{kw}是怎么描述的",
            "{kw}超过追责时效还能处理吗",
        ],
    },
}

# 固定句（不走槽位填充），贴近真实问句风格
_EXTRA_SAMPLES = {
    "rag": [
        "消防法第六十条内容是什么",
        "火灾事故责任认定依据哪条规定",
        "消防监督检查多久开展一次",
        "消防验收需要准备哪些材料",
        "公众聚集场所投入使用前要办什么手续",
        "临时查封的条件和期限是多久",
        "开业前消防安全检查合格证怎么办",
        "消防行政处罚的追诉时效是多久",
    ],
    "vision": [
        "看看这张图有没有隐患",
        "检查灭火器照片",
        "这个视频里安全出口通不通",
        "识别图片里的消防设施",
        "帮我分析这段视频里的消防隐患",
        "这张照片里的防火门状态正常吗",
        "看图判断这个灭火器压力表是否合格",
        "这张画面里应急照明灯亮不亮",
    ],
    "voice": [
        "听一下这段录音",
        "把执法记录仪音频转成文字",
        "这是昨天的执法录音，帮我整理成文字",
        "把这段通话录音转成书面记录",
        "帮我听一下询问过程的录音",
        "音频里当事人身份证号报一下",
        "把这段语音留言转成文字发我",
    ],
    "doc": [
        "帮我生成一份立案审批表",
        "写个询问笔录",
        "给我出一份责令限期改正通知书",
        "把昨天的检查情况写进消防监督检查记录",
        "帮我起草一份临时查封决定书",
        "写一份行政处罚决定书，罚款五千元",
        "帮我出一份听证告知书",
    ],
}


def _expand_templates() -> List[Tuple[str, str]]:
    """内置模板扩样：每类 25+ 模板 × 7 槽位 + 固定句，随机打乱（seed=42）。

    每类产出 ≥150 条（25×7=175 起步），返回 [(text, label), ...]。
    """
    rng = random.Random(42)
    samples: List[Tuple[str, str]] = []
    for label, bank in _TEMPLATE_BANK.items():
        seen = set()
        for text in _EXTRA_SAMPLES.get(label, []):
            if text not in seen:
                seen.add(text)
                samples.append((text, label))
        for tpl in bank["templates"]:
            for kw in bank["fillers"]:
                text = tpl.format(kw=kw)
                if text not in seen:  # 模板间撞句去重
                    seen.add(text)
                    samples.append((text, label))
    rng.shuffle(samples)
    return samples


# ==================== 种子数据读写 ====================
def load_seed(path: str) -> List[Tuple[str, str]]:
    """读取 jsonl 种子数据（每行 {"text","label"}），忽略空行/非法行。"""
    if not os.path.exists(path):
        return []
    samples: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[警告] {path} 第 {line_no} 行不是合法 JSON，已跳过")
                continue
            text = str(item.get("text", "")).strip()
            label = str(item.get("label", "")).strip().lower()
            if text and label in VALID_LABELS:
                samples.append((text, label))
            else:
                print(f"[警告] {path} 第 {line_no} 行 text/label 非法，已跳过")
    return samples


def write_jsonl(samples: List[Tuple[str, str]], path: str) -> None:
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for text, label in samples:
            f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")


# ==================== 训练与保存 ====================
def train(samples: List[Tuple[str, str]], out_path: str) -> None:
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        print("[错误] 缺少训练依赖：scikit-learn / joblib")
        print("请先执行：pip install scikit-learn joblib")
        print(f"详细信息：{exc}")
        sys.exit(1)

    texts = [t for t, _ in samples]
    labels = [lb for _, lb in samples]
    print(f"[训练] 样本总数 {len(texts)}，类别分布 {dict(Counter(labels))}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f"[训练] train={len(X_train)} test={len(X_test)}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\n===== classification_report =====")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"[保存] 模型已写入 {out_path}")
    print(f"[保存] 内置标签（pipeline.classes_）：{list(pipeline.classes_)}")
    print("[提示] common/router.py 的 L2 层会自动从 models/intent_clf.joblib 热加载")


# ==================== 主流程 ====================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="L2 意图分类器训练脚本（Tfidf char_wb + LogisticRegression）"
    )
    parser.add_argument("--data", default="data/intent_train.jsonl",
                        help="训练数据 jsonl 路径（相对路径基于项目根）")
    parser.add_argument("--out", default="models/intent_clf.joblib",
                        help="模型输出路径（相对路径基于项目根）")
    parser.add_argument("--regen-data", action="store_true",
                        help="重新生成「内置扩样 + 种子」数据并写回 --data 文件")
    args = parser.parse_args()

    data_path = args.data if os.path.isabs(args.data) else os.path.join(project_root(), args.data)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(project_root(), args.out)

    seed = load_seed(data_path)
    expanded = _expand_templates()
    exp_counter = Counter(lb for _, lb in expanded)
    print(f"[扩样] 内置模板扩样 {len(expanded)} 条，分布 {dict(exp_counter)}")
    for lb in VALID_LABELS:
        if exp_counter.get(lb, 0) < 150:
            print(f"[警告] 扩样后类别 {lb} 不足 150 条（{exp_counter.get(lb, 0)}）")

    if args.regen_data or not os.path.exists(data_path):
        dataset = expanded + seed
        write_jsonl(dataset, data_path)
        print(f"[数据] 已写出 {data_path}：扩样 {len(expanded)} 条 + 种子 {len(seed)} 条")
    else:
        dataset = seed
        print(f"[数据] 使用现有 {data_path}（{len(dataset)} 条）；"
              f"如需合并内置扩样请加 --regen-data")

    if len(dataset) < 10:
        print("[错误] 训练样本过少，无法训练")
        sys.exit(1)
    for lb in VALID_LABELS:
        n = sum(1 for _, lb2 in dataset if lb2 == lb)
        if n < 2:
            print(f"[错误] 类别 {lb} 样本不足 2 条，stratify 划分无法进行")
            sys.exit(1)

    train(dataset, out_path)


if __name__ == "__main__":
    main()
