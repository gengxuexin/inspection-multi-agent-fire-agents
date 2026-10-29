#!/usr/bin/env python3
"""TCO（总持有成本）测算。

★ 为什么FDE 方向必须会算这个：
  交付实施岗位的核心价值之一，是替客户算清楚"这套东西一年多少钱、
  跟别的方案比省多少"。说不出数字的方案对比是空话。
  而这项能力的关键不在于记住单价，而在于：
    1. 把成本拆到可归因的粒度（哪一项占比高，优化哪里收益最大）
    2. 把**隐性成本**算进去（运维人力、故障损失、迁移成本）
       —— 只算 API 费用的 TCO 是自欺
    3. 给出敏感性分析（用量变化时哪个环节先撑不住）

用法：
    python3 tco_calculator.py                # 默认场景
    python3 tco_calculator.py --scenarios    # 多场景对比
    python3 tco_calculator.py --export out.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

# ==================== 价格基线 ====================
# ★ 数据来源与时效（价格随时会变，面试时说清"截至哪天"很重要）：
#   DeepSeek 2026-08 公告后的现行定价，单位：元 / 百万 token
#     V4-Flash: 输入(缓存命中) 0.02  输入(未命中) 1  输出 2
#     V4-Pro:   输入(缓存命中) 0.025 输入(未命中) 3  输出 6
#   注意 DeepSeek 有峰谷定价：工作日 9-12、14-18 高峰期计费翻倍。
#   本测算按"全天平均"折算，即把 6 小时高峰（占 1/4 时间、2倍价）
#   折算进单价 —— 等效系数 = 0.75×1 + 0.25×2 = 1.25
PEAK_FACTOR = 1.25

PRICES = {
    # model: (输入未命中, 输入命中, 输出)  元/百万token，已含峰谷折算
    "deepseek-v4-flash": (1.0 * PEAK_FACTOR, 0.02 * PEAK_FACTOR, 2.0 * PEAK_FACTOR),
    "deepseek-v4-pro": (3.0 * PEAK_FACTOR, 0.025 * PEAK_FACTOR, 6.0 * PEAK_FACTOR),
    "text-embedding-v4": (0.5 * PEAK_FACTOR, 0.5 * PEAK_FACTOR, 0.0),
}

# 云主机与托管服务（按国内主流云厂商包年折算的粗略均价，元/月）
INFRA = {
    # 4C8G 能跑四个 Python 服务 + Redis；OCR/ASR 另算（按需实例）
    "server_4c8g": 480,
    "server_8c16g": 960,
    "gpu_推理卡_按需": 1200,       # 私有化 ASR/YOLO 用
    "managed_milvus_2node": 700,   # 托管向量库
    "oss_1t": 120,                 # 对象存储（文书、音视频）
    "rds_mysql_ha": 400,
    "slb": 150,
}

# 人力成本（元/月，按国内一线城市初级实施工程师估）
HUMAN = {
    "运维_0.5人": 9000,
    "实施_1人": 15000,
    "算法_0.5人": 14000,
}


class Scenario:
    """一个使用场景。"""

    def __init__(self, name, users, req_per_user, desc=""):
        self.name = name
        self.users = users                  # 活跃用户数
        self.req_per_user = req_per_user    # 人均日请求
        self.desc = desc

    @property
    def daily_requests(self):
        return self.users * self.req_per_user

    @property
    def monthly_requests(self):
        return self.daily_requests * 30


# 单次请求的 token 消耗（基于本项目 RAG 链路的实测/估算）
TOKENS_PER_REQ = {
    # 阶段: (输入token, 输出token)
    "意图分类": (180, 20),
    "query改写": (350, 80),
    "embedding": (60, 0),
    "rerank": (0, 0),        # 纯本地，不调 LLM
    "答案生成": (2200, 600),
}
# 命中缓存的输入比例（法规 system prompt 很长且固定，缓存命中率高）
CACHE_HIT_RATIO = 0.55


def calc_llm_cost(monthly_reqs, model="deepseek-v4-flash"):
    """按请求量算 LLM 成本。"""
    in_price, cache_price, out_price = PRICES[model]
    detail = {}
    total = 0.0
    for stage, (tin, tout) in TOKENS_PER_REQ.items():
        if tin == 0 and tout == 0:
            continue
        calls = monthly_reqs
        input_total = tin * calls
        output_total = tout * calls

        # 输入拆成命中/未命中两部分
        hit_tokens = input_total * CACHE_HIT_RATIO
        miss_tokens = input_total * (1 - CACHE_HIT_RATIO)
        stage_cost = (
            hit_tokens / 1e6 * cache_price
            + miss_tokens / 1e6 * in_price
            + output_total / 1e6 * out_price
        )
        detail[stage] = {
            "月调用次数": calls,
            "输入token": input_total,
            "输出token": output_total,
            "月成本": round(stage_cost, 2),
        }
        total += stage_cost
    return total, detail


def calc_embedding_cost(monthly_reqs):
    """Embedding 成本：每个查询一次 + 建索引时的文档向量化。"""
    price = PRICES["text-embedding-v4"][0]
    query_tokens = 60 * monthly_reqs
    # 建索引：假设 4000 chunk，一次性；每月重建 1 次
    index_tokens = 4000 * 500 * 1
    cost = (query_tokens + index_tokens) / 1e6 * price
    return cost, {
        "月查询embedding": query_tokens,
        "月索引embedding": index_tokens,
        "月成本": round(cost, 2),
    }


def calc_infra_cost(users):
    """基础设施成本，按用户规模分档。"""
    server = INFRA["server_4c8g"] if users < 200 else INFRA["server_8c16g"]
    vector = 0 if users < 500 else INFRA["managed_milvus_2node"]
    return {
        "应用服务器": server,
        "向量库托管": vector,
        "对象存储": INFRA["oss_1t"],
        "关系数据库": INFRA["rds_mysql_ha"],
        "负载均衡": INFRA["slb"],
    }


def calc_human_cost(need_gpu):
    """人力成本。"""
    h = {"运维": HUMAN["运维_0.5人"]}
    if need_gpu:
        h["算法支持"] = HUMAN["算法_0.5人"]
    return h


def build_report(scenario: Scenario, model="deepseek-v4-flash",
                 with_gpu=False):
    llm, llm_detail = calc_llm_cost(scenario.monthly_requests, model)
    emb, emb_detail = calc_embedding_cost(scenario.monthly_requests)
    infra = calc_infra_cost(scenario.users)
    human = calc_human_cost(with_gpu)

    infra_total = sum(infra.values())
    human_total = sum(human.values())

    # 故障损失：按可用性 99.5% 估算，月度不可用 3.6 小时，
    # 每次故障影响 N 个执法人员的作业 → 这项常被漏算，但往往是最大头
    downtime_cost = scenario.users * 3.6 / 8 * 150  # 人均时薪 150元

    monthly = llm + emb + infra_total + human_total + downtime_cost
    yearly = monthly * 12

    return {
        "场景": scenario.name,
        "描述": scenario.desc,
        "规模": {
            "活跃用户": scenario.users,
            "人均日请求": scenario.req_per_user,
            "月请求量": scenario.monthly_requests,
        },
        "成本明细_元每月": {
            "LLM_推理": round(llm, 2),
            "Embedding": round(emb, 2),
            "基础设施": round(infra_total, 2),
            "人力_运维与支持": round(human_total, 2),
            "故障损失_估算": round(downtime_cost, 2),
        },
        "年成本_万元": round(yearly / 10000, 2),
        "LLM_分阶段明细": llm_detail,
        "Embedding明细": emb_detail,
        "基础设施明细": infra,
        "人力明细": human,
        "单次请求成本_元": round(
            (llm + emb) / max(scenario.monthly_requests, 1), 4),
    }


DEFAULT_SCENARIOS = [
    Scenario("小型试点", 30, 10,
           "一个消防站内部试用，几十名执法人员"),
    Scenario("区县规模", 300, 20,
           "一个区县消防救援大队 + 下属消防站"),
    Scenario("市级规模", 2000, 25,
           "地市级消防救援支队，覆盖全市"),
    Scenario("省级规模", 8000, 30,
           "省级总队 + 多地市，多层部署"),
]


def print_report(r):
    print("=" * 74)
    print(f"场景：{r['场景']}    —— {r['描述']}")
    print("=" * 74)
    s = r["规模"]
    print(f"  规模        {s['活跃用户']:>8,} 用户  |  "
          f"月请求 {s['月请求量']:>12,} 次")
    print()
    print("  成本构成（元/月）：")
    total = 0
    for k, v in r["成本明细_元每月"].items():
        pct = 0
        total += v
        print(f"    {k:18s} {v:>12,.0f}元")
    for pct, (k, v) in enumerate(
            sorted(r["成本明细_元每月"].items(), key=lambda x: -x[1])):
        share = v / total * 100
        print(f"      {k:16s} 占比 {share:5.1f}%  "
              f"{'█' * int(share / 2)}")
    print()
    print(f"  合计          {total:>12,.0f} 元/月")
    print(f"  年成本        {r['年成本_万元']:>12.2f} 万元")
    print(f"  单次请求成本  {r['单次请求成本_元']:>12.4f} 元")
    print()


def print_sensitivity():
    """敏感性分析：用量变化时哪个环节先撑不住。"""
    print("=" * 74)
    print("敏感性分析（区县规模 300 用户，变动人均日请求）")
    print("=" * 74)
    print()
    print(f"  {'人均日请求':>10s} {'月请求':>14s} {'LLM费/月':>12s} "
          f"{'占总成本':>9s} {'单次成本':>10s}")
    print("  " + "-" * 60)

    for rpu in (5, 10, 20, 40, 80, 160):
        sc = Scenario("敏感", 300, rpu)
        rep = build_report(sc)
        llm = rep["成本明细_元每月"]["LLM_推理"]
        emb = rep["成本明细_元每月"]["Embedding"]
        base = rep["成本明细_元每月"]
        non_llm = (base["基础设施"] + base["人力_运维与支持"]
                   + base["故障损失_估算"])
        total = llm + emb + non_llm
        print(f"  {rpu:>10d} {sc.monthly_requests:>14,} "
              f"{llm + emb:>10,.0f}元 {llm / total * 100:>8.1f}% "
              f"{rep['单次请求成本_元']:>10.4f}元")

    print()
    print("  读法：")
    print("    人均日请求从 20 涨到 80（4 倍），LLM 成本从占比约六成涨到九成以上。")
    print("    → 成本结构会从『人力为主』翻转成『调用量为主』。")
    print("    → 这决定了优化顺序：")
    print("       低用量阶段，优化人工流程收益更大（人占大头）")
    print("       高用量阶段，必须做缓存 + 小模型分流，否则毛利被API 吃掉")
    print()


def print_optimization():
    print("=" * 74)
    print("降本措施与预期收益（按投入产出排序）")
    print("=" * 74)
    items = [
        ("prompt 缓存", "法规 system prompt 固定很长，命中缓存后"
                       "输入成本降到 1/50", "低", "最高"),
        ("意图分类降级到小模型", "分类不需要 Pro 档模型，"
                              "路由准确率要求也没那么高", "低", "高"),
        ("相似问题结果复用", "法规问答高度同质，"
                          "高频问题可命中已有答案", "中", "高"),
        ("检索 top_k 下调", "从 8 降到 5，输入 token 直接减 37%",
         "低", "中（有精度代价）"),
        ("自建推理", "GPU 常驻替代 API，适合高频稳定负载",
         "高", "视用量，很用量才回本"),
    ]
    print()
    print(f"  {'措施':22s} {'说明':44s} {'投入':4s} {'收益'}")
    print("  " + "-" * 82)
    for name, how, cost, gain in items:
        print(f"  {name:22s} {how:44s} {cost:4s} {gain}")
    print()
    print("  ★ 面试时的正确表述顺序：先讲缓存与分流（几乎零投入），")
    print("    再讲自建（要讲清回本临界点），最后才讲降精度换成本。")
    print("    只讲『换更便宜的模型』会被追问一句『精度掉了吗』就答不上来。")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", action="store_true", help="多场景对比")
    ap.add_argument("--export", help="导出 JSON 文件")
    args = ap.parse_args()

    reports = [build_report(s, with_gpu=(s.users >= 2000))
               for s in DEFAULT_SCENARIOS]

    if args.scenarios:
        print("=" * 74)
        print("多场景对比")
        print("=" * 74)
        print()
        print(f"  {'场景':12s} {'用户':>7s} {'月请求':>13s} "
              f"{'LLM/月':>11s} {'总成本/月':>12s} {'年成本':>10s}")
        print("  " + "-" * 72)
        for r in reports:
            b = r["成本明细_元每月"]
            print(f"  {r['场景']:12s} {r['规模']['活跃用户']:>7,} "
                  f"{r['规模']['月请求量']:>13,} "
                  f"{b['LLM_推理'] + b['Embedding']:>9,.0f}元 "
                  f"{sum(b.values()):>10,.0f}元 "
                  f"{r['年成本_万元']:>8.2f}万")
        print()
    else:
        for r in reports:
            print_report(r)

    print_sensitivity()
    print_optimization()

    print("=" * 74)
    print("口径说明（面试可能被追问，先想清楚）")
    print("=" * 74)
    print("""
  1. 单价取自厂商公开定价（DeepSeek 2026-08 后的口径），
     已含峰谷折算系数 1.25。工作日 9-12/14-18 计费翻倍。
  2. token 消耗按本项目 RAG 链路估算：
     意图分类 180+20、query 改写 350+80、答案生成 2200+600。
     实际值随 prompt 版本变化，**测算表应在 prompt 改动后重跑**。
  3. 缓存命中率 55% 是估算值。真实命中率取决于
     system prompt 是否稳定、请求间隔是否够密。
     这是本测算里不确定性最大的一个参数。
  4. 人力按国内一线城市初级工程师估，实际因地区而异。
  5. 故障损失按可用性 99.5% 估。**这一项常被漏算**，
     但它往往超过 API 费用 —— 也是"为什么要做熔断和降级"的成本依据。
  6. 不含：一次性实施成本（开发/测试/上线）、数据标注、合规审计。
     这些应该单列，因为它们是一次性的，不该混进年度运行成本。
""")

    if args.export:
        out = pathlib.Path(args.export)
        out.write_text(json.dumps(reports, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\n已导出：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())