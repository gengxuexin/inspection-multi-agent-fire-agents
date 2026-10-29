# agent1/testdata/permission_fixtures.py
"""权限过滤测试数据生成器。

★★ 为什么需要
--------------
实测索引里的元数据分布：

    secret_level: {0: 3953, 1: 66}   ← 等级 > 2 的：0 条
    valid_until:  全部为 0（永不过期）
    dept:         {通用: 3971, 消防支队: 48}

也就是说：``agent1/retrieval_backend.py`` 里那套权限过滤 expr
（``dept in [...] && secret_level <= clearance && (valid_until == 0 || valid_until > now)``）
**逻辑是对的，但当前语料里没有任何一条数据能触发它**。

后果是：过滤机制处于「空转」状态 —— 代码写了、测试也过了，
但从没在真实数据上验证过。等到真正接了涉密文档和废止法规，
才会第一次暴露问题，那时已经在生产环境了。

★★★ 解法
--------
造一批**带明确权限与时效标记**的测试语料 + 配套测试，
让过滤逻辑在真实数据上被验证一遍。

★ 为什么造数据而不是改生产语料
------------------------------
生产语料是 103 份真实消防法规，不能为了测试而塞假数据进去
（假数据会污染检索结果，进而影响评测集）。
所以测试数据独立存在，只在测试里用。
"""
from __future__ import annotations

import time
from typing import Any

# 清算等级含义：0=公开 1=内部 2=秘密 3=机密 4=绝密
CLEARANCE_PUBLIC = 0
CLEARANCE_INTERNAL = 1
CLEARANCE_SECRET = 2
CLEARANCE_CONFIDENTIAL = 3
CLEARANCE_TOP_SECRET = 4

NOW = int(time.time())
DAY = 86400


def build_permission_fixtures() -> list[dict[str, Any]]:
    """构造覆盖各种权限组合的测试文档。

    每个 fixture 是一个 chunk 的元数据 + 内容，
    字段与真实索引里的 metadata 完全一致。
    """
    return [
        # ---- 公开文档：任何 clearance 都能看 ----
        {
            "chunk_id": "fix_public_001",
            "page_content": "【测试】消防法公开条款。任何用户都应能看到这条。",
            "metadata": {
                "law": "测试法-公开",
                "article": "第一条",
                "source": "TEST_公开文档.pdf",
                "doc_id": "fix_doc_public",
                "dept": "通用",
                "role": "officer",
                "secret_level": CLEARANCE_PUBLIC,
                "valid_until": 0,          # 永不过期
                "is_catalog": False,
                "is_tier_table": False,
                "chars": 30,
            },
        },
        # ---- 内部文档：clearance >= 1 才可见 ----
        {
            "chunk_id": "fix_internal_001",
            "page_content": "【测试】内部文件内容。公开级用户不该看到这条。",
            "metadata": {
                "law": "测试法-内部",
                "article": "第二条",
                "source": "TEST_内部文档.pdf",
                "doc_id": "fix_doc_internal",
                "dept": "通用",
                "role": "officer",
                "secret_level": CLEARANCE_INTERNAL,
                "valid_until": 0,
                "is_catalog": False,
                "is_tier_table": False,
                "chars": 30,
            },
        },
        # ---- 秘密文档：clearance >= 2 才可见 ----
        {
            "chunk_id": "fix_secret_001",
            "page_content": "【测试】秘密级内容。低级权限用户不该看到这条。",
            "metadata": {
                "law": "测试法-秘密",
                "article": "第三条",
                "source": "TEST_秘密文档.pdf",
                "doc_id": "fix_doc_secret",
                "dept": "通用",
                "role": "officer",
                "secret_level": CLEARANCE_SECRET,
                "valid_until": 0,
                "is_catalog": False,
                "is_tier_table": False,
                "chars": 30,
            },
        },
        # ---- 机密文档：clearance >= 3 才可见 ----
        {
            "chunk_id": "fix_confidential_001",
            "page_content": "【测试】机密级内容。只有机密权限才可见。",
            "metadata": {
                "law": "测试法-机密",
                "article": "第四条",
                "source": "TEST_机密文档.pdf",
                "doc_id": "fix_doc_confidential",
                "dept": "通用",
                "role": "officer",
                "secret_level": CLEARANCE_CONFIDENTIAL,
                "valid_until": 0,
                "is_catalog": False,
                "is_tier_table": False,
                "chars": 30,
            },
        },
        # ---- 绝密文档：clearance >= 4 才可见 ----
        {
            "chunk_id": "fix_topsecret_001",
            "page_content": "【测试】绝密级内容。只有绝密权限才可见。",
            "metadata": {
                "law": "测试法-绝密",
                "article": "第五条",
                "source": "TEST_绝密文档.pdf",
                "doc_id": "fix_doc_topsecret",
                "dept": "通用",
                "role": "officer",
                "secret_level": CLEARANCE_TOP_SECRET,
                "valid_until": 0,
                "is_catalog": False,
                "is_tier_table": False,
                "chars": 30,
            },
        },
        # ---- 部门专属文档：只有「消防支队」可见 ----
        {
            "chunk_id": "fix_dept_001",
            "page_content": "【测试】支队专属文档。其他部门不该看到这条。",
            "metadata": {
                "law": "测试法-支队",
                "article": "第六条",
                "source": "TEST_支队文档.pdf",
                "doc_id": "fix_doc_dept",
                "dept": "消防支队",
                "role": "officer",
                "secret_level": CLEARANCE_PUBLIC,
                "valid_until": 0,
                "is_catalog": False,
                "is_tier_table": False,
                "chars": 30,
            },
        },
        # ---- 已过期文档：valid_until 在过去，任何人都不该看到 ----
        {
            "chunk_id": "fix_expired_001",
            "page_content": "【测试】已废止的条款。任何人都不该检索到这条。",
            "metadata": {
                "law": "测试法-已废止",
                "article": "第七条",
                "source": "TEST_已废止文档.pdf",
                "doc_id": "fix_doc_expired",
                "dept": "通用",
                "role": "officer",
                "secret_level": CLEARANCE_PUBLIC,
                "valid_until": NOW - 30 * DAY,     # 30 天前就失效了
                "is_catalog": False,
                "is_tier_table": False,
                "chars": 30,
            },
        },
        # ---- 即将过期文档：还在有效期内，应该能看到 ----
        {
            "chunk_id": "fix_soon_expire_001",
            "page_content": "【测试】即将失效但仍在有效期内的条款。应该能看到。",
            "metadata": {
                "law": "测试法-即将失效",
                "article": "第八条",
                "source": "TEST_即将失效文档.pdf",
                "doc_id": "fix_doc_soon_expire",
                "dept": "通用",
                "role": "officer",
                "secret_level": CLEARANCE_PUBLIC,
                "valid_until": NOW + 30 * DAY,     # 30 天后失效
                "is_catalog": False,
                "is_tier_table": False,
                "chars": 30,
            },
        },
    ]


def build_documents() -> list:
    """转成 LangChain Document（供 FAISS 入库）。"""
    from langchain_core.documents import Document
    return [Document(page_content=f["page_content"], metadata=f["metadata"])
            for f in build_permission_fixtures()]


# 期望可见性矩阵：fixture chunk_id → {clearance: 是否可见}
#
# ★ dept 维度单独处理
# -------------------
# 这张矩阵描述的是「**密级与时效**」的可见性（用「消防支队」这个视角）。
# 部门隔离是另一条独立的轴 —— fix_dept_001 是「消防支队」专属文档，
# 从消防支队的视角看它**是可见的**（secret_level=0，未过期）；
# 从别的部门看才不可见。
#
# 所以这里把 fix_dept_001 标成「密级维度全部可见」，
# 部门隔离由 test_dept_isolation 单独验证。
EXPECTED_VISIBILITY: dict[str, dict[int, bool]] = {
    "fix_public_001":        {0: True,  1: True,  2: True,  3: True,  4: True},
    "fix_internal_001":      {0: False, 1: True,  2: True,  3: True,  4: True},
    "fix_secret_001":        {0: False, 1: False, 2: True,  3: True,  4: True},
    "fix_confidential_001":  {0: False, 1: False, 2: False, 3: True,  4: True},
    "fix_topsecret_001":     {0: False, 1: False, 2: False, 3: False, 4: True},
    # 支队专属：从消防支队视角看可见（密级公开、未过期）
    "fix_dept_001":          {0: True,  1: True,  2: True,  3: True,  4: True},
    # 已废止：任何权限都不可见
    "fix_expired_001":       {0: False, 1: False, 2: False, 3: False, 4: False},
    # 即将失效但仍在有效期：可见
    "fix_soon_expire_001":   {0: True,  1: True,  2: True,  3: True,  4: True},
}
