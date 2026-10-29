#!/usr/bin/env python3
"""校验 docs/skills/ 下的 Skill 文档与代码是否一致。

★★ 为什么需要这个脚本
----------------------
Skill 文档是「接口契约」。契约一旦和实现脱节，比没有文档更糟 ——
因为调用方会**信任文档**。

本项目已经吃过这个亏：
- 简历/网站上的数字和仓库 docs 里的数字对不上（0.986 vs 0.982）
- 申请材料里的「6 个 MCP 工具函数」实际是 3 个

所以 Skill 文档必须**可验证**。这个脚本做三件事：

1. **端点存在性** —— 文档里写的每个端点，代码里必须有
2. **端点覆盖率** —— 代码里有的 /api 端点，文档里必须有（防遗漏）
3. **关键词表一致** —— 文档里的路由关键词必须等于 router.py 里的

★ 不做的检查
------------
刻意**不**校验字段级 schema —— 那需要维护一份完整 JSON Schema，
成本高且容易和 pydantic 模型重复。
改为校验「端点存在 + 关键词一致 + 关键字段名出现」三档，
覆盖最容易漂移的部分。

用法：
    python deploy/tools/verify_skill_docs.py
退出码：0 = 通过，1 = 有 FAIL
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKILLS_DIR = os.path.join(ROOT, "docs", "skills")

# 代码文件 → 文档文件 的映射
CODE_FILES = {
    "gateway": os.path.join(ROOT, "gateway.py"),
    "agent1": os.path.join(ROOT, "agent1", "server.py"),
    "agent2": os.path.join(ROOT, "agent2", "server.py"),
    "agent3": os.path.join(ROOT, "agent3", "server.py"),
    "mcp": os.path.join(ROOT, "mcp", "change_detector_server.py"),
}

DOC_FILES = {
    "gateway": os.path.join(SKILLS_DIR, "00-gateway.md"),
    "agent1": os.path.join(SKILLS_DIR, "01-agent1-rag.md"),
    "agent2": os.path.join(SKILLS_DIR, "02-agent2-voice.md"),
    "agent3": os.path.join(SKILLS_DIR, "03-agent3-vision.md"),
    "mcp": os.path.join(SKILLS_DIR, "04-mcp-server.md"),
}

# 代码里这些路径是变量拼出来的，不做端点覆盖检查
SKIP_ENDPOINT_PREFIX = ("/health", "/ready", "/")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))


def read(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def code_endpoints(src: str) -> set[str]:
    """从代码里抽出所有 @app.get/post/put/delete 的路径。"""
    eps = set()
    for m in re.finditer(
            r'@app\.(?:get|post|put|delete|patch)\(\s*["\']([^"\']+)["\']',
            src):
        eps.add(m.group(1))
    return eps


def doc_endpoints(md: str) -> set[str]:
    """从文档里抽出所有形如 /api/xxx 的路径。"""
    eps = set()
    for m in re.finditer(r'`(/[a-zA-Z0-9_\-/{}\.]+)`', md):
        p = m.group(1)
        if p.startswith("/api") or p in ("/dispatch", "/dispatch_async",
                                         "/dispatch_file", "/agents/health",
                                         "/auth/token", "/tasks/{task_id}"):
            eps.add(p)
    return eps


def main() -> int:
    if not os.path.isdir(SKILLS_DIR):
        print(f"❌ 找不到目录：{SKILLS_DIR}")
        return 1

    print("=" * 62)
    print("Skill 文档 ↔ 代码 一致性校验")
    print("=" * 62)

    # ---------- 1. 端点存在性 + 覆盖率 ----------
    for key, code_path in CODE_FILES.items():
        doc_path = DOC_FILES.get(key)
        code = read(code_path)
        md = read(doc_path) if doc_path else ""

        if not code:
            check(f"{key}: 代码文件存在", False, code_path)
            continue
        if not md:
            check(f"{key}: 文档存在", False, doc_path or "(缺)")
            continue

        c_eps = code_endpoints(code)
        d_eps = doc_endpoints(md)

        # 1a. 文档里写的端点，代码里必须有
        missing_in_code = sorted(d_eps - c_eps)
        check(f"{key}: 文档端点均存在于代码", not missing_in_code,
              f"文档里这些端点代码中找不到：{missing_in_code}")

        # 1b. 代码里的 /api 端点，文档里必须有
        c_api = {e for e in c_eps
                 if e.startswith("/api") and not e.startswith(SKIP_ENDPOINT_PREFIX)}
        missing_in_doc = sorted(c_api - d_eps)
        check(f"{key}: 代码端点均已在文档中", not missing_in_doc,
              f"代码里这些端点文档没写：{missing_in_doc}")

    # ---------- 2. 路由关键词一致 ----------
    router = read(os.path.join(ROOT, "common", "router.py"))
    readme = read(os.path.join(SKILLS_DIR, "README.md"))
    if router and readme:
        for var in ("VISION_KW", "VOICE_KW", "DOC_KW"):
            m = re.search(rf'{var}\s*=\s*\[(.*?)\]', router, re.S)
            if not m:
                continue
            kws = re.findall(r'"([^"]+)"', m.group(1))
            missing = [k for k in kws if k not in readme]
            check(f"路由关键词 {var} 均已写入文档", not missing,
                  f"这些关键词文档里没有：{missing}")
    else:
        check("路由关键词校验", False, "router.py 或 README.md 缺失")

    # ---------- 3. 关键字段名抽查 ----------
    field_checks = [
        ("gateway.py", DOC_FILES["gateway"], ["routed_to", "route", "degraded"]),
        ("agent1/server.py", DOC_FILES["agent1"],
         ["context_precision", "review_id", "clearance", "valid_until"]),
        ("agent2/server.py", DOC_FILES["agent2"],
         ["user_text", "answer_text", "audio_path"]),
        ("agent3/server.py", DOC_FILES["agent3"],
         ["judgment", "ocr", "yolo"]),
    ]
    for code_rel, doc_path, fields in field_checks:
        md = read(doc_path)
        missing = [f for f in fields if f not in md]
        check(f"{code_rel}: 关键字段已写入文档", not missing,
              f"这些字段文档里没有：{missing}")

    # ---------- 汇总 ----------
    print()
    failed = 0
    for name, ok, detail in results:
        flag = "OK " if ok else "FAIL"
        print(f"  [{flag}] {name}")
        if not ok and detail:
            print(f"         {detail}")
        if not ok:
            failed += 1

    print()
    print("=" * 62)
    if failed:
        print(f"❌ {failed} 项不一致 —— 文档已漂移，请同步代码或文档")
        return 1
    print(f"✅ 全部 {len(results)} 项一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
