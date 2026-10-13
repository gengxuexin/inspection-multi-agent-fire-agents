#!/usr/bin/env python3
# apply_rag_fix.py —— 把 RAG 检索质量改造同步到 agent1 的三个已有文件。
#
# 背景：Windows 侧 9P 挂载只能新建文件、不能覆盖已有文件，
#       因此用一个一次性脚本在 WSL 内部完成覆盖写入。
#
# 用法（在项目根目录执行）：
#     cd /home/geng/fire_agent_v2/fire_agents
#     python3 apply_rag_fix.py
#
# 幂等：重复执行安全。每个文件写入前先备份到 .rag_backup/。
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKUP = os.path.join(ROOT, ".rag_backup")

FILES = ["agent1/hybrid.py", "agent1/retrieval_backend.py",
         "agent1/build_index.py", "agent1/eval/ragas_eval.py"]

# Windows 侧已验证完成的最终版本
SOURCE_DIR = os.environ.get("RAG_FIX_SRC", "/mnt/c/Users/Lenovo/Desktop/fire_agents")


def main() -> int:
    if not os.path.isdir(SOURCE_DIR):
        print(f"[!] 找不到源目录：{SOURCE_DIR}")
        print("    请设置环境变量 RAG_FIX_SRC 指向含改造后文件的目录")
        return 1

    os.makedirs(BACKUP, exist_ok=True)
    applied, skipped = [], []

    for rel in FILES:
        src = os.path.join(SOURCE_DIR, rel)
        dst = os.path.join(ROOT, rel)

        if not os.path.isfile(src):
            print(f"[!] 源文件不存在，跳过：{rel}")
            skipped.append(rel)
            continue
        if not os.path.isfile(dst):
            print(f"[!] 目标文件不存在，跳过：{rel}")
            skipped.append(rel)
            continue

        # 备份一次即可（覆盖式备份，保留最近一次原始状态）
        bak = os.path.join(BACKUP, rel.replace("/", "__"))
        if not os.path.isfile(bak):
            shutil.copy2(dst, bak)

        with open(src, "r", encoding="utf-8") as f:
            new_content = f.read()
        with open(dst, "w", encoding="utf-8") as f:
            f.write(new_content)

        applied.append(rel)
        print(f"[✓] 已写入 {rel}（{len(new_content)} 字符）")

    print()
    print(f"完成：{len(applied)} 个文件已更新")
    if skipped:
        print(f"跳过：{skipped}")
    print(f"原始版本备份于：{BACKUP}")
    print()
    print("后续步骤：")
    print("  1) python3 -m pytest tests/test_chunking.py -q   # 48 项，应全过")
    print("  2) python3 agent1/build_index.py --mode full --rebuild   # 重建索引")
    print("  3) python3 agent1/eval/ragas_eval.py --top-k 5--output after.json")
    print("     与eval/report_local.json 的 top_k=20 结果对比")
    return 0


if __name__ == "__main__":
    sys.exit(main())