"""扫描知识库，统计文件类型和大小"""
import os
from collections import Counter

KB_DIR = r"D:\AI\ollama_deepseek\agent\knowledge_base\full"

ext_counter = Counter()
size_counter = Counter()
total_files = 0
total_size = 0

for root, _, files in os.walk(KB_DIR):
    for f in files:
        if f.startswith("~$") or f.startswith("."):
            continue
        ext = f.lower().split(".")[-1]
        path = os.path.join(root, f)
        size = os.path.getsize(path)
        ext_counter[ext] += 1
        size_counter[ext] += size
        total_files += 1
        total_size += size

print(f"总文件数：{total_files}")
print(f"总大小：{total_size/1024/1024:.1f} MB\n")
print("按类型分布：")
for ext, cnt in ext_counter.most_common():
    print(f"  .{ext:8s}  {cnt:5d} 个  {size_counter[ext]/1024/1024:.1f} MB")