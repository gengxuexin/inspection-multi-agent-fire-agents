#!/usr/bin/env python3
"""逐个确认 Dockerfile 里要用的系统包在基础镜像中是否真实存在。

为什么需要这个：`libsnd1`（Ubuntu 的包名）在 Debian 13 里根本不存在，
构建直接失败 exit 100。而 agent1 之所以构建成功，是因为它没装这个包 ——
**一个服务的构建通过，不代表另一个服务的包名也是对的**。

包名不能凭印象写：Debian/Ubuntu 同名库经常不一样
（libsnd1 vs libasound2、libglib2.0-0 vs libglib2.0-0t64），
而且 trixie 这类新版本还会在 t64 迁移里改包名。
"""
import subprocess

PKGS = [
    "curl", "ffmpeg", "libasound2", "libsnd1",
    "libgl1", "libglib2.0-0", "libglib2.0-0t64",
    "tesseract-ocr", "tesseract-ocr-chi-sim",
    "poppler-utils", "fonts-wqy-zenhei",
]

# agent2 在 Python 侧实际会 import 的音频库，决定要不要装 ALSA
NEEDS = {
    "agent1": ["curl", "tesseract-ocr", "tesseract-ocr-chi-sim",
               "poppler-utils", "libgl1", "libglib2.0-0", "fonts-wqy-zenhei"],
    "agent2": ["curl", "ffmpeg", "libasound2", "libgl1",
               "libglib2.0-0", "fonts-wqy-zenhei"],
    "agent3": ["curl", "ffmpeg", "libgl1", "libglib2.0-0", "fonts-wqy-zenhei"],
    "gateway": ["curl"],
}

script = "apt-get update -qq >/dev/null 2>&1\n"
for p in PKGS:
    script += (
        f'printf "%-24s %s\\n" "{p}" '
        f'"$(apt-cache policy {p} 2>/dev/null | grep -m1 Candidate | '
        f"awk '{{print $2}}')\"\n"
    )

out = subprocess.run(
    ["docker", "run", "--rm", "python:3.10-slim", "sh", "-c", script],
    capture_output=True, text=True, timeout=300,
)
print(out.stdout or out.stderr)

print("=== 各服务需要的包是否齐备 ===")
avail = {}
for line in out.stdout.splitlines():
    parts = line.split()
    if len(parts) >= 2:
        avail[parts[0]] = parts[1]

for svc, pkgs in NEEDS.items():
    bad = [p for p in pkgs if not avail.get(p)]
    if bad:
        print(f"  {svc:9s} 缺: {', '.join(bad)}")
    else:
        print(f"  {svc:9s} ✓ 全部可用")