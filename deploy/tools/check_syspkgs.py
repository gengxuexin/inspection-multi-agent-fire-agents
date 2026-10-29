#!/usr/bin/env python3
"""检查 Dockerfile 里 apt-get install 的包在基础镜像中是否真实存在。

★ 为什么这个检查值得单独存在：
  1. 包名不能凭印象写。`libsnd1`（Ubuntu）在 Debian 里叫 `libasound2`；
     而 python:3.10-slim 现在是 Debian 13 (trixie)，做了 64-bit time_t 迁移，
     `libasound2` / `libglib2.0-0` 也不存在了，实际是带 `t64` 后缀的版本。
     写错的表现是构建到 apt 那一步才 exit 100，前面几分钟全白等。
  2. **更隐蔽的一种**：某些包名会被 apt 当成虚拟包 / Provides 静默解析成功，
     构建照样通过，但真正缺的运行时库要到 import cv2 / 渲染图片时才暴露。
     所以"包能装上"不等于"运行时库齐全"。
  3. 这段逻辑必须用 Python 做。塞进 bash 会踩转义地狱 ——
     实测 `awk '{print $2}'` 的 $2 被外层 shell 吃掉，输出无法解析，
     结果把有问题的包判成没问题。**漏报比误报危险得多。**

用法：
    python3 check_syspkgs.py --base python:3.10-slim --dockerfiles deploy/docker
"""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys


def extract_apt_pkgs(dockerfile: pathlib.Path) -> list[str]:
    """从 Dockerfile 里抽出 apt-get install 的包名。

    ★ 这里踩过一个坑：最初写成「遇到 apt-get install 就取到 && 为止，
      然后立刻把 in_run 置 False」。但常见的写法是
          RUN apt-get update && apt-get install -y \\
                  curl \\
                  ffmpeg
      —— `apt-get install` 就在 RUN 的首行，处理完立刻收工，
      后面几行的包名（curl/ffmpeg）全被跳过，于是报告"无 apt 安装"。
      而"没检查到"和"检查通过"在输出上长得一样，是最危险的误报。

      正确做法：先把整个 RUN 块（按行尾 \\ 续行）拼成一条逻辑行，
      再从里面找 apt-get install。
    """
    text = dockerfile.read_text(encoding="utf-8", errors="ignore")

    # 1) 把物理行合并成逻辑行：RUN 开头，或上一行以 \ 结尾
    logical: list[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not buf and not line.upper().startswith("RUN "):
            continue                       # 只关心 RUN 指令
        line = re.sub(r"^RUN\s+", "", line, flags=re.I)
        buf += " " + line.rstrip("\\").strip()
        if not line.endswith("\\"):
            logical.append(buf)
            buf = ""
    if buf:
        logical.append(buf)

    # 2) 从每条逻辑行里取 apt-get install 到 && 之间的包名
    pkgs: list[str] = []
    for cmd in logical:
        if "apt-get install" not in cmd:
            continue
        after = cmd.split("apt-get install", 1)[1]
        for sep in ("&&", ";"):
            after = after.split(sep)[0]
        # 去掉 apt-get install 自身的选项：-y --no-install-recommends
        for tok in after.split():
            if tok.startswith("-"):
                continue
            if re.fullmatch(r"[a-z0-9][a-z0-9.+-]*", tok):
                pkgs.append(tok)
    return sorted(set(pkgs))


def query_packages(base_image: str, pkgs: list[str]) -> dict[str, str]:
    """在基础镜像里查每个包的 Candidate 版本。"""
    if not pkgs:
        return {}

    # ---- 缓存 ----
    # 这一步要起容器跑 apt-get update，实测 4 分钟。而 preflight 的意义是
    # "花几秒发现问题"，本身跑 4 分钟就本末倒置了。
    # 缓存键 = 基础镜像名 + 包列表；只要 Dockerfile 里的包没变就直接复用。
    cache_dir = pathlib.Path(
        os.environ.get("XDG_CACHE_HOME", pathlib.Path.home() / ".cache")
    ) / "fire-preflight"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(
            (base_image + "|" + "|".join(sorted(pkgs))).encode()
        ).hexdigest()[:16]
        cache_file = cache_dir / f"syspkgs-{key}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if data.get("image") == base_image:
                print(f"（命中缓存 {cache_file.name}，如需强制刷新请删"
                      f" {cache_dir}）", file=sys.stderr)
                return data["result"]
    except OSError:
        cache_file = None       # 缓存不可用不影响主流程

    # 循环体内统一用单引号拼装，避免宿主机 shell 参与变量展开
    script = "apt-get update -qq >/dev/null 2>&1\n"
    for p in pkgs:
        script += (
            "printf '%s\\t%s\\n' '" + p + "' "
            '"$(apt-cache policy ' + p + ' 2>/dev/null '
            "| grep -m1 Candidate | awk '{print $2}')\"\n"
        )
    try:
        out = subprocess.run(
            ["docker", "run", "--rm", base_image, "sh", "-c", script],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("docker run 超时", file=sys.stderr)
        return {}

    result: dict[str, str] = {}
    for line in out.stdout.splitlines():
        if "\t" in line:
            k, v = line.split("\t", 1)
            result[k.strip()] = v.strip()

    if cache_file is not None and result:
        try:
            cache_file.write_text(
                json.dumps({"image": base_image, "result": result},
                           ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="python:3.10-slim", help="基础镜像")
    ap.add_argument("--dockerfiles", default="deploy/docker", help="Dockerfile 所在目录")
    ap.add_argument("--verbose", action="store_true", help="打印每个包的版本")
    args = ap.parse_args()

    df_dir = pathlib.Path(args.dockerfiles).resolve()
    files = sorted(df_dir.glob("*.Dockerfile")) + sorted(df_dir.glob("Dockerfile*"))
    if not files:
        print(f"✗ 在 {df_dir} 下没找到 Dockerfile")
        return 1

    # 合并所有 Dockerfile 的包，一次性查（每包一次 docker run 太慢）
    all_pkgs: set[str] = set()
    for f in files:
        all_pkgs |= set(extract_apt_pkgs(f))

    avail = query_packages(args.base, sorted(all_pkgs))
    if not avail:
        print("✗ 无法查询基础镜像（docker 不可用？）")
        return 1

    problems = 0
    for f in files:
        pkgs = extract_apt_pkgs(f)
        if not pkgs:
            print(f"  ✓ {f.name}  无 apt 安装")
            continue

        missing, virtual = [], []
        for p in pkgs:
            v = avail.get(p, "")
            if not v:
                missing.append(p)
            elif v == "(none)":
                virtual.append(p)
            elif args.verbose:
                print(f"      {p} = {v}")

        name = f.name
        if missing:
            print(f"  ✗ {name}  包不存在: {', '.join(missing)}")
            print("      → 这些会让 docker build 以 'Unable to locate package' 失败")
            problems += 1
        if virtual:
            print(f"  ! {name}  候选为空: {', '.join(virtual)}")
            print("      → 可能是虚拟包被静默解析，构建能过但运行时库实际缺失")
            problems += 1
        if not missing and not virtual:
            print(f"  ✓ {name}  系统包齐备（{len(pkgs)} 个）")

    if problems:
        print(f"\n  共 {problems} 处问题。更换基础镜像前先确认包名，"
              f"trixie 类镜像普遍有 t64 后缀迁移。")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())