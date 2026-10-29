#!/usr/bin/env python3
"""校验 requirements 里声明的包在 PyPI 上是否真实存在。

★ 为什么需要这个检查：
  真实的失败案例是 `baiduocr>=2.0` —— 这个包名是照着 import 名猜的，
  但代码里根本没有这个 import（agent3/modules/baiduOCR.py 是用 requests
  直连百度 HTTP API）。构建时报
  "Could not find a version that satisfies the requirement baiduocr"，
  又是一次白等好几分钟。

  根因值得记下来：**依赖必须来自代码里真实的 import，不能靠模块名反推。**
  audit_deps.py 的对比方向是「代码 import → 包」，那个方向是安全的；
  反过来「包 → 代码」会把想象出来的依赖也带进来。

用法：
    python3 check_pypi.py            # 检查项目全部 requirements*.txt
    python3 check_pypi.py --offline  # 跳过（无网络环境）
"""
import argparse
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request


def collect_pkgs(root: pathlib.Path) -> dict[str, set[str]]:
    """收集所有 requirements 文件里声明的包名 → 声明位置。"""
    out: dict[str, set[str]] = {}
    for f in sorted(root.glob("requirements*.txt")):
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            # 注释必须先剥掉再取包名，否则 `pkg>=1.0  # 说明` 会带上多余内容
            line = line.split("#")[0].strip()
            if not line or line.startswith("-"):
                continue
            m = re.match(r"^([A-Za-z0-9._\-]+)", line)
            if m:
                name = m.group(1).lower().replace("_", "-")
                out.setdefault(name, set()).add(f.name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="项目根目录")
    ap.add_argument("--offline", action="store_true", help="跳过网络检查")
    ap.add_argument("--timeout", type=int, default=10)
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    pkgs = collect_pkgs(root)
    if not pkgs:
        print("  ✓ 没找到 requirements 声明")
        return 0

    if args.offline:
        print(f"  - 跳过 PyPI 校验（{len(pkgs)} 个包，offline 模式）")
        return 0

    missing = []
    # 并发查：串行 59 个包在网络慢时要几分钟，而 preflight 应该是秒级的事。
    # 用线程池而不是 asyncio —— 标准库 urllib 本身就是阻塞的，
    # 套asyncio 反而要另写客户端依赖。
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def probe(name: str):
        try:
            with urllib.request.urlopen(
                f"https://pypi.org/pypi/{name}/json", timeout=args.timeout
            ) as r:
                json.load(r)
            return name, None
        except urllib.error.HTTPError as e:
            return name, f"HTTP {e.code}"
        except Exception as e:                       # 网络问题，不算包不存在
            return name, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(probe, n): n for n in sorted(pkgs)}
        for fut in as_completed(futures):
            name, err = fut.result()
            if err:
                missing.append((name, err, sorted(pkgs[name])))

    missing.sort()

    if missing:
        print(f"  ✗ {len(missing)} 个包在 PyPI 上不存在：")
        for name, why, srcs in missing:
            print(f"      {name:26s} [{why}]  声明于: {', '.join(srcs)}")
            print(f"      → docker build 会以 'No matching distribution found' 失败")
        print()
        print("  常见成因：包名是照着 import 名猜的，代码里并没有这个 import。")
        print("  校验方式：deploy/tools/audit_deps.py 是「代码 import → 包」方向，")
        print("          反过来做会把想象出来的依赖带进来。")
        return 1

    print(f"  ✓ {len(pkgs)} 个包均存在于 PyPI")
    return 0


if __name__ == "__main__":
    sys.exit(main())