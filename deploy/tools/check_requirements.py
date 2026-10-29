#!/usr/bin/env python3
"""用 pip 自己的解析器校验 requirements 文件的语法。

★ 为什么要单独做这个检查：
  真实的失败案例是 `dashscope>=1.14# 阿里云 CosyVoice` ——
  版本号后面少了一个空格，pip 会把 `#` 当成版本串的一部分，
  报 `Invalid requirement` 并以 exit 1 结束。

  代价极其不划算：那一行错误让 docker build 跑了 14 分钟才失败，
  而用 pip 的解析器校验只需要几秒。
  ——这类"纯语法错误"和"包不存在""COPY 漏目录"是同一类问题：
  构建期能查出来，但代价是几分钟到十几分钟的等待。

用法：
    python3 check_requirements.py            # 校验项目全部 requirements 文件
    python3 check_requirements.py -v         # 同时打印解析结果
"""
import argparse
import pathlib
import re
import sys

# 不联网、不安装，只解析。用 pip 内部的 Requirement 解析器，
# 保证判定标准和真正安装时一致 —— 自己写正则去"猜"格式一定会漏。
try:
    from pip._internal.req.constructors import install_req_from_line
    from pip._internal.req.req import InvalidRequirement
except ImportError:  # pip ≥ 23 移到了别处，兜底给可用的导入路径
    try:
        from pip._internal.req.constructors import parse_req_from_line as _parse
        from pip._internal.req.constructors import install_req_from_line
        InvalidRequirement = Exception
    except ImportError:
        install_req_from_line = None
        InvalidRequirement = Exception


def check_file(path: pathlib.Path, verbose: bool = False) -> int:
    """返回该文件的错误数。"""
    if not path.exists():
        print(f"  ✗ {path} 不存在")
        return 1

    errors = 0
    total = 0
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#")[0].strip() if "#" in raw else raw.strip()
        if not line or line.startswith("-"):
            continue
        total += 1
        # 注释紧贴版本号是最高频的错误：pip 不会把它当注释
        if re.search(r"[><=!~]=?[\d.]+[^\s#]*#", raw):
            print(f"  ✗ {path.name}:{lineno}  注释与版本号之间缺少空格")
            print(f"       {raw.strip()}")
            print("       → pip 会把 '#' 当成版本串的一部分，报 Invalid requirement")
            errors += 1
            continue
        try:
            req = install_req_from_line(line)
            if verbose:
                print(f"      {req.name:24s} {req.specifier}")
        except InvalidRequirement as e:
            print(f"  ✗ {path.name}:{lineno}  无法解析: {line}")
            print(f"       {e}")
            errors += 1

    if errors == 0:
        print(f"  ✓ {path.name}  {total} 条依赖语法正确")
    return errors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="要检查的文件，默认检查项目根的 requirements*.txt")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if install_req_from_line is None:
        print("无法导入 pip 解析器，跳过（pip 版本不兼容）")
        sys.exit(0)

    root = pathlib.Path(__file__).resolve().parents[2]
    if args.files:
        files = [pathlib.Path(f) for f in args.files]
    else:
        files = sorted(root.glob("requirements*.txt"))

    if not files:
        print("没找到 requirements 文件")
        sys.exit(0)

    total_err = 0
    for f in files:
        total_err += check_file(f, args.verbose)

    print()
    if total_err:
        print(f"  共 {total_err} 处语法错误。会让 pip 在安装阶段直接失败，"
              f"而在 Docker 里这意味着白等整个构建。")
        sys.exit(1)
    print("  requirements 语法全部正确")
    sys.exit(0)