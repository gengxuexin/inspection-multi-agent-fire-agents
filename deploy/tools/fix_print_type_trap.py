"""修打印函数的类型陷阱（最后 3 个失败的共同根因）。

`checks.get("comparable") or {}`：
  当 checks["comparable"] 是布尔值时，`or` 会短路返回那个 bool，
  后面 `.get(...)` 就抛 'bool' object has no attribute 'get'。

★ 这类"用 or {} 做默认值"的写法在打印函数里特别危险：
  打印函数的输入是别人（测试、调用方、上一版实现）构造的 dict，
  字段类型不受你控制。而它崩了等于整轮评测白跑 ——
  分数已经算完了，只是没打出来。
  所以每个取值都要做类型校验，不能靠 or 兜底。
"""
import pathlib

p = pathlib.Path("agent1/eval/local_eval.py")
s = p.read_text(encoding="utf-8")

HELPER = '''def _as_dict(v) -> dict:
    """把任意值安全地变成 dict。

    ★ 打印函数的输入来自外部（测试、调用方、上一版实现），
      字段类型不受控。崩了等于整轮评测白跑——
      分数已经算完，只是没打出来。
      所以这里做类型校验，而不是用 `or {}` 兜底
      （后者遇到 bool 会短路返回 bool，随后 .get 就崩）。
    """
    return v if isinstance(v, dict) else {}


'''

REPLACEMENTS = [
    ('    checks = out.get("checks") or {}\n'
     '    cmp_ = checks.get("comparable") or {}',
     '    checks = _as_dict(out.get("checks"))\n'
     '    cmp_ = _as_dict(checks.get("comparable"))'),

    ('    vi = checks.get("variable_isolation") or {}',
     '    vi = _as_dict(checks.get("variable_isolation"))'),

    ('    nr = checks.get("null_run")\n'
     '    if not isinstance(nr, dict):\n'
     '        nr = {}\n'
     '    noise = checks.get("noise")\n'
     '    if not isinstance(noise, dict):\n'
     '        noise = {}\n'
     '    verdict = out.get("verdict") or {}\n'
     '    deltas = out.get("deltas") or {}',
     '    nr = _as_dict(checks.get("null_run"))\n'
     '    noise = _as_dict(checks.get("noise"))\n'
     '    verdict = _as_dict(out.get("verdict"))'),

    ('            floor_map = noise.get("noise_floor")\n'
     '            if not isinstance(floor_map, dict):\n'
     '                floor_map = {}',
     '            floor_map = _as_dict(noise.get("noise_floor"))'),

    ('            v = verdict.get(k)\n'
     '            if not v:\n'
     '                continue',
     '            v = _as_dict(verdict.get(k))\n'
     '            if not v:\n'
     '                continue'),

    ('        for k, v in verdict.items()\n'
     '                if v["exceeds_noise"]]',
     '        for k, v in verdict.items()\n'
     '                if _as_dict(v).get("exceeds_noise")]'),
]

n = 0
for old, new in REPLACEMENTS:
    if old in s:
        s = s.replace(old, new, 1)
        n += 1
    else:
        print(f"! 未匹配：{old.strip()[:52]}")

# sig 计数那行也要防护
s = s.replace(
    "    sig = sum(1 for v in verdict.values() if v.get(\"exceeds_noise\"))",
    "    sig = sum(1 for v in verdict.values()\n"
    "              if _as_dict(v).get(\"exceeds_noise\"))", 1)

# 插入 helper
if "_as_dict(v) -> dict:" not in s:
    anchor = "def print_baseline_report(out: dict) -> None:"
    s = s.replace(anchor, HELPER + anchor, 1)

p.write_text(s, encoding="utf-8")

import ast
ast.parse(s)
print(f"✓ 打印函数已加固（{n}/{len(REPLACEMENTS)} 处替换）")
print("✓ 语法正常")