"""演示页端到端验收。

★ 验收标准（每条都要真验，不能只看 HTTP 200）：
  1. 页面能加载，且关键元素都在
  2. API 返回的阶段数据完整（不是空数组）
  3. 三态溯源字段齐全（risk / verified / source_excerpt）
  4. fragment_count 与实际片段数一致 —— 防止前端显示硬编码的假数字
  5. 前端引用的字段后端都有（反向检查，最容易漏）
  6. 弃答路径能跑通
"""
import json
import pathlib
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8800"
HERE = pathlib.Path(__file__).resolve().parent

fail = 0


def check(label, ok, detail=""):
    global fail
    mark = "OK  " if ok else "FAIL"
    if not ok:
        fail += 1
    print(f"  [{mark}] {label}" + (f"  {detail}" if detail else ""))


print("=" * 66)
print("1. 页面加载")
print("=" * 66)
try:
    html = urllib.request.urlopen(f"{BASE}/", timeout=10).read().decode("utf-8")
    check("首页返回 200", True, f"{len(html)} 字节")
    for token in ("const SAMPLE", "id=\"stages\"", "renderStages",
                  "renderResult", "cite-ok", "guard"):
        check(f"包含 {token}", token in html)
    if token == "const SAMPLE":
        # 示例列表要真的用上 q/n，否则点了没反应
        check("示例列表用到 q/n",
              "s.q" in html and "s.n" in html)
except (urllib.error.URLError, Exception) as e:   # noqa: BLE001
    check("首页可访问", False, str(e))
    raise SystemExit(1)

print()
print("=" * 66)
print("1.5 前端 JS 语法门禁（node --check）")
print("=" * 66)

# ★ 这一段是被 bug 逼出来的：
#   页面曾因 const notes 重复声明 + cs 未定义，整个 <script> 无法解析。
#   症状是「页面能开、按钮点了没反应」，而 API 完全正常 ——
#   极易误判成后端问题，浪费大量时间。
#   所以门禁必须真的跑 JS 引擎，而不是只 grep HTML 字符串。
import re
import shutil
import subprocess
import tempfile

_js_m = re.search(r"<script>(.*?)</script>", html, re.S)
check("页面含 <script> 块", bool(_js_m))
if _js_m:
    _node = shutil.which("node")
    if not _node:
        check("node 可用（跳过语法门禁）", True, "未安装 node，跳过")
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                         encoding="utf-8") as _f:
            _f.write(_js_m.group(1))
            _js_path = _f.name
        _r = subprocess.run([_node, "--check", _js_path],
                            capture_output=True, text=True)
        _err = (_r.stderr or "").strip()
        check("前端 JS 语法通过 node --check", _r.returncode == 0,
              _err.splitlines()[0] if _err else "")
        if _r.returncode != 0:
            print("       ↑ 语法错误会让整个页面脚本失效，")
            print("         表现为「页面能开但点了没反应」，且 API 正常。")
        Path(_js_path).unlink(missing_ok=True)

# 引用溯源卡片必须有渲染逻辑，否则「引用可溯源」这个卖点根本没显示出来
check("引用卡片有渲染逻辑",
      "getElementById('cites')" in html
      and "getElementById('citeHint')" in html)
check("引用溯源读的是后端字段 r.citations",
      "r.citations" in html)
check("无重复 const notes 声明",
      len(re.findall(r"const notes\s*=", html)) <= 1,
      f"出现 {len(re.findall(r'const notes\s*=', html))} 次")

print()
print("=" * 66)
print("2. API 阶段数据完整性")
print("=" * 66)

#期望值说明：
#  "ok"          —— 期望能溯源，但**允许失败**：
#                   实测 LLM 输出不稳定，同一问题两次跑结果可能不同。
#                   这里只统计观测，不做通过判定 ——
#                   把不稳定当成 bug 去"修"，等于把风险藏起来。
#  "fabricated"  —— 期望看到编造，用来演示防线
#  "none"        —— 不引用条款号，只看链路完整性
CASES = [
    ("建筑消防设施包括哪些", "ok"),
    ("公众聚集场所投入使用需要什么手续", "ok"),
    ("消防救援机构有哪些职责", "fabricated"),
    ("消防安全重点单位有哪些消防安全义务？", "none"),
]

# 统计多次运行的可溯源率
trace_stats: list[tuple[str, int, int]] = []

for q, expect in CASES:
    print(f"\n  [{q}]  期望: {expect}")
    try:
        req = urllib.request.Request(
            f"{BASE}/api/demo/ask",
            data=json.dumps({"question": q, "top_k": 5}).encode(),
            headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=200).read())
    except Exception as e:                        # noqa: BLE001
        check("请求成功", False, f"{type(e).__name__}: {e}")
        continue

    stages = r.get("stages") or []
    names = [s["name"] for s in stages]
    check("阶段非空", len(stages) >= 3, f"{names}")
    check("含检索阶段", any("检索" in n for n in names))
    check("含生成阶段", any("生成" in n for n in names))
    check("含弃答判定", any("弃答" in n for n in names))

    frags = r.get("fragments") or []
    check("命中片段非空", len(frags) > 0, f"{len(frags)} 条")
    if frags:
        f0 = frags[0]
        for k in ("rank", "score", "source", "preview", "chars"):
            check(f"片段含 {k}", k in f0)
        ranks = [f["rank"] for f in frags]
        check("rank 连续", ranks == list(range(1, len(ranks) + 1)),
              str(ranks))

    print()
    print("       3. 溯源三态")
    check("fragment_count 存在", "fragment_count" in r,
          f"= {r.get('fragment_count')}")
    check("fragment_count 与实际一致",
          r.get("fragment_count") == len(frags),
          f"{r.get('fragment_count')} vs {len(frags)}")

    cs = r.get("citations") or []
    for c in cs:
        ok = ("risk" in c and "verified" in c
              and "source_excerpt" in c and "article" in c)
        if not ok:
            check("引用字段齐全", False, str(c)[:80])
            break
    else:
        check(f"全部 {len(cs)} 处引用字段齐全", True)

    risks = {c["risk"] for c in cs}
    check("risk 取值合法", risks <= {"ok", "fuzzy", "fabricated"},
          str(risks))
    n_ok = sum(1 for c in cs if c["risk"] == "ok")
    trace_stats.append((q, n_ok, len(cs)))

    if expect == "ok":
        # ★ 只观测不判定：LLM 引用不稳定是事实，不是 bug。
        #   同一问题跑两次可能一次 ok 一次 fabricated，
        #   把它当断言会让验收随机失败；
        #   假装它稳定又是在自欺。
        obs = "可溯源" if n_ok == len(cs) else (
            "部分可溯源" if n_ok else "全编造")
        print(f"       [观测] 引用可溯源性：{obs}（{n_ok}/{len(cs)}）")
        print("             ↑ LLM 输出不稳定，引用准确性需多次统计，"
              "不能靠单次演示下结论")
    elif expect == "fabricated":
        check("预期含编造（用于演示防线）",
              "fabricated" in risks or not cs, f"risks={risks}")

    print(f"       弃答={r.get('abstained')} "
          f"可溯源={r.get('verified_citations')}/{len(cs)} "
          f"总耗时={r.get('total_ms')}ms")

print()
print("=" * 66)
print("4. 前端引用字段 vs 后端返回（反向检查）")
print("=" * 66)

html = pathlib.Path(HERE / "index.html").read_text(encoding="utf-8")
# 前端访问 r.xxx 的字段
used = set(re.findall(r"\br\.([a-z_]+)", html))
print(f"  前端读取 {len(used)} 个字段: {sorted(used)}")

req = urllib.request.Request(
    f"{BASE}/api/demo/ask",
    data=json.dumps({"question": "建筑消防设施包括哪些",
                     "top_k": 5}).encode(),
    headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=200).read())
missing = sorted(f for f in used if f not in r)
check("前端字段全部由后端提供", not missing,
      f"缺失: {missing}" if missing else "")

# 阶段内的字段
stage_fields = set()
for s in r.get("stages") or []:
    stage_fields |= set(s.keys())
# ★ 只扫 renderStages 函数体内的 s.* —— 那个 s 才是 stage。
#   全文扫会把 SAMPLE 常量里的 s.q / s.n 也算进来
#   （那里的 s 来自示例列表 {q,n}），产生假报警。
fn_start = html.find("function renderStages")
fn_end = html.find("function renderResult")
render_src = (html[fn_start:fn_end]
              if fn_start > 0 and fn_end > fn_start else html)
used_stage = set(re.findall(r"\bs\.([a-z_]+)", render_src))
miss2 = sorted(f for f in used_stage if f not in stage_fields)
check("阶段字段齐全（限定 renderStages 作用域）", not miss2,
      f"缺失: {miss2}" if miss2 else f"实际: {sorted(stage_fields)}")

print()
print("=" * 66)
print("5. 弃答路径")
print("=" * 66)
try:
    req = urllib.request.Request(
        f"{BASE}/api/demo/ask",
        data=json.dumps({"question": "   ", "top_k": 5}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=30).read())
    check("空问题返回结构化错误", "error" in r, str(r)[:60])
except urllib.error.HTTPError as e:
    body = json.loads(e.read().decode("utf-8"))
    check("空问题返回 400 + error", "error" in body, str(body)[:60])
except Exception as e:                            # noqa: BLE001
    check("空问题处理", False, str(e))

print()
print("=" * 66)
print("6. 引用可溯源性汇总（观测项，非断言）")
print("=" * 66)
tot_ok = sum(a for _, a, _ in trace_stats)
tot_all = sum(b for _, _, b in trace_stats)
print()
for q, a, b in trace_stats:
    rate = f"{a/b*100:.0f}%" if b else "无引用"
    print(f"  {q[:30]:32s} {a}/{b}可溯源  {rate}")
print()
if tot_all:
    print(f"  合计 {tot_ok}/{tot_all} = {tot_ok/tot_all*100:.0f}%")
    print()
    print("  ★ 这个比例低，说明当前 prompt 下模型经常引用检索结果里")
    print("    不存在的条款号。消防执法场景下这是硬伤，")
    print("    所以项目里配套了三道防线（生成约束 / MCP 高风险标注 / 人工确认）。")
    print("    面试时讲这个比讲「引用准确率 100%」有说服力 ——")
    print("    前者展示的是判断力，后者一戳就破。")

print()
print("=" * 66)
print(f"结果：{'全部通过' if not fail else f'{fail} 项失败'}")
print("=" * 66)
raise SystemExit(1 if fail else 0)