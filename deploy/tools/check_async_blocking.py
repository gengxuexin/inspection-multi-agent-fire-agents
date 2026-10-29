"""用 AST 严格检查：async 端点里是否还有同步阻塞调用。

★ 为什么用 AST 而不是 grep
--------------------------
grep 会把 `def` 端点（本来就在线程池，没问题）也报出来，
也会漏掉多行调用的第二行。用 AST 解析才能准确知道：
「这个函数是不是 async」+「它里面调了哪些已知阻塞函数」。

判断依据是 FastAPI 的实际行为：
- `def` 端点     → 自动 threadpool，不阻塞事件循环 ✅
- `async def`   → 事件循环内执行，同步调用会卡住 worker ❌
"""
import ast
import sys

PATH = '/home/geng/fire_agent_v2/fire_agents/agent1/server.py'

# 已知会阻塞的同步函数（内含 LLM HTTP 调用 / 文件 IO / docx 渲染）
BLOCKING = {
    'rag_ask',
    'extract_legal_fields_from_case',
    'retrieve_legal_context',
    'generate_one',
    'analyze_case_file',
    'analyze_pdf',
    'render_from_facts',
    'rag_legal_advisor',
    'template_handler',
    'file_analyzer',
    'pdf_analyzer',
    'pdf_converter',
    'docx_to_pdf',
}

src = open(PATH, encoding='utf-8').read()
tree = ast.parse(src)

problems = []
checked = 0

for node in ast.walk(tree):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
    is_async = isinstance(node, ast.AsyncFunctionDef)
    # 只看端点（被 @app 装饰的）
    is_endpoint = any(
        isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
        and d.func.attr in ('get', 'post', 'put', 'delete', 'patch')
        for d in node.decorator_list
    )
    if not is_endpoint:
        continue
    checked += 1

    if not is_async:
        # def 端点：FastAPI 自动 threadpool，安全
        continue

    # async 端点：检查内部调用
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fname = None
            if isinstance(sub.func, ast.Name):
                fname = sub.func.id
            elif isinstance(sub.func, ast.Attribute):
                fname = sub.func.attr
            if fname in BLOCKING:
                # 排除已经在 await run_in_threadpool(...) 里的
                parent_ok = False
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call) and \
                            isinstance(inner.func, ast.Attribute) and \
                            inner.func.attr == 'run_in_threadpool':
                        if any(a is sub for a in inner.args):
                            parent_ok = True
                if not parent_ok:
                    problems.append((node.name, sub.lineno, fname))

print(f"检查了 {checked} 个端点")
if problems:
    print(f"\n❌ async 端点里还有 {len(problems)} 处阻塞调用：")
    for name, line, fname in problems:
        print(f"   {name}() L{line}  →  {fname}()")
    sys.exit(1)
else:
    print("✅ async 端点里没有裸的同步阻塞调用（全部已包 run_in_threadpool）")
    sys.exit(0)
