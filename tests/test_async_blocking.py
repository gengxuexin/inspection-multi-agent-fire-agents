# tests/test_async_blocking.py
"""检查 async 端点里有没有裸的同步阻塞调用。

★★ 为什么需要这个测试
----------------------
FastAPI 的行为有一条容易踩的规则：

- ``def`` 端点     → FastAPI **自动**丢到 threadpool，不阻塞事件循环 ✅
- ``async def`` 端点 → **在事件循环里直接执行**，
  里面的同步调用会**卡住整个 worker** ❌

本项目的 LLM 调用（``rag_ask`` / ``generate_one`` / ``extract_legal_fields_from_case`` …）
全是**同步阻塞**的 HTTP 请求，最长 30 秒。
放在 ``async def`` 端点里，就是「一个请求把所有其它请求一起堵死」。

★ 我在这次改造中真实踩过这个坑
------------------------------
1. 第一版用 grep 找阻塞调用 → 漏掉了多行调用的第二行
2. 改完发现 ``api_generate_doc`` 是 ``def`` 端点，
   本来就在线程池里，我却加了 ``await`` → 直接 SyntaxError
3. 改用 AST 解析后，又抓出一处 grep 漏掉的 ``api_retrieve``

所以这个测试用 AST 而不是文本匹配 ——
它能准确区分「async 端点」和「def 端点」，
也能识别多行调用。

★ 断言的意义
------------
这不是「代码风格检查」，是**并发正确性的门禁**：
新增一个 async 端点时如果忘了包线程池，这个测试会直接失败。
"""
import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "agent1", "server.py")

# 已知会阻塞的同步函数（内含 LLM HTTP 调用 / docx 渲染 / 文件 IO）
BLOCKING_FUNCS = {
    "rag_ask",
    "extract_legal_fields_from_case",
    "retrieve_legal_context",
    "generate_one",
    "analyze_case_file",
    "analyze_pdf",
    "render_from_facts",
    "rag_legal_advisor",
    "template_handler",
    "file_analyzer",
    "pdf_analyzer",
    "pdf_converter",
    "docx_to_pdf",
}


def _iter_endpoints(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_endpoint = any(
            isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
            and d.func.attr in ("get", "post", "put", "delete", "patch")
            for d in node.decorator_list
        )
        if is_endpoint:
            yield node


def _wrapped_in_threadpool(func_node, call_node):
    """该调用是否已经包在 run_in_threadpool(...) 里。"""
    for inner in ast.walk(func_node):
        if isinstance(inner, ast.Call) and \
                isinstance(inner.func, ast.Attribute) and \
                inner.func.attr == "run_in_threadpool":
            if any(a is call_node for a in inner.args):
                return True
    return False


def test_no_blocking_call_in_async_endpoint():
    """★ 核心断言：async 端点里不能有裸的同步阻塞调用。"""
    problems = []
    for node in _iter_endpoints(TARGET):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue    # def 端点由 FastAPI 自动线程池化，安全
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            if isinstance(sub.func, ast.Name):
                fname = sub.func.id
            elif isinstance(sub.func, ast.Attribute):
                fname = sub.func.attr
            else:
                continue
            if fname in BLOCKING_FUNCS:
                if not _wrapped_in_threadpool(node, sub):
                    problems.append(
                        f"{node.name}() L{sub.lineno} 调用 {fname}() 未包线程池")
    assert not problems, (
        "以下 async 端点里有裸的同步阻塞调用（会卡住整个 worker）：\n  "
        + "\n  ".join(problems)
    )


def test_threadpool_import_present():
    """确保 run_in_threadpool 被导入了 —— 否则上面的修复会 NameError。"""
    src = open(TARGET, encoding="utf-8").read()
    assert "from starlette.concurrency import run_in_threadpool" in src


def test_has_async_endpoints_to_check():
    """防止检查器因为「没有 async 端点」而空转。"""
    n = sum(1 for node in _iter_endpoints(TARGET)
            if isinstance(node, ast.AsyncFunctionDef))
    assert n > 0, "应该至少有 async 端点，否则这个测试没有意义"
