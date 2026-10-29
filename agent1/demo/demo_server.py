#!/usr/bin/env python3
"""RAG 链路演示服务。

    python agent1/demo/demo_server.py --port 8800

为什么单独一个服务而不挂到 agent1 里：
  agent1 有内部 Token 鉴权，而演示现场不该输token；
  且演示要能独立启动（agent1 起不来时至少能看链路）。
  两个进程互不阻塞 —— 演示环境里这条很重要。

★ 设计上的一条硬规矩：
  页面上的每个数字都来自 agent1/demo/trace_pipeline.py 的实跑结果，
  没有 mock、没有兜底的假数据。
  宁可显示"检索失败"，也不显示一个编出来的漂亮结果 ——
  演示时被抓到造假，比不演示严重得多。
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "agent1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent1.demo.trace_pipeline import (          # noqa: E402
    SAMPLE_QUESTIONS, trace_answer, trace_retrieval,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("demo")

HERE = pathlib.Path(__file__).resolve().parent

app = FastAPI(title="消防智能执法 · RAG 链路演示", docs_url="/api/docs")


@app.on_event("startup")
async def warmup():
    """启动时后台预热，不阻塞页面加载。

    ★ 为什么要预热（实测数据）：
      冷启动首次检索 6.5 秒，第二次起降到 0.6~1.2 秒。
      差值来自三处：embedding 模型加载、4019 片段的 jieba 分词、
      FAISS 索引载入。

      演示场景下 6 秒的空白足够让人以为服务坏了。
      预热把这笔开销挪到启动阶段，用一个跟演示问题无关的查询即可
      （embedding 模型和分词结果是可复用的，与查询内容无关）。

    ★ 顺带这也是个可讲的数字：
      "首包 6.5s、稳定后 1s 内" 是冷启动的典型特征。
      生产上要么常驻预热，要么就得承认首包延迟高 ——
      很多项目是 demo 时预热、上线后忘了，才暴露出首包问题。
    """
    import threading

    def _warm():
        try:
            t0 = time.perf_counter()
            trace_retrieval("消防救援机构有哪些职责", top_k=3)
            logger.info("预热完成，耗时 %.1f 秒",
                        time.perf_counter() - t0)
        except Exception as e:                     # noqa: BLE001
            # 预热失败不影响服务，只是首次查询会慢一点
            logger.warning("预热失败（不影响使用）：%s", e)

    threading.Thread(target=_warm, daemon=True).start()


class AskReq(BaseModel):
    question: str
    top_k: int = 5


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/samples")
def samples():
    return {"samples": SAMPLE_QUESTIONS}


@app.post("/api/demo/ask")
def ask(req: AskReq):
    """跑一次完整链路并返回全部中间态。

    ★ 异常一律返回结构化错误，不抛栈 ——
      演示时页面崩掉比返回一条错误信息糟糕得多。
    """
    q = (req.question or "").strip()
    if not q:
        return JSONResponse({"error": "问题为空"}, status_code=400)

    top_k = max(1, min(int(req.top_k or 5), 10))
    logger.info("演示查询：%s (top_k=%d)", q[:40], top_k)

    try:
        result = trace_answer(q, top_k=top_k)
        # 补一个溯源结论，页面直接展示即可
        cites = result.get("citations") or []
        unverified = [c for c in cites if not c.get("verified")]
        if cites:
            result["trace_summary"] = (
                f"{len(cites)} 处引用中 {len(cites) - len(unverified)} 处"
                f"能在检索片段里定位"
                + (f"，{len(unverified)} 处无法溯源（模型引用了检索结果里"
                   f"不存在的条款号）" if unverified else "")
            )
        else:
            result["trace_summary"] = "本轮答案未引用具体条款号"
        return result
    except Exception as e:                          # noqa: BLE001
        logger.exception("链路失败")
        # 检索阶段可能已经跑出一部分，保留它让页面还能显示点东西
        try:
            partial = trace_retrieval(q, top_k)
            partial["error"] = f"生成阶段失败：{type(e).__name__}: {e}"
            return JSONResponse(partial, status_code=200)
        except Exception as e2:                     # noqa: BLE001
            return JSONResponse(
                {"error": f"{type(e).__name__}: {e}（检索阶段同样失败：{e2}）"},
                status_code=500,
            )


@app.get("/")
def index():
    return FileResponse(str(HERE / "index.html"))


app.mount("/static", StaticFiles(directory=str(HERE)), name="static")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    import uvicorn
    logger.info("演示服务启动 http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())