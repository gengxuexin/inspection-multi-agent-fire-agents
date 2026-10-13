# common/tasks.py
"""异步任务状态管理：Redis 后端 + InMemory 回退。"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from common.config import get_settings

PENDING = "pending"
RUNNING = "running"
SUCCESS = "success"
FAILED = "failed"


def new_task_id() -> str:
    return f"task_{uuid.uuid4().hex[:12]}"


class InMemoryTaskStore:
    """内存任务存储（Redis 不可用时回退，进程重启丢失）。"""

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    async def create(self, task_id: str, payload: dict | None = None) -> dict:
        rec = {"task_id": task_id, "status": PENDING,
               "created_at": time.time(), "updated_at": time.time(),
               "payload": payload or {}, "result": None,
               "error": None, "progress": 0, "events": []}
        self._store[task_id] = rec
        return rec

    async def set_running(self, task_id: str) -> None:
        rec = self._store.get(task_id)
        if rec:
            rec["status"] = RUNNING
            rec["updated_at"] = time.time()

    async def save_result(self, task_id: str, result: Any) -> None:
        rec = self._store.get(task_id)
        if rec:
            rec["status"] = SUCCESS
            rec["result"] = result
            rec["progress"] = 100
            rec["updated_at"] = time.time()

    async def set_error(self, task_id: str, error: str) -> None:
        rec = self._store.get(task_id)
        if rec:
            rec["status"] = FAILED
            rec["error"] = error
            rec["updated_at"] = time.time()

    async def set_progress(self, task_id: str, progress: int) -> None:
        rec = self._store.get(task_id)
        if rec:
            rec["progress"] = max(0, min(100, int(progress)))
            rec["updated_at"] = time.time()

    async def add_event(self, task_id: str, event: str, data: dict | None = None) -> None:
        rec = self._store.get(task_id)
        if rec:
            rec.setdefault("events", []).append(
                {"ts": time.time(), "event": event, "data": data or {}})

    async def get(self, task_id: str) -> dict | None:
        return self._store.get(task_id)


class TaskStore:
    """Redis 任务存储。"""

    def __init__(self, redis_url: str) -> None:
        import redis.asyncio as aioredis
        self._r = aioredis.from_url(redis_url, decode_responses=True)

    async def create(self, task_id: str, payload: dict | None = None) -> dict:
        rec = {"task_id": task_id, "status": PENDING,
               "created_at": time.time(), "updated_at": time.time(),
               "payload": payload or {}, "result": None,
               "error": None, "progress": 0, "events": []}
        await self._r.set(f"task:{task_id}", json.dumps(rec, ensure_ascii=False))
        return rec

    async def set_running(self, task_id: str) -> None:
        await self._update(task_id, status=RUNNING)

    async def save_result(self, task_id: str, result: Any) -> None:
        await self._update(task_id, status=SUCCESS, result=result, progress=100)

    async def set_error(self, task_id: str, error: str) -> None:
        await self._update(task_id, status=FAILED, error=error)

    async def set_progress(self, task_id: str, progress: int) -> None:
        await self._update(task_id, progress=max(0, min(100, int(progress))))

    async def add_event(self, task_id: str, event: str, data: dict | None = None) -> None:
        rec = await self.get(task_id)
        if rec is None:
            return
        rec.setdefault("events", []).append(
            {"ts": time.time(), "event": event, "data": data or {}})
        await self._r.set(f"task:{task_id}", json.dumps(rec, ensure_ascii=False))

    async def get(self, task_id: str) -> dict | None:
        raw = await self._r.get(f"task:{task_id}")
        if raw:
            return json.loads(raw)
        return None

    async def _update(self, task_id: str, **fields: Any) -> None:
        rec = await self.get(task_id)
        if rec is None:
            return
        rec.update(fields)
        rec["updated_at"] = time.time()
        await self._r.set(f"task:{task_id}", json.dumps(rec, ensure_ascii=False))


_store_instance: Any = None


def get_task_store() -> InMemoryTaskStore | TaskStore:
    """获取任务存储单例：Redis 配置可用 → TaskStore，否则 InMemoryTaskStore。"""
    global _store_instance
    if _store_instance is not None:
        return _store_instance
    url = ""
    try:
        url = get_settings().redis_url
    except Exception:
        url = ""
    if url:
        try:
            import redis.asyncio  # noqa: F401
            _store_instance = TaskStore(url)
            return _store_instance
        except ImportError:
            pass
    _store_instance = InMemoryTaskStore()
    return _store_instance
