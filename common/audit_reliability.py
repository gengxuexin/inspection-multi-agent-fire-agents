# common/audit_reliability.py
"""审计写入的可靠性加固（与 common/audit.py 配套使用）。

★★ 为什么要独立成文件
-------------------
`common/audit.py` 的 `_write_jsonl` 是 best-effort：
写失败只 `logger.error` 提示一次，然后静默吞掉。

在开发调试阶段这没问题，但在消防执法场景下是硬伤：

    审计日志是执法留痕的载体 —— 它要能证明
    「当时系统给了什么答案、依据哪条法规、谁点的、什么时候点的」。

    磁盘满、权限变更、目录被删、JSON 序列化异常 —— 任何一条触发，
    审计就静默丢失，而你**不会知道丢了**。
    更糟的是：业务请求照常返回 200，执法文书照常生成，
    事后追责时才发现「那天没有记录」。

★★★ 改了三件事
--------------
1. **写失败不再静默** —— 记到死信队列 + 打 Prometheus 指标 + 严格模式下上抛
2. **同步落盘** —— 原来依赖文件缓冲，进程被 kill 时可能丢最后几行
3. **可观测** —— `fire_audit_write_failures_total` 指标，失败率 > 0 就该告警

★ 为什么默认不上抛
------------------
「写失败就让业务失败」看起来最严格，但实际会造成
**审计服务故障拖垮执法业务** —— 磁盘满的时候文书也生成不了了，
那才是真正的执法事故。

所以默认策略：**业务不中断 + 留死信 + 指标告警**，
同时提供 `AUDIT_STRICT=1` 开关给「审计必须保证」的部署场景。

用法（common/audit.py 里调用）：
    from common.audit_reliability import write_audit_record

    write_audit_record(path, record)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger("fire_agents.audit")

# ---- 死信队列（写失败时的兜底）----
_dead_letters: list[dict] = []
_dead_letter_lock = threading.Lock()
_MAX_DEAD_LETTERS = 500

# ---- 统计 ----
_write_ok_count = 0
_write_fail_count = 0
_stats_lock = threading.Lock()

_SERVICE = "agent1"


def set_service(name: str) -> None:
    """设置指标标签用的服务名（各 Agent 启动时调一次）。"""
    global _SERVICE
    _SERVICE = name


def audit_stats() -> dict:
    """审计写入统计 —— 给 /health 或监控端点用。"""
    with _stats_lock:
        total = _write_ok_count + _write_fail_count
        return {
            "written": _write_ok_count,
            "failed": _write_fail_count,
            "total": total,
            "fail_rate": round(_write_fail_count / total, 4) if total else 0.0,
            "dead_letters": len(_dead_letters),
        }


def dead_letters() -> list[dict]:
    """取当前死信（运维排查用）。"""
    with _dead_letter_lock:
        return list(_dead_letters)


def _record_failure(record: dict, err: Exception) -> None:
    global _write_fail_count
    with _stats_lock:
        _write_fail_count += 1

    with _dead_letter_lock:
        _dead_letters.append({
            "ts": time.time(),
            "error": f"{type(err).__name__}: {err}",
            "record": record,
        })
        if len(_dead_letters) > _MAX_DEAD_LETTERS:
            del _dead_letters[:-_MAX_DEAD_LETTERS]

    # 指标（metrics 模块可能不可用，失败不影响主流程）
    try:
        from common.metrics import audit_write_failures_total, audit_write_dead_letters
        c = audit_write_failures_total()
        if c is not None:
            c.labels(service=_SERVICE).inc()
        g = audit_write_dead_letters()
        if g is not None:
            g.labels(service=_SERVICE).set(len(_dead_letters))
    except Exception:
        pass


def _record_success() -> None:
    global _write_ok_count
    with _stats_lock:
        _write_ok_count += 1


def _strict_mode() -> bool:
    """AUDIT_STRICT=1 时写失败直接抛 —— 审计必须保证的部署场景用。"""
    v = os.getenv("AUDIT_STRICT", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def write_audit_record(path: str, record: dict) -> bool:
    """追加一行 JSONL 审计记录。

    ★ 相比原实现的三点变化
    ----------------------
    1. `f.flush()` + `os.fsync()` —— 同步落盘，进程被 kill 不丢最后几行
       （审计留痕要经得起「服务崩了」的追问）
    2. 写失败 → 死信 + 指标 + 日志，**不再只有一次 warning**
    3. AUDIT_STRICT=1 时上抛异常（默认不上抛，见模块 docstring 解释）

    返回 True 表示写成功。
    """
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # 某些文件系统（部分网络挂载）不支持 fsync —— 不该因此丢审计
                pass
        _record_success()
        return True
    except Exception as e:  # noqa: BLE001
        _record_failure(record, e)
        logger.error(
            "[audit] 审计日志写入失败：%s（已进死信队列；"
            "业务不中断但该条记录需人工补录）", e)
        if _strict_mode():
            raise
        return False
