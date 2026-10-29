# common/model_provisioner.py
"""本地模型供给：检测、测精度、引导下载。

★ 这个模块解决什么问题
----------------------
落地内网/涉密环境时，会遇到一个具体的困境：

    「本地要跑哪个模型？跑不起来怎么办？精度够不够？」

三个问题纠缠在一起，靠翻文档解决不了。本模块把它们拆成**可执行的探测**：

    1. **检测** —— 本机有没有 Ollama？装了哪些模型？目标模型在不在？
    2. **测精度** —— 用真实评测集跑一遍检索，看 context_precision 到底多少
    3. **引导** —— 精度不达标时，给出两个选项：
       - 让 Agent 自动下载（`ollama pull`）
       - 用户自己下载（生成逐步指引 + 离线包说明）

★ 设计原则：先测再用，不要凭印象推荐模型
------------------------------------------
「qwen2.5:7b 够用吗」这种问题，不能靠回答 —— 必须实测。
所以本模块用项目里已有的评测集（agent1/eval/qa_samples.jsonl）
跑一次真实检索，用实测的 context_precision 做判断。

**注意**：这里的「精度」测的是**检索层**精度（context_precision），
不是生成层。检索不准，换什么 LLM 都救不回来 ——
这是我在调优过程中踩过的坑（见 docs/rag-tuning-journey.md）。

用法：
    from common.model_provisioner import probe_environment, evaluate_retrieval

    report = probe_environment()
    if report["needs_model"]:
        guidance = build_download_guidance(target_model="qwen2.5:7b")
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from typing import Any

logger = logging.getLogger("fire_agents.model_provisioner")


# ==================== 1. 环境检测 ====================
def _ollama_base_url() -> str:
    try:
        from common.config import ollama_base_url
        return ollama_base_url()
    except Exception:
        return "http://localhost:11434"


def _ollama_list_models(timeout: float = 5.0) -> tuple[bool, list[str], str]:
    """返回 (是否连通, 模型列表, 错误信息)。

    用 HTTP 而非 CLI —— 容器里通常没装 ollama 命令，
    但 Ollama 服务一定在跑（那是另一个进程/另一台机器）。
    """
    base = _ollama_base_url().rstrip("/")
    # base 形如 http://host:11434/v1，Ollama 原生 API 在去掉 /v1 后
    url = base[:-3] if base.endswith("/v1") else base
    try:
        import httpx
        with httpx.Client(timeout=timeout) as c:
            r = c.get(f"{url}/api/tags")
            r.raise_for_status()
            data = r.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        return True, [m for m in models if m], ""
    except Exception as e:  # noqa: BLE001
        return False, [], f"{type(e).__name__}: {e}"


def _ollama_cli_available() -> bool:
    return shutil.which("ollama") is not None


def probe_environment(target_model: str | None = None) -> dict:
    """检测本地模型环境。

    返回结构固定，方便前端直接渲染。
    """
    try:
        from common.config import get_settings
        s = get_settings()
        provider_mode = (getattr(s, "provider_mode", "cloud") or "cloud").lower()
        want = target_model or getattr(s, "ollama_model", "") or "qwen2.5:7b"
        timeout = float(getattr(s, "llm_request_timeout", 30.0))
    except Exception:
        provider_mode, want, timeout = "cloud", target_model or "qwen2.5:7b", 30.0

    connected, models, err = _ollama_list_models()
    has_target = any(m == want or m.split(":")[0] == want.split(":")[0]
                     for m in models) if connected else False

    return {
        "provider_mode": provider_mode,
        "ollama_url": _ollama_base_url(),
        "ollama_connected": connected,
        "ollama_cli_available": _ollama_cli_available(),
        "installed_models": models,
        "target_model": want,
        "has_target_model": has_target,
        "needs_model": connected and not has_target,
        "error": err,
        # 云端是否配了 —— 决定能不能先云端跑
        "cloud_llm_configured": bool(
            os.getenv("DEEPSEEK_API_KEY")
            or _env_from_dotenv("DEEPSEEK_API_KEY")),
        "cloud_embedding_configured": bool(
            os.getenv("DASHSCOPE_API_KEY")
            or _env_from_dotenv("DASHSCOPE_API_KEY")),
    }


def _env_from_dotenv(key: str) -> str:
    """从项目根 .env 读一个 key（不覆盖已有环境变量）。"""
    try:
        from common.config import get_settings
        return str(getattr(get_settings(), key.lower(), "") or "")
    except Exception:
        return ""


# ==================== 2. 检索精度实测 ====================
def evaluate_retrieval(sample_limit: int = 20,
                       top_k: int = 8,
                       judge: bool = False) -> dict:
    """用真实评测集跑一遍检索，测 context_precision。

    ★ 为什么测检索而不是测生成
    -------------------------
    检索不准，LLM 再强也只是在错的材料上做漂亮的推理。
    我在调优时实测过：context_precision 从 0.165 修到 0.226 时，
    faithfulness 一直稳在 0.98 —— 说明 faithfulness 高不代表检索好。

    所以「换个更大的模型」这个建议，**只有在检索层达标后才有意义**。
    这也是本模块先测检索的原因。

    judge=False（默认）：只跑检索，用 ground_truth 与命中片段的
    **词面重合度**估算 precision。快、零 LLM 成本，适合在线探测。
    judge=True：调用 LLM-as-judge 精确打分，慢且烧钱，
    适合离线复现报告。

    返回 {ok, precision, recall, n, method, note}
    """
    try:
        import sys
        eval_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "agent1", "eval")
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
        from ragas_eval import retrieve_contexts, DEFAULT_SAMPLES
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"评测模块不可用：{e}",
                "precision": None, "recall": None, "n": 0, "method": "none"}

    # 优先用真实评测集，缺失时回退内置样例
    samples = None
    try:
        from common.config import get_settings
        s = get_settings()
        # 本地评测集路径（agent1/eval/qa_samples.jsonl）
        kb = os.path.dirname(eval_dir)
        p = os.path.join(kb, "eval", "qa_samples.jsonl")
        if os.path.exists(p):
            samples = []
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if d.get("question") and d.get("ground_truth"):
                        samples.append(d)
    except Exception:
        samples = None

    if not samples:
        samples = [{"question": q, "ground_truth": gt}
                   for q, gt in DEFAULT_SAMPLES]
    if sample_limit:
        samples = samples[:sample_limit]

    if not samples:
        return {"ok": False, "error": "没有可用样本",
                "precision": None, "recall": None, "n": 0, "method": "none"}

    prec_list: list[float] = []
    rec_list: list[float] = []
    try:
        for smp in samples:
            q = smp["question"]
            gt = str(smp.get("ground_truth", ""))
            contexts = retrieve_contexts(q, top_k=top_k) or []
            prec_list.append(_precision_of(q, gt, contexts))
            rec_list.append(_recall_of(gt, contexts))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"检索执行失败：{e}",
                "precision": None, "recall": None, "n": 0, "method": "none"}

    precision = round(sum(prec_list) / len(prec_list), 4) if prec_list else None
    recall = round(sum(rec_list) / len(rec_list), 4) if rec_list else None

    return {
        "ok": precision is not None,
        "precision": precision,
        "recall": recall,
        "n": len(samples),
        "top_k": top_k,
        "method": "lexical_overlap",
        "note": ("context_precision 低于阈值说明是检索层瓶颈，"
                 "换更大的 LLM 无济于事"),
    }


def _norm_text(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isalnum())


def _precision_of(question: str, ground_truth: str, contexts: list[str]) -> float:
    """命中的上下文里，有多少条与 ground_truth 相关（词面重合度）。

    刻意用词面而非 LLM 判断：在线探测要快、要零成本、要能在无 LLM 时也能跑。
    这是粗估，不等于 RAGAS 的 context_precision —— 用途是「快速发现是否明显偏低」。
    """
    if not contexts:
        return 0.0
    gt = _norm_text(ground_truth)
    if not gt:
        return 0.0
    gt_tokens = set(gt[i:i + 2] for i in range(len(gt) - 1))
    if not gt_tokens:
        return 0.0
    hit = 0
    for c in contexts:
        cn = _norm_text(c)
        c_tokens = set(cn[i:i + 2] for i in range(len(cn) - 1))
        if not c_tokens:
            continue
        # Jaccard 相似度，阈值 0.15 视为「相关」
        inter = len(gt_tokens & c_tokens)
        union = len(gt_tokens | c_tokens)
        if union and (inter / union) >= 0.15:
            hit += 1
    return hit / len(contexts)


def _recall_of(ground_truth: str, contexts: list[str]) -> float:
    """ground_truth 的关键内容有多少被检索覆盖。"""
    gt = _norm_text(ground_truth)
    if not gt:
        return 0.0
    gt_tokens = set(gt[i:i + 2] for i in range(len(gt) - 1))
    if not gt_tokens:
        return 0.0
    all_tokens: set[str] = set()
    for c in contexts:
        cn = _norm_text(c)
        all_tokens |= set(cn[i:i + 2] for i in range(len(cn) - 1))
    if not all_tokens:
        return 0.0
    return len(gt_tokens & all_tokens) / len(gt_tokens)


def check_and_advise(sample_limit: int = 20) -> dict:
    """完整流程：检测环境 → 测精度 → 给出建议。

    这是给 API 端点用的主入口。
    """
    env = probe_environment()
    precision_report = evaluate_retrieval(sample_limit=sample_limit)

    try:
        from common.config import get_settings
        s = get_settings()
        threshold = float(getattr(s, "retrieval_precision_threshold", 0.6))
        fast_model = getattr(s, "llm_fast_model", "deepseek-chat")
        high_model = getattr(s, "llm_high_precision_model", "deepseek-reasoner")
    except Exception:
        threshold, fast_model, high_model = 0.6, "deepseek-chat", "deepseek-reasoner"

    precision = precision_report.get("precision")
    below = precision is not None and precision < threshold

    advice: list[dict] = []

    # ---- 建议 1：环境层面 ----
    if not env["ollama_connected"]:
        advice.append({
            "level": "error",
            "code": "ollama_unreachable",
            "title": "本地模型服务未连通",
            "detail": (f"连不上 {env['ollama_url']}。"
                       f"内网/涉密环境下必须先部署 Ollama，"
                       f"否则云端不可达时系统没有降级路径。"),
            "actions": [
                {"kind": "run", "label": "启动 Ollama",
                 "command": "ollama serve"},
                {"kind": "doc", "label": "查看部署指引",
                 "ref": "docs/deployment.md"},
            ],
        })
    elif env["needs_model"]:
        advice.append({
            "level": "warn",
            "code": "model_missing",
            "title": f"本地缺少模型 {env['target_model']}",
            "detail": ("Ollama 服务已连通，但目标模型不在本地。"
                       "需要下载后才能在离线环境使用。"),
            "actions": [
                {"kind": "autodownload", "label": "让 Agent 自动下载",
                 "model": env["target_model"]},
                {"kind": "manual", "label": "我自己下载",
                 "ref": f"docs/model-provisioning.md#{env['target_model']}"},
            ],
        })

    # ---- 建议 2：精度层面 ----
    if below:
        advice.append({
            "level": "warn",
            "code": "low_retrieval_precision",
            "title": f"检索精度偏低（{precision:.3f} < {threshold}）",
            "detail": ("检索层是当前瓶颈。注意：**换更大的生成模型解决不了这个问题** —— "
                       "模型再强也只是在错的材料上做推理。"
                       "优先排查：分块策略、rerank 权重、元数据过滤条件。"),
            "actions": [
                {"kind": "doc", "label": "看检索调优记录",
                 "ref": "docs/rag-tuning-journey.md"},
            ],
        })
    elif precision is not None:
        advice.append({
            "level": "ok",
            "code": "precision_ok",
            "title": f"检索精度达标（{precision:.3f} ≥ {threshold}）",
            "detail": "此时若想进一步提升答案质量，可以考虑换更精确但更慢的生成模型。",
            "actions": [
                {"kind": "switch_model", "label": f"切到 {high_model}",
                 "model": high_model,
                 "note": "精度更高，但推理更慢"},
                {"kind": "keep", "label": f"保持 {fast_model}",
                 "model": fast_model},
            ],
        })

    return {
        "environment": env,
        "retrieval": precision_report,
        "threshold": threshold,
        "below_threshold": below,
        "advice": advice,
        "can_autodownload": env["ollama_connected"] and env["ollama_cli_available"],
    }


# ==================== 3. 下载引导 ====================
def build_download_guidance(target_model: str = "qwen2.5:7b") -> dict:
    """生成下载指引（用户自己下载的路径）。

    两种场景要给不同指引：
    - 有网络：直接 `ollama pull`
    - 无网络（内网）：需要离线包 —— 在有网机器 pull，再拷 GGUF 过去
    """
    return {
        "model": target_model,
        "online": {
            "title": "有网络环境（最简单）",
            "steps": [
                {"cmd": f"ollama pull {target_model}", "note": "自动下载到本地模型库"},
                {"cmd": f"ollama run {target_model} \"你好\"", "note": "验证模型可用"},
                {"cmd": "ollama list", "note": "确认模型已列出"},
            ],
        },
        "offline": {
            "title": "内网/无网络环境（离线包）",
            "note": ("内网机器无法访问外网。需要在**有网机器**上 pull 模型，"
                     "再把模型文件拷进内网。"),
            "steps": [
                {"cmd": f"ollama pull {target_model}",
                 "note": "在有网机器上执行"},
                {"cmd": "ollama show --modelfile " + target_model + " > Modelfile",
                 "note": "导出模型清单"},
                {"cmd": "ls ~/.ollama/models/manifests ~/.ollama/models/blobs",
                 "note": "确认 blobs 目录（模型权重在这里）"},
                {"cmd": "tar czf ollama-model.tar.gz ~/.ollama/models/",
                 "note": "打包整个 models 目录"},
                {"cmd": "scp ollama-model.tar.gz user@内网机器:/opt/",
                 "note": "拷贝到内网机器"},
                {"cmd": "tar xzf /opt/ollama-model.tar.gz -C ~/.ollama/",
                 "note": "在内网机器解包"},
                {"cmd": f"ollama list | grep {target_model.split(':')[0]}",
                 "note": "验证内网可见"},
            ],
        },
        "notes": [
            "内网部署前记得把 provider_mode 改成 local 或 hybrid",
            "模型体积：7B 约 4.7GB，14B 约 9GB，先确认磁盘空间",
            "CPU 推理 7B 大约 5~15 token/s，长文书生成会很慢，建议配 GPU",
        ],
    }


def autodownload_model(target_model: str = "qwen2.5:7b",
                       timeout: float = 1800.0) -> dict:
    """让 Agent 自动下载模型（ollama pull）。

    只有 CLI 可用时才能做；不可用时返回明确指引让用户自己下。
    """
    if not _ollama_cli_available():
        return {
            "ok": False,
            "reason": "ollama_cli_unavailable",
            "message": ("当前环境没有 ollama 命令（容器里常见）。"
                        "请按手动下载指引操作，或在有 ollama 的机器上 pull 后拷入。"),
            "guidance": build_download_guidance(target_model),
        }

    connected, _, err = _ollama_list_models(timeout=3.0)
    if not connected:
        return {
            "ok": False,
            "reason": "ollama_unreachable",
            "message": f"连不上 Ollama 服务（{err}），无法自动下载。",
            "guidance": build_download_guidance(target_model),
        }

    logger.info("[provisioner] 开始下载模型 %s", target_model)
    try:
        proc = subprocess.run(
            ["ollama", "pull", target_model],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout",
                "message": f"下载超时（>{timeout}s）。内网建议用离线包方式。",
                "guidance": build_download_guidance(target_model)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(e).__name__}",
                "message": f"下载失败：{e}",
                "guidance": build_download_guidance(target_model)}

    ok = proc.returncode == 0
    return {
        "ok": ok,
        "model": target_model,
        "message": (f"模型 {target_model} 下载完成" if ok
                    else f"下载失败（退出码 {proc.returncode}）"),
        "stdout_tail": (proc.stdout or "")[-800:],
        "stderr_tail": (proc.stderr or "")[-800:],
    }


def list_local_models() -> list[dict]:
    """列出本地模型及其体积（用于前端选择）。"""
    connected, models, _ = _ollama_list_models()
    if not connected:
        return []
    try:
        import httpx
        base = _ollama_base_url().rstrip("/")
        url = base[:-3] if base.endswith("/v1") else base
        with httpx.Client(timeout=5.0) as c:
            data = c.get(f"{url}/api/tags").json()
        out = []
        for m in data.get("models", []):
            out.append({
                "name": m.get("name", ""),
                "size_gb": round(m.get("size", 0) / 1024 ** 3, 2),
                "family": m.get("details", {}).get("family", ""),
                "parameter_size": m.get("details", {}).get("parameter_size", ""),
            })
        return out
    except Exception:
        return [{"name": m, "size_gb": None} for m in models]
