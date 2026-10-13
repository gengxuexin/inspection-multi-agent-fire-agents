# ~/fire_agents/multimodal_orchestrator.py
"""
logger = logging.getLogger("fire_agents")

多模态编排器
- 接收多个文件
- 按类型分发到对应 Agent
- 汇总结果，综合成案情
- 生成文书 + 案情报告
"""
import os
import json
import time
import httpx
import tempfile
import shutil
import asyncio
import traceback
import logging

AGENT1 = "http://127.0.0.1:8001"
AGENT2 = "http://127.0.0.1:8002"
AGENT3 = "http://127.0.0.1:8003"


# ==================== 文件类型判断 ====================
VISION_IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")
VISION_VID = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm")
VOICE_EXT  = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")
DOC_EXT    = (".pdf", ".docx", ".doc", ".txt", ".md")
TEMPLATE_EXT = (".docx",)


def classify_files(file_paths):
    """按后缀分类，返回 (vision, voice, doc, template)"""
    result = {"vision": [], "voice": [], "doc": [], "template": []}
    for p in file_paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in VISION_IMG or ext in VISION_VID:
            result["vision"].append(p)
        elif ext in VOICE_EXT:
            result["voice"].append(p)
        elif ext in DOC_EXT:
            result["doc"].append(p)
        else:
            # 未知类型，尝试当 doc 处理
            result["doc"].append(p)
    return result


# ==================== 各 Agent 调用 ====================
async def call_vision(client, img_path, is_video):
    """调用 agent3 分析图片/视频"""
    try:
        with open(img_path, "rb") as f:
            files = {"file": (os.path.basename(img_path), f)}
            if is_video:
                data = {"segment": "4", "fps": "1", "do_yolo": "false"}
                r = await client.post(
                    f"{AGENT3}/api/vision/analyze_video",
                    files=files, data=data, timeout=900,
                )
            else:
                data = {"do_yolo": "true", "do_ocr": "true"}
                r = await client.post(
                    f"{AGENT3}/api/vision/analyze_image",
                    files=files, data=data, timeout=600,
                )
        return {"file": os.path.basename(img_path), "result": r.json()}
    except Exception as e:
        return {"file": os.path.basename(img_path),
                "error": f"{type(e).__name__}: {e}"}


async def call_voice(client, audio_path):
    """调用 agent2 语音转写"""
    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f)}
            r = await client.post(
                f"{AGENT2}/api/voice/transcribe",
                files=files, data={"engine": "whisper"}, timeout=900,
            )
        return {"file": os.path.basename(audio_path), "result": r.json()}
    except Exception as e:
        return {"file": os.path.basename(audio_path),
                "error": f"{type(e).__name__}: {e}"}


async def call_doc_extract(client, doc_path):
    """调用 agent1 抽取文档里的案情"""
    try:
        with open(doc_path, "rb") as f:
            files = {"file": (os.path.basename(doc_path), f)}
            # 用 agent1 的文件分析接口（如果有）
            r = await client.post(
                f"{AGENT1}/api/case/extract",
                files=files, timeout=300,
            )
        return {"file": os.path.basename(doc_path), "result": r.json()}
    except Exception as e:
        return {"file": os.path.basename(doc_path),
                "error": f"{type(e).__name__}: {e}"}


# ==================== 综合所有模态结果 ====================
async def synthesize_case(vision_results, voice_results, doc_results):
    """用 LLM 把所有模态的结果综合成一份案情描述"""
    parts = []
    if vision_results:
        for v in vision_results:
            if "error" in v:
                parts.append(f"【图像/视频分析 {v['file']}】分析失败")
                continue
            res = v["result"].get("result", {})
            yolo = res.get("yolo", {})
            ocr = res.get("ocr", {})
            judgment = res.get("judgment", {})
            lines = [f"【视觉分析 - {v['file']}】"]
            if yolo.get("counts"):
                lines.append(f"  YOLO识别：{yolo['counts']}")
            if ocr.get("ocr_texts"):
                lines.append(f"  OCR文字：{' / '.join(ocr['ocr_texts'][:5])}")
            if judgment.get("overall_conclusion"):
                lines.append(f"  结论：{judgment['overall_conclusion']}")
            if judgment.get("alerts"):
                for a in judgment["alerts"]:
                    lines.append(f"  ⚠️ [{a.get('level')}] {a.get('title')}：{a.get('description', '')}")
            parts.append("\n".join(lines))

    if voice_results:
        for v in voice_results:
            if "error" in v:
                parts.append(f"【音频 {v['file']}】转写失败")
                continue
            text = v["result"].get("text", "")
            parts.append(f"【录音转写 - {v['file']}】\n  {text}")

    if doc_results:
        for v in doc_results:
            if "error" in v:
                parts.append(f"【文档 {v['file']}】抽取失败")
                continue
            res = v["result"]
            parts.append(f"【文档 - {v['file']}】\n  {json.dumps(res, ensure_ascii=False)[:500]}")

    return "\n\n".join(parts)


# ==================== 主流程 ====================
async def orchestrate(file_paths, template_path=None, user_query=""):
    """
    :param file_paths: 所有上传文件的本地路径（除模板外）
    :param template_path: 模板文件路径（可选）
    :param user_query: 用户的文字输入
    """
    t0 = time.time()
    
    # 1. 分类文件
    grouped = classify_files(file_paths)
    logger.info("[Orchestrator] 分类结果：vision=%d voice=%d doc=%d",
                 len(grouped["vision"]), len(grouped["voice"]), len(grouped["doc"]))

    # 2. 并行调用各 Agent
    async with httpx.AsyncClient(trust_env=False) as client:
        tasks = []
        
        # 图像/视频
        for p in grouped["vision"]:
            ext = os.path.splitext(p)[1].lower()
            is_video = ext in VISION_VID
            tasks.append(call_vision(client, p, is_video))
        
        # 音频
        for p in grouped["voice"]:
            tasks.append(call_voice(client, p))
        
        # 文档
        for p in grouped["doc"]:
            tasks.append(call_doc_extract(client, p))
        
        all_results = await asyncio.gather(*tasks, return_exceptions=False)

    # 3. 拆回去
    vision_results = all_results[:len(grouped["vision"])]
    voice_results = all_results[len(grouped["vision"]):
                                 len(grouped["vision"]) + len(grouped["voice"])]
    doc_results = all_results[len(grouped["vision"]) + len(grouped["voice"]):]

    # 4. 综合案情
    case_summary = await synthesize_case(vision_results, voice_results, doc_results)
    logger.info(f"[Orchestrator] 案情综合完成，长度 {len(case_summary)}")
    
    # 5. 用 agent1 抽结构化字段
    case_dict = {"case_context": case_summary, "case_id": "auto_" + str(int(time.time()))}
    
    # 6. 生成文书（如果有模板）
    docx_path = None
    # ★ 只处理 .docx 模板
    if not template_path.lower().endswith(".docx"):
        logger.warning(f"[Orchestrator] ⚠️ 模板必须是 docx，跳过：{template_path}")
    else:
        try:
            with open(template_path, "rb") as tf:
                files = {"template_file": (os.path.basename(template_path), tf)}
                data = {"case_dict_json": json.dumps(case_dict, ensure_ascii=False)}
                async with httpx.AsyncClient(trust_env=False, timeout=900) as c:
                    r = await c.post(
                        f"{AGENT1}/api/doc/generate_from_dict",
                        files=files, data=data,
                    )
                    doc_result = r.json()
                    if doc_result.get("status") == "success":
                        docx_path = doc_result.get("docx")
                    else:
                        logger.error(f"[Orchestrator] 文书生成失败：{doc_result}")
        except Exception as e:
            traceback.print_exc()
    # 7. 生成 Markdown 案情报告
    elapsed = time.time() - t0
    report = build_report(case_summary, vision_results, voice_results,
                          doc_results, docx_path, elapsed)

    return {
        "case_dict": case_dict,
        "case_summary": case_summary,
        "docx": docx_path,
        "report_md": report,
        "elapsed_sec": round(elapsed, 1),
        "stats": {
            "vision_files": len(grouped["vision"]),
            "voice_files": len(grouped["voice"]),
            "doc_files": len(grouped["doc"]),
        },
    }


def build_report(case_summary, vision_results, voice_results,
                 doc_results, docx_path, elapsed):
    lines = [
        "# 消防案件综合分析报告",
        f"\n生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"处理耗时：{elapsed:.1f} 秒",
        "\n---\n",
        "## 一、案情摘要\n",
        case_summary,
        "\n---\n",
        "## 二、视觉分析明细\n",
    ]
    for v in vision_results:
        lines.append(f"### {v['file']}\n")
        if "error" in v:
            lines.append(f"❌ {v['error']}\n")
        else:
            res = v["result"].get("result", {})
            lines.append(f"- YOLO：{res.get('yolo', {}).get('counts', {})}")
            lines.append(f"- OCR：{res.get('ocr', {}).get('ocr_texts', [])}")
            lines.append(f"- 结论：{res.get('judgment', {}).get('overall_conclusion', '')}\n")
    
    lines.append("\n---\n## 三、录音转写明细\n")
    for v in voice_results:
        if "error" in v:
            lines.append(f"- {v['file']}：❌ {v['error']}")
        else:
            lines.append(f"- {v['file']}：{v['result'].get('text', '')}")
    
    lines.append("\n---\n## 四、文档抽取明细\n")
    for v in doc_results:
        if "error" in v:
            lines.append(f"- {v['file']}：❌ {v['error']}")
        else:
            lines.append(f"- {v['file']}：{json.dumps(v['result'], ensure_ascii=False)[:300]}")
    
    if docx_path:
        lines.append(f"\n---\n## 五、生成文书\n\n✅ {docx_path}")
    
    return "\n".join(lines)