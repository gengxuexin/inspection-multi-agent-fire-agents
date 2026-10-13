# -*- coding: utf-8 -*-
"""
3.2.5 综合判断 Agent（VLM 融合版）
- 直接把原图传给 Qwen-VL
- 结合 YOLO/OCR 的结构化结果做综合判断
"""
import os
import json
import base64
from datetime import datetime
from openai import OpenAI
from config import (
    BASE_URL, VLM_MODEL, JUDGE_MODEL, ALERT_CATEGORIES, require_dashscope
)
import logging


logger = logging.getLogger("fire_agents.agent3.agent")

JUDGMENT_SYSTEM_PROMPT = """你是消防安全检查专家。你的任务是根据【图片画面】和【AI 辅助检测结果】，
进行综合判断，输出结构化告警。

【重要】你必须**自己看图**，AI 检测结果只是辅助参考，检测结果为空不代表没有隐患。
- 如果图中明显有明火、烟雾，即使 YOLO 没检测到，也要报警。
- 如果 YOLO 检测到的物体与图片不符（比如把卷状物误判成消防栓），以你的视觉判断为准。

【告警类型】只能从以下 6 类中选择：
- fire: 明火告警
- smoke: 烟雾告警
- blocked_passage: 消防通道堵塞告警
- missing_equipment: 缺少关键消防设备警示
- expired_equipment: 消防设备过期/失效
- other: 其他消防安全隐患

【告警等级】
- critical: 明火、浓烟、疏散通道严重堵塞
- high: 关键设备缺失（灭火器、消防栓）、通道占用
- medium: 次要设备缺失、管理疏漏
- low: 建议改进项
- normal: 无隐患

【输出格式】严格 JSON，不要任何其他文字：
{
  "alerts": [
    {
      "type": "fire",
      "level": "critical",
      "title": "简要标题（10字内）",
      "description": "详细描述（50字内）",
      "evidence": "来自画面哪里，或哪段检测结果",
      "suggestion": "整改建议（30字内）"
    }
  ],
  "overall_level": "critical",
  "overall_conclusion": "总体结论（100字内）",
  "stats": {
    "total_frames_analyzed": 1,
    "total_segments_analyzed": 0,
    "yolo_objects_detected": 0
  }
}
"""


def _image_to_data_url(image_path: str) -> str:
    """把本地图片转成 data URL（Qwen-VL 用）"""
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        mime = "image/jpeg"
    elif ext == "png":
        mime = "image/png"
    elif ext == "webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


class JudgmentAgent:
    """3.2.5 综合判断 Agent（VLM 融合版）"""

    def __init__(self):
        self.client = OpenAI(api_key=require_dashscope(), base_url=BASE_URL)

    def judge(self, yolo_result=None, video_result=None, frame_result=None,
              source_info="", image_path=None):
        """
        综合判断入口
        :param image_path: 单张图片的本地路径（有的话，直接传给 VLM 做视觉判断）
        """
        # ========== 1. 构建文本上下文 ==========
        parts = []
        if source_info:
            parts.append(f"【源信息】{source_info}")

        if yolo_result:
            counts = yolo_result.get('counts', {}) or {}
            if counts:
                parts.append(
                    f"【YOLO 检测结果】\n"
                    f"检测到的物体：{json.dumps(counts, ensure_ascii=False)}\n"
                    f"总目标数：{yolo_result.get('total_objects', 0)}\n"
                    f"分析帧数：{yolo_result.get('frames_analyzed', 0)}"
                )
            else:
                parts.append("【YOLO 检测结果】未检测到任何目标（不代表图中无隐患）")

        if video_result:
            if isinstance(video_result, dict):
                vtext = video_result.get("summary", "") or f"（共 {video_result.get('total', 0)} 段）"
            else:
                vtext = str(video_result)
            parts.append(f"【3.2.2 视频分段分析】\n{vtext}")

        if frame_result:
            if isinstance(frame_result, dict):
                ftext = frame_result.get("summary", "") or f"（共 {frame_result.get('total', 0)} 帧）"
            else:
                ftext = str(frame_result)
            parts.append(f"【3.2.3 抽帧分析】\n{ftext}")

        user_text = "\n\n".join(parts) if parts else "（无辅助检测结果，请仅根据图片判断）"

        # ========== 2. 构建消息（有图片就多模态） ==========
        if image_path and os.path.exists(image_path):
            data_url = _image_to_data_url(image_path)
            messages = [
                {"role": "system", "content": JUDGMENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": f"辅助检测结果：\n{user_text}\n\n请综合图片和检测结果做判断。"},
                    ],
                },
            ]
            model = VLM_MODEL   # 用视觉模型
            logger.info(f"[JudgmentAgent] 使用 VLM 融合判断（含原图）")
        else:
            messages = [
                {"role": "system", "content": JUDGMENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
            model = JUDGE_MODEL  # 纯文本模型
            logger.info(f"[JudgmentAgent] 使用纯文本判断（无原图）")

        # ========== 3. 调用 LLM ==========
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000,
            )
            result = json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError:
            result = {
                "alerts": [],
                "overall_level": "unknown",
                "overall_conclusion": "综合判断失败，模型未返回有效 JSON",
            }
        except Exception as e:
            logger.error(f"[JudgmentAgent] 调用失败：{e}")
            result = {
                "alerts": [],
                "overall_level": "error",
                "overall_conclusion": f"综合判断失败：{e}",
            }

        result["check_time"] = datetime.now().isoformat()
        result["source_info"] = source_info
        result["model_used"] = model
        return result

    @staticmethod
    def print_report(result):
        """打印可读报告"""
        logger.info("\n" + "=" * 60)
        logger.info("消防检查综合判断报告")
        logger.info("=" * 60)
        logger.info(f"检查时间：{result.get('check_time', '未知')}")
        logger.info(f"源信息：{result.get('source_info', '')}")
        logger.info(f"使用模型：{result.get('model_used', '')}")

        level = result.get("overall_level", "unknown")
        logger.info(f"\n【总体等级】{level.upper()}")

        alerts = result.get("alerts", [])
        if not alerts:
            logger.info("\n✅ 未发现明显隐患")
        else:
            logger.info(f"\n【告警列表】共 {len(alerts)} 条")
            for i, a in enumerate(alerts, 1):
                alert_name = ALERT_CATEGORIES.get(a.get("type", "other"), "其他")
                logger.info(f"\n  [{i}] {alert_name} | 等级：{a.get('level', '?')}")
                logger.info(f"      标题：{a.get('title', '')}")
                logger.info(f"      描述：{a.get('description', '')}")
                logger.info(f"      证据：{a.get('evidence', '')}")
                logger.info(f"      建议：{a.get('suggestion', '')}")

        logger.info(f"\n【总体结论】")
        logger.info(f"  {result.get('overall_conclusion', '')}")
        logger.info("=" * 60)


# ========== 测试 ==========
if __name__ == "__main__":
    agent = JudgmentAgent()
    result = agent.judge(
        yolo_result={"counts": {"fire_hose": 1}, "total_objects": 1, "frames_analyzed": 1},
        source_info="测试",
    )
    agent.print_report(result)