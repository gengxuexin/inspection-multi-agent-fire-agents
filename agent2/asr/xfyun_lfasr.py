# -*- coding: utf-8 -*-
"""讯飞录音文件转写（LFASR）客户端"""
import hashlib
import hmac
import base64
import json
import time
import os
import subprocess
import requests

from config import LFASR_APP_ID, LFASR_SECRET_KEY, LFASR_HOT_WORDS
import logging


logger = logging.getLogger("fire_agents.agent2")

class LFASRClient:

    UPLOAD_URL = "https://raasr.xfyun.cn/v2/api/upload"
    RESULT_URL = "https://raasr.xfyun.cn/v2/api/getResult"

    def __init__(self, app_id=None, secret_key=None):
        self.app_id = app_id or LFASR_APP_ID
        self.secret_key = secret_key or LFASR_SECRET_KEY
        if not self.app_id or not self.secret_key:
            raise RuntimeError(
                "讯飞 LFASR 密钥未配置，请设置 LFASR_APP_ID / "
                "LFASR_SECRET_KEY 环境变量"
            )

    # ---------- 签名 ----------
    def _generate_signature(self, params):
        """生成 LFASR 鉴权签名"""
        param_str = json.dumps(params, ensure_ascii=False)
        md5 = hashlib.md5(param_str.encode("utf-8")).hexdigest()
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            md5.encode("utf-8"),
            digestmod=hashlib.sha1,
        ).hexdigest()
        return signature

    # ---------- 热词处理 ----------
    def _clean_hotwords(self, hotwords):
        """清理热词字符串，只保留中文、英文、逗号"""
        if not hotwords:
            return ""
        import re
        cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z,]", "", hotwords)
        return cleaned

    # ---------- 音频时长 ----------
    def _get_duration(self, audio_path):
        """用 ffprobe 获取音频时长（毫秒）"""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="ignore",
            )
            return int(float(result.stdout.strip()) * 1000)
        except Exception:
            return 0

    # ---------- 上传 ----------
    def upload(self, audio_path):
        """上传音频到 LFASR 服务"""
        duration_ms = self._get_duration(audio_path)
        clean_hotwords = self._clean_hotwords(LFASR_HOT_WORDS)
        logger.info(f"[LFASR] 热词：{clean_hotwords or '(无)'}")

        params = {
            "appId": self.app_id,
            "signa": "",
            "ts": str(int(time.time())),
            "fileSize": str(os.path.getsize(audio_path)),
            "fileName": os.path.basename(audio_path),
            "duration": str(duration_ms),
        }
        params["signa"] = self._generate_signature(params)

        with open(audio_path, "rb") as f:
            files = {"content": f}
            resp = requests.post(
                self.UPLOAD_URL, data=params, files=files, timeout=60
            )
        return resp.json()

    # ---------- 获取结果 ----------
    def get_result(self, task_id):
        """轮询获取识别结果"""
        while True:
            params = {
                "appId": self.app_id,
                "signa": "",
                "ts": str(int(time.time())),
                "orderId": task_id,
            }
            params["signa"] = self._generate_signature(params)

            resp = requests.post(self.RESULT_URL, data=params, timeout=60)
            data = resp.json()

            if data.get("content", {}).get("orderInfo", {}).get(
                    "status") == 4:
                return data
            logger.info("[LFASR] 识别中... 状态：%s",
                         data.get("content", {}).get("orderInfo", {}).get("status"))
            time.sleep(3)

    # ---------- 解析结果 ----------
    def _parse_result(self, raw):
        """提取 LFASR 返回的纯文本"""
        try:
            order = raw.get("content", {}).get("orderResult", "{}")
            parsed = json.loads(order)
            lattice = parsed.get("lattice", [])
            words = []
            for item in lattice:
                text = self._extract_text_from_1best(item)
                if text:
                    words.append(text)
            return "".join(words)
        except Exception as e:
            logger.error(f"[LFASR] 解析异常：{e}")
            return ""

    def _extract_text_from_1best(self, item):
        """从 1best 结构里提取文字"""
        try:
            best = item.get("json_1best", {})
            ws_list = best.get("st", {}).get("rt", [])
            text = ""
            for ws_item in ws_list:
                for cw in ws_item.get("ws", []):
                    for w in cw.get("cw", []):
                        text += w.get("w", "")
            return text
        except Exception:
            return ""

    # ---------- 主接口 ----------
    def recognize(self, audio_path):
        """完整流程：上传 → 轮询 → 解析"""
        if not os.path.exists(audio_path):
            logger.info(f"❌ 找不到音频：{audio_path}")
            return None

        logger.info(f"[LFASR] 上传：{audio_path}")
        upload_resp = self.upload(audio_path)
        if upload_resp.get("code") != "0":
            logger.error(f"❌ 上传失败：{upload_resp}")
            return None

        task_id = upload_resp["content"]["orderId"]
        logger.info(f"[LFASR] 任务 ID：{task_id}")

        result = self.get_result(task_id)
        text = self._parse_result(result)
        logger.info("[LFASR] 识别结果：%s", text[:200] + ("..." if len(text) > 200 else ""))
        return {"text": text}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        logger.info("用法：python -m asr.xfyun_lfasr <音频路径>")
        sys.exit(1)
    client = LFASRClient()
    logger.info(client.recognize(sys.argv[1]))