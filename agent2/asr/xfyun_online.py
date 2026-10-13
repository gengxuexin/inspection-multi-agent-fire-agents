# -*- coding: utf-8 -*-
"""讯飞在线语音听写（WebSocket 流式版）"""
import hashlib
import hmac
import base64
import json
import time
import ssl
import threading
import websocket
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime

from config import XFYUN_APP_ID, XFYUN_API_KEY, XFYUN_API_SECRET
import logging


logger = logging.getLogger("fire_agents.agent2")

class XfyunASRClient:
    """科大讯飞在线语音听写（流式版）客户端"""

    def __init__(self, app_id=None, api_key=None, api_secret=None,
                 on_result=None):
        self.app_id = app_id or XFYUN_APP_ID
        self.api_key = api_key or XFYUN_API_KEY
        self.api_secret = api_secret or XFYUN_API_SECRET
        if not self.app_id or not self.api_key or not self.api_secret:
            raise RuntimeError(
                "讯飞密钥未配置，请设置 XFYUN_APP_ID / XFYUN_API_KEY / "
                "XFYUN_API_SECRET 环境变量"
            )
        self.on_result = on_result
        self.ws = None
        self.done = threading.Event()

    def _create_auth_url(self):
        """生成带鉴权参数的 WebSocket 连接 URL"""
        url = "wss://iat-api.xfyun.cn/v2/iat"
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = (
            f"host: iat-api.xfyun.cn\n"
            f"date: {date}\n"
            f"GET /v2/iat HTTP/1.1"
        )
        signature_sha = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature = base64.b64encode(signature_sha).decode("utf-8")

        authorization_origin = (
            f'api_key="{self.api_key}", '
            f'algorithm="hmac-sha256", '
            f'headers="host date request-line", '
            f'signature="{signature}"'
        )
        authorization = base64.b64encode(
            authorization_origin.encode("utf-8")
        ).decode("utf-8")

        params = {
            "authorization": authorization,
            "date": date,
            "host": "iat-api.xfyun.cn",
        }
        return url + "?" + urlencode(params)

    def _on_open(self, ws):
        logger.info("[讯飞] WebSocket 连接已建立")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            code = data.get("code", -1)

            if code != 0:
                logger.error(f"[讯飞] 错误码: {code}, 信息: {data.get('message', '')}")
                self.done.set()
                return

            result_data = data.get("data", {})
            if "result" in result_data:
                result = result_data["result"]
                text = ""
                for ws_item in result.get("ws", []):
                    for cw in ws_item.get("cw", []):
                        text += cw.get("w", "")
                if text and self.on_result:
                    self.on_result(text)

            if result_data.get("status") == 2:
                logger.info("[讯飞] 识别完成")
                self.done.set()

        except Exception as e:
            logger.error(f"[讯飞] 消息解析异常: {e}")

    def _on_error(self, ws, error):
        logger.error(f"[讯飞] WebSocket 错误: {error}")
        self.done.set()

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("[讯飞] WebSocket 连接已关闭")
        self.done.set()

    def recognize(self, pcm_file_path):
        """对 PCM/WAV 音频文件进行在线识别"""
        self.done.clear()
        auth_url = self._create_auth_url()

        self.ws = websocket.WebSocketApp(
            auth_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        wst = threading.Thread(
            target=self.ws.run_forever,
            kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}},
        )
        wst.daemon = True
        wst.start()

        time.sleep(1)

        first_frame = {
            "common": {"app_id": self.app_id},
            "business": {
                "language": "zh_cn",
                "domain": "iat",
                "accent": "mandarin",
                "vad_eos": 3000,
                "ptt": 1,
                "nunum": 0,
            },
            "data": {
                "status": 0,
                "format": "audio/L16;rate=16000",
                "encoding": "raw",
                "audio": "",
            },
        }

        frame_size = 1280

        try:
            with open(pcm_file_path, "rb") as f:
                header = f.read(4)
                if header == b"RIFF":
                    f.seek(44)
                else:
                    f.seek(0)

                audio_data = f.read(frame_size)
                if not audio_data:
                    logger.info("❌ 音频文件为空")
                    return

                first_frame["data"]["audio"] = base64.b64encode(
                    audio_data
                ).decode("utf-8")
                self.ws.send(json.dumps(first_frame))

                while not self.done.is_set():
                    audio_data = f.read(frame_size)
                    if not audio_data:
                        break

                    mid_frame = {
                        "data": {
                            "status": 1,
                            "format": "audio/L16;rate=16000",
                            "encoding": "raw",
                            "audio": base64.b64encode(
                                audio_data
                            ).decode("utf-8"),
                        }
                    }
                    try:
                        self.ws.send(json.dumps(mid_frame))
                    except websocket.WebSocketConnectionClosedException:
                        logger.info("[讯飞] 连接已关闭，停止发送")
                        self.done.set()
                        break

                    time.sleep(0.04)

                if not self.done.is_set():
                    end_frame = {
                        "data": {
                            "status": 2,
                            "format": "audio/L16;rate=16000",
                            "encoding": "raw",
                            "audio": "",
                        }
                    }
                    try:
                        self.ws.send(json.dumps(end_frame))
                    except websocket.WebSocketConnectionClosedException:
                        logger.info("[讯飞] 连接已关闭，无法发送结束帧")

        except Exception as e:
            logger.error(f"[讯飞] 发送异常: {e}")

        self.done.wait(timeout=120)

        try:
            self.ws.close()
        except Exception:
            pass