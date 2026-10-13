# -*- coding: utf-8 -*-
"""
3.1.1 百度在线 OCR
识别图片中的文字，提取酒店名称
"""
import base64
import requests
from config import BAIDU_API_KEY, BAIDU_SECRET_KEY


# ========== 1. 获取Access Token ==========
def get_access_token(api_key=None, secret_key=None):
    """获取百度OCR的临时访问令牌，有效期30天"""
    api_key = api_key or BAIDU_API_KEY
    secret_key = secret_key or BAIDU_SECRET_KEY
    if not api_key or not secret_key:
        raise RuntimeError(
            "未设置 BAIDU_API_KEY / BAIDU_SECRET_KEY 环境变量"
        )
    auth_url = (
        f"https://aip.baidubce.com/oauth/2.0/token"
        f"?grant_type=client_credentials"
        f"&client_id={api_key}&client_secret={secret_key}"
    )
    response = requests.get(auth_url)
    return response.json()["access_token"]


# ========== 2. 图片Base64编码 ==========
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ========== 3. 调用高精度文字识别API ==========
def general_ocr(access_token, image_path):
    """调用百度高精度文字识别接口（accurate_basic）"""
    ocr_url = (
        f"https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
        f"?access_token={access_token}"
    )
    image_data = image_to_base64(image_path)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    params = {"image": image_data}

    response = requests.post(ocr_url, headers=headers, data=params)
    result = response.json()

    if "words_result" not in result:
        print(f"识别失败：{result}")
        return []

    return [item["words"] for item in result["words_result"]]


# ========== 4. 从文本中提取酒店名称 ==========
def extract_hotel_name(texts):
    """从OCR结果中提取酒店名称"""
    hotel_keywords = ["酒店", "宾馆", "旅店", "招待所", "公寓", "度假村", "客栈"]
    fallback_keywords = ["大厦", "中心", "广场", "会所"]

    for text in texts:
        for kw in hotel_keywords:
            if kw in text:
                return text.strip()

    for text in texts:
        for kw in fallback_keywords:
            if kw in text:
                return text.strip()

    if texts:
        return max(texts, key=len).strip()

    return "未识别到酒店名称"


# ========== 5. 主流程 ==========
def main():
    image_path = r"D:\AI\ollama_deepseek\Agent4-图片和视频处理器\input\酒店外观_4.jpg"

    token = get_access_token()
    print("Access Token 获取成功")

    texts = general_ocr(token, image_path)
    print(f"\n===== OCR识别结果（共{len(texts)}条）=====")
    for i, t in enumerate(texts, 1):
        print(f"  [{i}] {t}")

    hotel_name = extract_hotel_name(texts)
    print(f"\n===== 提取结果 =====")
    print(f"酒店名称：{hotel_name}")

    return hotel_name


if __name__ == "__main__":
    main()