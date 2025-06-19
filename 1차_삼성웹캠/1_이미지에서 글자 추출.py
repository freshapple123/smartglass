import requests
import base64
import json
import os
from dotenv import load_dotenv

# 🔐 .env에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

def detect_text_rest(image_path):
    with open(image_path, "rb") as image_file:
        content = base64.b64encode(image_file.read()).decode("utf-8")

    url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"

    payload = {
        "requests": [
            {
                "image": {"content": content},
                "features": [{"type": "TEXT_DETECTION"}],
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    result = response.json()

    try:
        description = result["responses"][0]["textAnnotations"][0]["description"]
        print("📖 인식된 텍스트:\n", description)
    except (KeyError, IndexError):
        print("❌ 텍스트 인식 실패")
        print(json.dumps(result, indent=2))

# 사용 예시
detect_text_rest("output/right_page.jpg")  # 이미지 파일 경로 수정
