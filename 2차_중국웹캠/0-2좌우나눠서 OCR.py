import requests
import base64
import json
import os
import cv2
from dotenv import load_dotenv

# 🔐 .env에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# 이미지 → base64 인코딩
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode("utf-8")

# Google Vision OCR 요청
def ocr_with_google_api(cv2_image):
    content = encode_image_to_base64(cv2_image)
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
        return result["responses"][0]["textAnnotations"][0]["description"].strip()
    except (KeyError, IndexError):
        return None

# 책 양쪽 페이지를 나누어 OCR
def detect_text_from_split_book(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 이미지 불러오기 실패:", image_path)
        return

    height, width, _ = image.shape
    mid = width // 2

    left_img = image[:, :mid]
    right_img = image[:, mid:]

    print("🔍 왼쪽 페이지 OCR 중...")
    left_text = ocr_with_google_api(left_img)
    print("🔍 오른쪽 페이지 OCR 중...")
    right_text = ocr_with_google_api(right_img)

    print("\n📖 [왼쪽 페이지 텍스트]")
    print(left_text if left_text else "❌ 인식 실패")

    print("\n📖 [오른쪽 페이지 텍스트]")
    print(right_text if right_text else "❌ 인식 실패")

# 사용 예시
detect_text_from_split_book("captured_images_avg_sharp/avg+sharp.jpg")  # 실제 이미지 경로로 수정
