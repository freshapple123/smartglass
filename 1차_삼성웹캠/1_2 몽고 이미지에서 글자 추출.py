import requests
import base64
import json
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime

# 🔐 .env에서 API 키 및 MongoDB 정보 로드
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

# MongoDB 클라이언트 연결
client = MongoClient(MONGO_URI, tls=True)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

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

        # MongoDB에 저장
        doc = {
            "image_path": image_path,
            "text": description,
            "timestamp": datetime.now()
        }
        collection.insert_one(doc)
        print("✅ MongoDB에 저장 완료")

    except (KeyError, IndexError):
        print("❌ 텍스트 인식 실패")
        print(json.dumps(result, indent=2))

# 사용 예시
detect_text_rest("book_img1.jpg")  # 이미지 파일 경로 수정
