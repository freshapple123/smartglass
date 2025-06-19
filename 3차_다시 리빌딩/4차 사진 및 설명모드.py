import cv2
import os
import base64
from dotenv import load_dotenv
import openai
from pymongo import MongoClient
from datetime import datetime

# ✅ .env 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

# ✅ OpenAI 클라이언트
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ MongoDB 연결
mongo = MongoClient(MONGO_URI, tls=True)
db = mongo[MONGO_DB_NAME]
collection = db[MONGO_COLLECTION_NAME]

# ✅ 이미지 base64 인코딩 함수
def encode_image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ✅ GPT에 이미지 설명 요청
def ask_gpt_about_image(img_path, prompt="이 이미지에 대해 설명해줘"):
    base64_img = encode_image_to_base64(img_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]}
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

# ✅ 이미지와 설명을 MongoDB에 저장
def save_to_mongodb(image_path, description):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    doc = {
        "timestamp": datetime.now(),
        "description": description,
        "image_base64": image_data
    }
    collection.insert_one(doc)
    print("✅ MongoDB에 저장 완료")

# ✅ 웹캠 캡처 및 흐름 실행
def run_camera_capture():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    os.makedirs("captures", exist_ok=True)
    count = 0

    print("📷 w: 스샷 + GPT 분석 / ESC: 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Smart Glasses Camera", frame)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break
        elif key == ord('w'):
            img_path = f"captures/capture_{count}.jpg"
            cv2.imwrite(img_path, frame)
            print(f"📸 이미지 저장됨: {img_path}")

            print("🧠 GPT에 이미지 분석 요청 중...")
            description = ask_gpt_about_image(img_path)
            print(f"📝 GPT 응답: {description}")

            save_to_mongodb(img_path, description)
            count += 1

    cap.release()
    cv2.destroyAllWindows()

# ✅ 실행
run_camera_capture()
