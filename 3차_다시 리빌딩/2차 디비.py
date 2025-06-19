import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import requests
import base64
import json
from dotenv import load_dotenv
from difflib import SequenceMatcher
import io
from PIL import Image
from pymongo import MongoClient
from datetime import datetime

# 책 제목
book_title = "논어"

# 🔐 환경 변수 로드
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

# MongoDB 연결
client = MongoClient(MONGO_URI, tls=True)
db = client[MONGO_DB_NAME]
collection = db[MONGO_COLLECTION_NAME]

# 이미지 → base64
def encode_image_cv2_to_base64(cv2_img):
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Google OCR
def detect_text_from_array(cv2_img):
    content = encode_image_cv2_to_base64(cv2_img)
    url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"
    payload = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "TEXT_DETECTION"}],
        }]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    result = response.json()
    try:
        return result["responses"][0]["textAnnotations"][0]["description"].strip()
    except (KeyError, IndexError):
        return ""

# 텍스트 유사도 비교
def is_different_page(text1, text2, threshold=0.5):
    ratio = SequenceMatcher(None, text1, text2).ratio()
    return ratio < threshold

# 메인 함수
def auto_detect_and_ocr(model_path="yolo11n.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📷 해상도: {width}x{height}")

    base_dir = f"book_pages/{book_title}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "right"), exist_ok=True)

    text_file_path = os.path.join(base_dir, f"{book_title}.txt")
    image_counter = 1
    prev_left_text = ""
    prev_right_text = ""
    book_detected_time = None
    skip_until_time = 0
    preview_size = (854, 480)

    print(f"📘 책 스캔 시작 및 MongoDB 저장 (ESC 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        book_box = None
        for box in results.boxes:
            if int(box.cls[0]) == 73:  # book 클래스
                book_box = box
                break

        if book_box:
            x1, y1, x2, y2 = map(int, book_box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "BOOK DETECTED", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            now = time.time()
            if book_detected_time is None:
                book_detected_time = now

            if now - book_detected_time >= 3 and now >= skip_until_time:
                book_crop = frame[y1:y2, x1:x2]
                h, w, _ = book_crop.shape
                mid = w // 2
                left_page = book_crop[:, :mid]
                right_page = book_crop[:, mid:]

                left_text = detect_text_from_array(left_page)
                right_text = detect_text_from_array(right_page)

                if len(left_text.strip()) < 30 and len(right_text.strip()) < 30:
                    print("❌ 글자 수 부족 - 생략")
                    skip_until_time = now + 3
                    continue

                if is_different_page(left_text, prev_left_text) or is_different_page(right_text, prev_right_text):
                    print(f"\n✅ 새 페이지 감지:")
                    print(f"📖 왼쪽:\n{left_text}")
                    print(f"📖 오른쪽:\n{right_text}")

                    # 이미지 저장
                    left_img_path = os.path.join(base_dir, "left", f"page_{image_counter}_left.jpg")
                    right_img_path = os.path.join(base_dir, "right", f"page_{image_counter}_right.jpg")
                    cv2.imwrite(left_img_path, left_page)
                    cv2.imwrite(right_img_path, right_page)

                    # txt 파일 저장
                    with open(text_file_path, "a", encoding="utf-8") as f:
                        f.write(f"\n--- 왼쪽 페이지 ({image_counter}) ---\n{left_text}\n")
                        f.write(f"\n--- 오른쪽 페이지 ({image_counter}) ---\n{right_text}\n")

                    # MongoDB 저장 (단일 텍스트 필드)
                    full_text = f"{left_text}\n{right_text}"
                    doc = {
                        "book_title": book_title,
                        "text": full_text,
                        "timestamp": datetime.now()
                    }
                    collection.insert_one(doc)
                    print("📥 MongoDB 저장 완료")

                    prev_left_text = left_text
                    prev_right_text = right_text
                    image_counter += 1
                    book_detected_time = None
                    skip_until_time = 0
                else:
                    print("⏩ 페이지 동일 - 3초 후 재시도")
                    skip_until_time = now + 3
        else:
            book_detected_time = None
            skip_until_time = 0

        preview_frame = cv2.resize(frame, preview_size)
        cv2.imshow("Auto Book Scan", preview_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# 실행
auto_detect_and_ocr()
