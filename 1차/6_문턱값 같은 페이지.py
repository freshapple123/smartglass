'''
음.. 일단 안좋은거 같음 비교하는건 좋은데,,,
그냥 페이지를 넘기면 한장 찍었으면 좋겠다.
'''

import cv2
import numpy as np
from ultralytics import YOLO
import uuid
import os
import time
import requests
import base64
import json
from dotenv import load_dotenv
from difflib import SequenceMatcher

# 책 제목 설정 (테이블/파일명으로 사용됨)
book_title = "논어"

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
        return description.strip()
    except (KeyError, IndexError):
        return ""

def is_different_page(text1, text2, threshold=0.5):
    ratio = SequenceMatcher(None, text1, text2).ratio()
    return ratio < threshold

def auto_detect_and_ocr(model_path="yolo11n.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    save_dir = f"book_pages/{book_title}"
    os.makedirs(save_dir, exist_ok=True)

    text_file_path = f"{book_title}.txt"
    image_counter = 1
    prev_left_text = ""
    prev_right_text = ""
    last_capture_time = time.time()

    print(f"📘 책: {book_title} | 이미지 및 텍스트 저장 시작 (ESC로 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        book_box = None
        for box in results.boxes:
            if int(box.cls[0]) == 73:
                book_box = box
                break

        if book_box:
            x1, y1, x2, y2 = map(int, book_box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "BOOK DETECTED - AUTO MODE", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            now = time.time()
            if now - last_capture_time >= 3:
                book_crop = frame[y1:y2, x1:x2]
                h, w, _ = book_crop.shape
                mid = w // 2
                left_page = book_crop[:, :mid]
                right_page = book_crop[:, mid:]

                # OCR 텍스트 추출
                temp_left_path = os.path.join(save_dir, "temp_left.jpg")
                temp_right_path = os.path.join(save_dir, "temp_right.jpg")
                cv2.imwrite(temp_left_path, left_page)
                cv2.imwrite(temp_right_path, right_page)

                left_text = detect_text_rest(temp_left_path)
                right_text = detect_text_rest(temp_right_path)

                if is_different_page(left_text, prev_left_text) or is_different_page(right_text, prev_right_text):
                    # 번호 붙인 파일 저장
                    left_path = os.path.join(save_dir, f"{book_title}_left_{image_counter:03}.jpg")
                    right_path = os.path.join(save_dir, f"{book_title}_right_{image_counter:03}.jpg")
                    cv2.imwrite(left_path, left_page)
                    cv2.imwrite(right_path, right_page)

                    print(f"\n✅ 새 페이지 저장됨:")
                    print(f"📖 왼쪽: {left_path}\n{left_text}")
                    print(f"\n📖 오른쪽: {right_path}\n{right_text}")

                    # 텍스트 누적 저장
                    with open(text_file_path, "a", encoding="utf-8") as f:
                        f.write(f"\n--- 왼쪽 페이지 ({image_counter}) ---\n{left_text}\n")
                        f.write(f"\n--- 오른쪽 페이지 ({image_counter}) ---\n{right_text}\n")

                    prev_left_text = left_text
                    prev_right_text = right_text
                    image_counter += 1
                else:
                    print("⏩ 페이지 동일, 저장 생략")

                last_capture_time = now

        cv2.imshow("Auto Book Scan", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# 실행
auto_detect_and_ocr()
