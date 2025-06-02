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

# 책 제목 설정
book_title = "논어"

# 🔐 .env에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# OpenCV 이미지 → base64 인코딩
def encode_image_cv2_to_base64(cv2_img):
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# 전처리: 흑백 변환
def preprocess_grayscale(cv2_img):
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Google API는 BGR 필요

# Google OCR 요청
def detect_text_from_array(cv2_img):
    content = encode_image_cv2_to_base64(cv2_img)
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
    except:
        return ""

# 페이지 비교
def is_different_page(text1, text2, threshold=0.5):
    return SequenceMatcher(None, text1, text2).ratio() < threshold

# 메인
def auto_detect_and_ocr(model_path="yolo11n.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    save_dir = f"book_pages/{book_title}"
    os.makedirs(save_dir, exist_ok=True)

    image_counter = 1
    prev_left_text = ""
    prev_right_text = ""
    book_detected_time = None
    skip_until_time = 0

    # 텍스트 파일들 열기
    left_color_txt = open(os.path.join(save_dir, "left_color.txt"), "a", encoding="utf-8")
    right_color_txt = open(os.path.join(save_dir, "right_color.txt"), "a", encoding="utf-8")
    left_gray_txt = open(os.path.join(save_dir, "left_gray.txt"), "a", encoding="utf-8")
    right_gray_txt = open(os.path.join(save_dir, "right_gray.txt"), "a", encoding="utf-8")

    print(f"📘 책: {book_title} | OCR (컬러 + 흑백) 비교 저장 시작 (ESC로 종료)")

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
            now = time.time()
            if book_detected_time is None:
                book_detected_time = now

            if now - book_detected_time >= 3 and now >= skip_until_time:
                book_crop = frame[y1:y2, x1:x2]
                h, w = book_crop.shape[:2]
                mid = w // 2
                left_color = book_crop[:, :mid]
                right_color = book_crop[:, mid:]

                # 전처리: 흑백
                left_gray = preprocess_grayscale(left_color)
                right_gray = preprocess_grayscale(right_color)

                # OCR
                left_text_color = detect_text_from_array(left_color)
                right_text_color = detect_text_from_array(right_color)
                left_text_gray = detect_text_from_array(left_gray)
                right_text_gray = detect_text_from_array(right_gray)

                # 필터
                if all(len(t.strip()) < 30 for t in [left_text_color, right_text_color, left_text_gray, right_text_gray]):
                    print("❌ 모든 결과 글자 수 부족 - 생략")
                    skip_until_time = now + 3
                    continue

                # 중복 확인 (컬러 기준)
                if is_different_page(left_text_color, prev_left_text) or is_different_page(right_text_color, prev_right_text):
                    print(f"\n✅ 새 페이지 저장됨: {image_counter}")

                    # 이미지 저장
                    cv2.imwrite(os.path.join(save_dir, f"left_color_{image_counter:03d}.jpg"), left_color)
                    cv2.imwrite(os.path.join(save_dir, f"right_color_{image_counter:03d}.jpg"), right_color)
                    cv2.imwrite(os.path.join(save_dir, f"left_gray_{image_counter:03d}.jpg"), left_gray)
                    cv2.imwrite(os.path.join(save_dir, f"right_gray_{image_counter:03d}.jpg"), right_gray)

                    # 텍스트 저장
                    left_color_txt.write(f"\n--- Left Page {image_counter} ---\n{left_text_color}\n")
                    right_color_txt.write(f"\n--- Right Page {image_counter} ---\n{right_text_color}\n")
                    left_gray_txt.write(f"\n--- Left Page {image_counter} ---\n{left_text_gray}\n")
                    right_gray_txt.write(f"\n--- Right Page {image_counter} ---\n{right_text_gray}\n")

                    prev_left_text = left_text_color
                    prev_right_text = right_text_color
                    image_counter += 1
                    book_detected_time = None
                    skip_until_time = 0
                else:
                    print("⏩ 동일 페이지 - 생략")
                    skip_until_time = now + 3
        else:
            book_detected_time = None
            skip_until_time = 0

        cv2.imshow("Book OCR Preview", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("🛑 ESC 눌림 - 종료")
            break

    cap.release()
    cv2.destroyAllWindows()
    left_color_txt.close()
    right_color_txt.close()
    left_gray_txt.close()
    right_gray_txt.close()

# 실행
auto_detect_and_ocr()
