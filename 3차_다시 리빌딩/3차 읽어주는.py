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
from gtts import gTTS
import pygame

# 책 제목 설정
book_title = "논어"

# 🔐 .env에서 API 키 로드
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# 이미지 인코딩
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
        return ""

# 페이지 비교
def is_different_page(text1, text2, threshold=0.5):
    ratio = SequenceMatcher(None, text1, text2).ratio()
    return ratio < threshold

# 텍스트를 TTS로 재생
def speak_korean(text):
    try:
        if not text.strip():
            return
        tts = gTTS(text=text, lang="ko")
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        pygame.mixer.init()
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        pygame.mixer.quit()
    except Exception as e:
        print("❌ 음성 출력 오류:", e)

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
    print(f"📷 현재 카메라 해상도: {width}x{height}")

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

    print(f"📘 책: {book_title} | 텍스트 저장 및 낭독 시작 (ESC로 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        book_box = None
        for box in results.boxes:
            if int(box.cls[0]) == 73:  # 'book' 클래스
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
                    print("❌ 글자 수 부족 - 저장 생략 (넘기는 중일 수 있음)")
                    skip_until_time = now + 3
                    continue

                if is_different_page(left_text, prev_left_text) or is_different_page(right_text, prev_right_text):
                    print(f"\n✅ 새 페이지 텍스트 저장됨:")
                    print(f"📖 왼쪽:\n{left_text}")
                    print(f"\n📖 오른쪽:\n{right_text}")

                    # 이미지 저장
                    left_img_path = os.path.join(base_dir, "left", f"page_{image_counter}_left.jpg")
                    right_img_path = os.path.join(base_dir, "right", f"page_{image_counter}_right.jpg")
                    cv2.imwrite(left_img_path, left_page)
                    cv2.imwrite(right_img_path, right_page)

                    # 텍스트 저장
                    with open(text_file_path, "a", encoding="utf-8") as f:
                        f.write(f"\n--- 왼쪽 페이지 ({image_counter}) ---\n{left_text}\n")
                        f.write(f"\n--- 오른쪽 페이지 ({image_counter}) ---\n{right_text}\n")

                    # 음성 출력 (3000자 제한)
                    combined_text = f"왼쪽 페이지입니다. {left_text} 오른쪽 페이지입니다. {right_text}"
                    speak_korean(combined_text[:3000])

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
