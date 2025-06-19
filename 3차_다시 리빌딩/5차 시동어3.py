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
import speech_recognition as sr
from gtts import gTTS
import pygame
import threading

# 🔐 환경 변수 로드
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# 📌 전역 플래그
stop_flag = False

# 📢 음성 출력
def speak(text):
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

# 📌 OpenCV 이미지 → base64
def encode_image_cv2_to_base64(cv2_img):
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# 📌 Google OCR
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

# 📌 페이지 유사도 비교
def is_different_page(text1, text2, threshold=0.5):
    ratio = SequenceMatcher(None, text1, text2).ratio()
    return ratio < threshold

# 📌 시동어 + 책 제목 인식
def wait_for_voice_trigger_and_get_title():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("🎤 '씨큐' 또는 '시큐' 라고 말하면 시작합니다...")
        while True:
            try:
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio, language="ko-KR")
                print("📝 인식된 내용:", command)
                if "씨큐" in command or "시큐" in command or "cq" in command.lower():
                    print("✅ 시동어 인식됨.")
                    speak("책 제목을 말씀해주세요.")
                    audio = recognizer.listen(source)
                    title = recognizer.recognize_google(audio, language="ko-KR")
                    print("📚 인식된 책 제목:", title)
                    return title.strip()
                else:
                    print("🔁 다시 말해주세요 ('씨큐' 또는 '시큐')")
            except sr.UnknownValueError:
                print("❌ 인식 실패. 다시 시도 중...")
            except sr.RequestError as e:
                print(f"❌ 요청 실패: {e}")
                break

# 📌 '그만' 감지 스레드 (수정됨)
def listen_for_stop_word():
    global stop_flag
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("👂 OCR 중... '그만' 이라고 말하면 종료됩니다.")  # ✅ 한 번만 출력
        while not stop_flag:
            try:
                audio = recognizer.listen(source, phrase_time_limit=3)
                command = recognizer.recognize_google(audio, language="ko-KR")
                print("🎧 인식:", command)
                if "그만" in command:
                    stop_flag = True
                    speak("중단할게요.")
                    print("🛑 '그만' 감지됨 → OCR 중단")
                    break
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                print("❌ 음성 인식 오류")
                break

# 📘 OCR + 그만 감지 루프
def auto_detect_and_ocr_with_stop(book_title, model_path="yolo11n.pt"):
    global stop_flag
    stop_flag = False

    threading.Thread(target=listen_for_stop_word, daemon=True).start()

    model = YOLO(model_path)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    base_dir = f"book_pages/{book_title}"
    os.makedirs(os.path.join(base_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "right"), exist_ok=True)
    text_file_path = os.path.join(base_dir, f"{book_title}.txt")

    image_counter = 1
    prev_left_text = ""
    prev_right_text = ""
    book_detected_time = None
    skip_until_time = 0
    preview_size = (960, 540)

    print(f"📘 책: {book_title} | OCR 시작 (ESC 또는 '그만'으로 종료)")

    while not stop_flag:
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
                    print("❌ 글자 수 부족 - 저장 생략")
                    skip_until_time = now + 3
                    continue

                if is_different_page(left_text, prev_left_text) or is_different_page(right_text, prev_right_text):
                    left_img_path = os.path.join(base_dir, "left", f"page_{image_counter}_left.jpg")
                    right_img_path = os.path.join(base_dir, "right", f"page_{image_counter}_right.jpg")
                    cv2.imwrite(left_img_path, left_page)
                    cv2.imwrite(right_img_path, right_page)

                    with open(text_file_path, "a", encoding="utf-8") as f:
                        f.write(f"\n--- 왼쪽 페이지 ({image_counter}) ---\n{left_text}\n")
                        f.write(f"\n--- 오른쪽 페이지 ({image_counter}) ---\n{right_text}\n")

                    print(f"✅ 페이지 {image_counter} 저장 완료")
                    prev_left_text = left_text
                    prev_right_text = right_text
                    image_counter += 1
                    book_detected_time = None
                    skip_until_time = 0
                else:
                    print("⏩ 중복 페이지 - 3초 후 재시도")
                    skip_until_time = now + 3
        else:
            book_detected_time = None
            skip_until_time = 0

        preview_frame = cv2.resize(frame, preview_size)
        cv2.imshow("Auto Book Scan", preview_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("🛑 ESC 눌림 → OCR 중단")
            stop_flag = True
            break

    cap.release()
    cv2.destroyAllWindows()
    return True

# ✅ 메인 루프: OCR 종료 후 다시 시동어 대기
if __name__ == "__main__":
    while True:
        book_title = wait_for_voice_trigger_and_get_title()
        auto_detect_and_ocr_with_stop(book_title)
