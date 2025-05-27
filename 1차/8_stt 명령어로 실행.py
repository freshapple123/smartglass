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
import openai
from gtts import gTTS
import pygame

# 환경 변수 로드
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 책 제목
book_title = "논어"

# 음성 출력 함수
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

# 이미지 인코딩 함수
def encode_image_cv2_to_base64(cv2_img):
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# OCR 함수
def detect_text_from_array(cv2_img):
    content = encode_image_cv2_to_base64(cv2_img)
    url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"
    payload = {
        "requests": [{"image": {"content": content}, "features": [{"type": "TEXT_DETECTION"}]}]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    result = response.json()
    try:
        return result["responses"][0]["textAnnotations"][0]["description"].strip()
    except (KeyError, IndexError):
        return ""

# 페이지 비교 함수
def is_different_page(text1, text2, threshold=0.5):
    ratio = SequenceMatcher(None, text1, text2).ratio()
    return ratio < threshold

# 책 OCR + 그만 감지 통합 함수
def auto_detect_and_ocr_with_stop(model_path="yolo11n.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    save_dir = f"book_pages/{book_title}"
    os.makedirs(save_dir, exist_ok=True)
    text_file_path = f"{book_title}.txt"
    image_counter = 1
    prev_left_text = ""
    prev_right_text = ""
    book_detected_time = None
    skip_until_time = 0

    print(f"📘 책: {book_title} | 텍스트 저장 시작 (ESC 또는 '그만'으로 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 음성 인식 체크
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                speech = recognizer.recognize_google(audio, language="ko-KR")
                if "그만" in speech:
                    print("🛑 '그만' 인식됨. OCR 종료")
                    speak("책 읽기를 종료할게요.")
                    break
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                pass

        results = model(frame, verbose=False)[0]
        book_box = None
        for box in results.boxes:
            if int(box.cls[0]) == 73:
                book_box = box
                break

        if book_box:
            x1, y1, x2, y2 = map(int, book_box.xyxy[0])
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
                    with open(text_file_path, "a", encoding="utf-8") as f:
                        f.write(f"\n--- 왼쪽 페이지 ({image_counter}) ---\n{left_text}\n")
                        f.write(f"\n--- 오른쪽 페이지 ({image_counter}) ---\n{right_text}\n")
                    print(f"✅ 저장 완료: 페이지 {image_counter}")
                    prev_left_text = left_text
                    prev_right_text = right_text
                    image_counter += 1
                    book_detected_time = None
                    skip_until_time = 0
                else:
                    print("⏩ 페이지 동일 - 재시도 대기")
                    skip_until_time = now + 3
        else:
            book_detected_time = None
            skip_until_time = 0

        cv2.imshow("Auto Book Scan", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("🛑 ESC 누름. 종료")
            break

    cap.release()
    cv2.destroyAllWindows()

# GPT 대화 + 메뉴 처리 (Rule-based)
def gpt_menu_conversation():
    recognizer = sr.Recognizer()
    while True:
        try:
            with sr.Microphone() as source:
                print("\n🎤 메뉴를 말씀해주세요: '책읽기' 또는 '그만'")
                speak("책읽기, 또는 그만 중에서 말해주세요.")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            user_speech = recognizer.recognize_google(audio, language="ko-KR")
            print("🗣️ 인식된 말:", user_speech)

            if "책읽기" in user_speech:
                speak("책을 스캔할게요. 중간에 '그만'이라고 말하면 멈춰요.")
                auto_detect_and_ocr_with_stop()
                continue
            elif "그만" in user_speech:
                speak("종료할게요.")
                break
            else:
                speak("죄송해요. '책읽기'나 '그만' 중에서 말씀해주세요.")
        except sr.UnknownValueError:
            print("❌ 음성 인식 실패")
            speak("다시 말씀해주세요.")
        except Exception as e:
            print("⚠️ 오류 발생:", e)

# 웨이크워드 감지 루프
def listen_for_wakeword():
    recognizer = sr.Recognizer()
    print("🎤 웨이크워드 '복돌'을 기다리는 중...")

    while True:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio, language="ko-KR")
            print("📝 인식:", text)

            if "복돌" in text:
                speak("무엇을 도와드릴까요?")
                gpt_menu_conversation()
        except:
            continue

# 실행
listen_for_wakeword()
