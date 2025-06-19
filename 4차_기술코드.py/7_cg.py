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
from PIL import Image, ImageFont, ImageDraw
from gtts import gTTS
import pygame
import threading
from pymongo import MongoClient

# 🔐 환경 변수 로드
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")

stop_flag = False

# ✅ 한글 자막 출력
def put_korean_text(cv2_img, text, position=(50, 60), font_size=32, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 환경에 맞게 경로 조정
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ✅ TTS
def speak(text):
    try:
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
    except:
        pass

# ✅ OCR
def encode_image_cv2_to_base64(cv2_img):
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def detect_text_from_array(cv2_img):
    content = encode_image_cv2_to_base64(cv2_img)
    url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"
    payload = {
        "requests": [{
            "image": {"content": content},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    result = response.json()
    try:
        return result["responses"][0]["textAnnotations"][0]["description"].strip()
    except:
        return ""

def is_different_page(text1, text2, threshold=0.5):
    ratio = SequenceMatcher(None, text1, text2).ratio()
    return ratio < threshold

# ✅ 키보드 명령어 처리 쓰레드
def listen_keyboard(get_last_text_func):
    global stop_flag
    while not stop_flag:
        command = input("👉 명령 입력 (읽어 / 그만): ").strip()
        if command == "그만":
            stop_flag = True
            speak("중단할게요.")
        elif command == "읽어":
            left, right = get_last_text_func()
            speak(f"왼쪽 페이지입니다. {left}")
            speak(f"오른쪽 페이지입니다. {right}")
        else:
            print("⚠️ 지원되지 않는 명령입니다.")

# ✅ 메인 루프
def auto_detect_and_ocr_with_stop(book_title, model_path="yolo11n.pt"):
    global stop_flag
    stop_flag = False

    last_left_text = ""
    last_right_text = ""

    def get_last_text():
        return last_left_text, last_right_text

    threading.Thread(target=listen_keyboard, args=(get_last_text,), daemon=True).start()

    model = YOLO(model_path)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(f"{book_title}_녹화본.mp4", fourcc, 10.0, (FRAME_WIDTH, FRAME_HEIGHT))

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
    all_texts = []
    status_message = ""
    status_time = 0

    print(f"📘 책: {book_title} | OCR 시작")

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

        now = time.time()

        if book_box:
            x1, y1, x2, y2 = map(int, book_box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "BOOK DETECTED", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if book_detected_time is None:
                book_detected_time = now

            if now - book_detected_time >= 5 and now >= skip_until_time:
                book_crop = frame[y1:y2, x1:x2]
                h, w, _ = book_crop.shape
                mid = w // 2
                left_page = book_crop[:, :mid]
                right_page = book_crop[:, mid:]

                left_text = detect_text_from_array(left_page)
                right_text = detect_text_from_array(right_page)

                if len(left_text.strip()) < 30 and len(right_text.strip()) < 30:
                    status_message = "❌ 글자 수 부족 - 생략"
                    status_time = now
                    skip_until_time = now + 5
                    continue

                if is_different_page(left_text, prev_left_text) or is_different_page(right_text, prev_right_text):
                    lpath = os.path.join(base_dir, "left", f"page_{image_counter}_left.jpg")
                    rpath = os.path.join(base_dir, "right", f"page_{image_counter}_right.jpg")
                    cv2.imwrite(lpath, left_page)
                    cv2.imwrite(rpath, right_page)

                    with open(text_file_path, "a", encoding="utf-8") as f:
                        f.write(left_text + "\n" + right_text + "\n")

                    all_texts.append(left_text)
                    all_texts.append(right_text)
                    last_left_text = left_text
                    last_right_text = right_text
                    status_message = f"✅ 페이지 {image_counter} 저장됨"
                    status_time = now
                    prev_left_text = left_text
                    prev_right_text = right_text
                    image_counter += 1
                    book_detected_time = None
                    skip_until_time = 0
                else:
                    status_message = "⏩ 중복 페이지 생략"
                    status_time = now
                    skip_until_time = now + 5
        else:
            book_detected_time = None
            skip_until_time = 0

        # ✅ 자막 그리기
        if time.time() - status_time <= 2 and status_message:
            frame = put_korean_text(frame, status_message, font_size=42, position=(60, 60))

        # ✅ 최종 프레임 녹화
        video_writer.write(frame)

        # ✅ 프리뷰
        preview_frame = cv2.resize(frame, preview_size)
        cv2.imshow("Auto Book Scan", preview_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            stop_flag = True
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    if all_texts:
        try:
            client = MongoClient(MONGO_URI)
            db = client[MONGO_DB_NAME]
            collection = db[MONGO_COLLECTION_NAME]
            doc = {
                "title": book_title,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pages": all_texts
            }
            collection.insert_one(doc)
            print(f"✅ MongoDB에 '{book_title}' 저장 완료")
        except Exception as e:
            print(f"❌ MongoDB 저장 실패: {e}")

    return True

# ✅ 메인 루프
if __name__ == "__main__":
    title = input("📘 책 제목을 입력하세요: ").strip()
    if title:
        auto_detect_and_ocr_with_stop(title)
