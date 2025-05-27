import cv2
import numpy as np
from ultralytics import YOLO
import uuid
import os
import requests
import base64
import json
from dotenv import load_dotenv

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
        print(f"📖 OCR 결과 ({os.path.basename(image_path)}):\n{description}\n")
    except (KeyError, IndexError):
        print(f"❌ 텍스트 인식 실패: {image_path}")
        print(json.dumps(result, indent=2))

# 📸 웹캠에서 책 인식하고 스페이스바 누르면 저장 + OCR 실행
def detect_book_on_space(model_path="yolo11n.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    print("📷 웹캠 실행 중... 책 감지되면 스페이스바로 OCR까지 실행. ESC로 종료.")
    save_dir = "book_pages"
    os.makedirs(save_dir, exist_ok=True)

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
            cv2.putText(frame, "BOOK DETECTED (press SPACE)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Book Detection - Press SPACE", frame)
        key = cv2.waitKey(1)

        if key == 32 and book_box:
            x1, y1, x2, y2 = map(int, book_box.xyxy[0])
            book_crop = frame[y1:y2, x1:x2]
            h, w, _ = book_crop.shape
            mid = w // 2
            left_page = book_crop[:, :mid]
            right_page = book_crop[:, mid:]

            # 👉 전처리 없이 바로 저장
            left_path = os.path.join(save_dir, f"left_{uuid.uuid4().hex[:6]}.jpg")
            right_path = os.path.join(save_dir, f"right_{uuid.uuid4().hex[:6]}.jpg")
            cv2.imwrite(left_path, left_page)
            cv2.imwrite(right_path, right_page)
            print(f"✅ 저장 완료: {left_path}, {right_path}")
 
            # ✅ OCR 실행
            detect_text_rest(left_path)
            detect_text_rest(right_path)

        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# 실행
detect_book_on_space()
