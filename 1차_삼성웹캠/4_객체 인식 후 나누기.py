import cv2
import numpy as np
from ultralytics import YOLO
import uuid
import os

# 📌 전처리 함수: 선명도 및 대비 향상
def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 샤프닝 필터
    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)

    return enhanced

# 📸 웹캠으로 책 인식 후 스페이스바로 좌우 페이지 저장
def detect_book_on_space(model_path="yolo11n.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(1)  # 필요시 0 → 1로 변경

    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    print("📷 웹캠 실행 중... 책(book) 감지되면 스페이스바로 좌우 페이지 저장. ESC로 종료.")
    save_dir = "book_pages"
    os.makedirs(save_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        book_box = None
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 73:  # COCO 'book'
                book_box = box
                break

        if book_box:
            x1, y1, x2, y2 = map(int, book_box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "BOOK DETECTED (press SPACE to save)", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Book Detection - Press SPACE to save", frame)
        key = cv2.waitKey(1)

        if key == 32 and book_box:  # 스페이스바
            x1, y1, x2, y2 = map(int, book_box.xyxy[0])
            book_crop = frame[y1:y2, x1:x2]
            h, w, _ = book_crop.shape
            mid = w // 2
            left_page = book_crop[:, :mid]
            right_page = book_crop[:, mid:]

            # 전처리
            left_page = preprocess_for_ocr(left_page)
            right_page = preprocess_for_ocr(right_page)

            # 저장
            left_path = os.path.join(save_dir, f"left_{uuid.uuid4().hex[:6]}.jpg")
            right_path = os.path.join(save_dir, f"right_{uuid.uuid4().hex[:6]}.jpg")
            cv2.imwrite(left_path, left_page)
            cv2.imwrite(right_path, right_page)
            print(f"✅ 전처리 후 저장됨: {left_path}, {right_path}")

        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# 실행
detect_book_on_space()
