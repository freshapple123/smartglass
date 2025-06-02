import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# 📁 저장 폴더 준비
save_dir = "book_pages"
os.makedirs(save_dir, exist_ok=True)

# 📦 YOLO 모델 로드 (책 클래스 포함 모델 필요)
model = YOLO("yolo11n.pt")  # COCO 기준이면 'book'은 class_id = 73

# 📸 이미지 로드
img_path = "captured_images\capture_1748829161.jpg"  # 이미지 경로
img = cv2.imread(img_path)

# 📍 객체 탐지
results = model.predict(img)[0]

# 📘 'book' 클래스 ID (COCO 기준 73)
book_class_id = 73

# 📌 가장 큰 'book' 객체 찾기
book_boxes = []
for box in results.boxes:
    cls_id = int(box.cls[0])
    if cls_id == book_class_id:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        book_boxes.append((area, (x1, y1, x2, y2)))

# ❗ 책이 없으면 종료
if not book_boxes:
    print("❌ 책 객체를 찾지 못했습니다.")
    exit()

# 📚 가장 큰 책 객체 선택
_, (x1, y1, x2, y2) = max(book_boxes, key=lambda b: b[0])

# 🖼️ 책 부분만 잘라내기
book_crop = img[y1:y2, x1:x2]

# ✂️ 좌우로 반 나누기
h, w, _ = book_crop.shape
left_page = book_crop[:, :w//2]
right_page = book_crop[:, w//2:]

# 🧾 저장
timestamp = int(time.time())
cv2.imwrite(f"{save_dir}/page_left_{timestamp}.jpg", left_page)
cv2.imwrite(f"{save_dir}/page_right_{timestamp}.jpg", right_page)

print("✅ 좌/우 페이지 저장 완료!")
