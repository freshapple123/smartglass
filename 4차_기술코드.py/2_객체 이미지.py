import cv2
from ultralytics import YOLO
import numpy as np
import os

# 📌 모델 로드
model = YOLO("yolo11n.pt")  # 커스텀 YOLO 모델 경로

# 📌 이미지 로드
img_path = "화면캡쳐/captured_image.jpg"
image = cv2.imread(img_path)

if image is None:
    print("❌ 이미지 파일을 불러올 수 없습니다. 경로 확인하세요.")
    exit()

# 📌 YOLO 객체 감지
results = model(image)[0]

# 📌 가장 신뢰도 높은 'book' 객체만 추출
book_boxes = []
for box in results.boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    if cls_id == 73:  # 'book' 클래스
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        book_boxes.append((conf, (x1, y1, x2, y2)))

# 📌 신뢰도 높은 책 1개 추출 및 저장
if book_boxes:
    best_box = sorted(book_boxes, key=lambda x: x[0], reverse=True)[0][1]
    x1, y1, x2, y2 = best_box

    # 1. 잘라낸 책 이미지 저장
    cropped_book = image[y1:y2, x1:x2]
    cv2.imwrite("cropped_book.jpg", cropped_book)
    print("✅ 책 객체 잘림 이미지 저장 완료: cropped_book.jpg")

    # 2. 원본 이미지에 바운딩 박스 그려서 저장
    boxed_image = image.copy()
    cv2.rectangle(boxed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite("boxed_image.jpg", boxed_image)
    print("✅ 바운딩 박스 이미지 저장 완료: boxed_image.jpg")

    # 보기용 출력도 가능
    cv2.imshow("📕 Cropped Book", cropped_book)
    cv2.imshow("📦 Boxed Image", boxed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ 책 객체를 찾을 수 없습니다.")
