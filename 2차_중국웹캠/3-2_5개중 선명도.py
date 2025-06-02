import cv2
import os
import time
import numpy as np

# 저장 디렉토리
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# 웹캠 열기 (0번: 기본, 1번: 외장)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ 카메라 열기 실패")
    exit()

# 선명도 측정 함수
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# 가장 선명한 프레임 고르기
def get_sharpest_frame(cap, count=5):
    max_score = -1
    sharpest_frame = None
    for _ in range(count):
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = variance_of_laplacian(gray)
        if score > max_score:
            max_score = score
            sharpest_frame = frame.copy()
    return sharpest_frame, max_score

print("📷 카메라 실행 중 - w 키로 선명한 이미지 저장, ESC로 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    cv2.imshow("Camera Preview", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        print("🛑 종료")
        break
    elif key == ord('w'):
        print("🔍 선명한 프레임 선택 중...")
        best_frame, blur_score = get_sharpest_frame(cap, count=5)
        if best_frame is not None:
            timestamp = int(time.time())
            filename = os.path.join(save_dir, f"capture_{timestamp}_sharp.jpg")
            cv2.imwrite(filename, best_frame)
            print(f"✅ 저장됨: {filename} (선명도: {blur_score:.2f})")
        else:
            print("❌ 저장 실패: 프레임 없음")

cap.release()
cv2.destroyAllWindows()
