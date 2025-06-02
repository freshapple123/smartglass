import cv2
import os
import time

# 저장 디렉토리
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# 웹캠 열기 (0번 기본 카메라, 1이면 외장)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ 카메라 열기 실패")
    exit()

print("📷 카메라 실행 중 - w 키로 이미지 저장, ESC로 종료")

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
        timestamp = int(time.time())
        filename = os.path.join(save_dir, f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ 저장됨: {filename}")

cap.release()
cv2.destroyAllWindows()
