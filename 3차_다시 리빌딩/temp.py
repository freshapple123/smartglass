import cv2
import os
import time

# 저장 폴더
save_dir = "highres_screenshots"
os.makedirs(save_dir, exist_ok=True)

# 카메라 열기
cap = cv2.VideoCapture(1)  # 보통 0 또는 1

# 최대 해상도 설정 (가능한 경우)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # 예: 4K
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

# 설정된 해상도 출력
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📷 현재 해상도: {width}x{height}")

print("🖼 w 키로 스크린샷(PNG) 저장, ESC로 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 캡처 실패")
        break

    cv2.imshow("Camera Preview", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        print("🛑 종료")
        break
    elif key == ord('w'):
        timestamp = int(time.time())
        filename = os.path.join(save_dir, f"screenshot_{timestamp}.png")
        cv2.imwrite(filename, frame)  # PNG 저장 (무손실)
        print(f"✅ 저장됨: {filename}")

cap.release()
cv2.destroyAllWindows()
