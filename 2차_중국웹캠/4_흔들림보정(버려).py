import cv2
import os
import time
from vidstab import VidStab

# 저장 폴더
save_dir = "vidstab_images"
os.makedirs(save_dir, exist_ok=True)

# VidStab 객체 생성
stabilizer = VidStab()
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ 카메라 열기 실패")
    exit()

print("📷 vidstab 안정화 중 - w 키로 이미지 저장, ESC로 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 안정화된 프레임 생성
    stabilized_frame = stabilizer.stabilize_frame(input_frame=frame, smoothing_window=30)

    # 첫 프레임에는 안정화 불가하므로 원본 사용
    if stabilized_frame is None:
        stabilized_frame = frame

    cv2.imshow("VidStab Stabilized Camera", stabilized_frame)

    key = cv2.waitKey(1)
    if key == 27:
        print("🛑 ESC 눌림 - 종료")
        break
    elif key == ord('w'):
        filename = os.path.join(save_dir, f"vidstab_{int(time.time())}.jpg")
        cv2.imwrite(filename, stabilized_frame)
        print(f"✅ 저장됨: {filename}")

cap.release()
cv2.destroyAllWindows()
