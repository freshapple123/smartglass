import cv2
import os
import time
import numpy as np

# 저장 폴더
save_dir = "captured_images_avg_sharp"
os.makedirs(save_dir, exist_ok=True)

# 웹캠 열기
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ 웹캠 열기 실패")
    exit()

# 샤프닝 필터 함수 (약한 효과)
def sharpen_image(img):
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 2.5, -0.5],
                       [0, -0.5, 0]])
    return cv2.filter2D(img, -1, kernel)

# 평균 + 샤프닝 처리
def capture_averaged_sharpened_frame(cap, num_frames=10):
    frames = []
    print(f"📸 {num_frames}장 프레임 캡처 중...")
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)
        time.sleep(0.03)  # 프레임 간 시간 간격

    if not frames:
        return None

    avg = np.zeros_like(frames[0], dtype=np.float32)
    for f in frames:
        avg += f.astype(np.float32)
    avg /= len(frames)

    # 밝기 보정
    result = np.clip(avg * 1.1, 0, 255).astype(np.uint8)

    # 샤프닝 처리
    sharpened = sharpen_image(result)
    return sharpened

print("📷 영상 미리보기 중 - w 키로 평균+샤프닝 이미지 저장, ESC로 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera Preview", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        print("🛑 종료")
        break
    elif key == ord('w'):
        print("🔄 평균+샤프닝 처리 중...")
        processed_frame = capture_averaged_sharpened_frame(cap, num_frames=10)
        if processed_frame is not None:
            timestamp = int(time.time())
            filename = os.path.join(save_dir, f"capture_{timestamp}_avg_sharp.jpg")
            cv2.imwrite(filename, processed_frame)
            print(f"✅ 저장 완료: {filename}")
        else:
            print("❌ 저장 실패: 유효한 프레임 없음")

cap.release()
cv2.destroyAllWindows()
