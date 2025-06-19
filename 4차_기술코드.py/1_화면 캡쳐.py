import cv2
import os

# 📂 저장 경로 설정
save_dir = "captured_images_2"
os.makedirs(save_dir, exist_ok=True)

# 📸 웹캠 열기 (기본: 0, 외장: 1 등 상황에 맞게 조정)
cap = cv2.VideoCapture(1)

# ✅ 최대 해상도 요청
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)  # UHD 4K 시도
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

# 실제 적용된 해상도 확인
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"🎥 현재 해상도: {width}x{height}")

print("📷 카메라 실행 중 - 's' 키로 사진 저장, ESC 키로 종료")

# 💡 프리뷰용 크기
preview_size = (960, 540)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    # 👁 프리뷰는 축소해서 보기만 함
    preview_frame = cv2.resize(frame, preview_size)
    cv2.imshow("Camera Preview", preview_frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        filename = os.path.join(save_dir, "captured_image.jpg")
        cv2.imwrite(filename, frame)  # 🔹 원본 해상도로 저장
        print(f"✅ 이미지 저장됨: {filename}")

cap.release()
cv2.destroyAllWindows()
