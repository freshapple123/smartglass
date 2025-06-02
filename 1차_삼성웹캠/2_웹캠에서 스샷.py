import cv2
import uuid
import os

def capture_image_from_webcam(save_dir="captures"):
    # 저장 폴더 없으면 생성
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("❌ 카메라 열기 실패")
        return

    print("📷 웹캠 실행 중... 스페이스바 누르면 캡처 / ESC 누르면 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Webcam - Press SPACE to capture", frame)
        key = cv2.waitKey(1)

        if key == 32:  # 스페이스바 누르면 캡처
            filename = f"{save_dir}/capture_{uuid.uuid4().hex[:8]}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ 이미지 저장됨: {filename}")
            break

        elif key == 27:  # ESC 종료
            break

    cap.release()
    cv2.destroyAllWindows()
    return filename  # 저장된 이미지 경로 반환

# 실행 예시
if __name__ == "__main__":
    path = capture_image_from_webcam()
    print("📁 저장된 이미지 경로:", path)
