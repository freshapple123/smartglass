import cv2
import os

def split_and_save(image_path, output_dir="output"):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 이미지를 불러올 수 없습니다. 경로 확인:", image_path)
        return

    # 가로/세로 크기 가져오기
    h, w, _ = image.shape
    mid = w // 2

    # 좌우 이미지 분할
    left_img = image[:, :mid]
    right_img = image[:, mid:]

    # 저장 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 저장
    left_path = os.path.join(output_dir, "left_page.jpg")
    right_path = os.path.join(output_dir, "right_page.jpg")
    cv2.imwrite(left_path, left_img)
    cv2.imwrite(right_path, right_img)

    print(f"✅ 좌우 분할 완료:\n - {left_path}\n - {right_path}")

# 예시 사용
split_and_save("cropped_book.jpg")  # 여기에 네 이미지 경로 넣으면 됨
