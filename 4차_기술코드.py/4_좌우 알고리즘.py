import cv2
import numpy as np
import os

def smart_split_and_save(image_path, output_dir="output"):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 이미지를 불러올 수 없습니다. 경로 확인:", image_path)
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 세로 방향(열 방향)으로 평균 밝기 계산
    column_means = np.mean(gray, axis=0)  # shape: (width,)
    min_col = np.argmin(column_means)  # 가장 어두운 세로줄 (중앙 접힘부일 가능성 높음)

    # 너무 양쪽에 치우친 경우 예외처리 (가짜 접힘부 방지)
    w = image.shape[1]
    if min_col < w * 0.3 or min_col > w * 0.7:
        min_col = w // 2  # fallback: 중앙 기준

    # 좌우 분할
    left_img = image[:, :min_col]
    right_img = image[:, min_col:]

    # 저장
    os.makedirs(output_dir, exist_ok=True)
    left_path = os.path.join(output_dir, "left_page.jpg")
    right_path = os.path.join(output_dir, "right_page.jpg")
    cv2.imwrite(left_path, left_img)
    cv2.imwrite(right_path, right_img)

    print(f"✅ 중앙 어두운 라인 기준 분할 완료:\n - {left_path}\n - {right_path}")

# 사용 예시
smart_split_and_save("cropped_book.jpg")
