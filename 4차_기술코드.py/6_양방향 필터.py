import cv2

def bilateral_filter_and_save(image_path, output_path):
    # 이미지 읽기 (grayscale로 읽고 싶으면 0)
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 이미지를 불러올 수 없습니다. 경로 확인:", image_path)
        return

    # 양방향 필터 적용 (값은 상황에 따라 튜닝 가능)
    filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # 저장
    cv2.imwrite(output_path, filtered)
    print(f"✅ 양방향 필터 이미지 저장: {output_path}")

# 사용 예시
bilateral_filter_and_save("result2/book_page_clahe.jpg", "output_bilateral2.jpg")
