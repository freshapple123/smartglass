import cv2
import os
import numpy as np

def apply_clahe(image_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    이미지를 불러와 CLAHE를 적용하고 저장합니다.

    Args:
        image_path (str): 입력 이미지 파일 경로.
        output_path (str): 처리된 이미지를 저장할 경로.
        clip_limit (float): 대비(Contrast) 제한 값. 값이 클수록 대비가 강해집니다.
        tile_grid_size (tuple): 이미지를 나눌 그리드 크기.
    """
    # 1. 이미지를 흑백(Grayscale)으로 불러오기
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 이미지를 성공적으로 불러왔는지 확인
    if img is None:
        print(f"오류: '{image_path}' 경로에서 이미지를 불러올 수 없습니다.")
        print("파일 경로가 올바른지 확인해주세요.")
        return

    print(f"'{image_path}' 이미지 로딩 성공!")

    # 2. CLAHE 객체 생성
    # cv2.createCLAHE(clipLimit, tileGridSize)
    # - clipLimit: 대비 제한 값. 노이즈 증폭을 막는 역할을 합니다.
    # - tileGridSize: 이미지를 몇 개의 타일로 나눌지 결정합니다.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # 3. CLAHE 적용
    clahe_img = clahe.apply(img)
    
    print(f"CLAHE 적용 완료 (clipLimit={clip_limit}, tileGridSize={tile_grid_size}).")

    # 4. 결과 이미지 저장
    try:
        # 저장 경로의 디렉토리가 없으면 생성
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"'{output_dir}' 디렉토리 생성 완료.")
            
        cv2.imwrite(output_path, clahe_img)
        print(f"성공! 처리된 이미지가 '{output_path}' 경로에 저장되었습니다.")

    except Exception as e:
        print(f"오류: 이미지를 저장하는 데 실패했습니다. - {e}")

# --- 여기에서 경로와 파라미터를 수정하세요 ---
# 입력 이미지 경로 (가지고 있는 책 이미지 파일로 변경하세요)
INPUT_IMAGE_PATH = 'result/book_page_adaptive2.jpg' 

# 결과 이미지를 저장할 경로
OUTPUT_IMAGE_PATH = 'result2/book_page_clahe2.jpg'

# CLAHE 파라미터 (이 값들을 조정하며 최적의 결과를 찾아보세요)
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)
# ---------------------------------------------


if __name__ == "__main__":
    # 스크립트 실행 시 book_page.jpg 라는 이름의 샘플 이미지가 없으면,
    # 코드 실행을 위해 간단한 회색 샘플 이미지를 생성합니다.
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"'{INPUT_IMAGE_PATH}' 파일이 존재하지 않아, 테스트용 샘플 이미지를 생성합니다.")
        # 그라데이션이 있는 샘플 이미지 생성
        sample_img = np.zeros((400, 600), dtype=np.uint8)
        # 어두운 부분과 밝은 부분 만들기
        sample_img[:, :300] = np.linspace(20, 100, 300, dtype=np.uint8)
        sample_img[:, 300:] = np.linspace(150, 220, 300, dtype=np.uint8)
        # 그림자 효과를 위한 어두운 사각형 추가
        cv2.rectangle(sample_img, (50, 50), (250, 350), 30, -1)
        # 샘플 텍스트 추가
        cv2.putText(sample_img, "Text in shadow", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
        cv2.putText(sample_img, "Text in light", (320, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (20,20,20), 2)
        cv2.imwrite(INPUT_IMAGE_PATH, sample_img)

    apply_clahe(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH, clip_limit=CLIP_LIMIT, tile_grid_size=TILE_GRID_SIZE)
