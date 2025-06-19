import cv2
import os

def apply_adaptive_threshold(image_path, output_path):
    """
    이미지를 불러와 적응형 이진화를 적용하고 저장합니다.

    Args:
        image_path (str): 입력 이미지 파일 경로
        output_path (str): 처리된 이미지를 저장할 경로
    """
    # 1. 이미지 불러오기
    # cv2.IMREAD_GRAYSCALE 옵션으로 이미지를 바로 흑백으로 불러옵니다.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 이미지를 성공적으로 불러왔는지 확인
    if img is None:
        print(f"오류: '{image_path}' 경로에서 이미지를 불러올 수 없습니다.")
        print("파일 경로가 올바른지 확인해주세요.")
        return

    print(f"'{image_path}' 이미지 로딩 성공!")

    # 2. 적응형 이진화 적용
    # cv2.adaptiveThreshold(소스, 최댓값, 적응형 메소드, 스레시홀드 타입, 블록 크기, 감산 상수 C)
    # - maxValue: 임계값을 넘었을 때 적용할 값 (보통 255, 흰색)
    # - adaptiveMethod: 사용할 적응형 스레시홀딩 알고리즘
    #   - cv2.ADAPTIVE_THRESH_MEAN_C: 주변 영역의 평균값을 임계값으로 사용
    #   - cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 주변 영역의 가우시안 가중치 합을 임계값으로 사용 (더 좋은 결과)
    # - thresholdType: 스레시홀딩 타입 (cv2.THRESH_BINARY 또는 cv2.THRESH_BINARY_INV)
    # - blockSize: 임계값을 계산할 주변 영역의 크기 (홀수여야 함)
    # - C: 계산된 임계값에서 뺄 상수 (음수도 가능). 미세 조정을 위해 사용.
    block_size = 11
    C = 5 # 이 값을 조정하며 최적의 결과를 찾아보세요.
    
    binary_img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    print("적응형 이진화 적용 완료.")

    # 3. 결과 이미지 저장
    try:
        # 저장 경로의 디렉토리가 없으면 생성
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"'{output_dir}' 디렉토리 생성 완료.")
            
        cv2.imwrite(output_path, binary_img)
        print(f"성공! 처리된 이미지가 '{output_path}' 경로에 저장되었습니다.")

    except Exception as e:
        print(f"오류: 이미지를 저장하는 데 실패했습니다. - {e}")

# --- 여기에서 경로를 수정하세요 ---
# 입력 이미지 경로 (가지고 있는 책 이미지 파일로 변경하세요)
INPUT_IMAGE_PATH = 'output/right_page.jpg' 

# 결과 이미지를 저장할 경로
OUTPUT_IMAGE_PATH = 'result/book_page_adaptive2.jpg'
# ------------------------------------


if __name__ == "__main__":
    # 스크립트 실행 시 book_page.jpg 라는 이름의 샘플 이미지가 없으면,
    # 코드 실행을 위해 간단한 회색 샘플 이미지를 생성합니다.
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"'{INPUT_IMAGE_PATH}' 파일이 존재하지 않아, 테스트용 샘플 이미지를 생성합니다.")
        import numpy as np
        # 그라데이션이 있는 샘플 이미지 생성
        sample_img = np.zeros((400, 600), dtype=np.uint8)
        sample_img[:, :] = np.linspace(50, 200, 600, dtype=np.uint8)
        # 샘플 텍스트 추가
        cv2.putText(sample_img, "Sample Text on varying background", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.imwrite(INPUT_IMAGE_PATH, sample_img)


    apply_adaptive_threshold(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH)

