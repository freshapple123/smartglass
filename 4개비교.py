import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정 (환경에 맞게 수정)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows: 맑은 고딕
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

img_path = 'image.png'  # 분석할 이미지 파일명

# 1. 원본 읽기
img = cv2.imread(img_path)
if img is None:
    raise Exception("이미지 파일을 찾을 수 없습니다.")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. CLAHE 적용 (명암 대비 강화)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(gray)

# 🔧 수정: 강하게 적용
bilateral = cv2.bilateralFilter(clahe_img, d=15, sigmaColor=30, sigmaSpace=30)

# 4. 적응형 이진화 (글자/배경 분리)
adaptive = cv2.adaptiveThreshold(
    bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 21, 10
)

# 5. 시각화 (4단계 한눈에 보기)
def gray3(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

result_imgs = [
    img_rgb,
    gray3(clahe_img),
    gray3(bilateral),
    gray3(adaptive)
]
titles = ['원본', 'CLAHE', '양방향 필터', '적응형 이진화']

plt.figure(figsize=(16,6))
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(result_imgs[i])
    plt.title(titles[i], fontsize=15)
    plt.axis('off')
plt.tight_layout()
plt.savefig('compare_preprocessing_final2.png', dpi=200)
plt.show()

# 최종 이진화 결과는 adaptive 변수에 있음!
