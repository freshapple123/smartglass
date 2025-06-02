import cv2

# 웹캠 열기 (0번 카메라: 기본 카메라)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    # 프레임을 윈도우에 표시
    cv2.imshow('📷 웹캠', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
cv2.destroyAllWindows()
