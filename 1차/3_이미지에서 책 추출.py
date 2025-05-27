import cv2
from ultralytics import YOLO

def detect_book_from_webcam(model_path="yolo11n.pt"):
    # YOLO 모델 로드
    model = YOLO(model_path)

    # 웹캠 열기
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ 카메라 열기 실패")
        return

    print("📷 웹캠 실행 중... 'book' 감지 시 표시됩니다. ESC 키로 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 객체 탐지
        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 73:  # COCO 기준 'book'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"book {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO Book Detection", frame)

        # ESC 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# 실행
detect_book_from_webcam()
