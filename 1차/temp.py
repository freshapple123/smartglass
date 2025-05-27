from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # COCO pretrained model
print(model.names)  # {0: 'person', 1: 'bicycle', ..., 79: 'toothbrush'}
