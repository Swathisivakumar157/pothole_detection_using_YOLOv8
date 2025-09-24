from ultralytics import YOLO

model = YOLO("yolov9s.pt")

model.train(
    data=r"D:\Projects\Pothole_Detection\POTHOLE-DETECTION-2\data.yaml",
    imgsz=1280,
    epochs=50,
    batch=16,
    name="model",
)
