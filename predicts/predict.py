from ultralytics import YOLO

model = YOLO(r"D:\Projects\Pothole_Detection\models\best.pt")

# model.predict(source="test1.jpeg", save=True)

model.predict(source=r"D:\Projects\Pothole_Detection\test\test3.mp4", save=True)
