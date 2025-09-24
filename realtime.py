from ultralytics import YOLO

# from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Perform detection on the video source (webcam in this case)
results = model.predict(source="0", show=True)  # accepts all formats: img/folder/vid

# Print the detection results
print(results)

# Import YOLO module
