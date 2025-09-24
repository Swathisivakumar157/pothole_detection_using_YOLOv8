# pothole_detection_using_YOLOv8
Pothole Detection using YOLOv8 on Raspberry Pi

A real-time pothole detection system using YOLOv8 running on a Raspberry Pi 4 with a Zebronics Crystal Pro camera. This project captures live video from the camera, detects potholes using a trained YOLOv8 model, and displays annotated results with bounding boxes and confidence scores.

🔹 Features

Real-time pothole detection from Raspberry Pi camera feed

Annotates potholes with bounding boxes, class names, and confidence scores

Uses YOLOv8 for accurate detection

Compatible with Raspberry Pi 4 running Python and OpenCV

Can be extended for road maintenance alerts or data collection

🔹 Project Structure
pothole-detection/
│
├── pothole_detection.py      # Main detection script
├── pothole_new.pt            # Trained YOLOv8 model (do not commit large models!)
├── requirements.txt          # Python dependencies
├── .gitignore                # Files/folders to ignore
└── README.md                 # Project overview

🔹 Installation

Clone the repository

git clone https://github.com/yourusername/pothole-detection.git
cd pothole-detection


Set up Python environment (optional)

python3 -m venv venv
source venv/bin/activate


Install dependencies

pip install -r requirements.txt


requirements.txt example:

ultralytics
opencv-python
numpy

🔹 Usage

Ensure your Zebronics Crystal Pro camera is connected to the Raspberry Pi. Update your capture code in pothole_detection.py if using cv2.VideoCapture:

cap = cv2.VideoCapture(0)  # 0 is usually the default camera index


Update the YOLO model path:

model = YOLO(r"/home/pi/models/pothole_new.pt")


Run the detection script:

python3 pothole_detection.py


Press q to exit the live detection window.

🔹 How It Works

Captures frames from the Raspberry Pi camera using OpenCV

Passes frames to YOLOv8 for inference

Annotates the frame with bounding boxes and confidence scores

Displays the live annotated video

🔹 License

This project uses AGPL-3.0 License to comply with the Ultralytics YOLOv8 license. See LICENSE
 for details.

🔹 Future Improvements

Save detected frames automatically for further analysis

Add a dashboard or web interface to monitor potholes

Integrate with GPS for mapping pothole locations

Optimize performance for Raspberry Pi deployment

🔹Paper published
S. D. R S, J. S. A, S. S and T. Van S K, "Pothole Detection and Instance Segmentation Using Yolo V8," 2024 International Conference on IoT Based Control Networks and Intelligent Systems (ICICNIS), Bengaluru, India, 2024, pp. 1185-1190, doi: 10.1109/ICICNIS64247.2024.10823139.
keywords: {Instance segmentation;Training;Accuracy;Roads;Computational modeling;Visual systems;Real-time systems;Safety;Automobiles;Accidents;Object identification;computer vision;convolution neural networks;YOLO;bounding box;and;instance segmentation},
OpenCV Python Tutorials

Raspberry Pi Camera Guide
