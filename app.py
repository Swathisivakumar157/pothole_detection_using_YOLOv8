import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import tempfile
import numpy as np

# Load YOLO model
model = YOLO("model.pt")

# Streamlit app
st.title("YOLO Object Detection App")

# Sidebar options for different modes
option = st.sidebar.selectbox(
    "Select Mode", ("Upload Image", "Upload Video", "Real-time Camera")
)

if option == "Upload Image":
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the PIL image to an OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Add a button to show output after prediction
        if st.button("Show Output", key="show_image_output"):
            # Perform YOLO prediction
            results = model.predict(source=image, save=True, save_txt=True)

            # Display the processed image
            processed_image = results[0].plot()
            st.image(processed_image, caption="Processed Image", use_column_width=True)

elif option == "Upload Video":
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video_file.write(uploaded_video.read())
        temp_video_file.flush()
        temp_video_file.close()

        st.video(temp_video_file.name)

        # Add a button to show output after prediction
        if st.button("Show Output", key="show_video_output"):
            # Perform YOLO prediction on the video
            model.predict(source=temp_video_file.name, save=True)


elif option == "Real-time Camera":
    st.text("Using webcam...")

    # Start/Stop camera button toggle
    start_camera = st.button("Start Camera", key="start_camera_button")
    stop_camera = st.button("Stop Camera", key="stop_camera_button")

    if start_camera:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            # Frame display in real-time
            frame_placeholder = st.empty()

            # Run camera loop until "Stop Camera" is pressed
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture image from camera.")
                    break

                # YOLO prediction
                results = model.predict(source=frame)

                # Display the frame with detections
                processed_frame = results[0].plot()
                frame_placeholder.image(processed_frame, channels="BGR")

                # Check if "Stop Camera" is pressed
                if stop_camera:
                    cap.release()
                    break

            cap.release()
