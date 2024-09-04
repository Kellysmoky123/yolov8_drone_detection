import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np
import os
from PIL import Image

model = YOLO('best.pt')

st.title("Drone Detection using YOLOv8")
st.image('drone.jpg')
st.write('Drones are becoming increasingly prevalent in various industries, including agriculture, logistics, surveillance, and entertainment. However, the rise in drone usage also brings concerns related to safety, privacy, and security. Effective drone detection is essential to identify and monitor unauthorized or potentially harmful drone activity in restricted or sensitive areas. By leveraging advanced machine learning models like YOLO, this model offers real-time drone detection capabilities, ensuring that any drone entering a monitored airspace can be quickly identified and responded to. This technology is crucial for safeguarding public spaces, critical infrastructure, and personal privacy.')
uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    # Check if the uploaded file is an image
    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Proccessing Image. Please wait.") :
            img_array = np.array(image)
            results = model(img_array, conf=0.45)
            annotated_image = results[0].plot()
        st.image(annotated_image, caption="Detected Objects", use_column_width=True)

    # Check if the uploaded file is a video
    elif uploaded_file.type in ["video/mp4", "video/avi", "video/mov"]:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Open the video file
        cap = cv2.VideoCapture(tfile.name)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Check if the video was opened successfully
        if not cap.isOpened():
            st.error("Could not open the video file.")
        else:
            # Temporary file to save the processed video
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')

            # VideoWriter to save the processed video with VP8 encoding
            fourcc = cv2.VideoWriter_fourcc(*'VP80')  # 'VP80' is the FourCC code for VP8
            out = cv2.VideoWriter(output_file.name, fourcc, fps, (width, height))

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_text='Processing Video. Please wait.'
            progress = st.progress(0, progress_text)

            # Process each frame
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame, conf=0.45)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
                progress.progress((i + 1) / frame_count, progress_text)

            cap.release()
            out.release()

            st.video(output_file.name)

            os.remove(tfile.name)
            os.remove(output_file.name)
