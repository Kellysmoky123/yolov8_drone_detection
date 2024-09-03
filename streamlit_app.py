import streamlit as st
from ultralytics import YOLO
import tempfile
import numpy as np
import os
from PIL import Image
from moviepy.editor import VideoFileClip, ImageSequenceClip

# Load YOLOv8 model
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
        with st.spinner("Processing Image. Please wait."):
            img_array = np.array(image)
            results = model(img_array, conf=0.45)
            annotated_image = results[0].plot()
        st.image(annotated_image, caption="Detected Objects", use_column_width=True)

    # Check if the uploaded file is a video
    elif uploaded_file.type in ["video/mp4", "video/avi", "video/mov"]:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.seek(0)

        # Open the video file with MoviePy
        video = VideoFileClip(tfile.name)

        # Initialize the progress bar
        progress_text='Processing Video. Please wait.'
        progress = st.progress(0,progress_text)
        frame_count = video.reader.nframes

        # Create a list to hold the processed frames
        annotated_frames = []

        # Process each frame and update the progress bar
        for i, frame in enumerate(video.iter_frames()):
            results = model(frame, conf=0.45)
            annotated_frame = results[0].plot()
            annotated_frames.append(annotated_frame)
            progress.progress((i + 1) / frame_count,progress_text)

        # Create a video clip from the processed frames
        annotated_video = ImageSequenceClip(annotated_frames, fps=video.fps)

        # Save the processed video to a temporary file
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
        annotated_video.write_videofile(output_file.name, codec='libvpx')

        st.video(output_file.name)

        # Cleanup
        os.remove(tfile.name)
        os.remove(output_file.name)
