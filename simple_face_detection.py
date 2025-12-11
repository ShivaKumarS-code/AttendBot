import cv2
import numpy as np
import csv
import os
import streamlit as st
from datetime import datetime
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO("yolov8/yolov8n-face.pt")

# Streamlit app
st.title("Face Detection Attendance System (YOLO Only)")
st.write("Upload an image to detect faces")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get YOLO results
    results = yolo_model(img_rgb)
    
    # Draw boxes on detected faces
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = box.conf[0].item()
        
        # Draw rectangle
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_rgb, f'Face {confidence:.2f}', (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display results
    st.image(img_rgb, caption=f'Detected {len(results[0].boxes)} faces', use_container_width=True)
    
    st.success(f"Found {len(results[0].boxes)} face(s) in the image!")
