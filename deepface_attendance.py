import cv2
import numpy as np
import csv
import os
import streamlit as st
from datetime import datetime
from deepface import DeepFace
from PIL import Image

# Load reference images
image_dir = "photos/"
known_faces = {}

st.title("Face Recognition Attendance System (DeepFace)")

# Load known faces
with st.spinner("Loading reference faces..."):
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(image_dir, fname)
            name = fname.split('.')[0]
            known_faces[name] = file_path
            
st.success(f"Loaded {len(known_faces)} reference faces: {', '.join(known_faces.keys())}")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Save uploaded image temporarily
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    
    # Read image for display
    img = Image.open(uploaded_image)
    img_array = np.array(img)
    
    # Create or open CSV file
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    attendance_file = current_date + '.csv'
    
    students = list(known_faces.keys())
    attendance_records = []
    
    st.write("### Recognition Results:")
    
    # Try to match with each known face
    for name, ref_path in known_faces.items():
        try:
            result = DeepFace.verify(
                img1_path=temp_path,
                img2_path=ref_path,
                model_name="VGG-Face",
                enforce_detection=False
            )
            
            if result['verified']:
                st.success(f"✓ {name} - Present (Distance: {result['distance']:.4f})")
                if name in students:
                    students.remove(name)
                    attendance_records.append([name, 'present'])
            else:
                st.info(f"✗ {name} - Not matched (Distance: {result['distance']:.4f})")
        except Exception as e:
            st.warning(f"Could not process {name}: {str(e)}")
    
    # Display image
    st.image(img_array, caption='Uploaded Image', use_container_width=True)
    
    # Write to CSV
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Status"])
        writer.writerows(attendance_records)
        for name in students:
            writer.writerow([name, 'absent'])
    
    st.success(f"Attendance saved to {attendance_file}")
    
    # Show attendance summary
    st.write("### Attendance Summary:")
    st.write(f"Present: {len(attendance_records)}")
    st.write(f"Absent: {len(students)}")
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
