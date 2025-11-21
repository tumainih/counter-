import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(page_title="People Counter", layout="wide")
st.title("🧑‍🤝‍🧑 People Counter with YOLOv8 (Upload Multiple Images)")
st.markdown("Upload one or more images to detect people and crowd level: **Chini, Wastani, Kubwa**")

# -----------------------------
# Load YOLOv8 model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # small YOLOv8 model

model = load_model()

# -----------------------------
# Image uploader (multiple files)
# -----------------------------
uploaded_files = st.file_uploader(
    "📁 Chagua picha (can select multiple)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Convert uploaded file to OpenCV format
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Run detection
        results = model(frame)
        boxes = results[0].boxes

        # Count people (class 0 only)
        person_count = sum(1 for cls in boxes.cls if int(cls) == 0)

        # Determine crowd level
        if person_count < 5:
            level = "Chini"; color = (0, 255, 0)
        elif person_count < 15:
            level = "Wastani"; color = (0, 255, 255)
        else:
            level = "Kubwa"; color = (0, 0, 255)

        # Overlay count on image
        cv2.putText(frame, f"Watu: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Msongamano: {level}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Convert back to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show image with metrics
        st.image(
            frame_rgb,
            caption=f"{uploaded_file.name} → Watu: {person_count}, Msongamano: {level}",
            use_column_width=True
        )
