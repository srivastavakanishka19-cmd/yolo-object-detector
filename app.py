import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Pro AI Detector", layout="wide")
st.title("🛡️ Advanced Object Detection")

# Load the MEDIUM model for better accuracy
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt") 

model = load_model()

# --- Sidebar ---
conf_threshold = st.sidebar.slider("Sensitivity (Confidence)", 0.1, 1.0, 0.25)
run_webcam = st.checkbox("Start High-Accuracy Webcam")
FRAME_WINDOW = st.image([])

# --- Logic ---
cap = cv2.VideoCapture(0)

while run_webcam:
    ret, frame = cap.read()
    if not ret:
        st.error("Webcam not found.")
        break

    # Run Detection
    results = model.predict(frame, conf=conf_threshold, verbose=False)
    annotated_frame = results[0].plot()
    
    # Display the result
    FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

cap.release()