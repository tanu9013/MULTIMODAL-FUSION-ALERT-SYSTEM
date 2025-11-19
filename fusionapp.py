#this contain only detection through images , audio , and real time camera
import streamlit as st
import cv2
import numpy as np
import os
import librosa
import sounddevice as sd
import threading
import time
from ultralytics import YOLO

# CONFIGURATION

YOLO_MODEL_PATH = "yolov8n.pt"
SAMPLE_RATE = 22050
AUDIO_DURATION = 2.0
AUDIO_THRESHOLD = 0.05
DETECTION_CONFIDENCE = 0.5

# INITIALIZE MODELS

yolo_model = YOLO(YOLO_MODEL_PATH)

# Shared states
if "running" not in st.session_state:
    st.session_state.running = False
if "audio_alert" not in st.session_state:
    st.session_state.audio_alert = False
if "vision_alert" not in st.session_state:
    st.session_state.vision_alert = False

# AUDIO DETECTION THREAD

def audio_detection_loop():
    while st.session_state.running:
        try:
            audio_data = sd.rec(int(SAMPLE_RATE * AUDIO_DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            rms = np.mean(librosa.feature.rms(y=audio_data.flatten())[0])
            st.session_state.audio_alert = rms > AUDIO_THRESHOLD
            time.sleep(0.5)
        except Exception as e:
            st.error(f"Audio detection error: {e}")
            break
# VISION DETECTION FUNCTION

def detect_objects(frame):
    results = yolo_model(frame)
    boxes = results[0].boxes
    alert_triggered = False

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf > DETECTION_CONFIDENCE:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = yolo_model.names[cls_id]
            color = (0, 255, 0)
            if label.lower() in ["dog", "cat", "cow", "person", "animal"]:
                alert_triggered = True
                color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    st.session_state.vision_alert = alert_triggered
    return frame

# STREAMLIT FRONTEND
st.set_page_config(page_title="Multimodal Alert System", layout="wide")
st.title("🚦 Real-Time Multimodal Alert System")
st.markdown("Combining **YOLO Vision Detection** and **Audio Horn Detection** for Intelligent Traffic Alerts.")

col1, col2 = st.columns(2)
video_placeholder = col1.empty()
status_placeholder = col2.empty()

# START/STOP CONTROLS

start = st.button("▶️ Start Real-Time Detection", disabled=st.session_state.running)
stop = st.button("⏹️ Stop Detection", disabled=not st.session_state.running)

if start:
    st.session_state.running = True
    audio_thread = threading.Thread(target=audio_detection_loop, daemon=True)
    audio_thread.start()

    cap = cv2.VideoCapture(0)
    while st.session_state.running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not accessible.")
            break
        frame = cv2.flip(frame, 1)
        processed = detect_objects(frame)

        combined_alert = st.session_state.audio_alert and st.session_state.vision_alert

        video_placeholder.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB")

        status_placeholder.markdown(
            f"""
            ### 🧠 Fusion Status
            - 🎥 **Vision Alert:** {'🟥' if st.session_state.vision_alert else '🟩'}
            - 🔊 **Audio Alert:** {'🟥' if st.session_state.audio_alert else '🟩'}
            - 🚨 **Combined Alert:** {'🔥 ALERT DETECTED!' if combined_alert else '✅ Normal'}
            """
        )
        time.sleep(0.1)

    cap.release()
    st.session_state.running = False

if stop:
    st.session_state.running = False
    st.success("🛑 Real-time detection stopped.")

# BROWSE EXISTING IMAGE FILES

st.sidebar.markdown("---")
st.sidebar.header("🗂️ Analyze Existing Image Files")

browse_images = st.sidebar.checkbox("Enable browsing existing images")
if browse_images:
    folder_path = st.sidebar.text_input("Enter image folder path:")
    if folder_path and os.path.exists(folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            selected_image = st.selectbox("Select an image to analyze:", image_files)
            image_path = os.path.join(folder_path, selected_image)
            st.image(image_path, caption="Selected Image", use_container_width=True)

            if st.button("🔍 Analyze Selected Image"):
                results = yolo_model(image_path)
                annotated = results[0].plot()
                st.image(annotated, caption="Detected Objects", use_container_width=True)
                st.success("✅ Detection complete on existing image.")
        else:
            st.warning("No image files found.")
    else:
        st.info("Enter a valid image folder path to browse.")

# BROWSE EXISTING AUDIO FILES

st.sidebar.markdown("---")
st.sidebar.header("🎵 Analyze Existing Audio Files")

browse_audio = st.sidebar.checkbox("Enable browsing existing audio")
if browse_audio:
    audio_folder = st.sidebar.text_input("Enter audio folder path:")
    if audio_folder and os.path.exists(audio_folder):
        audio_files = [f for f in os.listdir(audio_folder) if f.endswith(('.wav', '.mp3', '.ogg'))]
        if audio_files:
            selected_audio = st.selectbox("Select an audio file:", audio_files)
            audio_path = os.path.join(audio_folder, selected_audio)
            st.audio(audio_path)

            if st.button("🎧 Analyze Selected Audio"):
                y, sr = librosa.load(audio_path, sr=None)
                energy = np.mean(librosa.feature.rms(y=y))
                if energy > AUDIO_THRESHOLD:
                    st.warning("🚨 High energy — possible horn or alert detected!")
                else:
                    st.info("✅ Normal audio detected.")
        else:
            st.warning("No audio files found.")
    else:
        st.info("Enter a valid audio folder path to browse.")
 