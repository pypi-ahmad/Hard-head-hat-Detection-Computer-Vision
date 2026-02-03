import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import os
import time
from pathlib import Path
from collections import Counter

# Configuration
st.set_page_config(page_title="Site Safety Compliance AI", page_icon="🏗️")

CLASSES_CONFIG = {
    'Hardhat': 'safe',
    'Mask': 'safe',
    'Safety Vest': 'safe',
    'NO-Hardhat': 'danger',
    'NO-Mask': 'danger',
    'NO-Safety Vest': 'danger',
    'Person': 'neutral',
    'Safety Cone': 'neutral',
    'machinery': 'neutral',
    'vehicle': 'neutral'
}

COLOR_MAP = {
    'safe': (0, 255, 0),      # Green
    'danger': (0, 0, 255),    # Red
    'neutral': (255, 0, 0)    # Blue
}

def get_latest_model_path():
    """Automatically finds the most recent model"""
    # PRIORITY 1: Check the deep path you found
    deep_path = Path("datasets/safety/results_yolov8n_100e/kaggle/working/runs/detect/train/weights/best.pt")
    if deep_path.exists():
        return str(deep_path)
        
    # PRIORITY 2: Check standard runs/detect
    base_dir = Path("runs/detect")
    if base_dir.exists():
        try:
            run_folders = sorted(list(base_dir.glob("safety_model*")), key=os.path.getmtime, reverse=True)
            for folder in run_folders:
                weights_path = folder / "weights" / "best.pt"
                if weights_path.exists():
                    return str(weights_path)
        except Exception:
            pass
    return None

def load_model(model_name, custom_path):
    if model_name == 'Custom Trained Model':
        if custom_path and os.path.exists(custom_path):
            return YOLO(custom_path), True
        else:
            st.sidebar.error(f"Custom model not found at {custom_path}")
            return YOLO('yolov8n.pt'), False
    
    model_map = {
        'YOLO26 Nano': 'yolov8n.pt',
        'YOLO26 Large': 'yolov8l.pt'
    }
    return YOLO(model_map.get(model_name, 'yolov8n.pt')), False

def format_reasons(reasons_list):
    if not reasons_list:
        return ""
    counts = Counter(reasons_list)
    return ", ".join([f"{k} ({v})" for k, v in counts.items()])

def draw_safety_box(image, result, conf_threshold, is_custom):
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    violation_reasons = []
    safe_reasons = []
    
    boxes = result.boxes
    names = result.names
    
    if boxes is not None:
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
                
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            
            # --- Logic Split ---
            if not is_custom:
                if cls_id != 0: continue
                status = 'neutral'
                cls_name = "Person"
            else:
                status = CLASSES_CONFIG.get(cls_name, 'neutral')
            
            # Track Reasons
            if status == 'danger':
                violation_reasons.append(cls_name)
            elif status == 'safe':
                safe_reasons.append(cls_name)
                
            color = COLOR_MAP.get(status, (255, 0, 0))
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            
            label = f"{cls_name}: {conf:.0%}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_cv, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(img_cv, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), violation_reasons, safe_reasons

def main():
    st.title("Site Safety Compliance AI 🏗️")
    
    # Sidebar
    st.sidebar.header("Model Settings")
    latest_custom_model = get_latest_model_path()
    model_options = ['YOLO26 Nano', 'YOLO26 Large']
    default_ix = 0
    
    if latest_custom_model:
        model_options.insert(0, 'Custom Trained Model')
        st.sidebar.success(f"Model Loaded!")
        default_ix = 0
    else:
        st.sidebar.warning("No custom model found.")
        default_ix = 0

    selected_model = st.sidebar.selectbox("Select Model", model_options, index=default_ix)
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4)
    
    # Load Model
    model, is_custom = load_model(selected_model, latest_custom_model)
    
    if not is_custom:
        st.warning("⚠️ Using Standard Model. Only 'Person' detection available.")

    # Input
    input_type = st.radio("Select Input", ["Image", "Video", "Webcam"])
    
    if input_type == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button("Analyze Safety"):
                results = model(image)
                annotated_img, v_reasons, s_reasons = draw_safety_box(image, results[0], confidence, is_custom)
                st.image(annotated_img, caption="Result", use_column_width=True)
                
                col1, col2 = st.columns(2)
                col1.metric("Violations 🛑", len(v_reasons))
                if v_reasons: col1.error(f"Reasons: {format_reasons(v_reasons)}")
                col2.metric("Safe Workers 🟢", len(s_reasons))
                if s_reasons: col2.success(f"Gear: {format_reasons(s_reasons)}")

    elif input_type == "Video":
        uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            vf = cv2.VideoCapture(tfile.name)
            
            # --- Video Controls ---
            col_ctrl1, col_ctrl2 = st.columns([1, 2])
            with col_ctrl1:
                run_video = st.checkbox("▶️ Play Video", value=True)
            with col_ctrl2:
                speed = st.select_slider("Playback Speed", 
                                         options=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], 
                                         value=1.0)

            stframe = st.empty()
            metric_ph = st.empty()
            
            while vf.isOpened() and run_video:
                # Speed Logic: Skip frames if speed > 1
                if speed > 1:
                    for _ in range(int(speed) - 1):
                        vf.read() # Skip frames
                
                ret, frame = vf.read()
                if not ret: break
                
                # Inference
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = model(frame_pil)
                annotated_frame, v_reasons, s_reasons = draw_safety_box(frame_pil, results[0], confidence, is_custom)
                stframe.image(annotated_frame)
                
                # Metrics
                with metric_ph.container():
                     col1, col2 = st.columns(2)
                     col1.metric("Violations", len(v_reasons))
                     if v_reasons: col1.error(format_reasons(v_reasons))
                     col2.metric("Safe", len(s_reasons))
                     if s_reasons: col2.success(format_reasons(s_reasons))
                
                # Speed Logic: Sleep if speed < 1
                if speed < 1.0:
                    time.sleep(0.1 * (1/speed)) # Add delay for slow mo

            vf.release()

    elif input_type == "Webcam":
        run_cam = st.checkbox("Start Webcam")
        if run_cam:
            cam = cv2.VideoCapture(0)
            stframe = st.empty()
            metric_ph = st.empty()
            while run_cam:
                ret, frame = cam.read()
                if not ret: break
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = model(frame_pil)
                annotated_frame, v_reasons, s_reasons = draw_safety_box(frame_pil, results[0], confidence, is_custom)
                stframe.image(annotated_frame)
                
                with metric_ph.container():
                     col1, col2 = st.columns(2)
                     col1.metric("Violations", len(v_reasons))
                     col2.metric("Safe", len(s_reasons))
            cam.release()

if __name__ == "__main__":
    main()