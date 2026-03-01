import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image, UnidentifiedImageError
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
    """Automatically finds the most recent available model."""
    candidates = []
    deep_path = Path("datasets/safety/results_yolov8n_100e/kaggle/working/runs/detect/train/weights/best.pt")
    if deep_path.exists():
        candidates.append(deep_path)

    base_dir = Path("runs/detect")
    if base_dir.exists():
        try:
            for folder in base_dir.glob("safety_model*"):
                weights_path = folder / "weights" / "best.pt"
                if weights_path.exists():
                    candidates.append(weights_path)
        except OSError as exc:
            st.sidebar.warning(f"Model discovery warning: {exc}")

    if candidates:
        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        return str(latest)
    return None

def load_model(model_name, custom_path):
    fallback_model = 'yolov8n.pt'

    if model_name == 'Custom Trained Model':
        if custom_path and os.path.exists(custom_path):
            try:
                return YOLO(custom_path), True
            except Exception as exc:
                st.sidebar.error(f"Custom model failed to load: {exc}")
        else:
            st.sidebar.error(f"Custom model not found at {custom_path}")
        try:
            return YOLO(fallback_model), False
        except Exception as exc:
            st.sidebar.error(f"Fallback model failed to load: {exc}")
            return None, False
    
    model_map = {
        'YOLO26 Nano': 'yolo26n.pt',
        'YOLO26 Large': 'yolo26l.pt'
    }
    selected_path = model_map.get(model_name, fallback_model)
    try:
        return YOLO(selected_path), False
    except Exception as exc:
        st.sidebar.error(f"Model load failed for {selected_path}: {exc}")
        try:
            return YOLO(fallback_model), False
        except Exception as fallback_exc:
            st.sidebar.error(f"Fallback model failed to load: {fallback_exc}")
            return None, False

def format_reasons(reasons_list):
    if not reasons_list:
        return ""
    counts = Counter(reasons_list)
    return ", ".join([f"{k} ({v})" for k, v in counts.items()])

def draw_safety_box(image, result, conf_threshold, is_custom):
    if image is None:
        raise ValueError("Input image cannot be None")
    if result is None:
        raise ValueError("Inference result cannot be None")

    if image.mode != 'RGB':
        image = image.convert('RGB')
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
            cls_name = names.get(cls_id, str(cls_id))
            
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
    
    if latest_custom_model:
        model_options.insert(0, 'Custom Trained Model')
        st.sidebar.success(f"Model Loaded!")
    else:
        st.sidebar.warning("No custom model found.")

    selected_model = st.sidebar.selectbox("Select Model", model_options, index=0)
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4)
    
    # Load Model
    model, is_custom = load_model(selected_model, latest_custom_model)
    if model is None:
        st.error("No usable model could be loaded. Check model files and retry.")
        return
    
    if not is_custom:
        st.warning("⚠️ Using Standard Model. Only 'Person' detection available.")

    # Input
    input_type = st.radio("Select Input", ["Image", "Video", "Webcam"])
    
    if input_type == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
            except (UnidentifiedImageError, OSError):
                st.error("Uploaded file is not a valid image.")
                return

            st.image(image, caption="Uploaded Image", use_container_width=True)
            if st.button("Analyze Safety"):
                try:
                    results = model(image)
                    annotated_img, v_reasons, s_reasons = draw_safety_box(image, results[0], confidence, is_custom)
                except Exception as exc:
                    st.error(f"Inference failed: {exc}")
                    return

                st.image(annotated_img, caption="Result", use_container_width=True)
                
                col1, col2 = st.columns(2)
                col1.metric("Violations 🛑", len(v_reasons))
                if v_reasons: col1.error(f"Reasons: {format_reasons(v_reasons)}")
                col2.metric("Safe Workers 🟢", len(s_reasons))
                if s_reasons: col2.success(f"Gear: {format_reasons(s_reasons)}")

    elif input_type == "Video":
        uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name

            vf = cv2.VideoCapture(video_path)
            if not vf.isOpened():
                st.error("Unable to open uploaded video file.")
                Path(video_path).unlink(missing_ok=True)
                return
            
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
            frame_skip_accumulator = 0.0
            
            try:
                while vf.isOpened() and run_video:
                    if speed > 1.0:
                        frame_skip_accumulator += speed - 1.0
                        while frame_skip_accumulator >= 1.0:
                            skipped_ret, _ = vf.read()
                            if not skipped_ret:
                                break
                            frame_skip_accumulator -= 1.0

                    ret, frame = vf.read()
                    if not ret:
                        break

                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    results = model(frame_pil)
                    annotated_frame, v_reasons, s_reasons = draw_safety_box(frame_pil, results[0], confidence, is_custom)
                    stframe.image(annotated_frame)

                    with metric_ph.container():
                         col1, col2 = st.columns(2)
                         col1.metric("Violations", len(v_reasons))
                         if v_reasons: col1.error(format_reasons(v_reasons))
                         col2.metric("Safe", len(s_reasons))
                         if s_reasons: col2.success(format_reasons(s_reasons))

                    if speed < 1.0:
                        time.sleep(0.1 * (1 / speed))
            finally:
                vf.release()
                Path(video_path).unlink(missing_ok=True)

    elif input_type == "Webcam":
        run_cam = st.checkbox("Start Webcam")
        if run_cam:
            cam = cv2.VideoCapture(0)
            stframe = st.empty()
            metric_ph = st.empty()
            try:
                while cam.isOpened():
                    ret, frame = cam.read()
                    if not ret:
                        break
                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    results = model(frame_pil)
                    annotated_frame, v_reasons, s_reasons = draw_safety_box(frame_pil, results[0], confidence, is_custom)
                    stframe.image(annotated_frame)

                    with metric_ph.container():
                         col1, col2 = st.columns(2)
                         col1.metric("Violations", len(v_reasons))
                         col2.metric("Safe", len(s_reasons))
            finally:
                cam.release()

if __name__ == "__main__":
    main()