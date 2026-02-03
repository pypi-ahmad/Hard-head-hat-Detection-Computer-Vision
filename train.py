from ultralytics import YOLO
import sys
from pathlib import Path
import torch

def train_safety_model():
    # Configuration
    model_name = 'yolo26l.pt' # Large Model
    
    data_path = Path("datasets/safety/data.yaml")
    project_dir = "runs/detect"
    run_name = "safety_model"
    
    print(f"Starting training with {model_name} on {data_path}...")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("⚠️ GPU Not Detected! Training will be slow.")
        device = 'cpu'

    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return

    # Train
    results = model.train(
        data=str(data_path),
        epochs=50,
        imgsz=640,
        batch=8,           # <--- CRITICAL: Keeps memory usage low for 8GB GPU
        device=device,     # <--- Forces GPU usage
        workers=2,         # <--- Reduces CPU overhead
        project=project_dir,
        name=run_name,
        save=True,
        plots=True
    )
    
    # Output best model path
    print(f"\nTraining Complete. Check runs/detect/ for the latest safety_model folder.")

if __name__ == "__main__":
    train_safety_model()