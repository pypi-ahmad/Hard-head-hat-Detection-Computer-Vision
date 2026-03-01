from ultralytics import YOLO
from pathlib import Path
import torch

def train_safety_model():
    # Configuration
    model_name = 'yolo26l.pt' # Large Model
    
    data_path = Path("datasets/safety/data.yaml")
    project_dir = "runs/detect"
    run_name = "safety_model"
    
    print(f"Starting training with {model_name} on {data_path}...")

    if not data_path.exists():
        print(f"Error: dataset config not found at {data_path}")
        return {"ok": False, "error": "missing_data_yaml"}
    
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
        return {"ok": False, "error": "model_load_failed"}

    # Train
    try:
        model.train(
            data=str(data_path),
            epochs=50,
            imgsz=640,
            batch=8,
            device=device,
            workers=2,
            project=project_dir,
            name=run_name,
            save=True,
            plots=True
        )
    except Exception as e:
        print(f"Training failed: {e}")
        return {"ok": False, "error": "train_failed"}
    
    # Output best model path
    best_model_path = Path(project_dir) / run_name / "weights" / "best.pt"
    print(f"\nTraining Complete. Best model: {best_model_path}")
    return {"ok": True, "best_model": str(best_model_path), "device": device}

if __name__ == "__main__":
    train_safety_model()