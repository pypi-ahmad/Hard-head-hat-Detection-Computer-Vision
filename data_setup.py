import os
import zipfile
from pathlib import Path
import yaml
import kaggle

# Configuration
DATASET_NAME = "snehilsanyal/construction-site-safety-image-dataset-roboflow"
DATASET_DIR = Path("datasets/safety")

CLASSES = [
    'Hardhat', 
    'Mask', 
    'NO-Hardhat', 
    'NO-Mask', 
    'NO-Safety Vest', 
    'Person', 
    'Safety Cone', 
    'Safety Vest', 
    'machinery', 
    'vehicle'
]

def find_folder(root_dir, folder_name):
    """Recursively search for a folder."""
    if not root_dir.exists():
        return None
    for path in root_dir.rglob(folder_name):
        if path.is_dir():
            return path
    return None

def setup_data():
    # 1. Download if 'train' folder doesn't exist anywhere
    train_dir = find_folder(DATASET_DIR, "train")
    
    if not train_dir:
        print(f"Dataset not found. Downloading {DATASET_NAME}...")
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(DATASET_NAME, path=DATASET_DIR, unzip=True)
            print("Download and extraction complete.")
            # Search again
            train_dir = find_folder(DATASET_DIR, "train")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure KAGGLE_API_TOKEN is set or kaggle.json is present.")
            return

    if not train_dir:
        print("Error: 'train' folder not found after download.")
        return

    # 2. Determine paths relative to DATASET_DIR
    try:
        relative_root = train_dir.parent.relative_to(DATASET_DIR)
    except ValueError:
        relative_root = Path(".")
    
    # Check if 'images' is inside 'train'
    if not (train_dir / "images").exists():
        print(f"Error: 'images' folder not found in {train_dir}")
        return

    train_rel = relative_root / "train" / "images"
    val_rel = relative_root / "valid" / "images"
    test_rel = relative_root / "test" / "images"

    # Verify valid exists
    if not (DATASET_DIR / val_rel).exists():
        val_dir = find_folder(DATASET_DIR, "valid") or find_folder(DATASET_DIR, "val")
        if val_dir:
             try:
                val_rel = val_dir.relative_to(DATASET_DIR) / "images"
             except ValueError:
                 print(f"Warning: Validation dir {val_dir} is not relative to {DATASET_DIR}")
        else:
             print("Warning: Validation set not found. Using train for validation.")
             val_rel = train_rel

    # Verify test exists
    if not (DATASET_DIR / test_rel).exists():
        test_dir = find_folder(DATASET_DIR, "test")
        if test_dir:
             test_rel = test_dir.relative_to(DATASET_DIR) / "images"
        else:
             test_rel = val_rel 

    # Yaml content
    yaml_data = {
        'path': str(DATASET_DIR.absolute()),
        'train': str(train_rel).replace(os.sep, '/'),
        'val': str(val_rel).replace(os.sep, '/'),
        'test': str(test_rel).replace(os.sep, '/'),
        'nc': len(CLASSES),
        'names': CLASSES
    }
    
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    print(f"Created/Updated data.yaml at {yaml_path}")
    print(f"Train path: {yaml_data['train']}")
    print("Data setup complete.")

if __name__ == "__main__":
    setup_data()
