import os
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

def find_split_folder(root_dir, split_names):
    """Find split folder containing both images/ and labels/ directories."""
    if not root_dir.exists():
        return None

    candidates = []
    for split_name in split_names:
        for path in root_dir.rglob(split_name):
            if path.is_dir() and (path / "images").exists() and (path / "labels").exists():
                candidates.append(path)

    if not candidates:
        return None

    candidates.sort(key=lambda item: (len(item.parts), str(item)))
    return candidates[0]

def setup_data():
    # 1. Download if train split doesn't exist
    train_dir = find_split_folder(DATASET_DIR, ("train",))
    malformed_train_dir = find_folder(DATASET_DIR, "train")
    if malformed_train_dir and not train_dir:
        print(f"Error: train split must contain both images/ and labels/: {malformed_train_dir}")
        return
    
    if not train_dir:
        print(f"Dataset not found. Downloading {DATASET_NAME}...")
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(DATASET_NAME, path=DATASET_DIR, unzip=True)
            print("Download and extraction complete.")
            train_dir = find_split_folder(DATASET_DIR, ("train",))
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure KAGGLE_API_TOKEN is set or kaggle.json is present.")
            return

    if not train_dir:
        print("Error: train split with images/labels not found after download.")
        return

    # 2. Determine paths relative to DATASET_DIR
    try:
        relative_root = train_dir.parent.relative_to(DATASET_DIR)
    except ValueError:
        relative_root = Path(".")
    
    if not (train_dir / "images").exists() or not (train_dir / "labels").exists():
        print(f"Error: train split is incomplete in {train_dir}")
        return

    train_rel = relative_root / "train" / "images"
    val_dir = find_split_folder(DATASET_DIR, ("valid", "val"))
    test_dir = find_split_folder(DATASET_DIR, ("test",))

    if not val_dir:
        print("Error: validation split with images/labels not found.")
        return

    if not test_dir:
        print("Error: test split with images/labels not found.")
        return

    try:
        val_rel = val_dir.relative_to(DATASET_DIR) / "images"
        test_rel = test_dir.relative_to(DATASET_DIR) / "images"
    except ValueError:
        print("Error: split directories must be inside DATASET_DIR.")
        return

    # Yaml content
    yaml_data = {
        'path': '.',
        'train': str(train_rel).replace(os.sep, '/'),
        'val': str(val_rel).replace(os.sep, '/'),
        'test': str(test_rel).replace(os.sep, '/'),
        'nc': len(CLASSES),
        'names': CLASSES
    }
    
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    print(f"Created/Updated data.yaml at {yaml_path}")
    print(f"Train path: {yaml_data['train']}")
    print("Data setup complete.")

if __name__ == "__main__":
    setup_data()
