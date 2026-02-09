# Safety Vision Pro 🦺

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![YOLO](https://img.shields.io/badge/YOLO-v8-green?style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia)

## 📖 About
**Safety Vision Pro** is an advanced **Health, Safety, and Environment (HSE)** compliance tool designed to detect Personal Protective Equipment (PPE) violations in real-time. Leveraging Computer Vision and the YOLOv8 architecture, it instantly identifies workers without hardhats, safety vests, or masks, providing actionable insights to maintain a zero-accident workplace.

---

## 🌟 Key Features

### 🎥 Advanced Player
Complete control over video playback with a custom-built interface:
- **Speed Control**: Adjust playback from **0.25x (Slow Mo)** to **2x (Fast Forward)**.
- **Play/Pause**: Toggle playback instantly for frame-by-frame analysis.

### 🏗️ Large File Support
Optimized for high-resolution site footage. The system is configured to handle **1GB+ video uploads** seamlessly via Streamlit's custom server configuration.

### 🧠 Auto-Training Pipeline
A unified workflow for model improvements:
- **One-Click Setup**: `data_setup.py` automatically authenticates with Kaggle, downloads the dataset, and configures `data.yaml`.
- **Optimization**: `train.py` is tuned for consumer hardware (8GB VRAM), utilizing **batch size 8** and the robust **YOLO Large** model.

### 🚀 Smart Inference
The application features intelligent model detection:
- **Auto-Switching**: Automatically detects if a custom trained model exists in `runs/detect` or specific results folders.
- **Fallback**: Seamlessly reverts to the standard YOLOv8 Nano model if no custom weights are found.

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/pypi-ahmad/Hard-head-hat-Detection-Computer-Vision.git
cd Hard-head-hat-Detection-Computer-Vision
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🛠️ Usage Guide

### Step 1: Data Setup
Initialize the project by downloading the dataset and generating the configuration files.
*Note: Ensure you have your `kaggle.json` API token ready or environment variables set.*
```bash
python data_setup.py
```

### Step 2: Train the Model
Launch the training process. The script is pre-configured for **50 epochs** (recommended for production) and a **batch size of 8** to maximize stability on 8GB GPUs.
```bash
python train.py
```

### Step 3: Run the Application
Start the Safety Vision Pro dashboard.
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
Hard-head-hat-Detection-Computer-Vision/
├── app.py                  # Main Streamlit dashboard application
├── data_setup.py           # Dataset downloader and config generator
├── train.py                # YOLO training script (Batch=8, Large Model)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Server config (maxUploadSize = 1000)
├── datasets/               # Dataset storage (auto-created)
│   └── safety/
│       ├── data.yaml
│       ├── train/
│       └── valid/
└── runs/                   # Training artifacts and weights
    └── detect/
```

---

## 🤖 Model Information

The system classifies objects into three safety categories with distinct visual indicators:

| Category | Color | Classes Included |
| :--- | :--- | :--- |
| **✅ Safe** | **Green** | `Hardhat`, `Mask`, `Safety Vest` |
| **🛑 Danger** | **Red** | `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest` |
| **🔵 Neutral** | **Blue** | `Person`, `Safety Cone`, `Machinery`, `Vehicle` |

---

*Built with ❤️ for Safer Workplaces*
