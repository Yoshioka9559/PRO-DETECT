# PRO-DETECT: Product Detection System

A comprehensive product detection system using YOLOv8 for training custom object detection models and a Flutter cross-platform app for real-time detection.

## 🚀 Project Overview

This project combines Python-based YOLO training/detection with a Flutter mobile/desktop application for detecting products (Betty Crocker, Haagen-Dazs, etc.) in images and video streams.

### Components

1. **Python YOLO Training & Detection** - Train custom YOLOv8 models and run detection
2. **Flutter App** - Cross-platform application (Windows, Android, iOS, Web) with detection API
3. **Backend API** - FastAPI server for model inference

## 📁 Project Structure

```
├── train_betty_haagen.py       # Train YOLO model on product dataset
├── train_poc.py                # Train proof-of-concept model
├── detect_gui.py               # GUI application for detection
├── detect_from_image.py        # CLI detection script
├── test_betty_haagen_webcam.py # Webcam detection testing
├── augment_dataset.py          # Dataset augmentation utilities
├── split_dataset.py            # Split dataset into train/val
├── split_dataset_stratified.py # Stratified dataset splitting
├── yolov8n.pt                  # Pretrained YOLO weights
├── POC/                        # Proof of concept dataset
├── flutter_app/                # Flutter mobile/desktop app
│   ├── lib/                    # Flutter source code
│   ├── backend/                # FastAPI backend
│   └── assets/                 # App assets
└── runs/                       # Training results and weights
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8+
- Flutter SDK 3.0+
- CUDA (optional, for GPU acceleration)

### Python Environment Setup

1. **Create a virtual environment:**
```bash
python -m venv yoloenv
```

2. **Activate the environment:**
```bash
# Windows
yoloenv\Scripts\activate

# Linux/Mac
source yoloenv/bin/activate
```

3. **Install dependencies:**
```bash
pip install ultralytics opencv-python pillow numpy
pip install fastapi uvicorn python-multipart  # For backend API
```

### Flutter App Setup

1. **Navigate to Flutter directory:**
```bash
cd flutter_app
```

2. **Install dependencies:**
```bash
flutter pub get
```

3. **Run the app:**
```bash
# Windows
flutter run -d windows

# Android
flutter run -d android

# Web
flutter run -d chrome
```

## 🎯 Usage

### Training a Model

Train on your custom dataset:

```bash
python train_betty_haagen.py
```

**Model Configuration:**
- **Epochs:** 50
- **Image Size:** 640x640
- **Batch Size:** 8
- **Base Model:** YOLOv8n

Results will be saved in `runs/detect/betty_haagen/`

### Running Detection

#### GUI Application

```bash
python detect_gui.py
```

Features:
- Upload images for detection
- Adjust confidence threshold
- View detection results with bounding boxes
- Export annotated images

#### Command Line Detection

```bash
python detect_from_image.py
```

#### Webcam Detection

```bash
python test_betty_haagen_webcam.py
```

### Backend API

1. **Start the FastAPI server:**
```bash
cd flutter_app/backend
python api.py
```

2. **API Endpoints:**
- `POST /detect` - Upload image for detection
- API runs at `http://localhost:8000`

### Flutter App

1. **Ensure backend is running** (see above)
2. **Launch the app** with `flutter run`
3. **Features:**
   - 📁 Upload images from gallery
   - 📹 Real-time webcam detection (coming soon)
   - 🔍 Adjust confidence threshold
   - 📊 View detection counts

## 📊 Dataset Management

### Augment Dataset

```bash
python augment_dataset.py
```

Applies augmentations:
- Rotation
- Brightness adjustment
- Flipping
- Scaling

### Split Dataset

```bash
# Random split
python split_dataset.py

# Stratified split (maintains class distribution)
python split_dataset_stratified.py
```

## 🔧 Configuration

### Dataset YAML Format

Create a `data.yaml` file in your dataset folder:

```yaml
train: ./train/images
val: ./val/images

nc: 2  # Number of classes
names: ['Betty Crocker', 'Haagen-Dazs']
```

## 📈 Model Performance

After training, find results in `runs/detect/betty_haagen/`:
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Last epoch weights
- `results.png` - Training metrics
- `confusion_matrix.png` - Confusion matrix


## 📝 License

This project is POC so uses open source apache lisence for YOLO as dummy data images are used, created by team ASIA. In case of commercial use its commercial lisence needs to be purchased.



