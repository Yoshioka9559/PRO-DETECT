# ProDetect Flutter App

A cross-platform product detection application using Flutter and YOLO.

## Setup Instructions

### Backend Setup (Python)

1. Install required packages:
```bash
cd backend
pip install fastapi uvicorn python-multipart ultralytics opencv-python pillow
```

2. Run the backend server:
```bash
python api.py
```
The API will be available at `http://localhost:8000`

### Flutter App Setup

1. Install Flutter dependencies:
```bash
flutter pub get
```

2. Run the app:
```bash
# For Windows
flutter run -d windows

# For Android
flutter run -d android

# For Web
flutter run -d chrome
```

## Features

- 📁 **Upload Image**: Select images from gallery for detection
- 📹 **Webcam**: Real-time detection from webcam (coming soon)
- 🔍 **Detect Products**: Run YOLO detection on images
- 🎚️ **Confidence Slider**: Adjust detection confidence threshold
- 📊 **Live Results**: View detection counts and details
- 🗑️ **Clear**: Reset the interface

## Color Theme

- Primary Blue: #0A4A8E
- Gold Accent: #FFD700
- Dark Blue: #084080
- White: #FFFFFF

## API Endpoints

- `GET /`: Health check
- `POST /detect`: Detect products in uploaded image
- `GET /webcam/status`: Check webcam availability
