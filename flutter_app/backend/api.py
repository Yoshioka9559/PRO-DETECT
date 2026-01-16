"""
FastAPI backend for ProDetect Flutter app
Handles YOLO model inference
"""
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI(title="ProDetect API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = YOLO('../../runs/detect/poc_final_training/weights/best.pt')

@app.get("/")
def root():
    return {"message": "ProDetect API is running"}

@app.post("/detect")
async def detect_products(
    file: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    """Detect products in uploaded image"""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run detection
        results = model(image, conf=confidence)
        result = results[0]
        
        # Count detections
        total_detections = len(result.boxes)
        class_counts = {}
        detection_details = []
        
        if len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])
                
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
                
                detection_details.append({
                    "class": class_name,
                    "confidence": conf
                })
        
        # Get annotated image
        annotated_image = result.plot()
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "total_detections": total_detections,
            "class_counts": class_counts,
            "detection_details": detection_details,
            "annotated_image": img_base64
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/webcam/status")
def webcam_status():
    """Check if webcam is available"""
    cap = cv2.VideoCapture(0)
    available = cap.isOpened()
    cap.release()
    return {"available": available}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
