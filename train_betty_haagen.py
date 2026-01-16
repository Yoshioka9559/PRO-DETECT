from ultralytics import YOLO

# Load pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Train on Betty Crocker and Haagen-Dazs dataset (corrected)
results = model.train(
    data='dataset_corrected/data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    name='betty_haagen',
    project='runs/detect'
)

print("\n✓ Training complete!")
print(f"✓ Results saved to: runs/detect/betty_haagen")
print(f"✓ Best weights: runs/detect/betty_haagen/weights/best.pt")
