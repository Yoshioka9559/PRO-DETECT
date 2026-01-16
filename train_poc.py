"""
Train YOLO model with pretrained weights on properly split POC dataset
"""
from ultralytics import YOLO

# Load pretrained model (much better starting point)
model = YOLO('yolov8n.pt')  # Use pretrained weights on COCO dataset

# Train the model on POC split dataset (260 train, 112 val)
results = model.train(
    data='POC_split/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='poc_final_training',
    project='runs/detect',
    patience=20,
    save=True,
    plots=True,
    device='cpu',  # Change to 0 for GPU if available
    workers=4,
    verbose=True
)

# Evaluate the model
metrics = model.val()

# Save the final model
model.save('runs/detect/poc_training/weights/best.pt')

print("\n✅ Training complete!")
print(f"📊 Results saved to: runs/detect/poc_training")
print(f"🎯 Best model saved to: runs/detect/poc_training/weights/best.pt")
