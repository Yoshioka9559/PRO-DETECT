from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('runs/detect/betty_haagen3/weights/best.pt')

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Webcam opened! Press 'q' to quit.")
print("Model trained to detect:")
print("  - Class 0: Betty Crocker Cake Mix")
print("  - Class 1: Haagen-Dazs Ice Cream")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference (lowered confidence to 0.25 for better detection)
    results = model(frame, conf=0.25)
    
    # Get annotated frame
    annotated_frame = results[0].plot()
    
    # Count detections
    total_objects = len(results[0].boxes)
    class_counts = {}
    
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Add total count
    cv2.putText(annotated_frame, f'Total: {total_objects}', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add class counts
    y_offset = 70
    for class_name, count in class_counts.items():
        cv2.putText(annotated_frame, f'{class_name}: {count}', 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 30
    
    cv2.imshow('Betty & Haagen-Dazs Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nTest complete!")
