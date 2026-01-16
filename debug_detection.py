from ultralytics import YOLO
import cv2

# Load model
model = YOLO('runs/detect/betty_haagen3/weights/best.pt')

# Test on single frame
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, frame = cap.read()
cap.release()

if ret:
    # Save test image
    cv2.imwrite('test_frame.jpg', frame)
    
    # Run with very low confidence to see what model detects
    results = model(frame, conf=0.1)  # Very low threshold
    
    print(f"\nDetections found: {len(results[0].boxes)}")
    
    if len(results[0].boxes) > 0:
        for i, box in enumerate(results[0].boxes):
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"  {i+1}. {class_name}: {conf:.2f} confidence")
    else:
        print("  No detections (even at 10% confidence)")
        print("\nPossible issues:")
        print("  - Phone screen vs physical product (different appearance)")
        print("  - Poor lighting or glare")
        print("  - Object too small/large in frame")
        print("  - Different angle than training images")
    
    # Show result
    annotated = results[0].plot()
    cv2.imwrite('test_result.jpg', annotated)
    print(f"\nTest images saved: test_frame.jpg, test_result.jpg")
