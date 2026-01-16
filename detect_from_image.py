from ultralytics import YOLO
import cv2
import sys
import os

def detect_in_image(image_path, model_path='runs/detect/betty_haagen3/weights/best.pt', conf=0.25):
    """
    Detect products in a given image file
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained YOLO model
        conf: Confidence threshold (default 0.25)
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Read the image
    print(f"Reading image from {image_path}...")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image from '{image_path}'")
        return
    
    # Run detection
    print(f"Running detection with confidence threshold: {conf}")
    results = model(image, conf=conf)
    
    # Get the result
    result = results[0]
    
    # Count detections by class
    class_counts = {}
    total_detections = 0
    
    if len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            
            # Count by class
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
            total_detections += 1
            
            # Print detection info
            print(f"  - {class_name}: {confidence:.2%} confidence")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Detection Summary:")
    print(f"{'='*50}")
    print(f"Total detections: {total_detections}")
    
    if class_counts:
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
    else:
        print("  No products detected!")
        print("\nPossible reasons:")
        print("  - Product not in frame or too small")
        print("  - Poor lighting or image quality")
        print("  - Object angle differs from training images")
        print("  - Try lowering confidence threshold")
    
    # Draw results on image
    annotated_image = result.plot()
    
    # Save the result
    output_path = image_path.rsplit('.', 1)[0] + '_detected.' + image_path.rsplit('.', 1)[1]
    cv2.imwrite(output_path, annotated_image)
    print(f"\nAnnotated image saved to: {output_path}")
    
    # Display the image
    cv2.imshow('Detection Result', annotated_image)
    print("\nPress any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_from_image.py <image_path> [confidence_threshold]")
        print("\nExample:")
        print("  python detect_from_image.py my_image.jpg")
        print("  python detect_from_image.py my_image.jpg 0.3")
        print("\nSupported formats: .jpg, .jpeg, .png, .bmp, etc.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Optional confidence threshold
    conf = 0.25
    if len(sys.argv) >= 3:
        try:
            conf = float(sys.argv[2])
            if not 0.0 <= conf <= 1.0:
                print("Warning: Confidence should be between 0.0 and 1.0. Using default 0.25")
                conf = 0.25
        except ValueError:
            print("Warning: Invalid confidence value. Using default 0.25")
    
    detect_in_image(image_path, conf=conf)
