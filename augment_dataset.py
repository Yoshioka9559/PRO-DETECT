"""
Data Augmentation for POC Dataset
Expands the dataset by creating augmented copies of existing images
Uses only OpenCV - no additional libraries needed
"""
import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import random

def augment_image_and_boxes(image, bboxes, class_labels):
    """Apply random augmentations to image and adjust bounding boxes"""
    h, w = image.shape[:2]
    aug_image = image.copy()
    aug_bboxes = bboxes.copy()
    
    # Horizontal flip (50% chance)
    if random.random() > 0.5:
        aug_image = cv2.flip(aug_image, 1)
        for i, bbox in enumerate(aug_bboxes):
            aug_bboxes[i][0] = 1.0 - bbox[0]  # Flip x_center
    
    # Brightness & Contrast
    alpha = random.uniform(0.7, 1.3)  # Contrast
    beta = random.randint(-30, 30)  # Brightness
    aug_image = cv2.convertScaleAbs(aug_image, alpha=alpha, beta=beta)
    
    # Add Gaussian noise (30% chance)
    if random.random() > 0.7:
        noise = np.random.normal(0, 15, aug_image.shape).astype(np.uint8)
        aug_image = cv2.add(aug_image, noise)
    
    # Gaussian blur (30% chance)
    if random.random() > 0.7:
        aug_image = cv2.GaussianBlur(aug_image, (5, 5), 0)
    
    # Hue/Saturation adjustment
    hsv = cv2.cvtColor(aug_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.8, 1.2), 0, 255)
    aug_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return aug_image, aug_bboxes, class_labels

# Paths
source_images = Path("POC/images")
source_labels = Path("POC/labels")
output_images = Path("POC_augmented/images")
output_labels = Path("POC_augmented/labels")

# Create output directories
output_images.mkdir(parents=True, exist_ok=True)
output_labels.mkdir(parents=True, exist_ok=True)

print("Starting data augmentation...")
print(f"Source images: {len(list(source_images.glob('*.jpg')))}")

# Number of augmentations per image
num_augmentations = 3

total_created = 0

for img_file in source_images.glob("*.jpg"):
    # Read image
    image = cv2.imread(str(img_file))
    
    # Read corresponding label
    label_file = source_labels / f"{img_file.stem}.txt"
    
    # Copy original first
    shutil.copy(img_file, output_images / img_file.name)
    if label_file.exists():
        shutil.copy(label_file, output_labels / label_file.name)
    
    # Parse bounding boxes
    bboxes = []
    class_labels = []
    
    if label_file.exists():
        with open(label_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
    
    # Create augmentations
    for aug_idx in range(num_augmentations):
        try:
            # Apply augmentation
            if bboxes:
                aug_image, aug_bboxes, aug_labels = augment_image_and_boxes(
                    image, np.array(bboxes), class_labels
                )
            else:
                aug_image = image.copy()
                # Apply same transformations without boxes
                if random.random() > 0.5:
                    aug_image = cv2.flip(aug_image, 1)
                alpha = random.uniform(0.7, 1.3)
                beta = random.randint(-30, 30)
                aug_image = cv2.convertScaleAbs(aug_image, alpha=alpha, beta=beta)
                aug_bboxes = []
                aug_labels = []
            
            # Save augmented image
            aug_img_name = f"{img_file.stem}_aug{aug_idx}.jpg"
            aug_img_path = output_images / aug_img_name
            cv2.imwrite(str(aug_img_path), aug_image)
            
            # Save augmented labels
            aug_label_path = output_labels / f"{img_file.stem}_aug{aug_idx}.txt"
            with open(aug_label_path, 'w') as f:
                for bbox, label in zip(aug_bboxes, aug_labels):
                    f.write(f"{label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            
            total_created += 1
            
        except Exception as e:
            print(f"Error augmenting {img_file.name}: {e}")
            continue
    
    if total_created % 50 == 0:
        print(f"Created {total_created} augmented images...")

print(f"\n✅ Augmentation complete!")
print(f"📊 Original images: {len(list(source_images.glob('*.jpg')))}")
print(f"📊 Total images (original + augmented): {len(list(output_images.glob('*.jpg')))}")
print(f"📁 Output: POC_augmented/")
