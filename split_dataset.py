"""
Split augmented dataset into train and validation sets
70% training, 30% validation
"""
import os
import shutil
from pathlib import Path
import random

# Set seed for reproducibility
random.seed(42)

# Paths
source_images = Path("POC_augmented/images")
source_labels = Path("POC_augmented/labels")

train_images = Path("POC_split/train/images")
train_labels = Path("POC_split/train/labels")
val_images = Path("POC_split/val/images")
val_labels = Path("POC_split/val/labels")

# Create directories
for path in [train_images, train_labels, val_images, val_labels]:
    path.mkdir(parents=True, exist_ok=True)

# Get all image files
all_images = list(source_images.glob("*.jpg"))
print(f"Total images: {len(all_images)}")

# Shuffle
random.shuffle(all_images)

# Split 70-30
split_idx = int(len(all_images) * 0.7)
train_imgs = all_images[:split_idx]
val_imgs = all_images[split_idx:]

print(f"Training images: {len(train_imgs)}")
print(f"Validation images: {len(val_imgs)}")

# Copy training files
print("\nCopying training files...")
for img_file in train_imgs:
    # Copy image
    shutil.copy(img_file, train_images / img_file.name)
    
    # Copy label
    label_file = source_labels / f"{img_file.stem}.txt"
    if label_file.exists():
        shutil.copy(label_file, train_labels / label_file.name)

# Copy validation files
print("Copying validation files...")
for img_file in val_imgs:
    # Copy image
    shutil.copy(img_file, val_images / img_file.name)
    
    # Copy label
    label_file = source_labels / f"{img_file.stem}.txt"
    if label_file.exists():
        shutil.copy(label_file, val_labels / label_file.name)

print("\n✅ Dataset split complete!")
print(f"📁 Train: POC_split/train/ ({len(train_imgs)} images)")
print(f"📁 Val: POC_split/val/ ({len(val_imgs)} images)")
