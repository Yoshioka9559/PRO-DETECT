"""
Split augmented dataset into train and validation sets WITH STRATIFICATION
Ensures each class (0, 1, 2) is proportionally represented in both train and val
70% training, 30% validation
"""
import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

# Paths
source_images = Path("POC_augmented/images")
source_labels = Path("POC_augmented/labels")

train_images = Path("POC_split/train/images")
train_labels = Path("POC_split/train/labels")
val_images = Path("POC_split/val/images")
val_labels = Path("POC_split/val/labels")

# Clear existing directories
print("Clearing existing split directories...")
for path in [train_images, train_labels, val_images, val_labels]:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

# Get all image files
all_images = list(source_images.glob("*.jpg"))
print(f"Total images: {len(all_images)}")

# Group images by their primary class (first class in label file)
class_to_images = defaultdict(list)
for img_file in all_images:
    label_file = source_labels / f"{img_file.stem}.txt"
    if label_file.exists():
        with open(label_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                class_id = first_line.split()[0]
                class_to_images[class_id].append(img_file)

# Print class distribution
print("\n📊 Original class distribution:")
for class_id in sorted(class_to_images.keys()):
    print(f"  Class {class_id}: {len(class_to_images[class_id])} images")

# Stratified split: 70-30 for each class
train_imgs = []
val_imgs = []

for class_id, images in class_to_images.items():
    random.shuffle(images)
    split_idx = int(len(images) * 0.7)
    train_imgs.extend(images[:split_idx])
    val_imgs.extend(images[split_idx:])
    print(f"  Class {class_id}: Train={split_idx}, Val={len(images)-split_idx}")

print(f"\n✅ Stratified split complete!")
print(f"Training images: {len(train_imgs)}")
print(f"Validation images: {len(val_imgs)}")

# Copy training files
print("\nCopying training files...")
for img_file in train_imgs:
    shutil.copy(img_file, train_images / img_file.name)
    label_file = source_labels / f"{img_file.stem}.txt"
    if label_file.exists():
        shutil.copy(label_file, train_labels / label_file.name)

# Copy validation files
print("Copying validation files...")
for img_file in val_imgs:
    shutil.copy(img_file, val_images / img_file.name)
    label_file = source_labels / f"{img_file.stem}.txt"
    if label_file.exists():
        shutil.copy(label_file, val_labels / label_file.name)

# Verify class distribution in splits
print("\n📊 TRAIN set class distribution:")
train_class_counts = defaultdict(int)
for label_file in train_labels.glob("*.txt"):
    with open(label_file, 'r') as f:
        for line in f:
            class_id = line.strip().split()[0]
            train_class_counts[class_id] += 1

for class_id in sorted(train_class_counts.keys()):
    print(f"  Class {class_id}: {train_class_counts[class_id]} instances")

print("\n📊 VAL set class distribution:")
val_class_counts = defaultdict(int)
for label_file in val_labels.glob("*.txt"):
    with open(label_file, 'r') as f:
        for line in f:
            class_id = line.strip().split()[0]
            val_class_counts[class_id] += 1

for class_id in sorted(val_class_counts.keys()):
    print(f"  Class {class_id}: {val_class_counts[class_id]} instances")

print(f"\n✅ Dataset split complete!")
print(f"📁 Train: POC_split/train/ ({len(train_imgs)} images)")
print(f"📁 Val: POC_split/val/ ({len(val_imgs)} images)")
