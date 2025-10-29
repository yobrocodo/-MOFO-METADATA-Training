#!/usr/bin/env python3
"""
Script to organize METADATA (BUSI_Corrected) dataset into MOFO format
"""

import os
import json
import shutil
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)

print("=" * 60)
print("Setting up METADATA Dataset for MOFO Training")
print("=" * 60)

# Paths
source_dir = Path("METADATA_extracted/BUSI_Corrected")
target_dir = Path("Multi-Organ Database/Dataset_METADATA")

# Create target directory structure
for split in ['Train', 'Valid', 'Test']:
    (target_dir / split / 'USImage').mkdir(parents=True, exist_ok=True)
    (target_dir / split / 'Mask').mkdir(parents=True, exist_ok=True)

# Categories to process (we'll combine all into one segmentation task)
categories = ['benign', 'malignant', 'normal']

# Collect all image-mask pairs
all_pairs = []

for category in categories:
    img_dir = source_dir / category
    mask_dir = source_dir / f"{category}_mask"
    
    if not img_dir.exists():
        print(f"Warning: {img_dir} does not exist, skipping...")
        continue
    
    # Get all images
    images = sorted(list(img_dir.glob("*.png")))
    
    for img_path in images:
        img_name = img_path.stem  # e.g., "benign (1)"
        
        # Find corresponding mask
        if mask_dir.exists():
            mask_path = mask_dir / f"{img_name}_mask.png"
            
            if mask_path.exists():
                all_pairs.append({
                    'image': img_path,
                    'mask': mask_path,
                    'category': category
                })
            else:
                print(f"Warning: No mask found for {img_name}, skipping...")
        else:
            print(f"Warning: Mask directory {mask_dir} does not exist")

print(f"\nFound {len(all_pairs)} valid image-mask pairs")

# Shuffle and split the data
random.shuffle(all_pairs)

# Split ratios: 70% train, 15% valid, 15% test
n_total = len(all_pairs)
n_train = int(0.70 * n_total)
n_valid = int(0.15 * n_total)

train_pairs = all_pairs[:n_train]
valid_pairs = all_pairs[n_train:n_train + n_valid]
test_pairs = all_pairs[n_train + n_valid:]

print(f"\nDataset split:")
print(f"  Training: {len(train_pairs)} samples")
print(f"  Validation: {len(valid_pairs)} samples")
print(f"  Test: {len(test_pairs)} samples")

# Function to copy files and create JSON entries
def process_split(pairs, split_name):
    json_entries = []
    
    for idx, pair in enumerate(pairs):
        # Create new filename with index for uniqueness
        category = pair['category']
        img_basename = pair['image'].stem
        new_name = f"{category}_{idx:04d}.png"
        
        # Copy image
        src_img = pair['image']
        dst_img = target_dir / split_name / 'USImage' / new_name
        shutil.copy2(src_img, dst_img)
        
        # Copy mask
        src_mask = pair['mask']
        dst_mask = target_dir / split_name / 'Mask' / new_name
        shutil.copy2(src_mask, dst_mask)
        
        # Create JSON entry
        json_entries.append({
            "USImage": f"{split_name}/USImage/{new_name}",
            "Mask": f"{split_name}/Mask/{new_name}"
        })
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(pairs)} files...")
    
    return json_entries

# Process each split
print("\nProcessing Training split...")
train_json = process_split(train_pairs, 'Train')

print("\nProcessing Validation split...")
valid_json = process_split(valid_pairs, 'Valid')

print("\nProcessing Test split...")
test_json = process_split(test_pairs, 'Test')

# Create JSON configuration file
dataset_config = {
    "Train": train_json,
    "Valid": valid_json,
    "Test": test_json
}

json_path = target_dir / "Dataset_METADATA.json"
with open(json_path, 'w') as f:
    json.dump(dataset_config, f, indent=4)

print(f"\n✓ Created JSON configuration: {json_path}")
print(f"  - Train entries: {len(train_json)}")
print(f"  - Valid entries: {len(valid_json)}")
print(f"  - Test entries: {len(test_json)}")

print("\n" + "=" * 60)
print("✓ Dataset setup complete!")
print("=" * 60)
print(f"\nDataset location: {target_dir}")
print("\nNext steps:")
print("  1. Update dataset_config.yaml to include this dataset")
print("  2. Run training with: python run_with_cuda_patch.py train.py")

