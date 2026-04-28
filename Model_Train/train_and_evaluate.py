"""
============================================================
YOLOv8s Training & Evaluation Script
PPE Detection - 7 Class Model
============================================================
Classes: Hardhat, Mask, NO-Hardhat, NO-Mask, 
         NO-Safety Vest, Person, Safety Vest
============================================================
"""

import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = r"c:\Users\LEGION\Desktop\Train_Dataset"
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
PROJECT_DIR = os.path.join(DATASET_DIR, "runs")
EXPERIMENT_NAME = "ppe_yolov8s"

# Original 10-class mapping (from the raw dataset)
# 0: Hardhat, 1: Mask, 2: NO-Hardhat, 3: NO-Mask,
# 4: NO-Safety Vest, 5: Person, 6: Safety Vest,
# 7: Safety Cone, 8: machinery, 9: vehicle

# Target 7-class mapping (what we want)
# 0: Hardhat, 1: Mask, 2: NO-Hardhat, 3: NO-Mask,
# 4: NO-Safety Vest, 5: Person, 6: Safety Vest

# Classes to REMOVE (original IDs)
CLASSES_TO_REMOVE = {7, 8, 9}  # Safety Cone, machinery, vehicle

# Class names for the 7-class model
CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask",
    "NO-Safety Vest", "Person", "Safety Vest"
]

# Training hyperparameters
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0  # GPU 0 (RTX 4060)


# ============================================================
# STEP 1: CLEAN LABELS (Remove unwanted classes)
# ============================================================

def clean_labels():
    """
    Remove annotations for classes 7, 8, 9 from all label files.
    Classes 0-6 stay with the same IDs (no remapping needed).
    """
    print("\n" + "=" * 60)
    print("STEP 1: CLEANING LABELS - Removing unwanted classes")
    print("=" * 60)

    total_removed = 0
    total_kept = 0
    files_modified = 0

    for split in ["train", "valid", "test"]:
        label_dir = os.path.join(DATASET_DIR, split, "labels")
        if not os.path.exists(label_dir):
            print(f"  [SKIP] {split}/labels not found")
            continue

        label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
        split_removed = 0
        split_kept = 0

        for label_file in label_files:
            filepath = os.path.join(label_dir, label_file)

            with open(filepath, "r") as f:
                lines = f.readlines()

            cleaned_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if class_id not in CLASSES_TO_REMOVE:
                    cleaned_lines.append(line)
                    split_kept += 1
                else:
                    split_removed += 1

            # Write back cleaned labels
            if len(cleaned_lines) != len(lines):
                files_modified += 1
                with open(filepath, "w") as f:
                    f.writelines(cleaned_lines)

        total_removed += split_removed
        total_kept += split_kept
        print(f"  [{split.upper()}] Kept: {split_kept} | Removed: {split_removed} annotations")

    print(f"\n  TOTAL - Kept: {total_kept} | Removed: {total_removed} | Files modified: {files_modified}")
    print("  Labels cleaned successfully!\n")


# ============================================================
# STEP 2: TRAIN YOLOv8s MODEL
# ============================================================

def train_model():
    """
    Train YOLOv8s on the PPE detection dataset.
    """
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING YOLOv8s MODEL")
    print("=" * 60)
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Classes: {len(CLASS_NAMES)}")
    print()

    # Load pretrained YOLOv8s model
    model = YOLO("yolov8s.pt")

    # Train the model
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        patience=10,         # Early stopping patience
        save=True,           # Save checkpoints
        save_period=10,      # Save every 10 epochs
        plots=True,          # Generate training plots
        verbose=True,
    )

    print("\n  Training complete!")
    print(f"  Best model saved at: {PROJECT_DIR}/{EXPERIMENT_NAME}/weights/best.pt")

    return model, results


# ============================================================
# STEP 3: MODEL EVALUATION
# ============================================================

def evaluate_model():
    """
    Evaluate the trained model and compute metrics.
    """
    print("\n" + "=" * 60)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 60)

    # Load the best trained model
    best_model_path = os.path.join(PROJECT_DIR, EXPERIMENT_NAME, "weights", "best.pt")
    if not os.path.exists(best_model_path):
        print(f"  [ERROR] Best model not found at: {best_model_path}")
        return
    
    model = YOLO(best_model_path)
    print(f"  Loaded best model: {best_model_path}")

    # Run validation on the test set
    print("\n  Running validation on TEST set...")
    val_results = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_DIR,
        name=f"{EXPERIMENT_NAME}_eval",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    # --------------------------------------------------------
    # Extract metrics from YOLO validation results
    # --------------------------------------------------------
    print("\n" + "-" * 60)
    print("  DETECTION METRICS (from YOLO validation)")
    print("-" * 60)

    # Per-class metrics
    box_results = val_results.box
    
    # mAP metrics
    map50 = box_results.map50       # mAP@0.5
    map50_95 = box_results.map      # mAP@0.5:0.95
    
    print(f"\n  mAP@0.5      : {map50:.4f}")
    print(f"  mAP@0.5:0.95 : {map50_95:.4f}")

    # Per-class AP values
    print(f"\n  {'Class':<20} {'Precision':>10} {'Recall':>10} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14}")
    print("  " + "-" * 66)

    # Get per-class metrics
    p = box_results.p        # Precision per class
    r = box_results.r        # Recall per class
    ap50 = box_results.ap50  # AP@0.5 per class
    ap = box_results.ap      # AP@0.5:0.95 per class

    for i, name in enumerate(CLASS_NAMES):
        if i < len(p):
            print(f"  {name:<20} {p[i]:>10.4f} {r[i]:>10.4f} {ap50[i]:>10.4f} {ap[i]:>14.4f}")

    # Mean values
    mean_precision = np.mean(p) if len(p) > 0 else 0
    mean_recall = np.mean(r) if len(r) > 0 else 0
    mean_f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0

    print("  " + "-" * 66)
    print(f"  {'MEAN':<20} {mean_precision:>10.4f} {mean_recall:>10.4f} {map50:>10.4f} {map50_95:>14.4f}")

    print("\n" + "=" * 60)
    print("  ALL DONE! Training and evaluation complete.")
    print("=" * 60)
    print(f"\n  Results saved in: {os.path.join(PROJECT_DIR, EXPERIMENT_NAME)}")
    print(f"  Best model: {os.path.join(PROJECT_DIR, EXPERIMENT_NAME, 'weights', 'best.pt')}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  YOLOv8s PPE DETECTION - TRAIN & EVALUATE")
    print("  Dataset: Construction Site Safety (7 Classes)")
    print("  Model: YOLOv8s (Small)")
    print(f"  GPU: NVIDIA RTX 4060 Laptop")
    print("=" * 60)

    # Step 1: Clean labels (remove classes 7, 8, 9)
    clean_labels()

    # Step 2: Train YOLOv8s
    model, results = train_model()

    # Step 3: Evaluate the trained model
    evaluate_model()
