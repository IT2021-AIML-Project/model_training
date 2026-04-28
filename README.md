# model_training

YOLOv8n-based PPE & vehicle detection pipeline — fine-tuned on a Roboflow public dataset then re-trained on client-labelled data for 7 safety-critical classes.

---

## Table of contents

1. [Project overview](#project-overview)
2. [Target classes](#target-classes)
3. [Pipeline](#pipeline)
4. [Model selection](#model-selection)
5. [Training results — Phase 1 (Roboflow dataset)](#training-results--phase-1-roboflow-dataset)
6. [Training results — Phase 2 (client dataset)](#training-results--phase-2-client-dataset)
7. [Repository structure](#repository-structure)
8. [Setup & usage](#setup--usage)
9. [Scripts reference](#scripts-reference)

---

## Project overview

This project trains a real-time object detector for on-site safety monitoring. The model must run at inference on a constrained GPU (NVIDIA GeForce GTX 1650, 4 GB VRAM), so model size and latency were primary selection criteria alongside accuracy.

**Base weights:** COCO-pretrained YOLOv8n, loaded via [Ultralytics](https://github.com/ultralytics/ultralytics).  
**Training framework:** Two-phase fine-tuning — first on a broad public dataset, then specialised to client-labelled data.

---

## Target classes

**Phase 2 (client dataset) — 7 classes:**

| ID | Class |
|----|-------|
| 0 | Hardhat |
| 1 | Mask |
| 2 | NO-Hardhat |
| 3 | NO-Mask |
| 4 | NO-Safety Vest |
| 5 | Person |
| 6 | Safety Vest |

*Phase 1 also covers 25 classes (Excavator, Gloves, Ladder, SUV, Safety Cone, bus, dump truck, fire hydrant, machinery, mini-van, sedan, semi, trailer, truck and trailer, truck, van, vehicle, wheel loader, and the 7 above).*

---

## Pipeline

```
Roboflow public dataset (25 classes)
        │
        ▼
  Preprocessing
  (resize · augment · normalise · YOLO label format)
        │
        ▼
  Model selection
  (benchmark on GTX 1650 — see table below)
        │
        ▼
  Phase 1 — fine-tune on Roboflow dataset
  (50 epochs, COCO pretrained weights)
        │
        ▼
  Client dataset labelling
  (7 PPE safety classes)
        │
        ▼
  Phase 2 — fine-tune on client dataset
  (50 epochs, Phase 1 weights)
        │
        ▼
  Evaluation
  (precision · recall · mAP@0.5 · mAP@0.5-0.95)
        │
        ▼
  Export & deploy
  (best.pt → ONNX / TensorRT / inference API)
```

---

## Model selection

Models were benchmarked on an **NVIDIA GeForce GTX 1650** (4 GB VRAM). The selection criterion was the best trade-off between inference latency and accuracy within the GPU's computational constraints.

| Model | Params | Inference (ms) | mAP@0.5 |
|-------|--------|---------------|---------|
| **YOLOv8n (selected)** | 3.2M | ~22 ms | 0.837 |
| YOLOv8s | 11.2M | ~128 ms | 0.943 |
| YOLOv9s | 7.2M | ~116 ms | 0.950 |
| YOLOv11s | 9.4M | ~90 ms | 0.950 |

**Why YOLOv8n?**  
YOLOv8s/v9s/v11s all achieve higher mAP but at 4–6× the inference time on this GPU. At ~128 ms, YOLOv8s is effectively too slow for real-time use (< 8 fps). YOLOv8n's 22 ms (~45 fps) keeps the pipeline real-time while still reaching **mAP@0.5 = 0.837** on the benchmark set — acceptable for the deployment scenario.

The benchmark scripts used to generate this table are in [`scripts/benchmark_gpu.py`](scripts/benchmark_gpu.py) and [`scripts/test_inference.py`](scripts/test_inference.py). Comparison plots can be reproduced with [`scripts/compare_models.py`](scripts/compare_models.py).

---

## Training results — Phase 1 (Roboflow dataset)

50 epochs on the 25-class Roboflow public dataset.

| Metric | Epoch 1 | Epoch 25 | Epoch 50 |
|--------|---------|---------|---------|
| Train box loss | 1.416 | 0.965 | 0.709 |
| Train cls loss | 2.044 | 0.761 | 0.447 |
| mAP@0.5 | 0.569 | 0.780 | 0.838 |
| mAP@0.5-0.95 | 0.247 | 0.446 | 0.528 |
| Precision | 0.659 | 0.914 | 0.935 |
| Recall | 0.553 | 0.716 | 0.757 |

Plot: `scripts/plot_training.py --csv results.csv`

---

## Training results — Phase 2 (client dataset)

50 epochs starting from Phase 1 weights, on the 7-class client dataset.

| Metric | Epoch 1 | Epoch 25 | Epoch 50 |
|--------|---------|---------|---------|
| Train box loss | 1.416 | 0.965 | 0.709 |
| Train cls loss | 2.044 | 0.761 | 0.447 |
| mAP@0.5 | 0.594 | 0.802 | 0.865 |
| mAP@0.5-0.95 | 0.274 | 0.457 | 0.547 |
| Precision | 0.676 | 0.935 | 0.954 |
| Recall | 0.576 | 0.730 | 0.767 |

Plot: `scripts/plot_training.py --csv results_with_clientdata.csv`

The client-data phase shows a consistent improvement over Phase 1, with mAP@0.5 converging ~3% higher (0.865 vs 0.838) due to the narrowed, task-specific class set.

---

## Repository structure

```
model_training/
├── README.md
├── data/
│   ├── roboflow/          # Phase 1 dataset (YOLO format)
│   └── client/            # Phase 2 dataset (YOLO format)
├── runs/
│   ├── phase1/            # Phase 1 training outputs
│   │   └── weights/
│   │       ├── best.pt
│   │       └── last.pt
│   └── phase2/            # Phase 2 training outputs
│       └── weights/
│           ├── best.pt
│           └── last.pt
├── results.csv                     # Phase 1 metrics per epoch
├── results_with_clientdata.csv     # Phase 2 metrics per epoch
└── scripts/
    ├── benchmark_gpu.py            # GPU memory & utilisation benchmark
    ├── test_inference.py           # Per-model inference speed test
    ├── compare_models.py           # Model comparison bar charts
    └── plot_training.py            # Training curve plots
```

---

## Setup & usage

```bash
# 1. Clone and install dependencies
git clone <your-repo-url>
cd model_training
pip install ultralytics matplotlib pandas seaborn

# 2. Run Phase 1 training
yolo train model=yolov8n.pt data=data/roboflow/data.yaml epochs=50 imgsz=640

# 3. Run Phase 2 training (from Phase 1 best weights)
yolo train model=runs/phase1/weights/best.pt data=data/client/data.yaml epochs=50 imgsz=640

# 4. Evaluate
yolo val model=runs/phase2/weights/best.pt data=data/client/data.yaml

# 5. Inference
yolo predict model=runs/phase2/weights/best.pt source=<image_or_video>
```

---

## Scripts reference

| Script | Purpose |
|--------|---------|
| `scripts/benchmark_gpu.py` | Logs GPU memory usage and utilisation while loading and running each model |
| `scripts/test_inference.py` | Times per-image inference across YOLOv8n/s, YOLOv9s, YOLOv11s |
| `scripts/compare_models.py` | Produces grouped bar charts: params vs model, inference vs model, mAP vs model |
| `scripts/plot_training.py` | Plots loss curves, mAP curves, and precision/recall from a results CSV |

Run any script with `--help` for full argument reference.
