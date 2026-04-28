"""
test_inference.py
-----------------
Times per-image inference for each candidate YOLO model and prints a
formatted comparison table. Optionally saves a bar chart.

Requires:
    pip install ultralytics torch matplotlib

Usage:
    python scripts/test_inference.py
    python scripts/test_inference.py --imgsz 640 --runs 100 --plot inference_speed.png
    python scripts/test_inference.py --source path/to/image.jpg --runs 30
"""

import argparse
import statistics
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO


MODELS = [
    ("YOLOv8n", "yolov8n.pt"),
    ("YOLOv8s", "yolov8s.pt"),
    ("YOLOv9s", "yolov9s.pt"),
    ("YOLOv11s", "yolo11s.pt"),
]

COL_W = 14


def time_model(name: str, weights: str, source, imgsz: int, runs: int, device: str):
    model = YOLO(weights)
    model.to(device)

    # Build dummy tensor or use provided source
    if source is None:
        img = torch.zeros(1, 3, imgsz, imgsz).to(device)
    else:
        img = source

    # Warmup
    for _ in range(5):
        model.predict(source=img, verbose=False)

    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(source=img, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "name":    name,
        "mean_ms": round(statistics.mean(times), 2),
        "std_ms":  round(statistics.stdev(times), 2),
        "p50_ms":  round(statistics.median(times), 2),
        "p95_ms":  round(sorted(times)[int(0.95 * len(times))], 2),
        "min_ms":  round(min(times), 2),
        "max_ms":  round(max(times), 2),
        "fps":     round(1000 / statistics.mean(times), 1),
    }


def print_table(rows: list[dict]):
    headers = ["Model", "Mean (ms)", "Std", "p50", "p95", "Min", "Max", "FPS"]
    keys    = ["name", "mean_ms", "std_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "fps"]
    print("\n" + "─" * (COL_W * len(headers)))
    print("".join(h.ljust(COL_W) for h in headers))
    print("─" * (COL_W * len(headers)))
    for row in rows:
        print("".join(str(row[k]).ljust(COL_W) for k in keys))
    print("─" * (COL_W * len(headers)) + "\n")


def save_plot(rows: list[dict], output: str):
    labels  = [r["name"] for r in rows]
    means   = [r["mean_ms"] for r in rows]
    stds    = [r["std_ms"] for r in rows]
    colors  = ["#1D9E75"] + ["#888780"] * (len(rows) - 1)  # highlight first

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(x, means, yerr=stds, color=colors, width=0.55,
                  capsize=4, ecolor="#444441", edgecolor="white", linewidth=0.8,
                  zorder=3)
    ax.set_title("Inference latency comparison (mean ± std)", fontsize=11, fontweight="500")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.2,
                f"{val:.1f} ms",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {output}")
    plt.show()


def main(imgsz: int, runs: int, source_path: str | None, plot: str | None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}  |  Image size: {imgsz}×{imgsz}  |  Runs per model: {runs}\n")

    source = None
    if source_path:
        source = source_path  # Ultralytics handles path loading

    results = []
    for name, weights in MODELS:
        print(f"Testing {name}...", end=" ", flush=True)
        try:
            row = time_model(name, weights, source, imgsz, runs, device)
            results.append(row)
            print(f"done — {row['mean_ms']} ms avg")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if results:
        print_table(results)
        if plot:
            save_plot(results, plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-model YOLO inference speed test")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--runs", type=int, default=50,
                        help="Number of timed runs per model (default: 50)")
    parser.add_argument("--source", default=None,
                        help="Path to a test image (uses a blank tensor if omitted)")
    parser.add_argument("--plot", default=None,
                        help="If set, save a bar chart to this path")
    args = parser.parse_args()
    main(args.imgsz, args.runs, args.source, args.plot)
