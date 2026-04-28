"""
benchmark_gpu.py
----------------
Logs GPU memory usage and utilisation while loading and performing a forward
pass with each candidate YOLO model. Results are printed to stdout and saved
to benchmark_results.csv.

Requires:
    pip install ultralytics torch pynvml

Usage:
    python scripts/benchmark_gpu.py
    python scripts/benchmark_gpu.py --imgsz 640 --runs 20 --output gpu_benchmark.csv
"""

import argparse
import csv
import time
import warnings
from pathlib import Path

import torch

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    warnings.warn("pynvml not found — GPU utilisation will not be recorded. "
                  "Install with: pip install pynvml")

from ultralytics import YOLO


MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov9s.pt",
    "yolo11s.pt",
]


def nvml_handle():
    if not NVML_AVAILABLE:
        return None
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def gpu_stats(handle):
    """Return (used_MiB, total_MiB, utilisation_pct) or (None, None, None)."""
    if handle is None:
        return None, None, None
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return mem.used // 1024**2, mem.total // 1024**2, util.gpu


def benchmark_model(model_name: str, imgsz: int, runs: int, handle):
    print(f"\n── {model_name} ──")

    # Memory before loading
    mem_before, mem_total, _ = gpu_stats(handle)

    model = YOLO(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Memory after loading
    mem_loaded, _, util_loaded = gpu_stats(handle)

    # Warmup
    dummy = torch.zeros(1, 3, imgsz, imgsz).to(device)
    for _ in range(3):
        _ = model.predict(source=dummy, verbose=False)

    # Timed runs
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.predict(source=dummy, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    mem_infer, _, util_infer = gpu_stats(handle)

    avg_ms  = sum(latencies) / len(latencies)
    min_ms  = min(latencies)
    max_ms  = max(latencies)
    mem_used_mb = (mem_loaded - mem_before) if mem_loaded and mem_before else None

    print(f"  Device           : {device.upper()}")
    print(f"  Image size       : {imgsz}×{imgsz}")
    print(f"  Runs             : {runs}")
    print(f"  Avg latency (ms) : {avg_ms:.1f}")
    print(f"  Min / Max (ms)   : {min_ms:.1f} / {max_ms:.1f}")
    if mem_used_mb is not None:
        print(f"  GPU mem (model)  : +{mem_used_mb} MiB  ({mem_loaded} / {mem_total} MiB total)")
    if util_infer is not None:
        print(f"  GPU utilisation  : {util_infer}%")

    return {
        "model":            model_name,
        "device":           device,
        "imgsz":            imgsz,
        "avg_ms":           round(avg_ms, 2),
        "min_ms":           round(min_ms, 2),
        "max_ms":           round(max_ms, 2),
        "gpu_mem_delta_MiB": mem_used_mb,
        "gpu_mem_total_MiB": mem_total,
        "gpu_util_pct":     util_infer,
    }


def main(imgsz: int, runs: int, output: str):
    handle = nvml_handle()
    results = []

    for model_name in MODELS:
        try:
            row = benchmark_model(model_name, imgsz, runs, handle)
            results.append(row)
        except Exception as exc:
            print(f"  ERROR loading {model_name}: {exc}")

    # Save CSV
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved → {output_path}")

    if NVML_AVAILABLE:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU benchmark for YOLO model selection")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--runs", type=int, default=50,
                        help="Number of timed inference runs per model (default: 50)")
    parser.add_argument("--output", default="benchmark_results.csv",
                        help="Output CSV path (default: benchmark_results.csv)")
    args = parser.parse_args()
    main(args.imgsz, args.runs, args.output)
