"""
plot_training.py
----------------
Plots training curves from a Ultralytics results.csv file.

Produces:
  - Loss curves (train box / cls / dfl loss)
  - Validation loss curves
  - mAP@0.5 and mAP@0.5-0.95 curves
  - Precision & Recall curves

Usage:
    # Phase 1
    python scripts/plot_training.py --csv results.csv

    # Phase 2 (client data)
    python scripts/plot_training.py --csv results_with_clientdata.csv --label "Phase 2 (client data)"

    # Overlay both phases
    python scripts/plot_training.py \
        --csv results.csv results_with_clientdata.csv \
        --label "Phase 1" "Phase 2 (client data)" \
        --output training_comparison.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


PALETTE = ["#1D9E75", "#E85D24", "#378ADD", "#BA7517"]  # teal, coral, blue, amber


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def add_curve(ax, epochs, values, label, color, lw=1.8, **kwargs):
    ax.plot(epochs, values, color=color, linewidth=lw, label=label, **kwargs)


def plot_all(csv_paths: list[str], labels: list[str], output: str, dpi: int):
    dfs = [load(p) for p in csv_paths]
    colors = PALETTE[: len(dfs)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Training metrics", fontsize=13, fontweight="500", y=1.01)

    panels = [
        # (ax_row, ax_col, y_col,                        ylabel,                title)
        (0, 0, "train/box_loss",           "Loss",         "Train — box loss"),
        (0, 1, "train/cls_loss",           "Loss",         "Train — cls loss"),
        (0, 2, "train/dfl_loss",           "Loss",         "Train — dfl loss"),
        (1, 0, "metrics/mAP50(B)",         "mAP",          "mAP@0.5"),
        (1, 1, "metrics/mAP50-95(B)",      "mAP",          "mAP@0.5-0.95"),
        (1, 2, None,                        "",             "Precision & Recall"),
    ]

    for r, c, col, ylabel, title in panels:
        ax = axes[r][c]
        ax.set_title(title, fontsize=10, fontweight="500", pad=6)
        ax.set_xlabel("Epoch", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        if col is not None:
            for df, label, color in zip(dfs, labels, colors):
                if col in df.columns:
                    add_curve(ax, df["epoch"], df[col], label, color)
        else:
            # Precision / Recall overlay
            for df, label, color in zip(dfs, labels, colors):
                if "metrics/precision(B)" in df.columns:
                    add_curve(ax, df["epoch"], df["metrics/precision(B)"],
                              f"{label} precision", color, lw=1.8)
                if "metrics/recall(B)" in df.columns:
                    add_curve(ax, df["epoch"], df["metrics/recall(B)"],
                              f"{label} recall", color, lw=1.8, linestyle="--")

        if len(dfs) > 1 or col is None:
            ax.legend(fontsize=7.5, frameon=False)

    plt.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {output}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Ultralytics training curves")
    parser.add_argument("--csv", nargs="+", required=True,
                        help="Path(s) to results CSV file(s)")
    parser.add_argument("--label", nargs="*",
                        help="Legend labels (one per CSV)")
    parser.add_argument("--output", default="training_curves.png",
                        help="Output image path (default: training_curves.png)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI (default: 150)")
    args = parser.parse_args()

    labels = args.label if args.label else [Path(p).stem for p in args.csv]
    plot_all(args.csv, labels, args.output, args.dpi)
