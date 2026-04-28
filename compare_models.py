"""
compare_models.py
-----------------
Produces grouped bar charts comparing YOLOv8n, YOLOv8s, YOLOv9s, YOLOv11s
across: parameter count, inference latency (ms), and mAP@0.5.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --output plots/model_comparison.png
    python scripts/compare_models.py --dpi 200
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Model data ──────────────────────────────────────────────────────────────
MODELS = {
    "YOLOv8n\n(selected)": {"params_M": 3.2,  "inference_ms": 22.0,  "mAP50": 0.837},
    "YOLOv8s":             {"params_M": 11.2, "inference_ms": 128.4, "mAP50": 0.943},
    "YOLOv9s":             {"params_M": 7.2,  "inference_ms": 116.0, "mAP50": 0.950},
    "YOLOv11s":            {"params_M": 9.4,  "inference_ms": 90.0,  "mAP50": 0.950},
}

COLORS = {
    "YOLOv8n\n(selected)": "#1D9E75",   # teal  — selected model
    "YOLOv8s":             "#888780",   # gray
    "YOLOv9s":             "#888780",
    "YOLOv11s":            "#888780",
}

HIGHLIGHT = "YOLOv8n\n(selected)"

# ── Helpers ──────────────────────────────────────────────────────────────────
def bar_chart(ax, labels, values, title, ylabel, colors, fmt=".1f", selected=None):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.55, zorder=3,
                  edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=11, fontweight="500", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ax.get_ylim()[1] * 0.01,
            f"{val:{fmt}}",
            ha="center", va="bottom", fontsize=8.5,
        )
    if selected is not None and selected in labels:
        idx = labels.index(selected)
        bars[idx].set_edgecolor("#0F6E56")
        bars[idx].set_linewidth(1.5)


# ── Main ─────────────────────────────────────────────────────────────────────
def main(output: str, dpi: int):
    labels = list(MODELS.keys())
    params  = [MODELS[m]["params_M"]     for m in labels]
    latency = [MODELS[m]["inference_ms"] for m in labels]
    mAP     = [MODELS[m]["mAP50"]        for m in labels]
    colors  = [COLORS[m]                 for m in labels]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        "Model comparison — NVIDIA GeForce GTX 1650",
        fontsize=12, fontweight="500", y=1.01,
    )

    bar_chart(axes[0], labels, params,  "Parameter count",   "Params (M)",      colors, fmt=".1f", selected=HIGHLIGHT)
    bar_chart(axes[1], labels, latency, "Inference latency", "Latency (ms)",    colors, fmt=".1f", selected=HIGHLIGHT)
    bar_chart(axes[2], labels, mAP,     "mAP@0.5",           "mAP@0.5",         colors, fmt=".3f", selected=HIGHLIGHT)

    # Force mAP y-axis to start near 0.8 for better readability
    axes[2].set_ylim(0.80, 0.97)

    legend_patch = mpatches.Patch(facecolor=COLORS[HIGHLIGHT], edgecolor="#0F6E56",
                                  linewidth=1.5, label="Selected model (YOLOv8n)")
    fig.legend(handles=[legend_patch], loc="lower center", ncol=1,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {output}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model comparison bar charts")
    parser.add_argument("--output", default="model_comparison.png",
                        help="Output image path (default: model_comparison.png)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI (default: 150)")
    args = parser.parse_args()
    main(args.output, args.dpi)
