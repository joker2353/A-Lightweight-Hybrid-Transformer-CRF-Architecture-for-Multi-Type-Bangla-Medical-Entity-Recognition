"""
generate_figures.py
Generates all five figures for the IEEE paper from real experimental data.
Saves to e:/MED-ER/figures/ as high-res PDF + PNG.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── colour palette (accessible, prints well in greyscale) ─────────────────────
C_TEACHER  = "#1f4e79"   # dark blue
C_CRFKD    = "#2e75b6"   # medium blue
C_KD       = "#70ad47"   # green
C_NOKD     = "#ed7d31"   # orange
C_QUANT    = "#7030a0"   # purple
C_GRID     = "#dddddd"

FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.6,
})

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  saved {name}.pdf / {name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Pipeline Architecture
# ══════════════════════════════════════════════════════════════════════════════
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.8, 2.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        ("Raw Dataset\n(6,895 sentences)", "#e2efda", 0.45),
        ("BIO\nPreprocessing", "#dce6f1", 1.55),
        ("BanglaBERT\nCRF Teacher (12L)", "#bdd7ee", 3.05),
        ("Pre-CRF\nEmission Logits", "#dce6f1", 4.75),
        ("TinyBanglaBERT\nKD Student (4L)", "#bdd7ee", 6.25),
        ("INT8 Dynamic\nQuantization", "#dce6f1", 7.75),
        ("Edge Deployment\n(6.28 ms / CPU)", "#e2efda", 9.3),
    ]

    box_w, box_h = 1.0, 0.52
    for label, color, cx in stages:
        bbox = FancyBboxPatch((cx - box_w/2, 0.24), box_w, box_h,
                               boxstyle="round,pad=0.04",
                               linewidth=0.8, edgecolor="#555555",
                               facecolor=color, zorder=3)
        ax.add_patch(bbox)
        ax.text(cx, 0.50, label, ha="center", va="center",
                fontsize=7.2, zorder=4, linespacing=1.3)

    # arrows
    arrow_xs = [
        (0.95, 1.05), (2.05, 2.55), (3.55, 4.25),
        (5.25, 5.75), (6.75, 7.25), (8.25, 8.80),
    ]
    for x0, x1 in arrow_xs:
        ax.annotate("", xy=(x1, 0.50), xytext=(x0, 0.50),
                    arrowprops=dict(arrowstyle="-|>", color="#333333",
                                   lw=1.0), zorder=5)

    # stage labels below
    labels_below = {2.55: "Fine-tune\n+CRF loss", 5.00: "KD soft\ntargets", 7.50: "No retraining"}
    for cx, txt in labels_below.items():
        ax.text(cx, 0.14, txt, ha="center", va="center",
                fontsize=6.5, color="#555555", style="italic")

    fig.tight_layout(pad=0.2)
    save(fig, "fig1_pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Training Convergence (F1 per epoch)
# ══════════════════════════════════════════════════════════════════════════════
def fig_training_curves():
    # CRF Teacher (from trainer_state log_history)
    teacher_epochs = [1, 2, 3, 4, 5]
    teacher_f1     = [31.47, 44.73, 46.42, 48.88, 50.86]   # val F1
    teacher_loss   = [31.78, 26.57, 25.79, 24.71, 28.08]   # val loss

    # CRF-KD Student
    student_epochs = [1, 2, 3, 4, 5, 6, 7]
    student_f1     = [28.80, 36.24, 40.82, 40.80, 42.19, 42.91, 44.26]
    student_loss   = [0.537, 0.412, 0.371, 0.354, 0.350, 0.337, 0.338]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6))

    # ── F1 curves ──────────────────────────────────────────────────────────
    ax1.plot(teacher_epochs, teacher_f1, "o-", color=C_TEACHER, lw=1.6,
             ms=5, label="CRF Teacher (12L)")
    ax1.plot(student_epochs, student_f1, "s--", color=C_CRFKD, lw=1.6,
             ms=5, label="CRF-KD Student (4L)")
    ax1.axhline(47.86, color=C_TEACHER, lw=0.9, ls=":", alpha=0.7)
    ax1.axhline(44.56, color=C_CRFKD,   lw=0.9, ls=":", alpha=0.7)
    ax1.text(5.05, 48.3, "Test 47.86%", fontsize=6.5, color=C_TEACHER)
    ax1.text(7.05, 45.0, "Test 44.56%", fontsize=6.5, color=C_CRFKD)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Macro F1 (%)")
    ax1.set_title("(a) Macro F1 Convergence")
    ax1.set_xlim(0.5, 8)
    ax1.set_ylim(20, 58)
    ax1.legend(loc="lower right")

    # ── Loss curves ────────────────────────────────────────────────────────
    ax2l = ax2
    ax2r = ax2.twinx()

    l1, = ax2l.plot(teacher_epochs, teacher_loss, "o-", color=C_TEACHER,
                    lw=1.6, ms=5, label="CRF Teacher loss")
    l2, = ax2r.plot(student_epochs, student_loss, "s--", color=C_CRFKD,
                    lw=1.6, ms=5, label="KD Student loss")
    ax2l.set_xlabel("Epoch")
    ax2l.set_ylabel("CRF NLL Val Loss", color=C_TEACHER)
    ax2r.set_ylabel("KD Val Loss (CE+KL)", color=C_CRFKD)
    ax2l.tick_params(axis="y", colors=C_TEACHER)
    ax2r.tick_params(axis="y", colors=C_CRFKD)
    ax2.set_title("(b) Validation Loss")
    ax2.set_xlim(0.5, 8)
    lines = [l1, l2]
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper right")

    fig.tight_layout(pad=0.5)
    save(fig, "fig2_training_curves")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Class-wise F1 Grouped Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig_classwise():
    classes = ["MED\n(n=307)", "ORG\n(n=130)", "DIS\n(n=199)",
               "HOR\n(n=28)",  "PHA\n(n=41)",  "CMT\n(n=225)"]

    teacher   = [76.34, 40.00, 29.21, 21.33, 29.27, 17.89]
    no_kd     = [71.12, 34.98, 27.98, 27.27, 17.07, 13.86]
    std_kd    = [72.48, 35.92, 27.08, 29.03, 21.95, 16.16]
    quant     = [72.50, 35.34, 27.41, 28.57, 24.69, 16.84]

    x     = np.arange(len(classes))
    width = 0.20
    offsets = [-1.5, -0.5, 0.5, 1.5]
    colours = [C_TEACHER, C_NOKD, C_KD, C_QUANT]
    labels  = ["CRF Teacher (12L)", "No-KD Student (4L)",
               "Std-KD Student (4L)", "Quantized CRF-KD (INT8)"]
    data    = [teacher, no_kd, std_kd, quant]

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    for i, (d, c, lbl, off) in enumerate(zip(data, colours, labels, offsets)):
        bars = ax.bar(x + off*width, d, width, label=lbl,
                      color=c, alpha=0.88, edgecolor="white", lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("Class-wise F1 Score Comparison Across Models")
    ax.set_ylim(0, 88)
    ax.legend(loc="upper right", ncol=2, fontsize=7.5)

    # annotate macro F1 at the right
    macro = {"CRF Teacher (12L)": 47.86, "No-KD Student (4L)": 40.90,
             "Std-KD Student (4L)": 41.96, "Quantized CRF-KD (INT8)": 42.20}
    for col, lbl, off in zip(colours, labels, offsets):
        ax.axhline(macro[lbl], color=col, lw=0.8, ls="--", alpha=0.55)

    fig.tight_layout(pad=0.5)
    save(fig, "fig3_classwise_f1")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Layer Truncation Analysis
# ══════════════════════════════════════════════════════════════════════════════
def fig_layer_truncation():
    layers_t = list(range(1, 13))
    f1_t     = [0.38, 0.59, 0.81, 1.18, 1.89, 4.98,
                12.91, 20.46, 28.12, 32.28, 37.77, 44.08]
    acc_t    = [9.41, 22.21, 37.39, 49.33, 56.75, 67.60,
                77.33, 80.39, 83.04, 84.19, 84.76, 85.50]

    layers_s = [1, 2, 3, 4]
    f1_s     = [0.93, 7.68, 23.99, 41.96]
    acc_s    = [44.83, 75.83, 82.83, 85.12]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8))

    # F1
    ax1.plot(layers_t, f1_t, "o-", color=C_TEACHER, lw=1.6, ms=5,
             label="Teacher (truncated, no retraining)")
    ax1.plot(layers_s, f1_s, "s-", color=C_KD, lw=1.6, ms=7,
             label="KD Student (after full distillation)")
    ax1.axvline(4, color="#aaaaaa", lw=0.9, ls="--")
    ax1.text(4.1, 2, "4-layer\nstudent", fontsize=7, color="#777777")
    ax1.set_xlabel("Number of Active Transformer Layers")
    ax1.set_ylabel("Macro F1 (%)")
    ax1.set_title("(a) Macro F1 vs Layer Count")
    ax1.set_xlim(0.5, 12.5)
    ax1.set_ylim(-2, 50)
    ax1.legend(fontsize=7, loc="upper left")

    # Accuracy
    ax2.plot(layers_t, acc_t, "o-", color=C_TEACHER, lw=1.6, ms=5,
             label="Teacher (truncated)")
    ax2.plot(layers_s, acc_s, "s-", color=C_KD, lw=1.6, ms=7,
             label="KD Student")
    ax2.axvline(4, color="#aaaaaa", lw=0.9, ls="--")
    ax2.set_xlabel("Number of Active Transformer Layers")
    ax2.set_ylabel("Token Accuracy (%)")
    ax2.set_title("(b) Token Accuracy vs Layer Count")
    ax2.set_xlim(0.5, 12.5)
    ax2.set_ylim(5, 90)
    ax2.legend(fontsize=7, loc="upper left")

    fig.tight_layout(pad=0.5)
    save(fig, "fig4_layer_truncation")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Efficiency vs Accuracy Trade-off (bubble chart)
# ══════════════════════════════════════════════════════════════════════════════
def fig_efficiency():
    models   = ["Teacher\n(12L, no CRF)", "CRF Teacher\n(12L+CRF)",
                "CRF-KD\nStudent (4L)", "Quantized\nCRF-KD (INT8)"]
    latency  = [54.14, 44.70, 15.73, 6.28]      # ms
    f1       = [44.08, 47.86, 44.56, 42.20]      # Macro F1 %
    size_mb  = [624.93, 627.24, 408.61, 327.62]  # model size
    colours  = [C_TEACHER, C_TEACHER, C_CRFKD, C_QUANT]
    markers  = ["D", "D", "o", "^"]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    for i, (m, lat, f, sz, col, mk) in enumerate(
            zip(models, latency, f1, size_mb, colours, markers)):
        sc = ax.scatter(lat, f, s=sz/3.5, c=col, marker=mk,
                        alpha=0.82, edgecolors="#333333", linewidths=0.7, zorder=4)
        offset = (-28, 4) if i == 0 else (3, 4)
        ax.annotate(m, xy=(lat, f), xytext=(lat + offset[0]*0.08, f + offset[1]*0.12),
                    fontsize=7.5, color="#222222",
                    ha="left" if i > 1 else "right")

    ax.set_xlabel("CPU Inference Latency (ms)  ← faster")
    ax.set_ylabel("Macro F1 (%)")
    ax.set_title("Efficiency vs. Accuracy Trade-off\n"
                 "(bubble area ∝ model storage size)")
    ax.set_xlim(-2, 65)
    ax.set_ylim(41, 50)
    ax.invert_xaxis()

    # Pareto frontier hint
    ax.annotate("", xy=(6.28, 42.20), xytext=(15.73, 44.56),
                arrowprops=dict(arrowstyle="-|>", color="#aaaaaa", lw=1.0,
                                connectionstyle="arc3,rad=-0.25"))

    # size legend
    for sz_ref, lbl in [(327, "327 MB"), (625, "625 MB")]:
        ax.scatter([], [], s=sz_ref/3.5, c="grey", alpha=0.5,
                   edgecolors="#666666", lw=0.7, label=lbl)
    ax.legend(title="Model size", loc="lower left", fontsize=7.5,
              title_fontsize=7.5, framealpha=0.9)

    fig.tight_layout(pad=0.5)
    save(fig, "fig5_efficiency_tradeoff")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    fig_pipeline()
    fig_training_curves()
    fig_classwise()
    fig_layer_truncation()
    fig_efficiency()
    print("Done. All figures saved to:", FIG_DIR)
