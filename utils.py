"""
Utility helpers: metrics, logging, plotting.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server / CI
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

import config


# ─── Metrics ───────────────────────────────────────────────────────────────────
def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC-AUC; returns 0.5 if degenerate."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_scores))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# ─── Logging ───────────────────────────────────────────────────────────────────
_T0 = time.perf_counter()


def log(msg: str) -> None:
    elapsed = time.perf_counter() - _T0
    print(f"[{elapsed:7.2f}s] {msg}")


# ─── Plotting ─────────────────────────────────────────────────────────────────
def _ensure_output_dir() -> str:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    return config.OUTPUT_DIR


def plot_roc_curves(y_true, admm_scores, fedavg_scores):
    """Plot ROC curves for both methods side-by-side."""
    out = _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(7, 6))

    for scores, label, color in [
        (admm_scores, "Async ADMM", "#2196F3"),
        (fedavg_scores, "Sync FedAvg", "#FF5722"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out, "roc_comparison.png")
    fig.savefig(path, dpi=config.PLOT_DPI)
    plt.close(fig)
    log(f"ROC plot saved -> {path}")


def plot_training_curves(admm_hist, fedavg_hist):
    """Plot loss & AUC over communication rounds."""
    out = _ensure_output_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss ---
    ax = axes[0]
    ax.plot(admm_hist["rounds"], admm_hist["loss"], label="Async ADMM",
            color="#2196F3", lw=2)
    ax.plot(fedavg_hist["rounds"], fedavg_hist["loss"], label="Sync FedAvg",
            color="#FF5722", lw=2)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- AUC ---
    ax = axes[1]
    ax.plot(admm_hist["rounds"], admm_hist["auc"], label="Async ADMM",
            color="#2196F3", lw=2)
    ax.plot(fedavg_hist["rounds"], fedavg_hist["auc"], label="Sync FedAvg",
            color="#FF5722", lw=2)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Test ROC-AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out, "training_curves.png")
    fig.savefig(path, dpi=config.PLOT_DPI)
    plt.close(fig)
    log(f"Training curves saved -> {path}")


def plot_communication_bar(admm_comm, fedavg_comm):
    """Bar chart comparing communication overhead."""
    out = _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(6, 5))

    methods = ["Sync FedAvg", "Async ADMM"]
    values = [fedavg_comm, admm_comm]
    colors = ["#FF5722", "#2196F3"]

    bars = ax.bar(methods, values, color=colors, width=0.5, edgecolor="white", lw=1.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                str(v), ha="center", va="bottom", fontweight="bold", fontsize=13)

    reduction = (1 - admm_comm / fedavg_comm) * 100 if fedavg_comm > 0 else 0
    ax.set_ylabel("Total Node-Server Communications")
    ax.set_title(f"Communication Overhead  (down {reduction:.1f}% reduction)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out, "communication_overhead.png")
    fig.savefig(path, dpi=config.PLOT_DPI)
    plt.close(fig)
    log(f"Communication bar chart saved -> {path}")
