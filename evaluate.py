"""
Evaluation & comparison script.

Compares Async ADMM against Sync FedAvg on:
  • ROC-AUC (predictive parity)
  • Communication overhead (target ≥ 60 % reduction)

Generates publication-ready plots and a summary table.
"""

import numpy as np
from model import LogisticRegressionModel
from utils import (compute_roc_auc, log, plot_roc_curves,
                   plot_training_curves, plot_communication_bar)


def evaluate(admm_history: dict, fedavg_history: dict,
             X_test: np.ndarray, y_test: np.ndarray):
    """
    Run head-to-head evaluation and produce plots.

    Parameters
    ----------
    admm_history, fedavg_history : dicts from .train()
    X_test, y_test : global test data
    """
    log("═══ Evaluation ═══")

    model = LogisticRegressionModel(X_test.shape[1])

    # ── Final predictions ────────────────────────────────────────────────────
    admm_scores = model.predict_proba(X_test, admm_history["final_z"])
    fedavg_scores = model.predict_proba(X_test, fedavg_history["final_w"])

    admm_auc = compute_roc_auc(y_test, admm_scores)
    fedavg_auc = compute_roc_auc(y_test, fedavg_scores)

    # ── Communication overhead ───────────────────────────────────────────────
    admm_comm = admm_history["total_comm"]
    fedavg_comm = fedavg_history["total_comm"]
    reduction = (1 - admm_comm / fedavg_comm) * 100 if fedavg_comm > 0 else 0

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"{'Metric':<35} {'Async ADMM':>12} {'Sync FedAvg':>12}")
    print("-" * 62)
    print(f"{'Final ROC-AUC':<35} {admm_auc:>12.4f} {fedavg_auc:>12.4f}")
    print(f"{'Total Communications':<35} {admm_comm:>12d} {fedavg_comm:>12d}")
    print(f"{'Communication Reduction':<35} {reduction:>11.1f}% {'(baseline)':>12}")
    print(f"{'AUC Difference':<35} {abs(admm_auc - fedavg_auc):>12.4f} {'':>12}")
    print("=" * 62)

    # ── Verdict ──────────────────────────────────────────────────────────────
    parity = abs(admm_auc - fedavg_auc) < 0.02
    target_met = reduction >= 60

    if parity and target_met:
        print("\n[OK] SUCCESS: Async ADMM achieves >=60% comm reduction with AUC parity.")
    elif parity:
        print(f"\n[!!] AUC parity achieved, but comm reduction is {reduction:.1f}% (target >=60%).")
    elif target_met:
        print(f"\n[!!] Comm reduction target met ({reduction:.1f}%), but AUC gap is {abs(admm_auc - fedavg_auc):.4f}.")
    else:
        print(f"\n[XX] Neither target met: comm reduction={reduction:.1f}%, AUC gap={abs(admm_auc - fedavg_auc):.4f}.")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_roc_curves(y_test, admm_scores, fedavg_scores)
    plot_training_curves(admm_history, fedavg_history)
    plot_communication_bar(admm_comm, fedavg_comm)

    return {
        "admm_auc": admm_auc,
        "fedavg_auc": fedavg_auc,
        "admm_comm": admm_comm,
        "fedavg_comm": fedavg_comm,
        "reduction_pct": reduction,
    }
