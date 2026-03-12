"""
Synchronous Federated Averaging (FedAvg) baseline.

All nodes train locally every round and synchronously send their parameters
to the server for averaging.  This serves as the communication-heavy baseline.
"""

import numpy as np
from model import LogisticRegressionModel
from utils import compute_roc_auc, log
import config


class FedAvgSync:
    """
    Standard synchronous Federated Averaging.

    Every round:
        1. Server broadcasts global model w to ALL nodes.
        2. Each node runs `local_epochs` SGD steps on its local data.
        3. Server averages all local models:  w ← (1/N) Σ wᵢ

    Communication per round = N (all nodes must participate).
    """

    def __init__(self, node_data, feature_dim: int):
        self.feature_dim = feature_dim
        self.node_data = node_data          # list[(X, y)]
        self.n_nodes = len(node_data)
        self.model = LogisticRegressionModel(feature_dim)

        # Global model
        self.w = np.zeros(feature_dim)

    # ── Training loop ────────────────────────────────────────────────────────
    def train(self, X_test: np.ndarray, y_test: np.ndarray,
              max_rounds: int = None):
        """
        Run synchronous FedAvg training.

        Returns
        -------
        history : dict  with keys 'rounds', 'loss', 'auc', 'total_comm'
        """
        max_rounds = max_rounds or config.FEDAVG_MAX_ROUNDS
        lr = config.FEDAVG_LOCAL_LR
        local_epochs = config.FEDAVG_LOCAL_EPOCHS
        total_comm = 0
        history = {"rounds": [], "loss": [], "auc": []}

        log("─── Synchronous FedAvg Training ───")

        for rnd in range(1, max_rounds + 1):
            local_weights = []

            # 1. Broadcast global model to ALL nodes → each trains locally
            for X_i, y_i in self.node_data:
                w_local = self.w.copy()
                for _ in range(local_epochs):
                    grad = self.model.gradient(X_i, y_i, w_local)
                    w_local -= lr * grad
                local_weights.append(w_local)
                total_comm += 1           # one upload per node per round

            # 2. Synchronous averaging
            self.w = np.mean(local_weights, axis=0)

            # 3. Evaluate on global test set
            test_loss = self.model.loss(X_test, y_test, self.w)
            y_scores = self.model.predict_proba(X_test, self.w)
            auc = compute_roc_auc(y_test, y_scores)

            history["rounds"].append(rnd)
            history["loss"].append(test_loss)
            history["auc"].append(auc)

            if rnd % 10 == 0 or rnd == 1:
                log(f"  Round {rnd:3d} | loss={test_loss:.4f} | "
                    f"AUC={auc:.4f} | comm={total_comm}")

        history["total_comm"] = total_comm
        history["final_w"] = self.w.copy()
        log(f"  [OK] FedAvg done -- total communications: {total_comm}")
        return history
