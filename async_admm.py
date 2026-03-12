"""
Asynchronous ADMM Orchestrator.

Coordinates local ADMM nodes and the central parameter server across
simulated asynchronous communication rounds.  Straggler nodes are handled
with a decay-weighted penalty in the aggregation step.
"""

import numpy as np
from admm_node import ADMMNode
from admm_server import ADMMServer
from model import LogisticRegressionModel
from utils import compute_roc_auc, log
import config


class AsyncADMM:
    """
    Runs the full Asynchronous ADMM training loop.

    Mathematical formulation
    ────────────────────────
    Global consensus problem:
        minimize  Σᵢ fᵢ(xᵢ)   subject to  xᵢ = z  ∀i

    Augmented Lagrangian:
        Lρ = Σᵢ [ fᵢ(xᵢ) + yᵢᵀ(xᵢ − z) + (ρ/2)‖xᵢ − z‖² ]

    Updates per round (asynchronous):
        1. Primal   :  xᵢ ← proximal_update(z, uᵢ, ρᵢ)   (only active nodes)
        2. Aggregate:  z  ← Σ wᵢ(xᵢ + uᵢ) / Σ wᵢ        wᵢ = ρ₀·α^delay
        3. Dual     :  uᵢ ← uᵢ + xᵢ − z                  (only active nodes)
    """

    def __init__(self, node_data, feature_dim: int):
        """
        Parameters
        ----------
        node_data : list[(X_train, y_train)]  per node
        feature_dim : int
        """
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(config.RANDOM_SEED + 100)

        # Create nodes
        self.nodes = []
        for i, (X, y) in enumerate(node_data):
            profile = config.NODE_PROFILE_ASSIGNMENT[i]
            node = ADMMNode(i, X, y, feature_dim, profile)
            self.nodes.append(node)

        # Central server
        self.server = ADMMServer(feature_dim, len(self.nodes))
        self.model = LogisticRegressionModel(feature_dim)

    # ── Training loop ────────────────────────────────────────────────────────
    def train(self, X_test: np.ndarray, y_test: np.ndarray,
              max_rounds: int = None):
        """
        Run the asynchronous ADMM training loop.

        Returns
        -------
        history : dict  with keys 'rounds', 'loss', 'auc', 'total_comm'
        """
        max_rounds = max_rounds or config.MAX_ROUNDS
        history = {"rounds": [], "loss": [], "auc": []}

        log("─── Async ADMM Training ───")

        for rnd in range(1, max_rounds + 1):
            # 1. Determine which nodes participate this round (async)
            active_nodes = [n for n in self.nodes
                            if n.should_participate(self.rng)]

            # Guarantee at least one node participates
            if not active_nodes:
                active_nodes = [self.rng.choice(self.nodes)]

            # 2. Each active node solves its local proximal sub-problem
            for node in active_nodes:
                delay = node.simulated_delay(rnd)
                rho_eff = self.server.decay_penalty(delay)
                node.local_update(self.server.z, rho_eff)
                node.last_update_round = rnd

            # 3. Server aggregates (z-update) with decay-weighted penalty
            z_new = self.server.aggregate(active_nodes, rnd)

            # 4. Dual variable update for active nodes
            for node in active_nodes:
                node.dual_update(z_new)

            # 5. Evaluate on global test set using consensus z
            test_loss = self.model.loss(X_test, y_test, self.server.z)
            y_scores = self.model.predict_proba(X_test, self.server.z)
            auc = compute_roc_auc(y_test, y_scores)

            history["rounds"].append(rnd)
            history["loss"].append(test_loss)
            history["auc"].append(auc)

            if rnd % 10 == 0 or rnd == 1:
                active_ids = [n.id for n in active_nodes]
                log(f"  Round {rnd:3d} | active={active_ids} | "
                    f"loss={test_loss:.4f} | AUC={auc:.4f} | "
                    f"comm={self.server.total_comm}")

        history["total_comm"] = self.server.total_comm
        history["final_z"] = self.server.z.copy()
        log(f"  [OK] Async ADMM done -- total communications: {self.server.total_comm}")
        return history
