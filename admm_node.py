"""
ADMM local worker node.

Each node owns a partition of the training data, maintains local primal
variables xᵢ and scaled dual variables uᵢ, and solves a proximal
sub-problem independently.
"""

import numpy as np
from model import LogisticRegressionModel
import config


class ADMMNode:
    """Simulated local ADMM worker."""

    def __init__(self, node_id: int, X_train: np.ndarray, y_train: np.ndarray,
                 feature_dim: int, latency_profile: str):
        self.id = node_id
        self.X = X_train
        self.y = y_train
        self.n_samples = len(y_train)
        self.model = LogisticRegressionModel(feature_dim)

        # Local primal variable xᵢ  and scaled dual variable uᵢ
        self.x = np.zeros(feature_dim)
        self.u = np.zeros(feature_dim)

        # Latency simulation
        profile = config.NODE_LATENCY_PROFILES[latency_profile]
        self.participate_prob = profile["participate_prob"]
        self.delay_range = profile["delay_range"]
        self.latency_profile = latency_profile

        # Communication tracking
        self.comm_count = 0          # number of uploads to server
        self.last_update_round = 0   # last global round this node sent an update

    # ── Proximal update ──────────────────────────────────────────────────────
    def local_update(self, z: np.ndarray, rho: float) -> np.ndarray:
        """
        Solve the local sub-problem:
            xᵢᵏ⁺¹ = argmin fᵢ(xᵢ) + (ρ/2)‖xᵢ − z + uᵢ‖²
        Returns the updated local parameters.
        """
        self.x = self.model.proximal_update(
            self.X, self.y, self.x, z, self.u, rho
        )
        self.comm_count += 1
        return self.x.copy()

    # ── Dual update ──────────────────────────────────────────────────────────
    def dual_update(self, z: np.ndarray) -> None:
        """uᵢᵏ⁺¹ = uᵢᵏ + xᵢᵏ⁺¹ − zᵏ⁺¹"""
        self.u += self.x - z

    # ── Participation check (async simulation) ───────────────────────────────
    def should_participate(self, rng: np.random.Generator) -> bool:
        """Stochastically decide if this node participates this round."""
        return rng.random() < self.participate_prob

    def simulated_delay(self, current_round: int) -> int:
        """Return how many rounds behind this node is."""
        return current_round - self.last_update_round
