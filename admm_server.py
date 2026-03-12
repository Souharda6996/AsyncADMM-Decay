"""
Central ADMM parameter server.

Maintains the global consensus variable z, aggregates local primal and dual
updates, and applies a decay-weighted penalty factor for straggler nodes.
"""

import numpy as np
import config


class ADMMServer:
    """Central parameter server for the ADMM consensus problem."""

    def __init__(self, feature_dim: int, n_nodes: int):
        self.d = feature_dim
        self.n_nodes = n_nodes

        # Global consensus variable z
        self.z = np.zeros(feature_dim)

        # Communication bookkeeping
        self.total_comm = 0      # total node→server messages received

    # ── Decay-weighted penalty ───────────────────────────────────────────────
    @staticmethod
    def decay_penalty(delay: int, rho_init: float = None,
                      alpha: float = None) -> float:
        """
        ρᵢ(τ) = ρ₀ · α^(delay)

        Nodes with higher delay get a smaller effective penalty, reducing
        the influence of stale updates.
        """
        rho_init = rho_init or config.RHO_INIT
        alpha = alpha or config.DECAY_ALPHA
        return rho_init * (alpha ** delay)

    # ── Global aggregation (z-update) ────────────────────────────────────────
    def aggregate(self, participating_nodes, current_round: int) -> np.ndarray:
        """
        zᵏ⁺¹ = Σᵢ wᵢ (xᵢ + uᵢ)  /  Σᵢ wᵢ

        where wᵢ = ρᵢ(delay_i) — decay-weighted by staleness.

        Parameters
        ----------
        participating_nodes : list[ADMMNode]
            Nodes that reported this round (subset in async mode).
        current_round : int
            Current global round number.

        Returns
        -------
        z : np.ndarray   (updated global consensus)
        """
        if not participating_nodes:
            return self.z.copy()

        weighted_sum = np.zeros(self.d)
        weight_total = 0.0

        for node in participating_nodes:
            delay = node.simulated_delay(current_round)
            w_i = self.decay_penalty(delay)
            weighted_sum += w_i * (node.x + node.u)
            weight_total += w_i
            self.total_comm += 1

        self.z = weighted_sum / weight_total
        return self.z.copy()
