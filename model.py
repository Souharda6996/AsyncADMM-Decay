"""
Logistic regression model implemented in pure NumPy.

Provides loss, gradient, prediction, and the proximal-update solver
used by each ADMM local node.
"""

import numpy as np
from utils import sigmoid
import config


class LogisticRegressionModel:
    """Stateless model utilities -- weights are stored externally."""

    MAX_WEIGHT = 20.0       # weight clipping bound
    MAX_GRAD_NORM = 5.0     # gradient clipping norm

    def __init__(self, feature_dim: int):
        self.d = feature_dim

    # -- Forward / predict ────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """P(y=1 | X, w) via sigmoid."""
        w_safe = np.clip(w, -self.MAX_WEIGHT, self.MAX_WEIGHT)
        return sigmoid(X @ w_safe)

    # -- Log-loss + L2 ────────────────────────────────────────────────────────
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """Binary cross-entropy + L2 regularisation."""
        p = np.clip(self.predict_proba(X, w), 1e-12, 1 - 1e-12)
        bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        l2 = 0.5 * config.L2_LAMBDA * np.dot(w, w)
        result = float(bce + l2)
        if np.isnan(result) or np.isinf(result):
            return 1e6  # fallback for degenerate cases
        return result

    # -- Gradient ─────────────────────────────────────────────────────────────
    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Gradient of the regularised log-loss with clipping."""
        p = self.predict_proba(X, w)
        grad = X.T @ (p - y) / len(y) + config.L2_LAMBDA * w
        # Replace any NaN/inf in gradient
        grad = np.nan_to_num(grad, nan=0.0, posinf=self.MAX_GRAD_NORM, neginf=-self.MAX_GRAD_NORM)
        # Clip gradient norm
        grad_norm = np.linalg.norm(grad)
        if grad_norm > self.MAX_GRAD_NORM:
            grad = grad * (self.MAX_GRAD_NORM / grad_norm)
        return grad

    # -- Proximal update (ADMM local sub-problem) ────────────────────────────
    def proximal_update(self, X: np.ndarray, y: np.ndarray,
                        w_init: np.ndarray, z: np.ndarray,
                        u: np.ndarray, rho: float,
                        lr: float = None, epochs: int = None) -> np.ndarray:
        """
        Solve:  argmin_x  f_i(x) + (rho/2) ||x - z + u||^2

        via gradient descent starting from w_init.
        """
        lr = lr or config.LOCAL_LR
        epochs = epochs or config.LOCAL_EPOCHS
        w = np.clip(w_init.copy(), -self.MAX_WEIGHT, self.MAX_WEIGHT)

        for _ in range(epochs):
            # Gradient of local loss
            g = self.gradient(X, y, w)
            # Gradient of the augmented-Lagrangian penalty
            g += rho * (w - z + u)
            # Clip combined gradient
            g = np.nan_to_num(g, nan=0.0, posinf=self.MAX_GRAD_NORM, neginf=-self.MAX_GRAD_NORM)
            g_norm = np.linalg.norm(g)
            if g_norm > self.MAX_GRAD_NORM:
                g = g * (self.MAX_GRAD_NORM / g_norm)
            w -= lr * g
            # Clip weights
            w = np.clip(w, -self.MAX_WEIGHT, self.MAX_WEIGHT)
        return w
