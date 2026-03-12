"""
Synthetic TalkingData-style click-stream dataset generator.

Produces imbalanced binary classification data mimicking ad-click fraud,
partitioned across N nodes with non-IID splits.
"""

import numpy as np
from sklearn.model_selection import train_test_split

import config


def _generate_raw_features(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate 7 raw categorical-like features normalised to [0, 1]."""
    ip       = rng.integers(0, 5000, size=n).astype(np.float64) / 5000
    app      = rng.integers(0, 800, size=n).astype(np.float64) / 800
    device   = rng.integers(0, 4000, size=n).astype(np.float64) / 4000
    os       = rng.integers(0, 900, size=n).astype(np.float64) / 900
    channel  = rng.integers(0, 500, size=n).astype(np.float64) / 500
    hour     = rng.integers(0, 24, size=n).astype(np.float64) / 24
    day      = rng.integers(0, 7, size=n).astype(np.float64) / 7
    return np.column_stack([ip, app, device, os, channel, hour, day])


def _engineer_features(X_raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create 5 count / interaction features with strong fraud signal."""
    n = X_raw.shape[0]
    ip_count  = np.round(X_raw[:, 0] * 50).astype(int)
    app_count = np.round(X_raw[:, 1] * 30).astype(int)

    ip_click   = np.bincount(ip_count, minlength=51)[ip_count].astype(np.float64)
    app_click  = np.bincount(app_count, minlength=31)[app_count].astype(np.float64)
    ip_app     = ip_click * app_click                        # interaction
    hour_sin   = np.sin(2 * np.pi * X_raw[:, 5])             # cyclical hour
    hour_cos   = np.cos(2 * np.pi * X_raw[:, 5])

    eng = np.column_stack([ip_click, app_click, ip_app, hour_sin, hour_cos])
    # Normalise each engineered feature to [0, 1]
    for j in range(eng.shape[1]):
        col = eng[:, j]
        mn, mx = col.min(), col.max()
        if mx - mn > 1e-12:
            eng[:, j] = (col - mn) / (mx - mn)
    return eng


def _generate_labels(X: np.ndarray, fraud_ratio: float,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Generate fraud labels with strong, learnable correlation to features.
    Uses a fixed true-weight vector with large coefficients so
    that logistic regression can actually separate the classes.
    """
    n, d = X.shape
    # Fixed true coefficients with strong signal
    w_true = np.array([
        2.5,    # ip   - high ip correlates with fraud
       -1.8,    # app  - certain apps have less fraud
        0.5,    # device
        1.2,    # os
       -2.0,    # channel - certain channels have less fraud
        3.0,    # click_hour_norm - late-night clicks are more fraud
        0.8,    # click_day
        2.0,    # ip_click_count - high click count = fraud
        1.5,    # app_click_count
        2.5,    # ip_app interaction - strong fraud signal
       -0.5,    # hour_sin
        0.3,    # hour_cos
    ])
    # Truncate or pad to match feature dim
    if len(w_true) < d:
        w_true = np.concatenate([w_true, np.zeros(d - len(w_true))])
    else:
        w_true = w_true[:d]

    logits = X @ w_true
    # Shift logits so the overall positive rate matches fraud_ratio
    target_intercept = -np.log(1 / fraud_ratio - 1)
    logits = logits - logits.mean() + target_intercept
    # Add a small amount of noise for realism
    logits += rng.standard_normal(n) * 0.3
    probs = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))
    y = (rng.random(n) < probs).astype(np.float64)
    return y


def _non_iid_partition(X: np.ndarray, y: np.ndarray, n_nodes: int,
                       alpha: float, rng: np.random.Generator):
    """Dirichlet-based non-IID split across nodes."""
    n = len(y)
    proportions = rng.dirichlet(np.full(n_nodes, alpha))
    # Convert proportions to integer counts that sum to n
    counts = (proportions * n).astype(int)
    counts[-1] = n - counts[:-1].sum()       # adjust rounding

    indices = rng.permutation(n)
    splits = []
    start = 0
    for c in counts:
        idx = indices[start: start + c]
        splits.append((X[idx], y[idx]))
        start += c
    return splits


# --- Public API ---
def generate_dataset():
    """
    Returns
    -------
    node_data : list[tuple[np.ndarray, np.ndarray]]
        (X_train, y_train) per node
    X_test, y_test : np.ndarray
        Global held-out test set
    feature_dim : int
    """
    rng = np.random.default_rng(config.RANDOM_SEED)

    X_raw = _generate_raw_features(config.NUM_SAMPLES, rng)
    X_eng = _engineer_features(X_raw, rng)
    X = np.hstack([X_raw, X_eng])
    y = _generate_labels(X, config.FRAUD_RATIO, rng)

    # Global train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_RATIO, stratify=y, random_state=config.RANDOM_SEED
    )

    # Partition training data across nodes (non-IID)
    node_data = _non_iid_partition(X_train, y_train, config.NUM_NODES,
                                   config.NON_IID_ALPHA, rng)

    feature_dim = X.shape[1]
    return node_data, X_test, y_test, feature_dim
