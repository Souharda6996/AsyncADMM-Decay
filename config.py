"""
Configuration for Asynchronous ADMM Distributed Fraud Detection.

All global constants and hyperparameters are defined here for easy tuning.
"""

import numpy as np

# ─── Random seed ───────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Dataset parameters ───────────────────────────────────────────────────────
NUM_SAMPLES = 100_000          # Total synthetic samples
FRAUD_RATIO = 0.02             # ~2 % positive (fraud) class
NUM_RAW_FEATURES = 7           # ip, app, device, os, channel, click_hour, click_day
NUM_ENGINEERED_FEATURES = 5    # count-based & interaction features
NUM_FEATURES = NUM_RAW_FEATURES + NUM_ENGINEERED_FEATURES  # 12
TEST_RATIO = 0.20              # Global test split

# ─── Distributed infrastructure ───────────────────────────────────────────────
NUM_NODES = 5                  # Simulated decentralised workers
NON_IID_ALPHA = 0.5            # Dirichlet α for non-IID partition (lower → more skew)

# ─── ADMM hyperparameters ─────────────────────────────────────────────────────
RHO_INIT = 1.5                 # Initial augmented-Lagrangian penalty rho_0
DECAY_ALPHA = 0.80             # Decay factor alpha for straggler penalty: rho_i = rho_0 * alpha^delay
MAX_ROUNDS = 60                # Global communication rounds
LOCAL_EPOCHS = 15              # SGD steps per local proximal update
LOCAL_LR = 0.10                # Learning rate for local proximal solver
L2_LAMBDA = 1e-4               # L2 regularisation strength

# ─── Async simulation ─────────────────────────────────────────────────────────
# Each node is assigned a latency class.  Probability that a node participates
# in a given async round depends on its class.
NODE_LATENCY_PROFILES = {
    "fast":     {"participate_prob": 0.60, "delay_range": (0, 1)},
    "medium":   {"participate_prob": 0.35, "delay_range": (1, 3)},
    "slow":     {"participate_prob": 0.15, "delay_range": (3, 6)},
}
# Assignment of latency profiles to the 5 nodes
NODE_PROFILE_ASSIGNMENT = ["fast", "fast", "medium", "medium", "slow"]

# ─── Federated Averaging baseline ─────────────────────────────────────────────
FEDAVG_MAX_ROUNDS = MAX_ROUNDS
FEDAVG_LOCAL_EPOCHS = 5               # FedAvg uses fewer local epochs (standard)
FEDAVG_LOCAL_LR = LOCAL_LR

# ─── Output / plotting ────────────────────────────────────────────────────────
OUTPUT_DIR = "results"
PLOT_DPI = 150
