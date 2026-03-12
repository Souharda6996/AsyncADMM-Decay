"""
Main entry point — Asynchronous ADMM vs Synchronous FedAvg
for Distributed Fraud Detection.

Usage:
    python main.py [--rounds N] [--nodes N] [--samples N]
"""

import argparse
import sys
import time

import config
from dataset import generate_dataset
from async_admm import AsyncADMM
from fedavg_baseline import FedAvgSync
from evaluate import evaluate
from utils import log


def parse_args():
    p = argparse.ArgumentParser(
        description="Async ADMM vs Sync FedAvg for Distributed Fraud Detection")
    p.add_argument("--rounds", type=int, default=config.MAX_ROUNDS,
                   help="Number of communication rounds")
    p.add_argument("--nodes", type=int, default=config.NUM_NODES,
                   help="Number of decentralised nodes")
    p.add_argument("--samples", type=int, default=config.NUM_SAMPLES,
                   help="Total synthetic dataset size")
    p.add_argument("--rho", type=float, default=config.RHO_INIT,
                   help="Initial ADMM penalty parameter rho_0")
    p.add_argument("--decay", type=float, default=config.DECAY_ALPHA,
                   help="Decay factor alpha for straggler penalty")
    return p.parse_args()


def main():
    args = parse_args()

    # Override config with CLI args
    config.MAX_ROUNDS = args.rounds
    config.FEDAVG_MAX_ROUNDS = args.rounds
    config.NUM_SAMPLES = args.samples
    config.RHO_INIT = args.rho
    config.DECAY_ALPHA = args.decay

    # Adjust node count if requested (re-slice profile assignment)
    if args.nodes != config.NUM_NODES:
        profiles = ["fast", "fast", "medium", "medium", "slow"]
        config.NUM_NODES = args.nodes
        config.NODE_PROFILE_ASSIGNMENT = (profiles * (args.nodes // len(profiles) + 1))[:args.nodes]

    print("=" * 62)
    print("  Asynchronous ADMM for Distributed Fraud Detection")
    print("  Theme-10: Augmented Lagrangian Consensus with Decay Penalty")
    print("=" * 62)
    print(f"  Nodes       : {config.NUM_NODES}")
    print(f"  Samples     : {config.NUM_SAMPLES:,}")
    print(f"  Rounds      : {config.MAX_ROUNDS}")
    print(f"  rho_0       : {config.RHO_INIT}")
    print(f"  Decay alpha : {config.DECAY_ALPHA}")
    print(f"  Features    : {config.NUM_FEATURES}")
    print("=" * 62 + "\n")

    # ── 1. Generate synthetic dataset ────────────────────────────────────────
    t0 = time.perf_counter()
    log("Generating synthetic TalkingData-style dataset …")
    node_data, X_test, y_test, feature_dim = generate_dataset()
    for i, (X, y) in enumerate(node_data):
        fraud_pct = y.mean() * 100
        log(f"  Node {i}: {len(y):>6,} samples  |  fraud rate {fraud_pct:.2f}%")
    log(f"  Test set: {len(y_test):,} samples  |  fraud rate {y_test.mean()*100:.2f}%")

    # ── 2. Train Async ADMM ──────────────────────────────────────────────────
    print()
    admm = AsyncADMM(node_data, feature_dim)
    admm_history = admm.train(X_test, y_test, max_rounds=config.MAX_ROUNDS)

    # ── 3. Train Sync FedAvg ─────────────────────────────────────────────────
    print()
    fedavg = FedAvgSync(node_data, feature_dim)
    fedavg_history = fedavg.train(X_test, y_test, max_rounds=config.FEDAVG_MAX_ROUNDS)

    # ── 4. Evaluate & compare ────────────────────────────────────────────────
    print()
    results = evaluate(admm_history, fedavg_history, X_test, y_test)

    elapsed = time.perf_counter() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    return 0 if results["reduction_pct"] >= 60 else 1


if __name__ == "__main__":
    sys.exit(main())
