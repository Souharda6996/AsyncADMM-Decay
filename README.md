# AsyncADMM-Decay

AsyncADMM-Decay: A decentralized fraud detection engine using asynchronous ADMM. Its decay-weighted penalty handles straggler nodes, reducing communication overhead by ≥60% vs FedAvg while maintaining ROC-AUC parity. A robust, staleness-aware solution for distributed consensus in high-latency networks.

---

**Theme-10**: Augmented Lagrangian Consensus with Decay-Weighted Penalty

## Mathematical Formulation

### Global Consensus Problem

We distribute the fraud-detection objective across **N** worker nodes, each
owning a local dataset partition. The global problem is:

```
minimise  Σᵢ fᵢ(xᵢ)
subject to  xᵢ = z  ∀ i ∈ {1, …, N}
```

where `fᵢ` is the regularised binary cross-entropy on node *i*, and `z` is the
global consensus variable maintained by the parameter server.

### Augmented Lagrangian

```
Lρ = Σᵢ [ fᵢ(xᵢ)  +  yᵢᵀ(xᵢ − z)  +  (ρ/2) ‖xᵢ − z‖² ]
```

Using scaled dual variables `uᵢ = yᵢ / ρ`:

```
Lρ = Σᵢ [ fᵢ(xᵢ)  +  (ρ/2) ‖xᵢ − z + uᵢ‖² ]  +  const
```

### ADMM Update Steps

| Step | Update | Description |
|------|--------|-------------|
| **Primal** | `xᵢᵏ⁺¹ = argmin fᵢ(x) + (ρ/2)‖x − zᵏ + uᵢᵏ‖²` | Proximal update (local SGD) |
| **Global** | `zᵏ⁺¹ = Σ wᵢ(xᵢ+uᵢ) / Σ wᵢ` | Weighted aggregation |
| **Dual**   | `uᵢᵏ⁺¹ = uᵢᵏ + xᵢᵏ⁺¹ − zᵏ⁺¹` | Dual ascent |

### Decay-Weighted Penalty (Novelty)

Straggler nodes report delayed updates. To down-weight stale information:

```
ρᵢ(τ) = ρ₀ · α ^ (t − τᵢ)
```

where `τᵢ` is the last round node *i* successfully communicated, and
`α ∈ (0, 1)` is the decay factor. Fresh updates keep full weight; stale ones
are exponentially dampened.

### Asynchronous Protocol

Each round, only a **random subset** of nodes participates (probability depends
on their latency class: *fast / medium / slow*). The server never blocks
waiting for all nodes — it aggregates whatever updates arrive, weighted by
their freshness.

## Architecture

```
┌─────────────┐
│ main.py     │  Entry point — CLI args, orchestration
└──────┬──────┘
       │
  ┌────▼────┐     ┌──────────────┐
  │ dataset │────▶│ Synthetic    │  100K TalkingData-like samples
  │  .py    │     │ click-stream │  non-IID Dirichlet split
  └─────────┘     └──────────────┘
       │
  ┌────▼──────────────────────────────────────┐
  │           Distributed Training            │
  │  ┌───────────┐       ┌───────────┐        │
  │  │async_admm │       │fedavg_    │        │
  │  │  .py      │       │baseline.py│        │
  │  └─────┬─────┘       └─────┬─────┘        │
  │        │                   │               │
  │  ┌─────▼─────┐     ┌──────▼──────┐        │
  │  │admm_node  │     │(inline nodes)│       │
  │  │  .py      │     └─────────────┘        │
  │  └─────┬─────┘                             │
  │  ┌─────▼──────┐                            │
  │  │admm_server │  Decay-weighted penalty    │
  │  │  .py       │  aggregation               │
  │  └────────────┘                            │
  └────────────────────────────────────────────┘
       │
  ┌────▼─────┐
  │evaluate  │  ROC-AUC comparison, comm overhead
  │  .py     │  plots → results/
  └──────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full comparison (default: 5 nodes, 100K samples, 50 rounds)
python main.py

# Customise
python main.py --rounds 80 --nodes 3 --samples 200000 --rho 1.5 --decay 0.9
```

## Output

The script produces:
- **Console summary table** with ROC-AUC and communication metrics
- **`results/roc_comparison.png`** — ROC curves for both methods
- **`results/training_curves.png`** — loss & AUC over rounds
- **`results/communication_overhead.png`** — bar chart of comm reduction

## Project Structure

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters & constants |
| `dataset.py` | Synthetic TalkingData generator + non-IID split |
| `model.py` | NumPy logistic regression + proximal solver |
| `admm_node.py` | Local ADMM worker (primal + dual updates) |
| `admm_server.py` | Central server (decay-weighted aggregation) |
| `async_admm.py` | Async ADMM orchestrator |
| `fedavg_baseline.py` | Synchronous FedAvg baseline |
| `evaluate.py` | Comparison metrics & plots |
| `utils.py` | Helpers (sigmoid, AUC, plotting) |
| `main.py` | CLI entry point |

## Key Results (Expected)

- **ROC-AUC parity**: Async ADMM ≈ Sync FedAvg (within ~2%)
- **Communication reduction**: ≥ 60% fewer node↔server messages
- **Convergence**: Both methods converge within 50 rounds
