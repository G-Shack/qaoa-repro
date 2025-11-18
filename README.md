# QAOA Reproduction & Knapsack Study

Beginner-friendly playground for reproducing the MaxCut and 2-SAT benchmarks from Willsch et al. (2023) and extending them with a 0/1 knapsack penalty experiment. Everything you need—problem generators, QAOA helpers, visualizations, notebooks, and tests—lives in this repository.

## Highlights

- **Turnkey problem instances** with known ground states for MaxCut, 2-SAT, and knapsack.
- **Reusable QAOA stack** (circuit builders, experiment orchestration, metrics, and Nelder–Mead/grid optimizers).
- **Visualization suite** covering parameter landscapes, convergence traces, energy histograms, penalty sweeps, and a publication-style dashboard.
- **Classical baselines** (greedy + dynamic programming) for honest comparisons against QAOA.
- **Two curated notebooks** that mirror Figure 2/6 and Table 1 style results from the paper and capture the knapsack extension.
- **Pytest coverage** for core utilities so you can refactor with confidence.

## 1. Getting Started

These steps assume macOS/Linux with Python 3.11+ installed.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional (recommended) sanity check:

```bash
python -m pytest
```

This runs 13 targeted tests spanning problem generation, statistics helpers, and knapsack penalties.

## 2. Repository Tour

```
src/
	problems/         # MaxCut, 2-SAT, knapsack generators + evaluation helpers
	qaoa/             # Cost Hamiltonians, experiments, statistics, knapsack analysis
	classical/        # Greedy + dynamic-programming knapsack solvers and tables
	visualization/    # Landscapes, histograms, penalty plots, dashboard builder
notebooks/
	01_qaoa_p1_grid_maxcut_knapsack.ipynb   # p=1 grid scans + histogram + penalty sweep
	02_qaoa_p_gt1_neldermead.ipynb          # p>1 Nelder–Mead, statistics, dashboard
tests/                                     # Pytest suite validating key modules
figures/                                   # Saved outputs (e.g., qaoa_dashboard.png)
```

Each module uses absolute imports so notebooks can import directly once `src/` is on `sys.path` (already handled in the notebooks).

## 3. Core Building Blocks

| Area | Key Module | Description |
| ---- | ---------- | ----------- |
| Problem generation | `problems.maxcut`, `problems.sat`, `problems.knapsack` | Random instance factories with ground-state metadata and brute-force references. |
| QAOA plumbing | `qaoa.circuits`, `qaoa.experiments`, `qaoa.statistics` | Build cost/driver layers, run depth studies or grids, and summarize repeated experiments. |
| Knapsack extension | `qaoa.knapsack_analysis` | Converts knapsack to Ising, performs λ sweeps, and evaluates feasibility/approximation ratios. |
| Visualization | `visualization.landscapes`, `visualization.distributions`, `visualization.dashboard` | Produce heatmaps, histograms, convergence traces, and the combined benchmarking dashboard. |
| Classical baselines | `classical.knapsack` | Greedy + DP solvers plus helper to assemble comparison tables. |

## 4. Running the Notebooks

1. **01_qaoa_p1_grid_maxcut_knapsack.ipynb**
	 - Reproduces the p=1 parameter landscape for MaxCut and overlays an energy histogram (Figure 6 analogue).
	 - Runs a λ penalty sweep for the knapsack mapping, plotting feasibility vs. approximation.

2. **02_qaoa_p_gt1_neldermead.ipynb**
	 - Executes Nelder–Mead optimizations for `p ∈ {1,2,3}` on MaxCut and 2-SAT.
	 - Adds statistical validation (`repeated_depth_experiments`) and creates the multi-panel dashboard saved to `figures/qaoa_dashboard.png`.

Tips:
- Run cells sequentially; the notebooks manage Python paths and caching.
- If you modify source modules, restart the kernel or add `importlib.reload` (already demonstrated for the dashboard helper).

## 5. Script Usage Examples

You can reuse the components in standalone scripts. Example: run a depth-3 MaxCut study.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from problems.maxcut import generate_maxcut_instance
from qaoa.circuits import maxcut_hamiltonian
from qaoa.experiments import run_depth_experiment

instance = generate_maxcut_instance(6, 0.5, weighted=False, seed=11)
hamiltonian = maxcut_hamiltonian(instance.weights, instance.n_nodes)
result = run_depth_experiment(
		hamiltonian,
		ground_states=instance.ground_state_bitstrings,
		energy_min=instance.ground_state_energy,
		energy_max=instance.energy_ceiling,
		p=3,
		n_restarts=5,
		shots=256,
)
print(result.analysis.success_probability)
```

Swap in `CostHamiltonian` for other Ising problems and reuse `visualization.*` utilities to plot the outputs.

## 6. Generating Key Figures

- **Energy histogram (Figure 6 analogue):** Run Notebook 1 up to the histogram cell using `visualization.distributions.plot_energy_distribution`.
- **Nelder–Mead convergence traces:** Notebook 2 uses `visualization.landscapes.plot_optimization_progress` for the highest depth per problem.
- **Dashboard:** Execute the final cell of Notebook 2 to create `figures/qaoa_dashboard.png`, which combines parameter grids, depth trends, knapsack metrics, and classical/QAOA comparison table.

