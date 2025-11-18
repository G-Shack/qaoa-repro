# qaoa-repro

End-to-end reproduction of the Willsch et al. QAOA benchmarks plus a knapsack penalty-extension study. The repo bundles:

- **Problem generators:** MaxCut, 2-SAT, and 0/1 knapsack (with exact ground states for small instances).
- **Reusable QAOA stack:** circuit builders, metric utilities, grid-search and gradient-free optimizers, and experiment orchestration helpers.
- **Visualization toolkit:** parameter landscapes, energy histograms, convergence plots, penalty trade-off curves, and dashboard assembly.
- **Classical baselines:** brute force, greedy, and dynamic-programming knapsack solvers for honest comparisons.
- **Notebooks:** ready-made workflows for reproducing Figures 2 & 6 analogs, knapsack penalty sweeps, and Table 1-style summaries.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running experiments

All reusable components live under `src/`. Typical workflow inside a notebook or script:

```python
from problems.maxcut import generate_maxcut_instance
from qaoa.circuits import maxcut_hamiltonian, CostHamiltonian
from qaoa.experiments import run_parameter_grid

instance = generate_maxcut_instance(n_nodes=6, edge_probability=0.5, seed=1)
hamiltonian = maxcut_hamiltonian(instance.weights, instance.n_nodes)
result = run_parameter_grid(
		hamiltonian,
		ground_states=instance.ground_state_bitstrings,
		energy_min=instance.ground_state_energy,
		energy_max=instance.energy_ceiling,
		gamma_values=np.linspace(0, 2*np.pi, 41),
		beta_values=np.linspace(0, np.pi, 41),
)
```

Use `run_depth_experiment` for `p>1` optimizations and `penalty_parameter_sweep` for the knapsack study.

## Tests

```bash
source .venv/bin/activate
python -m pytest
```

The suite covers problem generators, energy metrics, knapsack penalties, and basic circuit sanity checks.

## Repository structure

```
src/
	problems/        # MaxCut, 2-SAT, knapsack instances + helpers
	qaoa/            # Circuits, metrics, optimization, experiments
	classical/       # Greedy + DP knapsack baselines
	visualization/   # Heatmaps, histograms, dashboards
tests/             # Pytest suite described in guide
notebooks/         # Step-by-step reproductions and analysis
```

Refer to the guide inside the notebooks for the full reproduction/extension plan.
