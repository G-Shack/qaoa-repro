"""Parameter search helpers for QAOA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from .circuits import QAOACircuit
from .metrics import analyze_samples, compute_success_probability


@dataclass
class GridSearchResult:
    gamma_values: np.ndarray
    beta_values: np.ndarray
    energy_grid: np.ndarray
    success_grid: np.ndarray
    best_energy_params: Tuple[float, float]
    best_success_params: Tuple[float, float]


@dataclass
class OptimizationRun:
    gammas: np.ndarray
    betas: np.ndarray
    value: float
    success_probability: float
    history: List[Dict[str, float]]


def grid_search_p1(
    qaoa: QAOACircuit,
    ground_states: Iterable[str],
    energy_min: float,
    energy_max: float,
    gamma_values: Sequence[float],
    beta_values: Sequence[float],
    shots: int = 1024,
    progress_callback = None,
) -> GridSearchResult:
    beta_values = np.asarray(beta_values, dtype=float)
    gamma_values = np.asarray(gamma_values, dtype=float)
    energy_grid = np.zeros((len(beta_values), len(gamma_values)))
    success_grid = np.zeros_like(energy_grid)
    total_tasks = len(beta_values) * len(gamma_values)
    task = 0
    cost_fn = qaoa.cost.energy
    for i, beta in enumerate(beta_values):
        for j, gamma in enumerate(gamma_values):
            counts = qaoa.sample([gamma], [beta], shots=shots)
            analysis = analyze_samples(counts, cost_fn, ground_states, energy_min, energy_max)
            energy_grid[i, j] = analysis.energy_mean
            success_grid[i, j] = analysis.success_probability
            task += 1
            if progress_callback and task % 10 == 0:
                progress_callback(task, total_tasks)
    e_idx = np.unravel_index(np.argmin(energy_grid), energy_grid.shape)
    s_idx = np.unravel_index(np.argmax(success_grid), success_grid.shape)
    best_energy = (gamma_values[e_idx[1]], beta_values[e_idx[0]])
    best_success = (gamma_values[s_idx[1]], beta_values[s_idx[0]])
    return GridSearchResult(
        gamma_values=gamma_values,
        beta_values=beta_values,
        energy_grid=energy_grid,
        success_grid=success_grid,
        best_energy_params=best_energy,
        best_success_params=best_success,
    )


def optimize_parameters(
    qaoa: QAOACircuit,
    ground_states: Iterable[str],
    energy_min: float,
    energy_max: float,
    objective: str = "energy",
    initial_params: Sequence[float] | None = None,
    bounds: Sequence[Tuple[float, float]] | None = None,
    shots: int = 512,
    method: str = "Nelder-Mead",
    maxiter: int = 200,
) -> OptimizationRun:
    p = qaoa.p
    if initial_params is None:
        initial_params = np.random.uniform(0, 2 * np.pi, size=2 * p)
    params = np.asarray(initial_params, dtype=float)
    history: List[Dict[str, float]] = []

    def objective_fn(params_vec: np.ndarray) -> float:
        gammas = params_vec[:p]
        betas = params_vec[p:]
        counts = qaoa.sample(gammas, betas, shots=shots)
        energy = qaoa.expectation_from_counts(counts)
        success = compute_success_probability(counts, ground_states)
        history.append({
            "evaluation": len(history),
            "energy": energy,
            "success": success,
        })
        if objective == "energy":
            return energy
        elif objective == "success":
            return -success
        else:
            raise ValueError("objective must be 'energy' or 'success'")

    result = minimize(
        objective_fn,
        params,
        method=method,
        bounds=bounds,
        options={"maxiter": maxiter, "disp": False},
    )
    opt_params = result.x
    gammas = opt_params[:p]
    betas = opt_params[p:]
    final_counts = qaoa.sample(gammas, betas, shots=shots * 4)
    success = compute_success_probability(final_counts, ground_states)
    value = qaoa.expectation_from_counts(final_counts)
    history.append({
        "evaluation": len(history),
        "energy": value,
        "success": success,
    })
    return OptimizationRun(
        gammas=gammas,
        betas=betas,
        value=value,
        success_probability=success,
        history=history,
    )


def multiple_restarts(
    qaoa: QAOACircuit,
    ground_states: Iterable[str],
    energy_min: float,
    energy_max: float,
    n_restarts: int = 5,
    **kwargs,
) -> Tuple[OptimizationRun, List[OptimizationRun]]:
    runs: List[OptimizationRun] = []
    best_run: OptimizationRun | None = None
    for _ in range(n_restarts):
        run = optimize_parameters(qaoa, ground_states, energy_min, energy_max, **kwargs)
        runs.append(run)
        if best_run is None or run.value < best_run.value:
            best_run = run
    assert best_run is not None
    return best_run, runs
