"""Experiment orchestration utilities for QAOA studies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .circuits import CostHamiltonian, QAOACircuit
from .metrics import SampleAnalysis, analyze_samples
from .optimization import GridSearchResult, OptimizationRun, grid_search_p1, multiple_restarts


@dataclass
class ExperimentResult:
    qaoa: QAOACircuit
    parameters: Dict[str, np.ndarray]
    analysis: SampleAnalysis
    counts: Dict[str, int]
    grid: GridSearchResult | None = None
    optimization_runs: List[OptimizationRun] | None = None


def run_parameter_grid(
    hamiltonian: CostHamiltonian,
    ground_states: Iterable[str],
    energy_min: float,
    energy_max: float,
    gamma_values: Sequence[float],
    beta_values: Sequence[float],
    shots: int = 1024,
) -> ExperimentResult:
    qaoa = QAOACircuit(hamiltonian, p=1)
    grid = grid_search_p1(
        qaoa=qaoa,
        ground_states=ground_states,
        energy_min=energy_min,
        energy_max=energy_max,
        gamma_values=gamma_values,
        beta_values=beta_values,
        shots=shots,
    )
    optimal_gamma, optimal_beta = grid.best_energy_params
    counts = qaoa.sample([optimal_gamma], [optimal_beta], shots=shots * 4)
    analysis = analyze_samples(counts, qaoa.cost.energy, ground_states, energy_min, energy_max)
    return ExperimentResult(
        qaoa=qaoa,
        parameters={"gamma": np.array([optimal_gamma]), "beta": np.array([optimal_beta])},
        analysis=analysis,
        counts=counts,
        grid=grid,
        optimization_runs=None,
    )


def run_depth_experiment(
    hamiltonian: CostHamiltonian,
    ground_states: Iterable[str],
    energy_min: float,
    energy_max: float,
    p: int,
    n_restarts: int = 5,
    objective: str = "energy",
    shots: int = 512,
) -> ExperimentResult:
    qaoa = QAOACircuit(hamiltonian, p=p)
    best_run, runs = multiple_restarts(
        qaoa=qaoa,
        ground_states=ground_states,
        energy_min=energy_min,
        energy_max=energy_max,
        n_restarts=n_restarts,
        objective=objective,
        shots=shots,
    )
    gammas = best_run.gammas
    betas = best_run.betas
    counts = qaoa.sample(gammas, betas, shots=shots * 8)
    analysis = analyze_samples(counts, qaoa.cost.energy, ground_states, energy_min, energy_max)
    return ExperimentResult(
        qaoa=qaoa,
        parameters={"gamma": gammas, "beta": betas},
        analysis=analysis,
        counts=counts,
        grid=None,
        optimization_runs=runs,
    )
