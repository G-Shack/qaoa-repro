"""Statistical validation helpers for repeated QAOA runs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .experiments import run_depth_experiment


@dataclass
class StatisticalSummary:
    records: List[Dict[str, float]]
    mean_energy: float
    std_energy: float
    mean_success: float
    std_success: float
    mean_ratio: float
    std_ratio: float


def repeated_depth_experiments(
    hamiltonian,
    ground_states: Iterable[str],
    energy_min: float,
    energy_max: float,
    p: int,
    repeats: int = 10,
    n_restarts: int = 5,
    shots: int = 256,
) -> StatisticalSummary:
    """Run the same depth experiment multiple times and summarize statistics."""
    records: List[Dict[str, float]] = []
    for run_idx in range(repeats):
        experiment = run_depth_experiment(
            hamiltonian=hamiltonian,
            ground_states=ground_states,
            energy_min=energy_min,
            energy_max=energy_max,
            p=p,
            n_restarts=n_restarts,
            shots=shots,
        )
        records.append(
            {
                "run": float(run_idx),
                "energy": experiment.analysis.energy_mean,
                "success": experiment.analysis.success_probability,
                "ratio": experiment.analysis.approximation_ratio,
            }
        )
    energies = np.array([r["energy"] for r in records])
    successes = np.array([r["success"] for r in records])
    ratios = np.array([r["ratio"] for r in records])
    return StatisticalSummary(
        records=records,
        mean_energy=float(np.mean(energies)),
        std_energy=float(np.std(energies)),
        mean_success=float(np.mean(successes)),
        std_success=float(np.std(successes)),
        mean_ratio=float(np.mean(ratios)),
        std_ratio=float(np.std(ratios)),
    )
