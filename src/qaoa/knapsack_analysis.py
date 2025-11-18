"""Specialized analysis workflows for the knapsack extension."""
from __future__ import annotations

from typing import Dict, Iterable

from .circuits import CostHamiltonian
from .experiments import ExperimentResult, run_depth_experiment
from ..problems.knapsack import (
    KnapsackInstance,
    evaluate_knapsack_samples,
    knapsack_to_ising,
)


def penalty_parameter_sweep(
    instance: KnapsackInstance,
    lambda_values: Iterable[float],
    p: int = 1,
    shots: int = 512,
    n_restarts: int = 3,
) -> Dict[float, Dict[str, float]]:
    lambda_values = list(lambda_values)
    summary: Dict[float, Dict[str, float]] = {}
    optimal_bitstring = "".join(str(bit) for bit in instance.optimal_selection)
    for penalty in lambda_values:
        h, J, _ = knapsack_to_ising(instance.values, instance.weights, instance.capacity, penalty)
        hamiltonian = CostHamiltonian(h, J, instance.n_items)
        experiment = run_depth_experiment(
            hamiltonian,
            ground_states=[optimal_bitstring],
            energy_min=-instance.optimal_value,
            energy_max=0.0,
            p=p,
            n_restarts=n_restarts,
            shots=shots,
        )
        eval_stats = evaluate_knapsack_samples(
            experiment.counts,
            instance.values,
            instance.weights,
            instance.capacity,
        )
        best_value = eval_stats["best_feasible"]["value"] if eval_stats["best_feasible"] else 0.0
        summary[float(penalty)] = {
            "feasibility_rate": eval_stats["feasibility_rate"],
            "approx_ratio": best_value / instance.optimal_value if instance.optimal_value else 0.0,
            "best_value": best_value,
            "avg_violation": eval_stats["avg_violation"],
        }
    return summary
