"""Knapsack problem generation and penalty-based Ising conversion."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np

Bitstring = str


@dataclass
class KnapsackInstance:
    values: List[float]
    weights: List[float]
    capacity: float
    optimal_value: float
    optimal_selection: List[int]
    optimal_weight: float

    @property
    def n_items(self) -> int:
        return len(self.values)


def generate_knapsack_instance(
    n_items: int,
    capacity_ratio: float = 0.5,
    value_range: Tuple[int, int] = (5, 20),
    weight_range: Tuple[int, int] = (1, 5),
    seed: int | None = None,
) -> KnapsackInstance:
    if n_items < 2:
        raise ValueError("Need at least two items")
    rng = np.random.default_rng(seed)
    values = rng.integers(value_range[0], value_range[1] + 1, size=n_items).astype(float).tolist()
    weights = rng.integers(weight_range[0], weight_range[1] + 1, size=n_items).astype(float).tolist()
    capacity = capacity_ratio * float(sum(weights))
    best_selection, best_value, best_weight = brute_force_knapsack(values, weights, capacity)
    return KnapsackInstance(
        values=values,
        weights=weights,
        capacity=capacity,
        optimal_value=best_value,
        optimal_selection=best_selection,
        optimal_weight=best_weight,
    )


def brute_force_knapsack(values: Sequence[float], weights: Sequence[float], capacity: float) -> Tuple[List[int], float, float]:
    n_items = len(values)
    best_value = -float("inf")
    best_selection: List[int] = []
    best_weight = 0.0
    for config in range(2 ** n_items):
        selection = [(config >> i) & 1 for i in range(n_items)]
        total_weight = sum(weights[i] * selection[i] for i in range(n_items))
        if total_weight - capacity > 1e-9:
            continue
        total_value = sum(values[i] * selection[i] for i in range(n_items))
        if total_value > best_value + 1e-9:
            best_value = total_value
            best_selection = selection
            best_weight = total_weight
    if not best_selection:
        best_selection = [0] * n_items
        best_value = 0.0
        best_weight = 0.0
    return best_selection, best_value, best_weight


def evaluate_knapsack_solution(selection: Sequence[int], values: Sequence[float], weights: Sequence[float], capacity: float) -> Dict[str, float | bool]:
    total_weight = float(sum(weights[i] * selection[i] for i in range(len(selection))))
    total_value = float(sum(values[i] * selection[i] for i in range(len(selection))))
    feasible = total_weight <= capacity + 1e-9
    return {
        "value": total_value,
        "weight": total_weight,
        "feasible": feasible,
        "constraint_violation": max(0.0, total_weight - capacity),
    }


def knapsack_to_ising(values: Sequence[float], weights: Sequence[float], capacity: float, penalty_lambda: float) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    n_items = len(values)
    v = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    lam = float(penalty_lambda)
    B = 0.5 * np.sum(w) - capacity
    h = {}
    for i in range(n_items):
        h[i] = 0.5 * v[i] - lam * B * w[i]
    J: Dict[Tuple[int, int], float] = {}
    for i in range(n_items):
        for j in range(i + 1, n_items):
            J[(i, j)] = 0.5 * lam * w[i] * w[j]
    constant = -0.5 * np.sum(v) + lam * (B**2 + 0.25 * np.sum(w**2))
    return h, J, float(constant)


def evaluate_knapsack_samples(
    counts: Dict[str, int],
    values: Sequence[float],
    weights: Sequence[float],
    capacity: float,
) -> Dict[str, object]:
    total_shots = sum(counts.values())
    feasible_shots = 0
    best_feasible = None
    infeasible = []
    feasible = []
    for raw, freq in counts.items():
        bitstring = raw[::-1]
        selection = [1 if b == "1" else 0 for b in bitstring]
        stats = evaluate_knapsack_solution(selection, values, weights, capacity)
        stats["count"] = freq
        if stats["feasible"]:
            feasible_shots += freq
            feasible.append(stats)
            if best_feasible is None or stats["value"] > best_feasible["value"]:
                best_feasible = {
                    "bitstring": bitstring,
                    "selection": selection,
                    "value": stats["value"],
                    "weight": stats["weight"],
                    "count": freq,
                }
        else:
            infeasible.append(stats)
    feasibility_rate = feasible_shots / total_shots if total_shots else 0.0
    avg_violation = float(np.mean([entry["constraint_violation"] for entry in infeasible])) if infeasible else 0.0
    return {
        "feasibility_rate": feasibility_rate,
        "best_feasible": best_feasible,
        "feasible_samples": feasible,
        "infeasible_samples": infeasible,
        "avg_violation": avg_violation,
    }
