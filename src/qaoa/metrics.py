"""Metric helpers for QAOA experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

Bitstring = str


@dataclass
class SampleAnalysis:
    energy_mean: float
    energy_std: float
    success_probability: float
    approximation_ratio: float
    best_bitstring: Bitstring
    best_energy: float
    worst_bitstring: Bitstring
    worst_energy: float
    energy_distribution: Dict[float, float]


def bitstring_to_z(bitstring: Bitstring) -> List[int]:
    return [1 if b == "0" else -1 for b in bitstring]


def compute_energy_from_counts(
    counts: Dict[str, int],
    cost_fn,
) -> Tuple[float, float, Dict[float, float]]:
    total = sum(counts.values())
    energies: List[float] = []
    distribution: Dict[float, float] = {}
    for raw, freq in counts.items():
        bitstring = raw[::-1]
        energy = cost_fn(bitstring)
        energies.extend([energy] * freq)
        distribution[energy] = distribution.get(energy, 0.0) + freq / total
    arr = np.array(energies, dtype=float)
    return float(arr.mean()), float(arr.std()), distribution


def compute_success_probability(counts: Dict[str, int], ground_states: Iterable[Bitstring]) -> float:
    total = sum(counts.values())
    gs = set(ground_states)
    success = 0.0
    for raw, freq in counts.items():
        if raw[::-1] in gs:
            success += freq / total
    return success


def compute_approximation_ratio(energy: float, energy_min: float, energy_max: float) -> float:
    if energy_min == energy_max:
        return 1.0
    ratio = (energy - energy_max) / (energy_min - energy_max)
    return float(np.clip(ratio, 0.0, 1.0))


def analyze_samples(
    counts: Dict[str, int],
    cost_fn,
    ground_states: Iterable[Bitstring],
    energy_min: float,
    energy_max: float,
) -> SampleAnalysis:
    mean_energy, std_energy, distribution = compute_energy_from_counts(counts, cost_fn)
    success = compute_success_probability(counts, ground_states)
    ratio = compute_approximation_ratio(mean_energy, energy_min, energy_max)
    best_raw = min(counts.items(), key=lambda kv: (cost_fn(kv[0][::-1]), -kv[1]))[0]
    worst_raw = max(counts.items(), key=lambda kv: (cost_fn(kv[0][::-1]), kv[1]))[0]
    best_bitstring = best_raw[::-1]
    worst_bitstring = worst_raw[::-1]
    best_energy = cost_fn(best_bitstring)
    worst_energy = cost_fn(worst_bitstring)
    return SampleAnalysis(
        energy_mean=mean_energy,
        energy_std=std_energy,
        success_probability=success,
        approximation_ratio=ratio,
        best_bitstring=best_bitstring,
        best_energy=best_energy,
        worst_bitstring=worst_bitstring,
        worst_energy=worst_energy,
        energy_distribution=distribution,
    )
