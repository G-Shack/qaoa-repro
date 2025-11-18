"""Plotting helpers for energy histograms and penalty sweeps."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_energy_distribution(
    distribution: Dict[float, float],
    ground_energy: float,
    comparison: Dict[float, float] | None = None,
    save_path: str | None = None,
) -> None:
    energies = np.array(sorted(distribution))
    primary = np.array([distribution[e] for e in energies])
    width = 0.4 * np.min(np.diff(energies)) if len(energies) > 1 else 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(energies, primary, width=width, align="center", label="QAOA")
    if comparison:
        comp = np.array([comparison.get(float(e), 0.0) for e in energies])
        ax.bar(energies, comp, width=width, align="center", alpha=0.6, label="Comparison")
    ax.axvline(ground_energy, color="red", linestyle="--", label="Ground Energy")
    ax.set_xticks(energies)
    ax.set_xticklabels([f"{e:.2f}" for e in energies], rotation=45)
    ax.set_ylabel("Probability")
    ax.set_title("Energy Distribution of Samples")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_penalty_tradeoff(
    lambda_values: Iterable[float],
    feasibility: Iterable[float],
    approximation: Iterable[float],
    save_path: str | None = None,
) -> None:
    lambda_values = list(lambda_values)
    feasibility = list(feasibility)
    approximation = list(approximation)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    color = "tab:blue"
    ax1.set_xlabel("Penalty λ")
    ax1.set_ylabel("Feasibility Rate", color=color)
    ax1.plot(lambda_values, feasibility, marker="o", color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Approximation Ratio", color=color)
    ax2.plot(lambda_values, approximation, marker="s", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax1.set_xscale("log")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
