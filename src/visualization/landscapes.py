"""Visualization helpers for parameter landscapes and convergence."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_parameter_landscape(
    gamma_values: np.ndarray,
    beta_values: np.ndarray,
    energy_grid: np.ndarray,
    success_grid: np.ndarray,
    problem_label: str,
    save_path: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    extent = [gamma_values[0] / np.pi, gamma_values[-1] / np.pi, beta_values[0] / np.pi, beta_values[-1] / np.pi]
    im0 = axes[0].imshow(success_grid, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[0].set_title("Success Probability")
    axes[0].set_xlabel("γ / π")
    axes[0].set_ylabel("β / π")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(energy_grid, origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
    axes[1].set_title("Energy Expectation")
    axes[1].set_xlabel("γ / π")
    fig.colorbar(im1, ax=axes[1])

    fig.suptitle(f"{problem_label} Parameter Landscape")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_optimization_progress(history: Iterable[Dict[str, float]], ground_energy: float, save_path: str | None = None) -> None:
    history = list(history)
    iterations = list(range(len(history)))
    energies = [entry["energy"] for entry in history]
    successes = [entry["success"] for entry in history]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(iterations, energies, marker="o")
    ax1.axhline(ground_energy, color="red", linestyle="--", label="Ground")
    ax1.set_ylabel("Energy")
    ax1.set_title("Energy Convergence")
    ax1.legend()

    ax2.plot(iterations, successes, marker="s", color="green")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Success Probability")
    ax2.set_title("Success Probability Evolution")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
