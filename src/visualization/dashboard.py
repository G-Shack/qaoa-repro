"""Composite dashboards summarizing experiments."""
from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_results_dashboard(
    maxcut_panels: Dict[str, np.ndarray],
    sat_panels: Dict[str, np.ndarray],
    knapsack_metrics: Dict[str, List[float]],
    comparison_table: pd.DataFrame,
    save_path: str,
) -> None:
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(maxcut_panels["success"], origin="lower", aspect="auto", cmap="viridis")
    ax.set_title("MaxCut Success")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(maxcut_panels["energy"], origin="lower", aspect="auto", cmap="coolwarm")
    ax.set_title("MaxCut Energy")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(maxcut_panels.get("p_values", []), maxcut_panels.get("success_vs_p", []), marker="o")
    ax.set_title("MaxCut Depth Study")
    ax.set_xlabel("p")
    ax.set_ylabel("Success")

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(sat_panels["success"], origin="lower", aspect="auto", cmap="viridis")
    ax.set_title("2-SAT Success")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(sat_panels["energy"], origin="lower", aspect="auto", cmap="coolwarm")
    ax.set_title("2-SAT Energy")

    ax = fig.add_subplot(gs[1, 2])
    ax.bar(range(len(sat_panels.get("instances", []))), sat_panels.get("instance_success", []))
    ax.set_title("2-SAT Instance Comparison")

    ax = fig.add_subplot(gs[2, 0])
    lambdas = knapsack_metrics.get("lambda", [])
    feasibility = knapsack_metrics.get("feasibility", [])
    ax.plot(lambdas, feasibility, marker="o")
    ax.set_xscale("log")
    ax.set_title("Knapsack Feasibility")
    ax.set_xlabel("λ")
    ax.set_ylabel("Feasibility")

    ax = fig.add_subplot(gs[2, 1])
    approximation = knapsack_metrics.get("approximation", [])
    ax.plot(lambdas, approximation, marker="s", color="red")
    ax.set_xscale("log")
    ax.set_title("Knapsack Approximation")
    ax.set_xlabel("λ")
    ax.set_ylabel("Approximation")

    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    table = ax.table(
        cellText=np.round(comparison_table.values, 3),
        colLabels=comparison_table.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.suptitle("QAOA Benchmarking Dashboard", fontsize=20, fontweight="bold")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
