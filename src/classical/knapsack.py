"""Classical baseline solvers for 0/1 knapsack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class KnapsackSolution:
    selection: List[int]
    value: float
    weight: float
    feasible: bool = True
    method: str = ""


def greedy_knapsack(values: Sequence[float], weights: Sequence[float], capacity: float) -> KnapsackSolution:
    order = sorted(
        range(len(values)),
        key=lambda i: values[i] / weights[i],
        reverse=True,
    )
    selection = [0] * len(values)
    total_weight = 0.0
    total_value = 0.0
    for idx in order:
        if total_weight + weights[idx] <= capacity + 1e-9:
            selection[idx] = 1
            total_weight += weights[idx]
            total_value += values[idx]
    return KnapsackSolution(selection, total_value, total_weight, True, "greedy")


def dynamic_programming_knapsack(values: Sequence[float], weights: Sequence[float], capacity: float) -> KnapsackSolution:
    values = list(map(float, values))
    weights = list(map(float, weights))
    cap = int(round(capacity))
    n = len(values)
    dp = np.zeros((n + 1, cap + 1))
    for i in range(1, n + 1):
        for w in range(cap + 1):
            dp[i, w] = dp[i - 1, w]
            weight = int(round(weights[i - 1]))
            if weight <= w:
                dp[i, w] = max(dp[i, w], dp[i - 1, w - weight] + values[i - 1])
    w = cap
    selection = [0] * n
    for i in range(n, 0, -1):
        if dp[i, w] != dp[i - 1, w]:
            selection[i - 1] = 1
            w -= int(round(weights[i - 1]))
    total_weight = sum(weights[i] * selection[i] for i in range(n))
    total_value = sum(values[i] * selection[i] for i in range(n))
    return KnapsackSolution(selection, total_value, total_weight, total_weight <= capacity + 1e-9, "dynamic_programming")


def comparison_table(
    optimal_value: float,
    optimal_method: str,
    baseline_solutions: List[KnapsackSolution],
    qaoa_rows: List[Dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for solution in baseline_solutions:
        rows.append(
            {
                "Method": solution.method,
                "Value": solution.value,
                "Time (ms)": np.nan,
                "Success Rate": 1.0,
                "Approx Ratio": solution.value / optimal_value if optimal_value else 0.0,
            }
        )
    rows.extend(qaoa_rows)
    return pd.DataFrame(rows)
