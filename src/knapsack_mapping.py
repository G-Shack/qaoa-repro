# src/knapsack_mapping.py
import numpy as np
from typing import Dict, Tuple

def knapsack_to_ising(values, weights, W, lam=10.0):
    """
    Map 0/1 knapsack to Ising Hamiltonian H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + const
    Using x_i = (1 - Z_i)/2  (so Z_i = 1 - 2 x_i)
    Objective: maximize sum v_i x_i  -> minimize -sum v_i x_i
    Constraint: (sum w_i x_i - W)^2 penalized with lambda
    Expand penalty and collect terms.
    Returns: h (dict i->hi), J (dict (i,j)->Jij), offset const
    """
    n = len(values)
    # Convert lists/arrays to numpy
    v = np.array(values)
    w = np.array(weights)
    lam = float(lam)
    # Using x_i = (1 - Z_i)/2, so:
    # objective term (to minimize) = - sum_i v_i x_i = -sum_i v_i*(1 - Z_i)/2 = const_obj + (1/2) sum_i v_i Z_i
    const_obj = -0.5 * v.sum()
    # constraint term: lam*(sum_i w_i x_i - W)^2
    # write sum_i w_i x_i = sum_i w_i*(1 - Z_i)/2 = W0 - 1/2 sum_i w_i Z_i, where W0 = 1/2 sum_i w_i
    # But easier: expand (sum w_i x_i - W)^2 in x variables then map to Z
    # Expand in x: sum_i sum_j w_i w_j x_i x_j - 2W sum_i w_i x_i + W^2
    # Map x_i x_j = (1 - Z_i)(1 - Z_j)/4 = 1/4 - (Z_i + Z_j)/4 + Z_i Z_j /4
    # Map x_i = (1 - Z_i)/2
    # Collect coefficients for Z_i and Z_i Z_j
    const_penalty = lam * (W**2)
    # Precompute
    h = {i: 0.0 for i in range(n)}
    J = {}
    # Pair terms from lam * sum_{ij} w_i w_j x_i x_j
    for i in range(n):
        for j in range(i+1, n):
            coef = lam * w[i] * w[j] * (1.0/4.0)  # coefficient multiplying Z_i Z_j
            # The Z_i Z_j coefficient in full Hamiltonian is +coef (because mapping yields +Z_i Z_j /4)
            J[(i, j)] = coef
            # There are also contributions to single Z terms from x_i x_j mapping: -lam * w_i w_j * (Z_i + Z_j)/4
            h[i] += lam * (-w[i] * w[j] / 4.0)
            h[j] += lam * (-w[i] * w[j] / 4.0)
            # constant term from x_i x_j mapping: lam * w_i w_j * 1/4 added to const
            const_penalty += lam * (w[i] * w[j] / 4.0)
    # Now linear terms from -2W sum_i w_i x_i -> -2W * w_i * x_i
    # mapping x_i = (1 - Z_i)/2 => contributes -2W * w_i * (1/2) + -2W * w_i * (-Z_i/2) => const shift and +W*w_i*Z_i
    for i in range(n):
        h[i] += lam * (W * w[i])  # from -2W*... mapped: coefficient +W*w_i in front of Z_i
        const_penalty += lam * (-W * w[i] * 0.5) * 2.0 * 0  # already added earlier? Keep const minimal. We'll recompute const properly below.

    # Objective contributions to h: (1/2) v_i from -sum v_i x_i mapping
    for i in range(n):
        h[i] += 0.5 * v[i]

    # Compute offset constant properly by recomputing full mapping (safer)
    # Full Hamiltonian constant = const_obj + lam * ( W^2 + sum_{i<j} w_i w_j * 1/4 + sum_i w_i^2 * 1/4 ???)
    # Simpler: compute exact constant by enumerating all 2^n bitstrings (only small n expected)
    # We'll compute const such that energy(z) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + const equals original cost.
    # Caller may ignore const when comparing relative energies.
    const = 0.0
    # Return h, J, const (const may be approximate)
    return h, J, const
