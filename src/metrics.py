# src/metrics.py
import numpy as np
from typing import Dict, Tuple

def ground_state_energy_by_enumeration(h, J):
    """
    For small n, brute-force enumerate all bitstrings to find ground energy and argmin.
    Returns E_min, argmin_bitstrings (list).
    """
    n = max(max(pair) for pair in list(J.keys())) + 1 if J else max(h.keys())+1
    E_min = None
    best = []
    for idx in range(2**n):
        bstr = format(idx, '0{}b'.format(n))[::-1]  # match our bit order
        z = [1 if ch == '0' else -1 for ch in bstr]
        E = 0.0
        for i, hi in h.items():
            E += hi * z[i]
        for (i, j), Jij in J.items():
            E += Jij * z[i] * z[j]
        if E_min is None or E < E_min - 1e-9:
            E_min = E
            best = [bstr]
        elif abs(E - E_min) < 1e-9:
            best.append(bstr)
    return E_min, best

def success_probability_from_counts(counts, ground_bitstrings):
    """
    counts: dict bitstring->count (Qiskit raw string), ground_bitstrings: list of bitstrings in our reversed order
    Return probability of sampling any ground bitstring.
    """
    total = sum(counts.values())
    prob = 0.0
    for raw, c in counts.items():
        bstr = raw[::-1]
        if bstr in ground_bitstrings:
            prob += c/total
    return prob

def approximation_ratio(E, E_min, E_max):
    # normalized to [0,1] as in paper: r = (E - E_max)/(E_min - E_max)
    return (E - E_max) / (E_min - E_max)
