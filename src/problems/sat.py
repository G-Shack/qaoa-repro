"""Random 2-SAT instance generation and Ising conversion."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np

Bitstring = str
Clause = Tuple[Tuple[int, bool], Tuple[int, bool]]


@dataclass
class TwoSATInstance:
    n_variables: int
    clauses: List[Clause]
    h: Dict[int, float]
    J: Dict[Tuple[int, int], float]
    ground_energy: float
    ground_states: List[Bitstring]


def evaluate_clause(bitstring: Bitstring, clause: Clause) -> bool:
    (var1, neg1), (var2, neg2) = clause
    val1 = bitstring[var1] == "1"
    val2 = bitstring[var2] == "1"
    if neg1:
        val1 = not val1
    if neg2:
        val2 = not val2
    return val1 or val2


def evaluate_2sat_assignment(bitstring: Bitstring, clauses: Sequence[Clause]) -> bool:
    return all(evaluate_clause(bitstring, clause) for clause in clauses)


def _clause_to_coeffs(clause: Clause) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    (var1, neg1), (var2, neg2) = clause
    s1 = -1.0 if neg1 else 1.0
    s2 = -1.0 if neg2 else 1.0
    h = {var1: -s1 / 4.0, var2: -s2 / 4.0}
    J = {(min(var1, var2), max(var1, var2)): s1 * s2 / 4.0}
    const = 0.25
    return h, J, const


def _enumerate_ground_states(
    n_variables: int,
    clauses: Sequence[Clause] | None,
    h,
    J,
    require_satisfaction: bool,
) -> Tuple[float, List[Bitstring]]:
    best_energy = None
    best_bitstrings: List[Bitstring] = []
    for bits in product("01", repeat=n_variables):
        bitstring = "".join(bits)
        if require_satisfaction and clauses is not None and not evaluate_2sat_assignment(bitstring, clauses):
            continue
        z = [1 if b == "0" else -1 for b in bitstring]
        energy = 0.0
        for i, coeff in h.items():
            energy += coeff * z[i]
        for (i, j), coeff in J.items():
            energy += coeff * z[i] * z[j]
        if best_energy is None or energy < best_energy - 1e-9:
            best_energy = energy
            best_bitstrings = [bitstring]
        elif abs(energy - best_energy) <= 1e-9:
            best_bitstrings.append(bitstring)
    assert best_energy is not None
    return best_energy, best_bitstrings


def generate_2sat_instance(
    n_variables: int,
    n_clauses: int,
    ensure_satisfiable: bool = True,
    seed: int | None = None,
) -> TwoSATInstance:
    if n_variables < 2:
        raise ValueError("Need at least two variables")
    if n_clauses < 1:
        raise ValueError("Need at least one clause")

    rng = np.random.default_rng(seed)
    while True:
        clauses: List[Clause] = []
        for _ in range(n_clauses):
            var1 = rng.integers(0, n_variables)
            var2 = rng.integers(0, n_variables)
            while var2 == var1:
                var2 = rng.integers(0, n_variables)
            neg1 = bool(rng.integers(0, 2))
            neg2 = bool(rng.integers(0, 2))
            clauses.append(((int(var1), neg1), (int(var2), neg2)))
        if not ensure_satisfiable or any(
            evaluate_2sat_assignment(format(idx, f"0{n_variables}b"), clauses)
            for idx in range(2**n_variables)
        ):
            break
    h: Dict[int, float] = {}
    J: Dict[Tuple[int, int], float] = {}
    const = 0.0
    for clause in clauses:
        clause_h, clause_J, clause_const = _clause_to_coeffs(clause)
        const += clause_const
        for idx, coeff in clause_h.items():
            h[idx] = h.get(idx, 0.0) + coeff
        for (i, j), coeff in clause_J.items():
            key = (min(i, j), max(i, j))
            J[key] = J.get(key, 0.0) + coeff
    ground_energy, ground_states = _enumerate_ground_states(
        n_variables,
        clauses,
        h,
        J,
        require_satisfaction=ensure_satisfiable,
    )
    return TwoSATInstance(
        n_variables=n_variables,
        clauses=clauses,
        h=h,
        J=J,
        ground_energy=ground_energy,
        ground_states=ground_states,
    )
