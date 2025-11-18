"""QAOA circuit builders for different problem families."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

Bitstring = str


@dataclass(frozen=True)
class CostHamiltonian:
    """Sparse Ising Hamiltonian H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j."""

    h: Dict[int, float]
    J: Dict[Tuple[int, int], float]
    n_qubits: int

    def energy(self, bitstring: Bitstring) -> float:
        z = [1 if b == "0" else -1 for b in bitstring]
        energy = 0.0
        for i, coeff in self.h.items():
            energy += coeff * z[i]
        for (i, j), coeff in self.J.items():
            energy += coeff * z[i] * z[j]
        return energy


class QAOACircuit:
    """Construct and simulate QAOA circuits for a given cost Hamiltonian."""

    def __init__(self, cost_hamiltonian: CostHamiltonian, p: int = 1):
        if p <= 0:
            raise ValueError("Circuit depth p must be positive")
        self.cost = cost_hamiltonian
        self.p = p

    @property
    def n_qubits(self) -> int:
        return self.cost.n_qubits

    def _apply_cost_layer(self, circuit: QuantumCircuit, gamma: float) -> None:
        for qubit, coeff in self.cost.h.items():
            if abs(coeff) > 1e-12:
                circuit.rz(2.0 * gamma * coeff, qubit)
        for (u, v), coeff in self.cost.J.items():
            if abs(coeff) > 1e-12:
                circuit.cx(u, v)
                circuit.rz(2.0 * gamma * coeff, v)
                circuit.cx(u, v)

    def _apply_mixer_layer(self, circuit: QuantumCircuit, beta: float) -> None:
        for qubit in range(self.n_qubits):
            circuit.rx(2.0 * beta, qubit)

    def build(self, gammas: Sequence[float], betas: Sequence[float], measure: bool = False) -> QuantumCircuit:
        if len(gammas) != self.p or len(betas) != self.p:
            raise ValueError("gamma and beta lists must have length p")
        circuit = QuantumCircuit(self.n_qubits)
        circuit.h(range(self.n_qubits))
        for gamma, beta in zip(gammas, betas):
            self._apply_cost_layer(circuit, gamma)
            self._apply_mixer_layer(circuit, beta)
        if measure:
            circuit.measure_all()
        return circuit

    def get_statevector(self, gammas: Sequence[float], betas: Sequence[float]) -> Statevector:
        circuit = self.build(gammas, betas, measure=False)
        return Statevector.from_instruction(circuit)

    def sample(self, gammas: Sequence[float], betas: Sequence[float], shots: int = 2048) -> Dict[str, int]:
        circuit = self.build(gammas, betas, measure=True)
        backend = AerSimulator()
        tqc = transpile(circuit, backend)
        job = backend.run(tqc, shots=shots)
        result = job.result()
        return result.get_counts()

    def expectation_from_statevector(self, statevector: Statevector) -> float:
        energy = 0.0
        n = self.n_qubits
        for idx, amplitude in enumerate(statevector.data):
            prob = float(np.abs(amplitude) ** 2)
            bitstring = format(idx, f"0{n}b")[::-1]
            energy += prob * self.cost.energy(bitstring)
        return energy

    def expectation_from_counts(self, counts: Dict[str, int]) -> float:
        total = sum(counts.values())
        energy = 0.0
        for raw, freq in counts.items():
            bitstring = raw[::-1]
            energy += (freq / total) * self.cost.energy(bitstring)
        return energy


def maxcut_hamiltonian(weights: Dict[Tuple[int, int], float], n_nodes: int) -> CostHamiltonian:
    h: Dict[int, float] = {}
    J: Dict[Tuple[int, int], float] = {}
    for (u, v), weight in weights.items():
        J[(u, v)] = weight
    return CostHamiltonian(h=h, J=J, n_qubits=n_nodes)
