import numpy as np

from qaoa.circuits import CostHamiltonian, QAOACircuit
from qaoa.metrics import compute_approximation_ratio, compute_energy_from_counts


def simple_cost():
    h = {}
    J = {(0, 1): 1.0}
    return CostHamiltonian(h, J, 2)


def test_initial_state_uniform():
    hamiltonian = simple_cost()
    qaoa = QAOACircuit(hamiltonian, p=1)
    sv = qaoa.get_statevector([0.0], [0.0])
    amplitudes = sv.data
    probs = np.abs(amplitudes) ** 2
    assert np.allclose(probs, np.ones(4) / 4)


def test_energy_from_counts():
    hamiltonian = simple_cost()
    qaoa = QAOACircuit(hamiltonian, p=1)
    counts = {"00": 1024}
    mean, std, distribution = compute_energy_from_counts(counts, hamiltonian.energy)
    assert np.isclose(mean, 1.0)
    counts = {"01": 1024}
    mean, _, _ = compute_energy_from_counts(counts, hamiltonian.energy)
    assert np.isclose(mean, -1.0)


def test_approximation_ratio_bounds():
    assert compute_approximation_ratio(-10, -10, 5) == 1.0
    assert compute_approximation_ratio(5, -10, 5) == 0.0
