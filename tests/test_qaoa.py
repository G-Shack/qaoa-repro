import numpy as np

from qaoa.circuits import CostHamiltonian, QAOACircuit
from qaoa.metrics import compute_approximation_ratio, compute_energy_from_counts
from qaoa.statistics import repeated_depth_experiments


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


def test_repeated_depth_experiments_runs():
    hamiltonian = simple_cost()
    summary = repeated_depth_experiments(
        hamiltonian=hamiltonian,
        ground_states=["01", "10"],
        energy_min=-1.0,
        energy_max=1.0,
        p=1,
        repeats=2,
        n_restarts=1,
        shots=64,
    )
    assert len(summary.records) == 2
    assert summary.std_energy >= 0.0
