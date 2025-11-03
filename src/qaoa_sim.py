# src/qaoa_sim.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit_aer import AerSimulator
from typing import List, Tuple, Dict

def rz(angle):
    """Return an RZ gate angle (just for clarity)."""
    return angle

def build_Uc_from_hJ(h: Dict[int,float], J: Dict[Tuple[int,int],float], gamma: float, n_qubits: int):
    """
    Build a QuantumCircuit that implements U_C(gamma)=exp(-i * gamma * H_C),
    using RZ and CNOT pairs for ZZ terms. Returns a QuantumCircuit (no measurement).
    h: dict {i: hi}
    J: dict {(i,j): Jij} with i<j
    """
    qc = QuantumCircuit(n_qubits)
    # single-qubit Z rotations from h_i
    for i, hi in h.items():
        theta = 2 * gamma * hi  # RZ(2*gamma*h_i) implements exp(-i gamma h_i Z)
        qc.rz(theta, i)
    # two-qubit ZZ rotations: implement exp(-i gamma Jij Z_i Z_j)
    # Standard decomposition: CNOT(i,j); RZ(2*gamma*Jij) on j; CNOT(i,j)
    for (i, j), Jij in J.items():
        theta = 2 * gamma * Jij
        qc.cx(i, j)
        qc.rz(theta, j)
        qc.cx(i, j)
    return qc

def build_Ub(n_qubits: int, beta: float):
    """
    Build the mixing unitary U_B(beta) = exp(-i beta sum X_i) as RX(2beta) on each qubit.
    """
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.rx(2*beta, i)
    return qc

def build_qaoa_circuit(h, J, gammas: List[float], betas: List[float], n_qubits: int):
    """
    Build full QAOA circuit (no measurements) for given lists gammas, betas of length p.
    """
    p = len(gammas)
    qc = QuantumCircuit(n_qubits)
    # initial H on all qubits
    for i in range(n_qubits):
        qc.h(i)
    # alternate
    for layer in range(p):
        Uc = build_Uc_from_hJ(h, J, gammas[layer], n_qubits)
        qc.append(Uc.to_instruction(), range(n_qubits))
        Ub = build_Ub(n_qubits, betas[layer])
        qc.append(Ub.to_instruction(), range(n_qubits))
    return qc

def statevector_from_circuit(qc: QuantumCircuit):
    """
    Return statevector from a circuit using Qiskit Aer statevector simulator (exact).
    """
    sv = Statevector.from_instruction(qc)
    return sv

def expected_cost_from_statevector(sv: Statevector, h: Dict[int,float], J: Dict[Tuple[int,int],float]) -> float:
    """
    Compute expected energy <H_C> for a given statevector.
    h, J: as above. Basis ordering: computational (|0> is +1 eigenvalue of Z by our mapping convention)
    We'll map computational basis |0>=z=+1, |1>=z=-1 (standard).
    """
    n = int(np.log2(len(sv.data)))
    energy = 0.0
    # iterate basis states
    for idx, amp in enumerate(sv.data):
        prob = np.abs(amp)**2
        bitstr = format(idx, '0{}b'.format(n))[::-1]  # qiskit convention, qubit 0 is least significant
        z = [1 if b == '0' else -1 for b in bitstr]
        cost = 0.0
        # single qubit fields
        for i, hi in h.items():
            cost += hi * z[i]
        for (i, j), Jij in J.items():
            cost += Jij * z[i] * z[j]
        energy += prob * cost
    return energy

def sample_bitstrings(qc: QuantumCircuit, shots=2048):
    """
    Append measurement to qc and run on Aer qasm_simulator to get counts.
    Returns counts dict mapping bitstring -> counts (Qiskit ordering).
    """
    qc_meas = qc.copy()
    n = qc.num_qubits
    qc_meas.measure_all()
    simulator = AerSimulator()
    tqc = transpile(qc_meas, simulator)
    job = simulator.run(tqc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return counts

def energy_from_counts(counts: Dict[str,int], h, J) -> float:
    """
    Compute expected energy from measured counts.
    counts keys are bitstrings in Qiskit format (e.g., '0101' with qubit 0 as least-significant).
    """
    total_shots = sum(counts.values())
    energy = 0.0
    n = len(next(iter(counts.keys())))
    for bitstr, c in counts.items():
        # qiskit returns bitstrings with qubit 0 as rightmost; we reverse to match our z mapping
        bstr = bitstr[::-1]
        z = [1 if ch == '0' else -1 for ch in bstr]
        cost = 0.0
        for i, hi in h.items():
            cost += hi * z[i]
        for (i, j), Jij in J.items():
            cost += Jij * z[i] * z[j]
        energy += (c/total_shots) * cost
    return energy

def best_sampled_solution(counts):
    """
    Return the most frequent bitstring (in our ordering) and its count.
    """
    best = max(counts.items(), key=lambda kv: kv[1])
    bitstr = best[0][::-1]
    return bitstr, best[1]
