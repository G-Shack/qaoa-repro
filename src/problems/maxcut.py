"""MaxCut problem generation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

Bitstring = str


@dataclass
class MaxCutInstance:
    """Container describing a MaxCut problem instance."""

    graph: nx.Graph
    weights: Dict[Tuple[int, int], float]
    ground_state_energy: float
    ground_state_bitstrings: List[Bitstring]
    energy_ceiling: float

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def edge_list(self) -> List[Tuple[int, int]]:
        return list(self.graph.edges())


def _ensure_connected_graph(n_nodes: int, edge_probability: float, rng: np.random.Generator) -> nx.Graph:
    """Generate an Erdős–Rényi graph that is connected."""
    while True:
        g = nx.Graph()
        g.add_nodes_from(range(n_nodes))
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() <= edge_probability:
                    g.add_edge(i, j)
        if nx.is_connected(g):
            return g


def _generate_weights(graph: nx.Graph, weighted: bool, rng: np.random.Generator) -> Dict[Tuple[int, int], float]:
    weights: Dict[Tuple[int, int], float] = {}
    for u, v in graph.edges():
        if weighted:
            weights[(u, v)] = float(rng.uniform(0.2, 1.0))
        else:
            weights[(u, v)] = 1.0
    return weights


def _cut_energy(bitstring: Bitstring, weights: Dict[Tuple[int, int], float]) -> float:
    z = [1 if b == "0" else -1 for b in bitstring]
    energy = 0.0
    for (u, v), weight in weights.items():
        energy += weight * z[u] * z[v]
    return energy


def _bruteforce_ground_state(n_nodes: int, weights: Dict[Tuple[int, int], float]) -> Tuple[float, List[Bitstring]]:
    best_energy = None
    best_bitstrings: List[Bitstring] = []
    for idx in range(2 ** n_nodes):
        bitstring = format(idx, f"0{n_nodes}b")
        energy = _cut_energy(bitstring, weights)
        if best_energy is None or energy < best_energy - 1e-9:
            best_energy = energy
            best_bitstrings = [bitstring]
        elif abs(energy - best_energy) <= 1e-9:
            best_bitstrings.append(bitstring)
    assert best_energy is not None
    return best_energy, best_bitstrings


def generate_maxcut_instance(
    n_nodes: int,
    edge_probability: float,
    weighted: bool = False,
    seed: int | None = None,
) -> MaxCutInstance:
    """Create a reproducible MaxCut instance and compute its ground state exactly."""
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    if not 0.0 < edge_probability <= 1.0:
        raise ValueError("edge_probability must be in (0, 1]")

    rng = np.random.default_rng(seed)
    graph = _ensure_connected_graph(n_nodes, edge_probability, rng)
    weights = _generate_weights(graph, weighted, rng)
    ground_energy, ground_states = _bruteforce_ground_state(n_nodes, weights)
    energy_ceiling = sum(weights.values())
    return MaxCutInstance(
        graph=graph,
        weights=weights,
        ground_state_energy=ground_energy,
        ground_state_bitstrings=ground_states,
        energy_ceiling=energy_ceiling,
    )


def triangle_graph_instance(weighted: bool = False) -> MaxCutInstance:
    graph = nx.complete_graph(3)
    weights = {(u, v): (1.0 if not weighted else 0.5) for u, v in graph.edges()}
    ground_energy, ground_states = _bruteforce_ground_state(graph.number_of_nodes(), weights)
    energy_ceiling = sum(weights.values())
    return MaxCutInstance(graph, weights, ground_energy, ground_states, energy_ceiling)


def square_cycle_instance(weighted: bool = False) -> MaxCutInstance:
    graph = nx.cycle_graph(4)
    weights = {(u, v): (1.0 if not weighted else 0.5) for u, v in graph.edges()}
    ground_energy, ground_states = _bruteforce_ground_state(graph.number_of_nodes(), weights)
    energy_ceiling = sum(weights.values())
    return MaxCutInstance(graph, weights, ground_energy, ground_states, energy_ceiling)
