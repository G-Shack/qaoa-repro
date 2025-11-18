import math

from problems.maxcut import generate_maxcut_instance, square_cycle_instance, triangle_graph_instance
from problems.sat import evaluate_2sat_assignment, generate_2sat_instance
from problems.knapsack import (
    brute_force_knapsack,
    evaluate_knapsack_solution,
    generate_knapsack_instance,
    knapsack_to_ising,
)


def test_triangle_graph_ground_state():
    instance = triangle_graph_instance()
    assert math.isclose(instance.ground_state_energy, -1.0)
    assert len(instance.ground_state_bitstrings) == 6


def test_square_graph_ground_state():
    instance = square_cycle_instance()
    assert math.isclose(instance.ground_state_energy, -4.0)


def test_random_maxcut_connected():
    instance = generate_maxcut_instance(5, 0.6, seed=42)
    assert instance.graph.number_of_edges() > 0
    assert instance.graph.number_of_nodes() == 5


def test_2sat_generation():
    instance = generate_2sat_instance(4, 6, ensure_satisfiable=True, seed=3)
    bitstring = instance.ground_states[0]
    assert evaluate_2sat_assignment(bitstring, instance.clauses)


def test_knapsack_bruteforce_matches_mapping():
    instance = generate_knapsack_instance(4, seed=7)
    selection, value, weight = brute_force_knapsack(instance.values, instance.weights, instance.capacity)
    assert value == instance.optimal_value
    assert weight <= instance.capacity + 1e-9

    h, J, const = knapsack_to_ising(instance.values, instance.weights, instance.capacity, penalty_lambda=20)
    assert len(h) == instance.n_items
    assert all(i < j for i, j in J)


def test_knapsack_feasibility_check():
    values = [10, 20, 30]
    weights = [1, 2, 3]
    capacity = 4
    stats = evaluate_knapsack_solution([1, 1, 0], values, weights, capacity)
    assert stats["feasible"]
    stats = evaluate_knapsack_solution([1, 1, 1], values, weights, capacity)
    assert not stats["feasible"]
