from problems.knapsack import evaluate_knapsack_solution, knapsack_to_ising


def test_penalty_conversion_discourages_infeasible():
    values = [10, 20]
    weights = [2, 3]
    capacity = 3
    h, J, const = knapsack_to_ising(values, weights, capacity, penalty_lambda=10)
    feasible = [1, 0]  # take first item only
    infeasible = [1, 1]

    def energy(selection):
        z = [1 if bit == 0 else -1 for bit in selection]
        total = const
        for i, coeff in h.items():
            total += coeff * z[i]
        for (i, j), coeff in J.items():
            total += coeff * z[i] * z[j]
        return total

    assert energy(feasible) < energy(infeasible)


def test_feasibility_detection_various():
    values = [5, 10, 15]
    weights = [1, 2, 3]
    capacity = 3
    cases = [([1, 0, 0], True), ([0, 1, 0], True), ([1, 1, 1], False)]
    for selection, expected in cases:
        stats = evaluate_knapsack_solution(selection, values, weights, capacity)
        assert stats["feasible"] == expected
