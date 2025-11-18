from qaoa.metrics import compute_success_probability


def test_success_probability():
    counts = {"00": 500, "11": 500}
    ground_states = ["00"]
    prob = compute_success_probability(counts, ground_states)
    assert abs(prob - 0.5) < 1e-9
