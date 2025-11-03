# src/instances.py
import networkx as nx
import numpy as np
from typing import Tuple, List

def generate_regular_graph(n=6, deg=3, seed=None):
    # networkx: random_regular_graph(d, n)
    return nx.random_regular_graph(deg, n, seed=seed)

def generate_random_knapsack(n=6, value_range=(5,20), weight_range=(1,5), W=None, seed=None):
    rng = np.random.default_rng(seed)
    values = rng.integers(value_range[0], value_range[1]+1, size=n).tolist()
    weights = rng.integers(weight_range[0], weight_range[1]+1, size=n).tolist()
    if W is None:
        # set capacity to around half total weight
        W = int(0.5 * sum(weights))
    return values, weights, W
