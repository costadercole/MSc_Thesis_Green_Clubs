"""
Network construction.

Returns a NetworkX graph and a weight matrix W (N x N numpy array) where
W[i, j] = w_ij as defined in eq. (3.5): normalised so that each row sums to 1
over the neighbours of i (zero for non-neighbours).
"""

import numpy as np
import networkx as nx


def build_network(N: int, k: int, topology: str, seed: int) -> tuple[nx.Graph, np.ndarray]:
    """
    Returns (G, W).

    topology: "ring" | "er" | "ba"
      - ring : k-regular ring lattice (Watts-Strogatz with p=0)
      - er   : Erdős–Rényi with p = k / (N - 1)
      - ba   : Barabási–Albert with m = k // 2  →  mean degree ≈ k
    """
    rng = np.random.default_rng(seed)

    if topology == "ring":
        G = nx.watts_strogatz_graph(N, k, p=0, seed=int(rng.integers(1 << 31)))
    elif topology == "er":
        p_er = k / (N - 1)
        G = nx.erdos_renyi_graph(N, p_er, seed=int(rng.integers(1 << 31)))
    elif topology == "ba":
        m = max(1, k // 2)
        G = nx.barabasi_albert_graph(N, m, seed=int(rng.integers(1 << 31)))
    else:
        raise ValueError(f"Unknown topology '{topology}'. Choose 'ring', 'er', or 'ba'.")

    W = _weight_matrix(G, N)
    return G, W


def _weight_matrix(G: nx.Graph, N: int) -> np.ndarray:
    """Uniform weight matrix: W[i,j] = 1/deg(i) for (i,j) in E, else 0."""
    W = np.zeros((N, N))
    for i in G.nodes():
        neighbours = list(G.neighbors(i))
        if neighbours:
            w = 1.0 / len(neighbours)
            for j in neighbours:
                W[i, j] = w
    return W


def effective_degree(G: nx.Graph) -> float:
    """Mean degree of G; used as k for ER and BA topologies."""
    degrees = [d for _, d in G.degree()]
    return float(np.mean(degrees))
