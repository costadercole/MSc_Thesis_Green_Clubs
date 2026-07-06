"""
Network construction.

Returns a NetworkX graph and a weight matrix W (N x N numpy array) where
W[i, j] = w_ij as defined in eq. (3.5): normalised so that each row sums to 1
over the neighbours of i (zero for non-neighbours).
"""

import numpy as np
import networkx as nx


def build_network(N: int, k: int, topology: str, seed: int, m: int | None = None) -> tuple[nx.Graph, np.ndarray]:
    """
    Returns (G, W).

    topology: "ring" | "er" | "ba"
      - ring : k-regular ring lattice (Watts-Strogatz with p=0). Watts-Strogatz
               connects each node to k // 2 neighbours on each side, so the
               REALISED degree is 2 * (k // 2): exact only for even k. Odd k
               silently rounds down (k=3 -> degree 2), so callers that need an
               exact degree must pass an even k.
      - er   : Erdős–Rényi with p = k / (N - 1)
      - ba   : Barabási–Albert with mean degree ≈ 2*m. By default
               m = max(1, k // 2), but note this map is NOT injective:
               k=1 and k=2 both give m=1. Pass `m` explicitly to select an
               exact attachment parameter (e.g. to distinguish m=1 from m=2)
               instead of deriving it from k.
    """
    rng = np.random.default_rng(seed)

    if topology == "ring":
        G = nx.watts_strogatz_graph(N, k, p=0, seed=int(rng.integers(1 << 31)))
    elif topology == "er":
        p_er = k / (N - 1)
        G = nx.erdos_renyi_graph(N, p_er, seed=int(rng.integers(1 << 31)))
    elif topology == "ba":
        m_eff = m if m is not None else max(1, k // 2)
        G = nx.barabasi_albert_graph(N, m_eff, seed=int(rng.integers(1 << 31)))
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
