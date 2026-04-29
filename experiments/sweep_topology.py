"""
Compare outcomes across ring-lattice, Erdős–Rényi, and Barabási–Albert topologies
at the same mean degree and baseline parameters.
"""

from dataclasses import replace
from params import BASELINE
from model.simulation import run
from analysis.metrics import steady_state, classify_outcome


TOPOLOGIES = ["ring", "er", "ba"]


def sweep(k_values: list[int] | None = None) -> list[dict]:
    if k_values is None:
        k_values = [4, 6, 10]

    results = []
    for topo in TOPOLOGIES:
        for k in k_values:
            p = replace(BASELINE, topology=topo, k=k)
            res = run(p)
            ss = steady_state(res)
            outcome = classify_outcome(ss["h_ss"], ss["s_ss"])
            row = {"topology": topo, "k": k, **ss, "outcome": outcome}
            results.append(row)
            print(f"  {topo:6s}  k={k}  →  h_ss={ss['h_ss']:.3f}  s_ss={ss['s_ss']:.3f}  [{outcome}]")

    return results


if __name__ == "__main__":
    sweep()
