"""
Run one simulation with baseline parameters and display key plots.
"""

import matplotlib.pyplot as plt
from params import BASELINE
from model.simulation import run
from analysis.metrics import steady_state, classify_outcome
from analysis.plots import plot_time_series, plot_phase_portrait, plot_firm_distribution


def main():
    print("Running baseline simulation...")
    results = run(BASELINE)

    ss = steady_state(results)
    outcome = classify_outcome(ss["h_ss"], ss["s_ss"])
    print(f"\nSteady state:  h = {ss['h_ss']:.3f} ± {ss['h_std']:.3f}")
    print(f"               s = {ss['s_ss']:.3f} ± {ss['s_std']:.3f}")
    print(f"Outcome: {outcome}")

    plot_time_series(results, title="Baseline — h(t) and s(t)")
    plot_phase_portrait(results, title="Baseline — phase portrait")
    plot_firm_distribution(results, step=-1, title="Baseline — firm distribution (final step)")
    plt.show()


if __name__ == "__main__":
    main()
