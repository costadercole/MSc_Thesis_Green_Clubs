"""
Accounting tests for the welfare-decomposition layer (analysis/welfare.py),
in the spirit of the Appendix-A Cournot-solver unit tests.

Checks at a representative market state:
  W1 — the four components sum exactly to W_i (and to W_tot);
  W2 — per-capita x population = total, jurisdiction by jurisdiction;
  W3 — re-solved p* equals the value the simulation recorded;
  W4 — decomposed W_pc equals model.per_capita_welfare (the imitation fitness).
"""

import numpy as np
import pytest
from dataclasses import replace

from params import BASELINE
from model.simulation import run
from model.market import solve_market
from model.jurisdictions import fiscal_revenues, per_capita_welfare
from analysis.welfare import welfare_components, aggregate, welfare_at_state


@pytest.fixture(scope="module")
def state():
    r = run(replace(BASELINE, mu=0.1, lam=1.0, nu=1.0, T=400, seed=42))
    return r, (r["f_H"][-1], r["f_L"][-1], r["sigma"][-1], r["P"], r["W"])


def test_W1_components_sum_to_total(state):
    _, (f_H, f_L, sigma, P, W) = state
    _, comp, _ = welfare_at_state(f_H, f_L, sigma, P, W, BASELINE)
    recon = comp["cs_tot"] + comp["fisc_tot"] + comp["host_tot"] - comp["dmg_tot"]
    np.testing.assert_allclose(recon, comp["W_tot"], atol=1e-6)
    # and at the aggregate level
    agg = aggregate(comp, sigma)["all"]
    np.testing.assert_allclose(
        agg["cs_tot"] + agg["fisc_tot"] + agg["host_tot"] - agg["dmg_tot"],
        agg["W_tot"], rtol=1e-9)


def test_W2_percapita_times_pop_equals_total(state):
    _, (f_H, f_L, sigma, P, W) = state
    _, comp, _ = welfare_at_state(f_H, f_L, sigma, P, W, BASELINE)
    np.testing.assert_allclose(comp["W_pc"] * comp["P"], comp["W_tot"], atol=1e-6)


def test_W3_resolved_price_matches_recorded(state):
    r, (f_H, f_L, sigma, P, W) = state
    _, _, p_star = welfare_at_state(f_H, f_L, sigma, P, W, BASELINE)
    np.testing.assert_allclose(p_star, r["p_star"][-1], atol=1e-9)


def test_W4_matches_imitation_fitness(state):
    _, (f_H, f_L, sigma, P, W) = state
    _, comp, _ = welfare_at_state(f_H, f_L, sigma, P, W, BASELINE)
    p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, BASELINE)
    TR = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, BASELINE)
    ref = per_capita_welfare(f_H, sigma, P, p_star, TR, BASELINE, q_H, f_L, q_L)
    np.testing.assert_allclose(comp["W_pc"], ref, atol=1e-9)
