"""
Test suite for the static per-period Cournot equilibrium.

All expected values are derived from first principles; no simulation output is used.

Convention used throughout (mirrors model/market.py):
  q_H[i, j] = per-H-firm quantity shipped from source j to destination market i
  q_L[i, j] = per-L-firm quantity shipped from source j to destination market i

Inverse demand in market i:  p_i = a - (b / P_i) * Q_i
Cournot equilibrium:         p*_i = (a + Σ_{k∈A_i} c̃_k * n_k) / (N_active + 1)
                             q*_{ji} = (P_i / b) * (p*_i - c̃_{ji})

Groups
------
A  – closed-economy Cournot (autarky)
B  – trade, transport costs, tariffs, BCA
C  – accounting / conservation identities
D  – degenerate edge cases
"""

import numpy as np
import pytest

from params import Params
from model.market import (
    _cournot_solve,
    _delivered_cost,
    _mc_eff,
    solve_market,
    firm_variable_profits,
    consumer_surplus,
)
from model.jurisdictions import fiscal_revenues, per_capita_welfare


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_params(**kwargs) -> Params:
    """
    Return a Params with deterministic, small-scale defaults suitable for
    hand-verification, then override with any keyword arguments.

    Defaults
    --------
    a=10, b=1, c_H=2, c_L=4, t=0, g=100 (prohibitive), tau=0, tau_BA=0,
    delta_loc=0, delta_glob=0, F=0.
    """
    p = Params()
    p.a          = 10.0
    p.b          = 1.0
    p.c_H        = 2.0
    p.c_L        = 4.0
    p.t          = 0.0
    p.g          = 100.0   # prohibitive → autarky unless overridden
    p.tau        = 0.0
    p.tau_BA     = 0.0
    p.delta_loc  = 0.0
    p.delta_glob = 0.0
    p.F          = 0.0
    for k, v in kwargs.items():
        setattr(p, k, v)
    return p


def autarky_market(n: int, firm_type: str = "L"):
    """
    Single jurisdiction with n identical firms of the given type.

    Returns (f_H, f_L, sigma, P, W, params).
    sigma=0 (lax), P=1, W=zeros(1,1), g=100 (no trade).
    """
    f_H    = np.array([float(n) if firm_type == "H" else 0.0])
    f_L    = np.array([float(n) if firm_type == "L" else 0.0])
    sigma  = np.array([0])
    P      = np.array([1.0])
    W      = np.zeros((1, 1))
    params = make_params(t=0.0)
    return f_H, f_L, sigma, P, W, params


# ---------------------------------------------------------------------------
# Group A: closed-economy Cournot (no trade)
# ---------------------------------------------------------------------------

class TestA_ClosedEconomy:
    """Autarky Cournot equilibrium — no transport, tariff, or BCA."""

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_A1_symmetric_n_firm_price(self, n):
        """
        n identical L-firms (cost c_L) in a single autarky market.

        p* = (a + n·c_L) / (n+1)
        """
        a, c = 10.0, 4.0
        f_H, f_L, sigma, P, W, par = autarky_market(n, "L")

        p_star, _, _ = solve_market(f_H, f_L, sigma, P, W, par)

        expected_p = (a + n * c) / (n + 1)
        assert p_star[0] == pytest.approx(expected_p, rel=1e-9), (
            f"n={n}: p*={p_star[0]:.6f}, expected {expected_p:.6f}"
        )

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_A1_symmetric_n_firm_quantity(self, n):
        """
        Per-firm quantity: q* = (P/b) · (a − c_L) / (n+1)
        """
        a, b, c, P = 10.0, 1.0, 4.0, 1.0
        f_H, f_L, sigma, P_arr, W, par = autarky_market(n, "L")

        _, _, q_L = solve_market(f_H, f_L, sigma, P_arr, W, par)

        expected_q = (P / b) * (a - c) / (n + 1)
        assert q_L[0, 0] == pytest.approx(expected_q, rel=1e-9), (
            f"n={n}: q*={q_L[0,0]:.6f}, expected {expected_q:.6f}"
        )

    def test_A2_asymmetric_two_firm(self):
        """
        One H-firm (c1=c_H=2) and one L-firm (c2=c_L=4) in a single lax market, t=0.

        p*   = (a + c1 + c2) / 3
        q*_H = (P/b) · (a + c2 − 2·c1) / 3
        q*_L = (P/b) · (a + c1 − 2·c2) / 3
        """
        a, b, c1, c2, P = 10.0, 1.0, 2.0, 4.0, 1.0
        par   = make_params(c_H=c1, c_L=c2, t=0.0)
        f_H   = np.array([1.0])
        f_L   = np.array([1.0])
        sigma = np.array([0])
        P_arr = np.array([P])
        W     = np.zeros((1, 1))

        p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P_arr, W, par)

        expected_p  = (a + c1 + c2) / 3
        expected_q1 = (P / b) * (a + c2 - 2 * c1) / 3
        expected_q2 = (P / b) * (a + c1 - 2 * c2) / 3

        # Both must be positive with these numbers (a=10, c1=2, c2=4)
        assert expected_q1 > 0 and expected_q2 > 0, "Test setup error: both firms should be active"

        assert p_star[0] == pytest.approx(expected_p,  rel=1e-9)
        assert q_H[0, 0] == pytest.approx(expected_q1, rel=1e-9)
        assert q_L[0, 0] == pytest.approx(expected_q2, rel=1e-9)

    def test_A3_high_cost_firm_eliminated(self):
        """
        Three candidate groups: c1=c2=1, c3=100, a=10.

        Monotone elimination round 1:
          p_candidate = (10 + 1 + 1 + 100) / 4 = 28
          firm 3: 100 < 28 → False → eliminated
        Round 2: p = (10 + 1 + 1) / 3 = 4 → both active.

        Expected: p*=4, mask=[T, T, F].
        """
        p, mask = _cournot_solve([(1.0, 1), (1.0, 1), (100.0, 1)], a=10.0)

        assert p    == pytest.approx(4.0, rel=1e-9)
        assert mask[0] is True
        assert mask[1] is True
        assert mask[2] is False, "Firm 3 (c=100) must be eliminated by monotone procedure"

    def test_A4_marginal_firm_just_active(self):
        """
        With c1=c2=1 and a=10, the two-firm equilibrium price is p_2 = (10+1+1)/3 = 4.

        Setting c3 = 3.99 < 4:
          Candidate price with all 3: p3 = (10+1+1+3.99)/4 = 3.9975
          3.99 < 3.9975 → firm 3 is active.
        """
        a      = 10.0
        p_2    = (a + 1.0 + 1.0) / 3        # = 4.0
        c3     = p_2 - 0.01                  # 3.99 → must be active

        p, mask = _cournot_solve([(1.0, 1), (1.0, 1), (c3, 1)], a)

        expected_p3 = (a + 1.0 + 1.0 + c3) / 4
        assert mask[2] is True, f"c3={c3} < p_2={p_2}: firm 3 must be active"
        assert p == pytest.approx(expected_p3, rel=1e-9)
        assert expected_p3 - c3 > 0, "Active firm 3 must produce positive quantity"

    def test_A4_marginal_firm_at_boundary_inactive(self):
        """
        c3 = p_2 = 4 exactly.

        Elimination uses strict <: 4 < 4 is False → firm 3 eliminated.
        Returns two-firm equilibrium p* = 4.
        """
        a   = 10.0
        p_2 = (a + 1.0 + 1.0) / 3   # = 4.0
        c3  = p_2                    # exactly at boundary

        p, mask = _cournot_solve([(1.0, 1), (1.0, 1), (c3, 1)], a)

        assert mask[2] is False, (
            f"c3={c3} == p_2={p_2}: strict inequality means firm 3 is INACTIVE"
        )
        assert p == pytest.approx(p_2, rel=1e-9)

    def test_A4_boundary_flip(self):
        """
        Verify the flip is sharp: infinitesimally below → active, at/above → inactive.
        This catches off-by-one and weak-vs-strict inequality bugs.
        """
        a   = 10.0
        p_2 = (a + 1.0 + 1.0) / 3

        _, mask_below = _cournot_solve([(1.0, 1), (1.0, 1), (p_2 - 1e-9, 1)], a)
        _, mask_at    = _cournot_solve([(1.0, 1), (1.0, 1), (p_2,         1)], a)
        _, mask_above = _cournot_solve([(1.0, 1), (1.0, 1), (p_2 + 1e-9,  1)], a)

        assert mask_below[2] is True,  "Strictly below p_2: firm must be active"
        assert mask_at[2]    is False, "Exactly at p_2: strict < eliminates the firm"
        assert mask_above[2] is False, "Above p_2: firm must be inactive"


# ---------------------------------------------------------------------------
# Group B: trade
# ---------------------------------------------------------------------------

class TestB_Trade:
    """Transport costs, tariffs, and BCA — delivered-cost and equilibrium tests."""

    def test_B1_free_trade_identical_jurisdictions(self):
        """
        Two symmetric lax jurisdictions, g=tau=tau_BA=t=0, each with n L-firms.
        Each market's candidate set has 2n firms at cost c_L, so:

          p* = (a + 2n·c_L) / (2n + 1)   (same in both markets)
        """
        n    = 2
        a, c = 10.0, 4.0
        N    = 2
        par  = make_params(c_L=c, g=0.0, tau=0.0, tau_BA=0.0, t=0.0)
        f_H  = np.zeros(N)
        f_L  = np.full(N, float(n))
        sigma = np.zeros(N, dtype=int)
        P     = np.ones(N)
        W     = np.array([[0.0, 1.0], [1.0, 0.0]])

        p_star, _, _ = solve_market(f_H, f_L, sigma, P, W, par)

        expected_p = (a + 2 * n * c) / (2 * n + 1)
        assert p_star[0] == pytest.approx(expected_p, rel=1e-9)
        assert p_star[1] == pytest.approx(expected_p, rel=1e-9)

    def test_B2_prohibitive_transport_recovers_autarky(self):
        """
        With g large, imported goods have cost >> a and are eliminated.
        Each market's equilibrium must equal the autarky Cournot price.
        """
        n    = 3
        a, c = 10.0, 4.0
        N    = 2
        par  = make_params(c_L=c, g=1000.0, tau=0.0, tau_BA=0.0, t=0.0)
        f_H  = np.zeros(N)
        f_L  = np.full(N, float(n))
        sigma = np.zeros(N, dtype=int)
        P     = np.ones(N)
        W     = np.array([[0.0, 1.0], [1.0, 0.0]])

        p_star, _, _ = solve_market(f_H, f_L, sigma, P, W, par)

        expected_p_autarky = (a + n * c) / (n + 1)
        assert p_star[0] == pytest.approx(expected_p_autarky, rel=1e-9)
        assert p_star[1] == pytest.approx(expected_p_autarky, rel=1e-9)

    def test_B3_tariff_direction(self):
        """
        tau is charged only on lax→strict cross-border shipments.
        sigma=[1, 0]: jurisdiction 0 = strict, jurisdiction 1 = lax.
        g=2, tau=3, tau_BA=0, t=0.

        Expected delivered costs (H-firms):
          j=1(lax)→i=0(strict): c_H + g + tau = 2 + 2 + 3 = 7
          j=0(strict)→i=1(lax): c_H + g        = 2 + 2     = 4   (no tau: wrong direction)
          j=1→i=1 (domestic):   c_H             = 2             (no transport)
          j=0→i=0 (domestic):   c_H + t·σ_0    = 2             (t=0)
        """
        par   = make_params(c_H=2.0, t=0.0, g=2.0, tau=3.0, tau_BA=0.0)
        sigma = np.array([1, 0])
        c_H   = par.c_H

        assert _delivered_cost(True, j=1, i=0, sigma=sigma, p=par) == \
               pytest.approx(c_H + 2.0 + 3.0, rel=1e-9), "lax→strict must pay tau"

        assert _delivered_cost(True, j=0, i=1, sigma=sigma, p=par) == \
               pytest.approx(c_H + 2.0, rel=1e-9), "strict→lax must NOT pay tau"

        assert _delivered_cost(True, j=1, i=1, sigma=sigma, p=par) == \
               pytest.approx(c_H, rel=1e-9), "domestic (lax) must not pay transport"

        assert _delivered_cost(True, j=0, i=0, sigma=sigma, p=par) == \
               pytest.approx(c_H, rel=1e-9), "domestic (strict, t=0) must not pay transport"

    def test_B4_BCA_only_on_H_firms(self):
        """
        H-firm from lax→strict pays tau + tau_BA; L-firm pays tau only.
        sigma=[1,0], g=0, tau=3, tau_BA=5.

          H: c_H + tau + tau_BA = 2 + 3 + 5 = 10
          L: c_L + tau          = 4 + 3     = 7
          Difference: 3 = tau_BA + c_H − c_L = 5 + 2 − 4
        """
        par   = make_params(c_H=2.0, c_L=4.0, t=0.0, g=0.0, tau=3.0, tau_BA=5.0)
        sigma = np.array([1, 0])

        c_H_exp = _delivered_cost(True,  j=1, i=0, sigma=sigma, p=par)
        c_L_exp = _delivered_cost(False, j=1, i=0, sigma=sigma, p=par)

        assert c_H_exp == pytest.approx(par.c_H + par.tau + par.tau_BA, rel=1e-9), \
               "H-firm must pay tau + tau_BA"
        assert c_L_exp == pytest.approx(par.c_L + par.tau, rel=1e-9), \
               "L-firm must pay tau only (no tau_BA)"
        assert c_H_exp - c_L_exp == \
               pytest.approx(par.tau_BA + par.c_H - par.c_L, rel=1e-9)

    def test_B5_carbon_tax_on_location_not_destination(self):
        """
        Carbon tax is levied at the ORIGIN based on the origin jurisdiction's policy.
        sigma=[1,0]: jurisdiction 0=strict, jurisdiction 1=lax.
        t=4, g=1, tau=0, tau_BA=3.

        H-firm in strict (j=0):
          - Domestic i=0:  mc = c_H + t      = 6   (pays carbon tax)
          - Export to i=1: mc = c_H + t + g  = 7   (carbon tax travels with the firm)

        H-firm in lax (j=1):
          - Domestic i=1:  mc = c_H           = 2   (no carbon tax)
          - Export to i=0: mc = c_H + g + tau_BA = 6  (BCA at border, not carbon tax)
        """
        par   = make_params(c_H=2.0, t=4.0, g=1.0, tau=0.0, tau_BA=3.0)
        sigma = np.array([1, 0])
        c_H   = par.c_H

        # Strict-origin firm
        assert _delivered_cost(True, j=0, i=0, sigma=sigma, p=par) == \
               pytest.approx(c_H + par.t, rel=1e-9), \
               "H in strict, domestic: pays carbon tax"

        assert _delivered_cost(True, j=0, i=1, sigma=sigma, p=par) == \
               pytest.approx(c_H + par.t + par.g, rel=1e-9), \
               "H in strict, exporting to lax: carbon tax applies at origin (not at border)"

        # Lax-origin firm
        assert _delivered_cost(True, j=1, i=1, sigma=sigma, p=par) == \
               pytest.approx(c_H, rel=1e-9), \
               "H in lax, domestic: no carbon tax"

        assert _delivered_cost(True, j=1, i=0, sigma=sigma, p=par) == \
               pytest.approx(c_H + par.g + par.tau_BA, rel=1e-9), \
               "H in lax, exporting to strict: BCA at border (not carbon tax)"


# ---------------------------------------------------------------------------
# Group C: conservation / accounting identities
# ---------------------------------------------------------------------------

class TestC_Accounting:
    """
    These identities must hold in every well-formed simulation period.
    They are tested on a fixed hand-constructed equilibrium.

    Setup (used by all C tests):
      N=2, sigma=[1,0], W connected, P=[1,1]
      f_H=[1,2], f_L=[2,1], t=3, g=1, tau=2, tau_BA=1
      (parameters chosen so all firms are active in both markets)
    """

    @pytest.fixture
    def eq(self):
        """Compute and return a 2-jurisdiction equilibrium."""
        N   = 2
        par = make_params(c_H=2.0, c_L=4.0, t=3.0, g=1.0, tau=2.0, tau_BA=1.0)
        f_H = np.array([1.0, 2.0])
        f_L = np.array([2.0, 1.0])
        sigma = np.array([1, 0])
        P     = np.array([1.0, 1.0])
        W     = np.array([[0.0, 1.0], [1.0, 0.0]])

        p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, par)
        TR = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, par)
        return dict(f_H=f_H, f_L=f_L, sigma=sigma, P=P, W=W, par=par,
                    p_star=p_star, q_H=q_H, q_L=q_L, TR=TR)

    def test_C1_market_clearing(self, eq):
        """
        Demand identity: p*_i = a − (b / P_i) · Q_i

        where Q_i = Σ_j [ f_H[j]·q_H[i,j] + f_L[j]·q_L[i,j] ].
        """
        par, p_star, q_H, q_L, P = (
            eq["par"], eq["p_star"], eq["q_H"], eq["q_L"], eq["P"]
        )
        N = len(p_star)

        for i in range(N):
            Q_i = sum(
                eq["f_H"][j] * q_H[i, j] + eq["f_L"][j] * q_L[i, j]
                for j in range(N)
            )
            p_demand = par.a - (par.b / P[i]) * Q_i
            assert p_demand == pytest.approx(p_star[i], rel=1e-6), (
                f"Market {i}: p* = {p_star[i]:.4f}, "
                f"demand identity gives {p_demand:.4f}"
            )

    def test_C2_tariff_revenue(self):
        """
        TR_tariff_i = tau · Σ_{j: lax nbr} [ f_H[j]·q_H[i,j] + f_L[j]·q_L[i,j] ]

        Isolated with tau_BA=0, t=0, only L-firms to keep the arithmetic clean.
        Hand calculation:
          sigma=[1,0], g=0, tau=1.
          Market 0: domestic c_L=4, import c_L+tau=5.
          p*(0) = (10+4+5)/3 = 19/3; q_L[0,1] = 19/3−5 = 4/3.
          TR[0] = 1·1·(4/3) = 4/3.
        """
        N   = 2
        par = make_params(c_H=2.0, c_L=4.0, t=0.0, g=0.0, tau=1.0, tau_BA=0.0)
        f_H = np.zeros(N)
        f_L = np.array([1.0, 1.0])
        sigma = np.array([1, 0])
        P     = np.ones(N)
        W     = np.array([[0.0, 1.0], [1.0, 0.0]])

        p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, par)
        TR = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, par)

        # Hand-computed: TR[0] = tau * f_L[1] * q_L[0,1]
        expected_TR0 = par.tau * f_L[1] * q_L[0, 1]
        assert TR[0] == pytest.approx(expected_TR0, rel=1e-9)
        assert TR[1] == pytest.approx(0.0, abs=1e-12), "Lax jurisdiction has no revenue"

        # Cross-check the specific value
        # p*(0) = (10+4+5)/3 = 19/3; q_L[0,1] = 19/3 − 5 = 4/3
        assert q_L[0, 1] == pytest.approx(4.0 / 3.0, rel=1e-9)
        assert TR[0]     == pytest.approx(4.0 / 3.0, rel=1e-9)

    def test_C3_BCA_revenue(self):
        """
        TR_BCA_i = tau_BA · Σ_{j: lax nbr, H} f_H[j] · q_H[i,j]

        Isolated with tau=0, t=0, only H-firms.
        Hand calculation:
          sigma=[1,0], g=0, tau_BA=2, t=0.
          H in j=0 (strict): mc = c_H+t·1 = 2 (t=0). Domestic: cost=2.
          H in j=1 (lax): mc = c_H = 2. Export to i=0: cost = 2 + tau_BA = 4.
          Market 0 candidates: [(2,1),(4,1)] → p*=(10+2+4)/3=16/3.
          q_H[0,0]=16/3−2=10/3, q_H[0,1]=16/3−4=4/3.
          TR[0] = tau_BA · f_H[1] · q_H[0,1] = 2·1·4/3 = 8/3.
        """
        N   = 2
        par = make_params(c_H=2.0, c_L=4.0, t=0.0, g=0.0, tau=0.0, tau_BA=2.0)
        f_H = np.array([1.0, 1.0])
        f_L = np.zeros(N)
        sigma = np.array([1, 0])
        P     = np.ones(N)
        W     = np.array([[0.0, 1.0], [1.0, 0.0]])

        p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, par)
        TR = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, par)

        expected_TR0 = par.tau_BA * f_H[1] * q_H[0, 1]
        assert TR[0] == pytest.approx(expected_TR0, rel=1e-9)
        assert TR[1] == pytest.approx(0.0, abs=1e-12)

        # Cross-check specific values
        assert p_star[0] == pytest.approx(16.0 / 3.0, rel=1e-9)
        assert q_H[0, 1] == pytest.approx(4.0  / 3.0, rel=1e-9)
        assert TR[0]     == pytest.approx(8.0  / 3.0, rel=1e-9)

    def test_C4_carbon_tax_base_includes_exports(self):
        """
        Carbon tax is on total H-output from the strict jurisdiction, not just
        what is sold domestically.

        Setup: sigma=[1,0], 2 H-firms in j=0 (strict), no other firms, g=0, t=3.
          H in j=0 (strict): mc = c_H + t = 2+3 = 5.
          Both markets see 2 H-firms from j=0 at cost 5 (no tariff: strict→lax).
          p*(0) = (10 + 5·2)/3 = 20/3;  q_H[0,0] = 20/3−5 = 5/3 per firm.
          p*(1) = (10 + 5·2)/3 = 20/3;  q_H[1,0] = 20/3−5 = 5/3 per firm.
          Total H-output from j=0 = 5/3 + 5/3 = 10/3 per firm.
          TR[0] = t · f_H[0] · (10/3) = 3·2·(10/3) = 20.
        """
        N   = 2
        par = make_params(c_H=2.0, c_L=4.0, t=3.0, g=0.0, tau=0.0, tau_BA=0.0)
        f_H = np.array([2.0, 0.0])
        f_L = np.zeros(N)
        sigma = np.array([1, 0])
        P     = np.ones(N)
        W     = np.array([[0.0, 1.0], [1.0, 0.0]])

        p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, par)
        TR = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, par)

        total_H_output_from_0 = sum(q_H[m, 0] for m in range(N))
        expected_TR0 = par.t * f_H[0] * total_H_output_from_0

        assert TR[0] == pytest.approx(expected_TR0, rel=1e-9)
        assert TR[1] == pytest.approx(0.0, abs=1e-12)

        # Cross-check
        assert q_H[0, 0] == pytest.approx(5.0 / 3.0, rel=1e-9), "domestic sales"
        assert q_H[1, 0] == pytest.approx(5.0 / 3.0, rel=1e-9), "exports included in tax base"
        assert TR[0]     == pytest.approx(20.0,       rel=1e-9)

    def test_C5_consumer_surplus_formula(self):
        """
        consumer_surplus(p_star, a, b) must equal (a − p*)² / (2b) exactly.
        Checked against a range of hand-constructed prices.
        """
        a, b  = 10.0, 1.0
        prices = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
        expected = (a - prices) ** 2 / (2 * b)
        computed = consumer_surplus(prices, a, b)

        np.testing.assert_allclose(computed, expected, rtol=1e-12)

    def test_C5_consumer_surplus_end_to_end(self):
        """
        CS computed inside per_capita_welfare must match the standalone formula
        when TR=0, delta_loc=0, delta_glob=0.
        """
        N   = 2
        par = make_params(c_L=4.0, t=0.0, tau=0.0, tau_BA=0.0,
                          delta_loc=0.0, delta_glob=0.0, g=100.0)
        f_H = np.zeros(N)
        f_L = np.array([2.0, 3.0])
        sigma = np.zeros(N, dtype=int)
        P     = np.ones(N)
        W     = np.zeros((N, N))
        TR    = np.zeros(N)

        p_star, _, _ = solve_market(f_H, f_L, sigma, P, W, par)
        w = per_capita_welfare(f_H, sigma, P, p_star, TR, par)

        expected_cs = (par.a - p_star) ** 2 / (2 * par.b)
        np.testing.assert_allclose(w, expected_cs, rtol=1e-9)

    def test_C6_host_benefit(self):
        """
        The host-benefit term adds exactly phi*(E_H + E_L)/P_i to per-capita
        welfare, where E_X[i] is total type-X output produced in i.  Two checks:
        (a) the gap between a phi>0 run and an otherwise-identical phi=0 run
            equals phi*(E_H + E_L)/P_i to tolerance;
        (b) phi=0 exactly recovers the damage-only welfare.
        """
        N    = 2
        phi0 = 7.0
        common = dict(c_L=5.0, t=0.0, tau=0.0, tau_BA=0.0,
                      delta_loc=0.0, delta_glob=0.0, g=100.0)
        par0 = make_params(phi=0.0,  **common)
        par1 = make_params(phi=phi0, **common)

        f_H   = np.array([2.0, 1.0])
        f_L   = np.array([1.0, 3.0])
        sigma = np.zeros(N, dtype=int)     # lax → H firms produce (no carbon tax)
        P     = np.array([10.0, 4.0])      # uneven sizes → exercises the /P_i scaling
        W     = np.zeros((N, N))
        TR    = np.zeros(N)

        p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, par0)

        w0 = per_capita_welfare(f_H, sigma, P, p_star, TR, par0, q_H, f_L, q_L)
        w1 = per_capita_welfare(f_H, sigma, P, p_star, TR, par1, q_H, f_L, q_L)

        E_H    = f_H * q_H.sum(axis=0)
        E_L    = f_L * q_L.sum(axis=0)
        P_safe = np.maximum(P, 1.0)
        expected_gain = phi0 * (E_H + E_L) / P_safe

        # (a) the welfare gain is exactly the host-benefit term
        np.testing.assert_allclose(w1 - w0, expected_gain, rtol=1e-9)
        # (b) phi=0 recovers the damage-only value (term inactive)
        w0_damage_only = per_capita_welfare(f_H, sigma, P, p_star, TR, par0, q_H)
        np.testing.assert_allclose(w0, w0_damage_only, rtol=1e-9)


# ---------------------------------------------------------------------------
# Group D: degenerate / sanity
# ---------------------------------------------------------------------------

class TestD_Degenerate:
    """D1–D2: edge cases and end-to-end sanity checks."""

    def test_D1_neutral_payoffs_strict_equals_lax(self):
        """
        With t=tau=tau_BA=delta_loc=delta_glob=0 and g=0, the 'strict' and 'lax'
        labels are inert: identical firm compositions must produce identical welfare.

        Both markets face the same 4 firms (2H + 2L) at the same costs.
        TR = 0 everywhere. Damage = 0. So welfare = consumer surplus.
        Since p*[0] == p*[1], welfare[0] must equal welfare[1].
        """
        N   = 2
        par = make_params(t=0.0, tau=0.0, tau_BA=0.0,
                          delta_loc=0.0, delta_glob=0.0, g=0.0)
        f_H = np.array([1.0, 1.0])
        f_L = np.array([1.0, 1.0])
        sigma = np.array([1, 0])   # one strict, one lax — both inert
        P     = np.array([1.0, 1.0])
        W     = np.array([[0.0, 1.0], [1.0, 0.0]])

        p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, par)
        TR = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, par)
        w  = per_capita_welfare(f_H, sigma, P, p_star, TR, par)

        assert p_star[0] == pytest.approx(p_star[1], rel=1e-9), \
               "Symmetric markets must clear at the same price"
        assert TR[0] == pytest.approx(0.0, abs=1e-12), "No policy → no revenue"
        assert TR[1] == pytest.approx(0.0, abs=1e-12)
        assert w[0] == pytest.approx(w[1], rel=1e-9), (
            f"With all policy params zero, strict welfare {w[0]:.6f} "
            f"!= lax welfare {w[1]:.6f}"
        )

    def test_D2_empty_market_no_crash(self):
        """
        A jurisdiction with no firms and no neighbours must not raise.

        Documented behaviour: solve_market returns p_star = a (demand intercept)
        because the price array is initialised to a and no candidate loop runs.
        All quantities are zero.
        """
        par = make_params()
        f_H = np.array([0.0])
        f_L = np.array([0.0])
        sigma = np.array([0])
        P     = np.array([1.0])
        W     = np.zeros((1, 1))

        p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, par)

        assert p_star[0] == pytest.approx(par.a, rel=1e-9), (
            "Empty market: price defaults to demand intercept a "
            "(no goods traded, Q=0, p=a from inverse demand)"
        )
        assert q_H.sum() == pytest.approx(0.0, abs=1e-12)
        assert q_L.sum() == pytest.approx(0.0, abs=1e-12)

    def test_D2_empty_market_fiscal_revenue_zero(self):
        """fiscal_revenues must return 0 for an empty market (no crash, no NaN)."""
        par = make_params(t=5.0, tau=3.0, tau_BA=2.0)
        f_H   = np.array([0.0])
        f_L   = np.array([0.0])
        sigma = np.array([1])     # strict but empty
        P     = np.array([1.0])
        W     = np.zeros((1, 1))
        q_H   = np.zeros((1, 1))
        q_L   = np.zeros((1, 1))

        TR = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, par)

        assert TR[0] == pytest.approx(0.0, abs=1e-12)
        assert not np.any(np.isnan(TR))
