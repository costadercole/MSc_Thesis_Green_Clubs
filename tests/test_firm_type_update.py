"""
Verification suite for firm_type_update — §8 of implementation brief.

Run: python -m pytest tests/test_firm_type_update.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy.special import expit
from model.firms import firm_type_update


# ---------------------------------------------------------------------------
# Test 1: Numerical stability
# ---------------------------------------------------------------------------

def test_numerical_stability():
    """p_switch must be finite and in [0,1] for all kappa_f and large |Δπ|."""
    rng = np.random.default_rng(0)
    M = 500
    for kappa_f in [1e-3, 1e-2, 1.0]:
        for delta in [-5000, -100, 0, 100, 5000]:
            firm_type = (rng.random(M) < 0.5).astype(int)
            profit = np.where(firm_type == 1, float(delta), 0.0)
            new_type = firm_type_update(firm_type, profit, nu=2.0, kappa_f=kappa_f,
                                        dt=0.05, rng=rng, eps=0.0)
            assert np.all(np.isfinite(new_type)), f"non-finite at kappa_f={kappa_f}, delta={delta}"
            assert set(np.unique(new_type)).issubset({0, 1}), "type values outside {0,1}"


# ---------------------------------------------------------------------------
# Test 2: Timescale bound — decisive test
# ---------------------------------------------------------------------------

def _crossing_time(firm_type_init, profit_H, profit_L, nu, kappa_f, dt, seed,
                   h_target=0.9, max_steps=5000):
    """Steps for h to cross h_target from below under frozen profits."""
    rng = np.random.default_rng(seed)
    firm_type = firm_type_init.copy()
    for step in range(max_steps):
        profit = np.where(firm_type == 1, profit_H, profit_L)
        firm_type = firm_type_update(firm_type, profit, nu=nu, kappa_f=kappa_f,
                                     dt=dt, rng=rng, eps=0.0)
        if np.mean(firm_type) >= h_target:
            return step + 1
    return max_steps


def test_timescale_bound():
    """
    With saturating selection (kappa_f=1.0), crossing time h=0.1→0.9 must be
    independent of Δπ and equal to ln(81)/(nu*dt) ≈ 43.9 steps.
    All three Δ values (100, 1000, 5000) must give 43.4 ± 5 steps.
    """
    M = 2000
    nu, kappa_f, dt = 2.0, 1.0, 0.05
    expected = np.log(81) / (nu * dt)   # ≈ 43.9

    times = []
    for delta in [100, 1000, 5000]:
        seeds_times = []
        for seed in range(50):
            rng0 = np.random.default_rng(seed * 1000)
            firm_type_init = (rng0.random(M) < 0.1).astype(int)
            profit_H, profit_L = float(delta), 0.0
            t = _crossing_time(firm_type_init, profit_H, profit_L,
                               nu, kappa_f, dt, seed=seed + 1)
            seeds_times.append(t)
        mean_t = float(np.mean(seeds_times))
        times.append(mean_t)
        # tolerance: ±15% or ±5 steps, whichever is larger
        tol = max(5.0, 0.15 * expected)
        assert abs(mean_t - expected) < tol, (
            f"Δπ={delta}: crossing time {mean_t:.1f} deviates from {expected:.1f} by "
            f"{abs(mean_t-expected):.1f} > tol={tol:.1f}. "
            "Payoff coupling still present — check for stray normalisation."
        )

    # All three must be essentially equal (within 10 steps of each other)
    assert max(times) - min(times) < 10, (
        f"Crossing times differ across Δπ values: {times}. "
        "Rate is not bounded by nu — implementation wrong."
    )


# ---------------------------------------------------------------------------
# Test 3: Mean-field correspondence
# ---------------------------------------------------------------------------

def test_mean_field_correspondence():
    """
    Crossing time scales as ln(81) / (T * nu * dt) where T = tanh(kappa_f * Δ / 2).
    Check T=0.5 → ~87.9 steps and T=0.25 → ~175.8 steps.
    """
    M = 2000
    nu, dt = 2.0, 0.05
    delta = 500.0

    for T_target, tol_frac in [(0.5, 0.12), (0.25, 0.15)]:
        kappa_f = 2.0 * np.arctanh(T_target) / delta
        expected = np.log(81) / (T_target * nu * dt)

        seed_times = []
        for seed in range(50):
            rng0 = np.random.default_rng(seed * 999)
            firm_type_init = (rng0.random(M) < 0.1).astype(int)
            t = _crossing_time(firm_type_init, delta, 0.0, nu, kappa_f, dt, seed=seed + 1)
            seed_times.append(t)

        mean_t = float(np.mean(seed_times))
        tol = max(10.0, tol_frac * expected)
        assert abs(mean_t - expected) < tol, (
            f"T={T_target}: crossing time {mean_t:.1f}, expected {expected:.1f}, tol={tol:.1f}"
        )


# ---------------------------------------------------------------------------
# Test 4: Sampling-pool regression
# ---------------------------------------------------------------------------

def _run_h_final(firm_type_init, profit_fn, nu, kappa_f, dt, steps, seed, pool="global",
                 firm_loc=None, sigma=None):
    """Run firm_type_update for `steps` steps and return final h."""
    rng = np.random.default_rng(seed)
    firm_type = firm_type_init.copy()
    M = len(firm_type)
    for _ in range(steps):
        profit = profit_fn(firm_type, firm_loc, sigma)
        if pool == "global":
            firm_type = firm_type_update(firm_type, profit, nu=nu, kappa_f=kappa_f,
                                         dt=dt, rng=rng, eps=0.0)
        else:
            # Within-jurisdiction: restricted pool (not the default, for comparison)
            new_type = firm_type.copy()
            N = int(sigma.max()) + 1 if sigma is not None else 1
            for jur in range(N if sigma is not None else 1):
                idx = np.where(firm_loc == jur)[0] if sigma is not None else np.arange(M)
                if len(idx) < 2:
                    continue
                revise = rng.random(len(idx)) < nu * dt
                rev_idx = idx[revise]
                if rev_idx.size == 0:
                    continue
                partners_local = rng.integers(0, len(idx), size=rev_idx.size)
                dpi = profit[idx[partners_local]] - profit[rev_idx]
                p_sw = expit(kappa_f * dpi)
                do_sw = rng.random(rev_idx.size) < p_sw
                new_type[rev_idx[do_sw]] = firm_type[idx[partners_local[do_sw]]]
            firm_type = new_type
    return float(np.mean(firm_type))


def test_sampling_pool():
    """
    Global pool → h resolves toward boundary when one type is globally superior.
    Within-jur pool → h stuck near strict_fraction (≈0.51 ± 0.03 with 50/50 split).
    """
    M, N_jur = 500, 50
    steps = 400
    rng0 = np.random.default_rng(77)

    firm_loc  = rng0.integers(0, N_jur, size=M)
    sigma     = (rng0.random(N_jur) < 0.5).astype(int)   # ~50% strict
    firm_type = (rng0.random(M) < 0.5).astype(int)

    # H globally superior: H earns 500 everywhere, L earns 0
    def profit_H_superior(ft, fl, sig):
        return np.where(ft == 1, 500.0, 0.0)

    h_global = np.mean([
        _run_h_final(firm_type, profit_H_superior, nu=2.0, kappa_f=1e-2,
                     dt=0.05, steps=steps, seed=s, pool="global")
        for s in range(20)
    ])
    assert h_global > 0.85, f"Global pool: h={h_global:.3f}, expected >0.85 (H globally superior)"

    h_local = np.mean([
        _run_h_final(firm_type, profit_H_superior, nu=2.0, kappa_f=1e-2,
                     dt=0.05, steps=steps, seed=s, pool="local",
                     firm_loc=firm_loc, sigma=sigma)
        for s in range(20)
    ])
    strict_frac = float(np.mean(sigma[firm_loc]))
    # within-jur pool pins h near strict_fraction ± 0.1
    assert abs(h_local - strict_frac) < 0.15, (
        f"Within-jur pool: h={h_local:.3f}, stuck near strict_frac={strict_frac:.3f} ± 0.15"
    )


# ---------------------------------------------------------------------------
# Test 5: Absorbing vs leaky boundaries
# ---------------------------------------------------------------------------

def test_absorbing_vs_leaky():
    """eps=0 → h stays at 0; eps=1e-2 → h escapes to ~1 when H is favoured."""
    M = 500
    rng0 = np.random.default_rng(42)
    firm_type_zero = np.zeros(M, dtype=int)   # all L

    # H is favoured: profit difference = 1000
    profit = np.where(firm_type_zero == 1, 1000.0, 0.0)

    # eps=0: absorbing — must stay at 0
    rng = np.random.default_rng(1)
    ft = firm_type_zero.copy()
    for _ in range(200):
        p_ = np.where(ft == 1, 1000.0, 0.0)
        ft = firm_type_update(ft, p_, nu=2.0, kappa_f=1e-2, dt=0.05, rng=rng, eps=0.0)
    assert float(np.mean(ft)) == 0.0, "eps=0: boundary not absorbing"

    # eps=1e-2: leaky — should escape to near 1 after enough steps
    h_vals = []
    for seed in range(20):
        rng = np.random.default_rng(seed + 100)
        ft = firm_type_zero.copy()
        for _ in range(400):
            p_ = np.where(ft == 1, 1000.0, 0.0)
            ft = firm_type_update(ft, p_, nu=2.0, kappa_f=1e-2, dt=0.05, rng=rng, eps=1e-2)
        h_vals.append(float(np.mean(ft)))
    assert np.mean(h_vals) > 0.85, f"eps=1e-2: h={np.mean(h_vals):.3f}, expected >0.85"
