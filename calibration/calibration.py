"""
Phase-1 calibration module.

Computes three target moments for a given structural parameter vector:
  leakage_pct    — carbon leakage from relocation experiment  (target  5–20 %)
  import_pen_pct — quantity-weighted import penetration        (target  8–12 %)
  price_cv_pct   — coefficient of variation of p*_i           (target 10–30 %)
  price_iqr_pct  — IQR / median of p*_i                       (informational)

Free calibration parameters:
  delta_c  = c_L − c_H   (abatement gap; c_H anchored at cfg.c_H)
  t_ratio  = t / delta_c (carbon tax as multiple of the abatement gap)
  g                      (per-unit additive transport cost)

Held fixed (from params.py):
  a, b, c_H, F, mu, kappa, tau, tau_BA, mu_P, sigma_P, N, M, k, topology, h0.

Leakage experiment
------------------
1. Build a random network + initial firm distribution.
2. Solve the all-lax goods-market equilibrium → baseline H output per jurisdiction.
3. Switch a contiguous bloc (theta_strict × N jurisdictions) to strict.
4. Iterate: solve_market → relocate_firms until ‖Δf_H‖₁ < CONV_TOL for
   CONV_WINDOW consecutive periods (or T_MAX_RELOC periods).
5. Leakage = sum(max(ΔH_lax, 0)) / sum(max(−ΔH_strict, 0)) × 100 %
   where ΔH = total H-firm output (all destinations) post-shock − pre-shock.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import params as cfg
from params import Params
from model.network import build_network
from model.market import solve_market, firm_variable_profits
from model.firms import count_firms, relocate_firms

# ── Fixed structural anchors ───────────────────────────────────────────────────
C_H = cfg.c_H
A   = cfg.a
B   = cfg.b
F   = cfg.F

# ── Leakage experiment controls ────────────────────────────────────────────────
THETA_STRICT    = 0.30    # fraction of jurisdictions switched to strict
                          # Leakage is the pure competitive channel (no relocation):
                          # impose the tax, solve the goods-market equilibrium, measure.
                          # Relocation leakage is a treatment-variable effect (governed
                          # by mu) and is analysed separately in the main sweep.
MIN_TYPE_GROUPS = 3       # minimum active type-groups per market (assertion)

# ── Relocation leakage experiment controls ─────────────────────────────────────
T_MAX_RELOC  = 150   # max relocation periods
CONV_WINDOW  = 5     # consecutive periods within CONV_TOL to declare convergence
CONV_TOL     = 0.5   # L1 norm of Δf_H (absolute firm counts) to declare convergence

# ── Calibration targets ────────────────────────────────────────────────────────
TARGETS = {
    "leakage":    ( 5,  20),
    "import_pen": ( 8,  12),
    "price_cv":   (10,  30),
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_params(c_L: float, t: float, g: float, relocate: bool, seed: int) -> Params:
    return Params(
        N=cfg.N, M=cfg.M, k=cfg.k, topology=cfg.topology,
        a=A, b=B, c_H=C_H, c_L=c_L, t=t, g=g,
        tau=cfg.tau, tau_BA=cfg.tau_BA,
        delta_loc=cfg.delta_loc, delta_glob=cfg.delta_glob, F=F,
        mu=cfg.mu, lam=cfg.lam, kappa=cfg.kappa,
        relocate=relocate, dt=cfg.dt, T=cfg.T, seed=seed,
        mu_P=cfg.mu_P, sigma_P=cfg.sigma_P, s0=0.5, h0=cfg.h0,
    )


def _build_state(seed: int):
    """Build independent network + firm/population arrays for one replication."""
    rng       = np.random.default_rng(seed)
    P         = np.exp(rng.normal(cfg.mu_P, cfg.sigma_P, size=cfg.N))
    firm_loc  = rng.integers(0, cfg.N, size=cfg.M)
    firm_type = (rng.random(cfg.M) < cfg.h0).astype(int)
    _, W      = build_network(cfg.N, cfg.k, cfg.topology, seed)
    return rng, P, firm_loc, firm_type, W


def _H_out(f_H: np.ndarray, q_H: np.ndarray) -> np.ndarray:
    """
    Total H-type output produced in each jurisdiction j (summed over all
    destination markets i).  H_out[j] = f_H[j] × Σ_i q_H[i, j].
    """
    return f_H * q_H.sum(axis=0)


def _import_pen(f_H, f_L, q_H, q_L, W) -> float:
    """
    Aggregate import penetration (%).

    imports / (imports + domestic) summed over all markets.
    Quantities q already scale with population P_i, so this is implicitly
    population-weighted (OECD STAN spirit).
    """
    num = den = 0.0
    N = len(f_H)
    for i in range(N):
        den += f_H[i] * q_H[i, i] + f_L[i] * q_L[i, i]
        for j in range(N):
            if j != i and W[i, j] > 0:
                imp  = f_H[j] * q_H[i, j] + f_L[j] * q_L[i, j]
                num += imp
                den += imp
    return (num / den * 100.0) if den > 0 else float("nan")


def _price_disp(p_star: np.ndarray):
    """Returns (CV%, IQR/median%) of equilibrium prices."""
    mn = float(np.mean(p_star))
    if mn <= 0:
        return float("nan"), float("nan")
    cv  = float(np.std(p_star) / mn * 100.0)
    q25, q75 = np.percentile(p_star, [25, 75])
    iqr = float((q75 - q25) / float(np.median(p_star)) * 100.0)
    return cv, iqr


def _type_groups_per_market(f_H, f_L, W) -> float:
    """
    Mean number of active firm-type groups per market (domestic + trading
    neighbours).  Used to assert the market is non-degenerate.
    """
    N = len(f_H)
    counts = []
    for i in range(N):
        g = int(f_H[i] > 0) + int(f_L[i] > 0)
        for j in range(N):
            if W[i, j] > 0 and j != i:
                g += int(f_H[j] > 0) + int(f_L[j] > 0)
        counts.append(g)
    return float(np.mean(counts))


# ─────────────────────────────────────────────────────────────────────────────
# Core: single replication
# ─────────────────────────────────────────────────────────────────────────────

def run_one(delta_c: float, t_ratio: float, g: float, seed: int = 42) -> dict:
    """
    Compute all three moments for a single stochastic replication.

    Parameters
    ----------
    delta_c : c_L − c_H
    t_ratio : t / delta_c
    g       : per-unit transport cost
    seed    : RNG seed (controls network, firm placement, and relocation draws)

    Returns
    -------
    dict with keys leakage_pct, import_pen_pct, price_cv_pct, price_iqr_pct,
    n_active_mean, t_conv, c_L, t, error.
    error is None on success, a string on failure.
    """
    c_L = C_H + delta_c
    t   = t_ratio * delta_c

    rng, P, firm_loc, firm_type, W = _build_state(seed)

    p_static = _make_params(c_L, t, g, relocate=False, seed=seed)

    f_H, f_L  = count_firms(firm_loc, firm_type, cfg.N)
    sigma_lax = np.zeros(cfg.N, dtype=int)

    # ── a) All-lax baseline: import penetration ────────────────────────────
    p_lax, q_H_lax, q_L_lax = solve_market(f_H, f_L, sigma_lax, P, W, p_static)

    n_active = _type_groups_per_market(f_H, f_L, W)
    if n_active < MIN_TYPE_GROUPS:
        return {
            "error": f"too_few_active({n_active:.1f})",
            "leakage_pct": float("nan"), "import_pen_pct": float("nan"),
            "price_cv_pct": float("nan"), "price_iqr_pct": float("nan"),
            "n_active_mean": n_active,
        }

    import_pen = _import_pen(f_H, f_L, q_H_lax, q_L_lax, W)

    # ── b) Mixed-policy snapshot: price dispersion ─────────────────────────
    # First half of jurisdictions strict (deterministic; same across reps)
    sigma_mix = np.zeros(cfg.N, dtype=int)
    sigma_mix[: cfg.N // 2] = 1
    p_mix, _, _ = solve_market(f_H, f_L, sigma_mix, P, W, p_static)
    price_cv, price_iqr = _price_disp(p_mix)

    # ── c) Leakage experiment ───────────────────────────────────────────────
    n_strict  = max(1, round(THETA_STRICT * cfg.N))
    sigma_exp = np.zeros(cfg.N, dtype=int)
    sigma_exp[:n_strict] = 1          # contiguous bloc: first n_strict jurisdictions

    H_pre = _H_out(f_H, q_H_lax)   # baseline output, all-lax (same firm dist)

    # Pure competitive leakage: impose the policy shock, solve goods-market
    # equilibrium with the SAME firm distribution (no relocation).
    # Relocation leakage is governed by mu (treatment variable) and is not
    # calibrated here — it is swept in the main analysis.
    _, qH_post, _ = solve_market(f_H, f_L, sigma_exp, P, W, p_static)
    H_post        = _H_out(f_H, qH_post)

    lax_m    = sigma_exp == 0
    strict_m = sigma_exp == 1
    abate    = float(np.sum(np.maximum(H_pre[strict_m]  - H_post[strict_m], 0.0)))
    leak_abs = float(np.sum(np.maximum(H_post[lax_m]    - H_pre[lax_m],    0.0)))
    leakage  = (leak_abs / abate * 100.0) if abate > 1e-9 else float("nan")

    return {
        "leakage_pct":    leakage,
        "import_pen_pct": import_pen,
        "price_cv_pct":   price_cv,
        "price_iqr_pct":  price_iqr,
        "n_active_mean":  n_active,
        "c_L":            c_L,
        "t":              t,
        "error":          None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public: multi-rep aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_moments(delta_c: float, t_ratio: float, g: float,
                    n_reps: int = 3, base_seed: int = 42) -> dict:
    """
    Run n_reps replications and return mean ± std for each moment.

    Returns a dict suitable for writing to CSV or printing.
    The keys *_ok are 1 if the mean falls in the target band, 0 otherwise.
    """
    reps  = [run_one(delta_c, t_ratio, g, base_seed + r) for r in range(n_reps)]
    valid = [r for r in reps if r["error"] is None]

    def _ms(key):
        vals = [r[key] for r in valid if not np.isnan(r.get(key, float("nan")))]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

    lk_m,  lk_s  = _ms("leakage_pct")
    ip_m,  ip_s  = _ms("import_pen_pct")
    cv_m,  cv_s  = _ms("price_cv_pct")
    iqr_m, iqr_s = _ms("price_iqr_pct")
    na_m,  _     = _ms("n_active_mean")

    def _ok(v, lo, hi):
        return int(not np.isnan(v) and lo <= v <= hi)

    return {
        "delta_c": delta_c, "t_ratio": t_ratio, "g": g,
        "leakage_mean":     lk_m,  "leakage_std":    lk_s,
        "import_pen_mean":  ip_m,  "import_pen_std": ip_s,
        "price_cv_mean":    cv_m,  "price_cv_std":   cv_s,
        "price_iqr_mean":   iqr_m, "price_iqr_std":  iqr_s,
        "n_active_mean":    na_m,
        "n_valid":          len(valid),
        "leakage_ok":   _ok(lk_m,  *TARGETS["leakage"]),
        "import_ok":    _ok(ip_m,  *TARGETS["import_pen"]),
        "price_cv_ok":  _ok(cv_m,  *TARGETS["price_cv"]),
        "all_pass":     int(_ok(lk_m,  *TARGETS["leakage"]) and
                            _ok(ip_m,  *TARGETS["import_pen"]) and
                            _ok(cv_m,  *TARGETS["price_cv"])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Relocation leakage diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def measure_relocation_leakage(
    delta_c: float,
    t_ratio: float,
    g: float,
    mu: float,
    seed: int = 42,
) -> dict:
    """
    Decompose total leakage into competitive and relocation channels.

    1. Competitive leakage: same firm distribution, goods-market re-equilibration
       only (no relocation). Identical to the calibration experiment.
    2. Total leakage: run relocate_firms loop to steady state, then re-solve the
       goods market.
    3. Relocation leakage = total − competitive.

    Returns a dict with:
      competitive_pct, relocation_pct, total_pct,
      abatement_competitive, abatement_total,
      t_conv (periods to convergence; T_MAX_RELOC if not converged),
      converged (bool).
    """
    c_L = C_H + delta_c
    t   = t_ratio * delta_c

    rng, P, firm_loc, firm_type, W = _build_state(seed)

    # Build Params objects with mu set explicitly
    p_competitive = _make_params(c_L, t, g, relocate=False, seed=seed)
    p_reloc = _make_params(c_L, t, g, relocate=True,  seed=seed)
    p_reloc = Params(
        **{k: (mu if k == "mu" else getattr(p_reloc, k))
           for k in p_reloc.__dataclass_fields__}
    )

    f_H, f_L = count_firms(firm_loc, firm_type, cfg.N)

    sigma_lax = np.zeros(cfg.N, dtype=int)
    n_strict  = max(1, round(THETA_STRICT * cfg.N))
    sigma_exp = np.zeros(cfg.N, dtype=int)
    sigma_exp[:n_strict] = 1

    # ── Baseline: all-lax equilibrium (same firm distribution for both channels)
    _, q_H_lax, _ = solve_market(f_H, f_L, sigma_lax, P, W, p_competitive)
    H_pre = _H_out(f_H, q_H_lax)

    # ── Competitive channel: impose policy, no relocation ─────────────────────
    _, q_H_comp, _ = solve_market(f_H, f_L, sigma_exp, P, W, p_competitive)
    H_comp = _H_out(f_H, q_H_comp)

    lax_m    = sigma_exp == 0
    strict_m = sigma_exp == 1

    def _leakage(H_post):
        abate    = float(np.sum(np.maximum(H_pre[strict_m] - H_post[strict_m], 0.0)))
        leak_abs = float(np.sum(np.maximum(H_post[lax_m]   - H_pre[lax_m],    0.0)))
        leakage  = (leak_abs / abate * 100.0) if abate > 1e-9 else float("nan")
        return leakage, abate

    comp_pct, abate_comp = _leakage(H_comp)

    # ── Relocation channel: run to steady state ────────────────────────────────
    firm_loc_r  = firm_loc.copy()
    rng_r       = np.random.default_rng(seed + 1000)
    f_H_r, f_L_r = count_firms(firm_loc_r, firm_type, cfg.N)

    # Initial market solve under policy (starting point for relocation loop)
    p_star_r, q_H_r, q_L_r = solve_market(f_H_r, f_L_r, sigma_exp, P, W, p_reloc)
    pi_H_r, _ = firm_variable_profits(q_H_r, q_L_r, P, p_reloc)

    conv_streak = 0
    t_conv      = T_MAX_RELOC
    converged   = False

    for period in range(T_MAX_RELOC):
        f_H_prev = f_H_r.copy()

        firm_loc_r = relocate_firms(
            firm_loc_r, firm_type, sigma_exp, pi_H_r, p_reloc, rng_r, W
        )
        f_H_r, f_L_r = count_firms(firm_loc_r, firm_type, cfg.N)

        p_star_r, q_H_r, q_L_r = solve_market(f_H_r, f_L_r, sigma_exp, P, W, p_reloc)
        pi_H_r, _ = firm_variable_profits(q_H_r, q_L_r, P, p_reloc)

        delta = float(np.sum(np.abs(f_H_r - f_H_prev)))
        if delta < CONV_TOL:
            conv_streak += 1
            if conv_streak >= CONV_WINDOW:
                t_conv    = period + 1
                converged = True
                break
        else:
            conv_streak = 0

    if not converged:
        import warnings
        warnings.warn(
            f"measure_relocation_leakage: did not converge within {T_MAX_RELOC} "
            f"periods (mu={mu:.3f}, delta_c={delta_c}, t_ratio={t_ratio}). "
            f"Last Δf_H={delta:.3f}."
        )

    # Final H output after relocation steady state
    H_total = _H_out(f_H_r, q_H_r)
    total_pct, abate_total = _leakage(H_total)
    reloc_pct = (float(total_pct) - float(comp_pct)) if not (
        np.isnan(total_pct) or np.isnan(comp_pct)
    ) else float("nan")

    return {
        "mu":                  mu,
        "competitive_pct":     comp_pct,
        "relocation_pct":      reloc_pct,
        "total_pct":           total_pct,
        "abatement_competitive": abate_comp,
        "abatement_total":     abate_total,
        "t_conv":              t_conv,
        "converged":           converged,
    }
