"""
Welfare-decomposition layer (normative output).

Per-period jurisdiction welfare (eq. 3.42, the same object the Fermi imitation
step uses as fitness):

    W_i / P_i = cs_i  +  TR_i/P_i  +  phi*(E^H_i + E^L_i)/P_i  -  D_i
    D_i       = delta_loc * E^H_i/P_i  +  delta_glob * (sum_j E^H_j)/(sum_j P_j)

This module returns each of the four components SEPARATELY (consumer surplus,
fiscal revenue, host benefit, environmental damage), in both per-capita and total
(x P_i) form, and aggregates them — globally and split by the strict / lax blocs.

Conventions reused from the model (do not duplicate dynamics):
  - solve_market (monotone elimination)        -> p*, q_H, q_L
  - fiscal_revenues (eqs 3.29-3.31)            -> TR_i
  - the welfare formula matches model.jurisdictions.per_capita_welfare exactly.

CRITICAL distinction kept throughout:
  - per-capita welfare W_i/P_i  = what AGENTS respond to (imitation fitness)
  - total social welfare W_tot  = sum_i W_i = what the ANALYST optimises
"""

import numpy as np
from model.market import solve_market
from model.jurisdictions import fiscal_revenues


def welfare_components(f_H, f_L, sigma, P, p_star, q_H, q_L, W, p):
    """
    Decompose welfare at one market state into its four components.

    Returns a dict of per-jurisdiction arrays (length N).  Each component is given
    per-capita ('*_pc') and total ('*_tot' = per-capita x P_i).  W_pc and W_tot are
    the assembled welfare; by construction the four totals sum to W_tot.
    """
    P_safe = np.maximum(P, 1.0)

    # per-capita components
    cs_pc   = (p.a - p_star) ** 2 / (2 * p.b)
    TR      = fiscal_revenues(f_H, f_L, sigma, W, q_H, q_L, p)
    fisc_pc = TR / P_safe

    E_H = f_H * q_H.sum(axis=0)          # total H-output produced in i (eq. host/damage base)
    E_L = f_L * q_L.sum(axis=0)
    phi = getattr(p, "phi", 0.0)
    host_pc = phi * (E_H + E_L) / P_safe

    dloc_pc = p.delta_loc * E_H / P_safe
    dglob_pc = p.delta_glob * E_H.sum() / P_safe.sum()      # scalar, identical across i
    dglob_pc = np.full_like(P_safe, dglob_pc)
    dmg_pc = dloc_pc + dglob_pc

    W_pc = cs_pc + fisc_pc + host_pc - dmg_pc

    out = dict(
        cs_pc=cs_pc, fisc_pc=fisc_pc, host_pc=host_pc,
        dmg_loc_pc=dloc_pc, dmg_glob_pc=dglob_pc, dmg_pc=dmg_pc, W_pc=W_pc,
    )
    # totals (x population)
    for k in list(out.keys()):
        out[k.replace("_pc", "_tot")] = out[k] * P_safe
    out["P"] = P_safe
    return out


def aggregate(comp, sigma):
    """
    Aggregate per-jurisdiction components into global and per-bloc totals.

    Sign convention: damage is reported as a positive magnitude; W_tot already nets it.
    """
    P = comp["P"]
    strict = sigma == 1
    lax = sigma == 0

    def block(mask):
        Psum = float(P[mask].sum())
        d = dict(
            n=int(mask.sum()), pop=Psum,
            W_tot=float(comp["W_tot"][mask].sum()),
            cs_tot=float(comp["cs_tot"][mask].sum()),
            fisc_tot=float(comp["fisc_tot"][mask].sum()),
            host_tot=float(comp["host_tot"][mask].sum()),
            dmg_tot=float(comp["dmg_tot"][mask].sum()),
            dmg_loc_tot=float(comp["dmg_loc_tot"][mask].sum()),
            dmg_glob_tot=float(comp["dmg_glob_tot"][mask].sum()),
        )
        d["W_pc"] = d["W_tot"] / Psum if Psum > 0 else float("nan")
        return d

    allmask = np.ones_like(sigma, dtype=bool)
    res = dict(all=block(allmask), strict=block(strict), lax=block(lax))
    return res


def welfare_at_state(f_H, f_L, sigma, P, W, p):
    """Solve the market at one (firms, policy) state and return aggregated welfare."""
    p_star, q_H, q_L = solve_market(f_H, f_L, sigma, P, W, p)
    comp = welfare_components(f_H, f_L, sigma, P, p_star, q_H, q_L, W, p)
    return aggregate(comp, sigma), comp, p_star


def steady_state_welfare(results, p, tail=0.2, n_samples=40):
    """
    Average aggregated welfare over the final `tail` fraction of a run.

    `results` is the dict returned by model.simulation.run (it records f_H, f_L,
    p_star, sigma per step plus P and W). The market is re-solved at up to
    `n_samples` evenly-spaced steps inside the window (steady state is ~stationary,
    so a stride keeps the cost low without biasing the mean); the four components
    are exact and p* is checked against the recorded value in the unit tests.

    Returns a flat dict of scalars suitable for one CSV row.
    """
    T = len(results["h"])
    start = int(T * (1 - tail))
    P, W = results["P"], results["W"]
    steps = np.unique(np.linspace(start, T - 1, min(n_samples, T - start)).astype(int))

    keys_all = ["W_tot", "cs_tot", "fisc_tot", "host_tot", "dmg_tot",
                "dmg_loc_tot", "dmg_glob_tot", "W_pc"]
    acc = {f"{blk}_{k}": [] for blk in ("all", "strict", "lax") for k in keys_all}
    acc.update({f"{blk}_{q}": [] for blk in ("strict", "lax") for q in ("n", "pop")})

    for step in steps:
        f_H = results["f_H"][step]; f_L = results["f_L"][step]
        sigma = results["sigma"][step]
        agg, _, _ = welfare_at_state(f_H, f_L, sigma, P, W, p)
        for blk in ("all", "strict", "lax"):
            for k in keys_all:
                acc[f"{blk}_{k}"].append(agg[blk][k])
            if blk != "all":
                acc[f"{blk}_n"].append(agg[blk]["n"])
                acc[f"{blk}_pop"].append(agg[blk]["pop"])

    def _safe_mean(v):
        # An empty bloc (e.g. no strict jurisdictions in a race-to-bottom run)
        # yields all-NaN W_pc samples; nanmean of those is a (correct) NaN but
        # warns "Mean of empty slice". Skip the mean when nothing is finite.
        arr = np.asarray(v, dtype=float)
        return float(np.nanmean(arr)) if arr.size and np.isfinite(arr).any() else float("nan")

    return {k: _safe_mean(v) for k, v in acc.items()}
