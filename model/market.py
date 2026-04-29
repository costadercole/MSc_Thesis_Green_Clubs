"""
Local goods market clearing — equations (3.7)–(3.15).

All functions operate on per-jurisdiction arrays and are called every time step
before firm profits or jurisdiction welfare are computed.
"""

import numpy as np


def equilibrium_prices(
    f_H: np.ndarray,   # (N,) number of high-emission firms per jurisdiction
    f_L: np.ndarray,   # (N,) number of low-emission firms per jurisdiction
    sigma: np.ndarray, # (N,) regulatory policy: 1 = strict (S), 0 = lax (L)
    P: np.ndarray,     # (N,) population per jurisdiction
    c_H: float,
    c_L: float,
    t: float,
    a: float,
    b: float,
) -> np.ndarray:
    """
    Closed-form equilibrium price for each jurisdiction (eq. 3.12).

    p*_i = (a * P_i + c_bar_i * f_i) / (P_i + f_i)

    Jurisdictions with no firms get price = a (no supply → consumers pay max WTP).
    """
    f_i = f_H + f_L
    c_eff_H = c_H + t * sigma          # (N,) effective marginal cost of H-firms
    c_bar = np.where(
        f_i > 0,
        (f_H * c_eff_H + f_L * c_L) / np.maximum(f_i, 1),
        c_L,
    )                                   # (N,) average effective marginal cost
    p_star = np.where(
        f_i > 0,
        (a * P + c_bar * f_i) / (P + f_i),
        a,
    )
    return p_star


def firm_variable_profits(
    p_star: np.ndarray,  # (N,)
    f_H: np.ndarray,     # (N,)
    f_L: np.ndarray,     # (N,)
    sigma: np.ndarray,   # (N,)
    P: np.ndarray,       # (N,)
    c_H: float,
    c_L: float,
    t: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-firm variable profit for H- and L-type firms in each jurisdiction (eq. 3.15).

    pi_var = (p* - mc_eff)^2 / (b / P)

    Returns (pi_H, pi_L), each shape (N,).  Negative values are clipped to 0
    (firms that cannot cover marginal cost produce nothing).
    """
    c_eff_H = c_H + t * sigma
    pi_H = np.maximum(0.0, (p_star - c_eff_H) ** 2 / (b / np.maximum(P, 1)))
    pi_L = np.maximum(0.0, (p_star - c_L) ** 2 / (b / np.maximum(P, 1)))
    return pi_H, pi_L


def equilibrium_prices_with_trade(
    f_H: np.ndarray,    # (N,)
    f_L: np.ndarray,    # (N,)
    sigma: np.ndarray,  # (N,)
    P: np.ndarray,      # (N,)
    W: np.ndarray,      # (N, N) weight matrix
    c_H: float,
    c_L: float,
    t: float,
    c_trade: float,
    tau_BA: float,
    a: float,
    b: float,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Equilibrium prices with import competition (Option B trade dynamics).

    Algorithm:
      1. Start from domestic-only p* (eq. 3.12).
      2. For each jurisdiction i, check each neighbour j: exporting firm from j
         to i is profitable iff its marginal cost + c_trade (+ tau_BA if lax→strict)
         is below the current p*_i.
      3. Add profitable exporters to the effective supply in market i, recompute p*.
      4. Repeat until p* converges.

    Returns (p_star, export_counts) where export_counts[i] = number of
    effective foreign firms competing in jurisdiction i's market.

    Border carbon adjustment:
      - A lax-jurisdiction firm exporting to a strict jurisdiction pays tau_BA
        per unit on top of c_trade (import tariff levied by the strict importer).
      - Strict→lax and strict→strict exports pay only c_trade.
    """
    c_eff_H = c_H + t * sigma          # (N,) domestic effective cost of H-firms
    c_eff_L = np.full(N := len(f_H), c_L)

    # initial domestic-only prices
    p_star = equilibrium_prices(f_H, f_L, sigma, P, c_H, c_L, t, a, b)

    export_counts = np.zeros(N, dtype=float)

    for _ in range(max_iter):
        p_prev = p_star.copy()
        export_counts[:] = 0.0

        for i in range(N):
            extra_supply_num = 0.0   # Σ mc_export (numerator contribution)
            extra_supply_den = 0.0   # number of extra firms

            for j in range(N):
                if W[i, j] == 0 or i == j:
                    continue
                w_ij = W[i, j]  # weight; used only to decide whether link exists (w>0)

                # H-firm from j exporting to i
                bca_H = tau_BA if (sigma[j] == 0 and sigma[i] == 1) else 0.0
                mc_H_export = c_H + t * sigma[j] + c_trade + bca_H
                if mc_H_export < p_star[i]:
                    n_H_export = f_H[j] * w_ij   # scaled by link weight
                    extra_supply_num += mc_H_export * n_H_export
                    extra_supply_den += n_H_export
                    export_counts[i] += n_H_export

                # L-firm from j exporting to i
                bca_L = 0.0   # L-firms face no BCA (no carbon to adjust)
                mc_L_export = c_L + c_trade + bca_L
                if mc_L_export < p_star[i]:
                    n_L_export = f_L[j] * w_ij
                    extra_supply_num += mc_L_export * n_L_export
                    extra_supply_den += n_L_export
                    export_counts[i] += n_L_export

            # Recompute p*_i including exporters
            f_i = f_H[i] + f_L[i]
            c_bar_dom = np.where(
                f_i > 0,
                (f_H[i] * c_eff_H[i] + f_L[i] * c_L) / max(f_i, 1),
                c_L,
            )
            total_n = f_i + extra_supply_den
            if total_n > 0:
                c_bar_eff = (c_bar_dom * f_i + extra_supply_num) / total_n
                p_star[i] = (a * P[i] + c_bar_eff * total_n) / (P[i] + total_n)
            else:
                p_star[i] = a

        if np.max(np.abs(p_star - p_prev)) < tol:
            break

    return p_star, export_counts


def consumer_surplus(
    p_star: np.ndarray,  # (N,)
    a: float,
    b: float,
) -> np.ndarray:
    """Per-consumer surplus in each jurisdiction (eq. 3.20)."""
    return (a - p_star) ** 2 / (2 * b)


def environmental_damage(
    f_H: np.ndarray,  # (N,)
    P: np.ndarray,    # (N,)
    delta: float,
) -> np.ndarray:
    """Per-capita environmental damage in each jurisdiction (eq. 3.21)."""
    return delta * f_H / np.maximum(P, 1)
