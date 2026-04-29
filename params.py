# ─────────────────────────────────────────────────────────────────────────────
#  params.py  —  edit this file to configure the simulation
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

# Simulation control
T    = 200   # number of periods to run
seed = 42    # random seed for reproducibility

# Network
N        = 50      # number of jurisdictions  (must be > k+1 for ring)
k        = 4       # degree (ring: exact; ER / BA: mean degree)  (must be > 2)
topology = "ring"  # "ring" | "er" | "ba"

# Goods market
a   = 20.0   # demand intercept
b   = 0.5    # slope of per-consumer demand
c_H = 2.0    # marginal cost, high-emission firm
c_L = 4.0    # marginal cost, low-emission firm
t   = 5.0    # carbon tax rate (applied to H-firms in strict jurisdictions)

# Trade and environment
tau     = 2.0   # bilateral fiscal tariff rate (between strict and lax neighbours)
c_trade = 1.0   # per-unit transport / iceberg cost paid by any exporting firm
tau_BA  = 0.0   # per-unit border carbon adjustment on lax→strict exports (set >0 to activate)
delta   = 0.3   # environmental damage parameter
F       = 0.5   # firm fixed cost per period
w_bar   = 1.0   # wage scaling parameter

# Dynamics
mu  = 1.0   # firm mobility rate          (eq. 3.17)
lam = 0.5   # policy imitation rate       (eq. 3.31)
dt  = 1.0   # time step size

# Initial condition parameters
M       = 500   # total number of firms
s0      = 0.5   # initial fraction of strict jurisdictions
h0      = 0.8   # initial fraction of high-emission firms
mu_P    = 7.0   # log-normal population mean  (eq. 3.1)
sigma_P = 1.5   # log-normal population std   (eq. 3.1)

# ─────────────────────────────────────────────────────────────────────────────
#  Generated initial conditions  (derived from the parameters above + seed)
#  Re-generated automatically whenever you change N, M, s0, h0, mu_P, sigma_P.
# ─────────────────────────────────────────────────────────────────────────────

_rng0      = np.random.default_rng(seed)
P_init     = list(np.exp(_rng0.normal(mu_P, sigma_P, size=N)))
sigma_init = list((_rng0.random(N) < s0).astype(int))
_locs      = _rng0.integers(0, N, size=M)
_types     = _rng0.random(M) < h0
firms_init = [(_locs[m], "H" if _types[m] else "L") for m in range(M)]

# ─────────────────────────────────────────────────────────────────────────────
#  Params dataclass — used internally by model/ functions; no need to edit
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class Params:
    # Network
    N: int = 50
    k: int = 4
    topology: str = "ring"
    # Firms
    M: int = 500
    c_H: float = 2.0
    c_L: float = 4.0
    F: float = 0.5
    # Policy
    t: float = 5.0
    # Goods market
    a: float = 20.0
    b: float = 0.5
    # Tariff / trade
    tau: float = 2.0
    c_trade: float = 1.0
    tau_BA: float = 0.0
    # Population
    mu_P: float = 7.0
    sigma_P: float = 1.5
    # Welfare
    delta: float = 0.3
    w_bar: float = 1.0
    # Dynamics
    mu: float = 1.0
    lam: float = 0.5
    pi_ref: float = 1.0
    # Simulation
    T: int = 200
    dt: float = 1.0
    seed: int = 42
    s0: float = 0.5
    h0: float = 0.8


BASELINE = Params()
