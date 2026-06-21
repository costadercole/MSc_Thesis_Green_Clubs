import numpy as np

# Simulation control
T    = 1000   # number of periods to run
seed = 42    # random seed for reproducibility

# Network
N        = 50      # number of jurisdictions  (must be > k+1 for ring)
k        = 3       # degree (ring: exact; ER / BA: mean degree)  (must be > 2)
topology = "er"  # "ring" | "er" | "ba"

# Goods market
a   = 20.0   # demand intercept
b   = 1.0    # slope of per-consumer demand
c_H = 4.0    # marginal cost, high-emission firm
c_L = 6.0    # marginal cost, low-emission firm  (Δc = 2.0, calibrated — Set B)
t   = 3.0    # carbon tax rate — t/Δc=1.5 (calibrated); partial crowd-out: H weakly present in strict (c_H+t=7.0 < p*_L-only=8.33)

# Trade and environment
tau    = 0.5    # bilateral fiscal tariff rate (between strict and lax neighbours)
g      = 2.3    # per-unit additive transport cost (calibrated — Set B; hits import-penetration 8–12%)
tau_BA = 0.5    # per-unit border carbon adjustment on lax→strict exports
delta_loc  = 1000.0   # local environmental damage parameter   (calibrated — contested-regime baseline)
delta_glob = 250.0    # global environmental damage parameter  (delta_glob = delta_loc / 4)
phi        = 1500.0   # host economic benefit per unit of local output (jobs + non-carbon tax base) — calibrated contested point
F      = 0.0    # firm fixed cost per period

# Dynamics
mu      = 1.2    # firm mobility rate              (eq. 3.36)
lam     = 2.0    # policy imitation rate           (eq. 3.41)
nu      = 2.0    # firm type-revision rate         (eq. 3.42); start at nu ≈ lam
kappa   = 1.0    # selection intensity, jurisdiction Fermi update (eq. 3.41)
kappa_f = 1e-3   # selection intensity, firm type Fermi update   (eq. 3.42)
eps     = 1e-4   # spontaneous mutation rate (small: keeps boundaries leaky without swamping selection)
dt      = 0.05   # time step size

# Firm relocation toggle
relocate = True   # relocation enabled by default

# Initial condition parameters
M       = 500   # total number of firms
s0      = 0.5   # initial fraction of strict jurisdictions
h0      = 0.5   # initial fraction of high-emission firms
mu_P    = 7.0   # log-normal population mean  (eq. 3.1)
sigma_P = 0.0   # population std — set to 0 for homogeneous jurs (heterogeneity dominates TR/P)

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


from dataclasses import dataclass, field

@dataclass
class Params:
    # Network
    N: int = 50
    k: int = 3
    topology: str = "er"
    # Firms
    M: int = 500
    c_H: float = 4.0
    c_L: float = 6.0      # Δc = 2.0 (calibrated — Set B)
    F: float = 0.0
    # Policy
    t: float = 3.0        # t/Δc=1.5 (calibrated); partial crowd-out: H weakly present in strict (c_H+t=7.0 < p*_L-only=8.33)
    # Goods market
    a: float = 20.0
    b: float = 1.0
    # Tariff / trade
    tau: float = 0.5
    g: float = 2.3        # calibrated (Set B); hits import-penetration 8–12%
    tau_BA: float = 0.5
    # Population
    mu_P: float = 7.0
    sigma_P: float = 0.0
    # Welfare / damage
    delta_loc:  float = 1000.0   # calibrated — contested-regime baseline
    delta_glob: float = 250.0    # delta_glob = delta_loc / 4
    phi:        float = 1500.0   # host benefit per unit of local output (jobs + tax base) — calibrated contested point
    # Dynamics
    mu: float = 1.2
    lam: float = 2.0
    nu: float = 2.0
    kappa: float = 1.0
    kappa_f: float = 1e-3
    eps: float = 1e-4
    relocate: bool = True
    # Simulation
    T: int = 1000
    dt: float = 0.05
    seed: int = 42
    s0: float = 0.5
    h0: float = 0.5
    T_burnin: int = 50   # relocation-only steps before policy updates begin


BASELINE = Params()
