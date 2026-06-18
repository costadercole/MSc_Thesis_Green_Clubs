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
c_L = 6.0    # marginal cost, low-emission firm
t   = 5.0    # carbon tax rate (applied to H-firms in strict jurisdictions)

# Trade and environment
tau    = 3.0  # bilateral fiscal tariff rate (between strict and lax neighbours)
g      = 2.4   # per-unit additive transport cost paid by any exporting firm
tau_BA = 5   # per-unit border carbon adjustment on lax→strict exports
delta_loc  = 2000.0   # local environmental damage parameter (per-jurisdiction H-firm density)
delta_glob = 500.0    # global environmental damage parameter (economy-wide H-firm density)
F      = 0.0   # firm fixed cost per period (set to zero; keep parameter for later use)

# Dynamics
mu    = 1.2   # firm mobility rate          (eq. 3.17)
lam   = 2.0   # policy imitation rate
kappa = 1.0   # selection intensity for Fermi replicator (eq. 3.53)
dt    = 0.1   # time step size

# Firm relocation toggle
relocate = False   # set False to disable firm relocation entirely

# Initial condition parameters
M       = 500   # total number of firms
s0      = 0.5   # initial fraction of strict jurisdictions
h0      = 0.5   # initial fraction of high-emission firms
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


from dataclasses import dataclass, field

@dataclass
class Params:
    # Network
    N: int = 50
    k: int = 4
    topology: str = "ring"
    # Firms
    M: int = 500
    c_H: float = 4.0
    c_L: float = 6.0
    F: float = 0.0
    # Policy
    t: float = 5.0
    # Goods market
    a: float = 20.0
    b: float = 1.0
    # Tariff / trade
    tau: float = 18.0
    g: float = 2.4        # additive transport cost
    tau_BA: float = 0.0
    # Population
    mu_P: float = 7.0
    sigma_P: float = 1.5
    # Welfare / damage
    delta_loc:  float = 2000.0  # local damage  (per-jurisdiction H-firm density)
    delta_glob: float = 500.0   # global damage (economy-wide H-firm density)
    # Dynamics
    mu: float = 1.2
    lam: float = 0.5
    kappa: float = 1.0    # Fermi selection intensity
    relocate: bool = False # toggle firm relocation on/off
    # Simulation
    T: int = 200
    dt: float = 1.0
    seed: int = 42
    s0: float = 0.5
    h0: float = 0.5


BASELINE = Params()
