
<!--EXP_A-->
## Experiment A — CBAM (border adjustment as a policy lever)

Held fixed: lambda=nu=1, phi=1500.0, delta_loc=1000.0, delta_glob=250.0, c_L=6.0, t=3.0, g=2.3. mu log-spaced 0.02–20. Because lambda=1, mu* = critical (mu/lambda)*.


### Lever: tau_BA
| lever | mu* = (mu/lambda)*  (0 = race at all mu; ≥20 = green at all mu) |
|---|---|
| 0.00 | 0 (race ∀μ) |
| 0.14 | 0 (race ∀μ) |
| 0.27 | 0 (race ∀μ) |
| 0.41 | 20.000 |
| 0.55 | 20.000 |
| 0.68 | 20.000 |
| 0.82 | 20.000 |
| 0.95 | 20.000 |
| 1.09 | 20.000 |
| 1.23 | 20.000 |
| 1.36 | 20.000 |
| 1.50 | 20.000 |
- minimum effective tau_BA to secure the green club at mu/lambda = 1: **0.41**
- minimum effective tau_BA to secure the green club at mu/lambda = 3: **0.41**
- per-unit effectiveness: d log10(mu*)/dtau_BA = **-0.000** decades per unit

### Lever: tau
| lever | mu* = (mu/lambda)*  (0 = race at all mu; ≥20 = green at all mu) |
|---|---|
| 0.00 | 0 (race ∀μ) |
| 0.14 | 0 (race ∀μ) |
| 0.27 | 0 (race ∀μ) |
| 0.41 | 0.225 |
| 0.55 | 20.000 |
| 0.68 | 20.000 |
| 0.82 | 20.000 |
| 0.95 | 20.000 |
| 1.09 | 20.000 |
| 1.23 | 20.000 |
| 1.36 | 20.000 |
| 1.50 | 20.000 |
- minimum effective tau to secure the green club at mu/lambda = 1: **0.55**
- minimum effective tau to secure the green club at mu/lambda = 3: **0.55**
- per-unit effectiveness: d log10(mu*)/dtau = **0.953** decades per unit

**Carbon-targeted vs generic:** tau_BA moves the boundary -0.00× as much per unit as tau (slopes -0.000 vs 0.953 decades/unit).
<!--/EXP_A-->

<!--EXP_B-->
## Experiment B — coalition nucleation (minimum viable club)

Held fixed: lambda=nu=1, h0=0.9 (dirty start), instruments tau=tau_BA=0.5, phi=1500.0, delta_loc=1000.0 (ON the structural boundary (contested)). mu log-spaced 0.02–20.


### Minimum viable club  s0*(mu)
| mu = (mu/lambda) | s0* (smallest green-club coalition) |
|---|---|
| 0.020 | 0.91 |
| 0.037 | 0.91 |
| 0.070 | 0.91 |
| 0.132 | 1.00 |
| 0.247 | 0.91 |
| 0.462 | 0.91 |
| 0.866 | 0.91 |
| 1.622 | 0.91 |
| 3.040 | 0.91 |
| 5.696 | 0.91 |
| 10.673 | 0.91 |
| 20.000 | 0.82 |
- at mu/lambda=1: s0* ≈ 0.91 (mu=0.87); at mu/lambda=3: s0* ≈ 0.91 (mu=3.04)

### Tipping (hysteresis) at mu=lambda=1, 3 seeds
- tipping s0 (mean final s crosses 0.5): **0.91**
- stochastic transition-band width (seeds disagree): **0.09** in s0
<!--/EXP_B-->

<!--EXP_B_GREEN-->
## Experiment B — coalition nucleation (minimum viable club)

Held fixed: lambda=nu=1, h0=0.9 (dirty start), instruments tau=tau_BA=0.5, phi=1350.0, delta_loc=1000.0 (GREEN-SIDE of the boundary). mu log-spaced 0.02–20.


### Minimum viable club  s0*(mu)
| mu = (mu/lambda) | s0* (smallest green-club coalition) |
|---|---|
| 0.020 | 0.73 |
| 0.037 | 0.73 |
| 0.070 | 0.91 |
| 0.132 | 0.73 |
| 0.247 | 0.82 |
| 0.462 | 0.91 |
| 0.866 | 0.91 |
| 1.622 | 0.91 |
| 3.040 | 0.91 |
| 5.696 | 0.91 |
| 10.673 | 0.91 |
| 20.000 | 0.82 |
- at mu/lambda=1: s0* ≈ 0.91 (mu=0.87); at mu/lambda=3: s0* ≈ 0.91 (mu=3.04)

### Tipping (hysteresis) at mu=lambda=1, 3 seeds
- tipping s0 (mean final s crosses 0.5): **0.91**
- stochastic transition-band width (seeds disagree): **0.09** in s0
<!--/EXP_B_GREEN-->
