# 1D Radial Transport Solvers

Two solvers sharing the same nonlinear flux model structure. Previous iterations (simple, coupled, original single-field, original two-species) are in `archive/`.

---

## Overview

| Driver | Model | Fields | Gradient type | Features |
|--------|-------|--------|---------------|----------|
| `flux_transport_driver_clean.py` | `flux_transport_model_clean.py` | `p` | absolute `g = -dp/dx` | Bremsstrahlung + alpha heating, initial_only power balance, gradient smoothing, video output |
| `flux_transport_driver_two_species_clean.py` | `flux_transport_model_two_species_clean.py` | `p_e, p_i, n` | log `kappa = -d(ln f)/dx` | Stix alpha e/i split, Q_ei coupling, quasi-neutrality, gradient smoothing, video output |

Both use RK4 time integration. Data is saved to `.npz` for re-plotting without re-solving (`SKIP_SOLVE = True`).

---

## 1. Single-Field Pressure Solver

**Files:** `flux_transport_driver_clean.py` + `flux_transport_model_clean.py`

Evolves a single pressure field:
```
dp/dt = -div(Q) + S_ext + P_alpha - P_brem
```

Temperature inferred via fixed background density: `T_keV = (p / n_ref) * T_ref_keV`.

### Run control

| Flag | Default | Description |
|------|---------|-------------|
| `SKIP_SOLVE` | `False` | `True` loads `plots/run_data.npz` instead of running solver |
| `num_video_frames` | 300 | Frames saved from solver (used for videos + time-series) |
| `num_plot_snapshots` | 8 | Subset shown on static overlay plots |
| `power_balance_mode` | `"initial_only"` | Scale source once at t=0 to match edge flux |

### Key transport parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chi0` | 10.0 | NL transport coefficient |
| `chi_RR` | 0.05 | Background diffusivity |
| `g_c` | 4.0 | Critical gradient (NL flux -> 0) |
| `g_stiff` | 3 | Stiff transport onset |
| `n_stiff` | 2 | Stiffness exponent |
| `g_MHD` | `0.9 * g_c` | MHD cliff onset |
| `chi_MHD` | 2 | MHD cliff stiffness |
| `nu4` | 1e-4 | Hyperviscosity coefficient |
| `n_smooth` | 2 | Binomial smoothing passes on face gradients before flux evaluation |
| `power_balance` | 0.95 | Target ratio: int S dx = power_balance * Q_edge |

### Physics sources

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C_brem` | 0.01 | Bremsstrahlung coefficient (0 to disable) |
| `C_alpha` | 4e-6 | Alpha heating coefficient (0 to disable) |
| `T_ref_keV` | 10.0 | Model T=1 -> T_ref_keV keV |
| `n_ref` | 1.0 | Background density in model units |

### Output

Plots saved as PDFs to `plots/`. Videos saved as MP4:
- `pressure_evolution.mp4` — p(x,t) profile animation
- `source_components.mp4` — integrated source components & edge flux traced over time

Data saved to `plots/run_data.npz`.

---

## 2. Two-Species (Electron + Ion) Solver

**Files:** `flux_transport_driver_two_species_clean.py` + `flux_transport_model_two_species_clean.py`

Evolves three fields with quasi-neutrality (n_e = n_i = n):
```
dp_e/dt = -div(Q_e) + S_pe + Q_ei - P_brem + P_alpha_e
dp_i/dt = -div(Q_i) + S_pi - Q_ei + P_alpha_i
dn/dt   = -div(Gamma) + S_n
```

Heat flux driven by log temperature gradient: `kappa_T = -d(ln T)/dx`. Particle flux driven by log density gradient: `kappa_n = -d(ln n)/dx`. Electron-ion coupling: `Q_ei = n (T_i - T_e) / tau_ei`.

### Run control

| Flag | Default | Description |
|------|---------|-------------|
| `SKIP_SOLVE` | `False` | `True` loads `plots_2sp/run_data.npz` |
| `num_video_frames` | 300 | Frames saved from solver |
| `num_plot_snapshots` | 8 | Subset for static plots |
| `power_balance_mode` | `None` | `None` or `"initial_only"` |

### Transport parameters (per species)

| Parameter | Electron heat | Ion heat | Particle |
|-----------|:---:|:---:|:---:|
| chi0 | `chi0_e` = 10 | `chi0_i` = 3 | `chi0_n_e` = 1 |
| chi_RR | `chi_RR_e` = 0.05 | `chi_RR_i` = 0.05 | `chi_RR_n_e` = 0.01 |
| g_c | `g_c_e` = 4 | `g_c_i` = 3 | `g_c_n_e` = 2 |
| g_stiff | `g_stiff_e` = 3 | `g_stiff_i` = 2.5 | `g_stiff_n_e` = 1.5 |

Plus `tau_ei = 0.05` (e-i coupling), `n_smooth = 2` (gradient smoothing).

### Physics sources

Alpha heating uses Bosch-Hale D-T reactivity with Stix critical energy formula for electron/ion power split. Bremsstrahlung is electron-only radiation loss.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C_brem` | 0.03 | Bremsstrahlung coefficient |
| `C_alpha` | 1e-3 | Alpha heating coefficient |
| `T_ref_keV` | 10.0 | Model T=1 -> keV |
| `n_ref_20` | 1.0 | Model n=1 -> 10^20 m^-3 |

### Initial profiles

Power-law: `f = f_ped + A * (1 - (x/L)^m)`, amplitude scaled so max log-gradient hits a target multiple of kappa_crit. Controlled by `initial_state_e`, `initial_state_i`, `initial_state_n` (`"supercritical"` or `"subcritical"`).

### Output

Plots saved as PDFs to `plots_2sp/`. Videos saved as MP4:
- `temperature_evolution.mp4` — T_e and T_i profiles overlaid
- `pressure_evolution.mp4` — p_e and p_i profiles overlaid
- `density_evolution.mp4` — n(x,t) profile
- `edge_fluxes_and_sources.mp4` — edge fluxes (Q_e, Q_i, Gamma) and total sources traced over time

Data saved to `plots_2sp/run_data.npz`.

---

## Flux Model (shared)

All solvers use the same nonlinear saturation flux model, blended spatially via smooth windows:

**NL (nonlinear):** `Q = chi0 * g * max(1 - g/g_c, 0)` — peaks at `g = g_c/2`, drops to zero at `g_c`.

**Core (stiff):** `Q = chi_core * g * (1 + (g/g_stiff)^n_stiff)` — super-linear above stiffness onset.

**Linear:** `Q = chi_core * g`

**Background:** `Q_RR = chi_RR * g` (always added).

**MHD cliff (optional, single-field only):** `Q_MHD = chi_MHD * max(g - g_MHD, 0)` — prevents blowup beyond NL zero.

Region selection via `flux_models = ["nl"]`. Multiple regions blended with `make_windows(x, boundaries, deltas)`.

### Gradient smoothing

Both solvers apply `n_smooth` passes of a (1/4, 1/2, 1/4) binomial filter to face gradients before flux evaluation. This suppresses grid-scale staircases in the antidiffusion region (falling branch of Q(g) where dQ/dg < 0). Set `n_smooth = 0` to disable.

---

## Archive

Previous solver iterations are in `archive/`:
- `flux_transport_model_simple.py` / `flux_transport_driver_simple.py` — minimal single-field
- `flux_transport_model.py` / `flux_transport_driver.py` — single-field with PI/schedule controller
- `flux_transport_model_coupled.py` / `flux_transport_driver_coupled.py` — coupled p + n (single species)
- `flux_transport_model_two_species.py` / `flux_transport_driver_two_species.py` — four-field (p_e, n_e, p_i, n_i) with PI power balance
