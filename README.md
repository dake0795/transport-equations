# 1D Radial Transport Solvers — Instructions

Three progressively more complete 1D radial transport solvers, each with its own driver and model file.

---

## Overview

| Driver | Model | Fields evolved | Gradient type | Physics sources |
|--------|-------|---------------|---------------|-----------------|
| `flux_transport_driver.py` | `flux_transport_model.py` | `p` | absolute `g = -dp/dx` | Gaussian + feedback |
| `flux_transport_driver_coupled.py` | `flux_transport_model_coupled.py` | `p, n` | absolute `g = -df/dx` | Gaussian + feedback |
| `flux_transport_driver_two_species.py` | `flux_transport_model_two_species.py` | `p_e, n_e, p_i, n_i` | log `κ = -d(ln f)/dx` | Gaussian + bremsstrahlung + alpha heating |

All solvers use the same nonlinear flux model structure, with time integration via Euler, RK4, or Crank-Nicolson (single-field only).

---

## 1. Single-Field Pressure Solver

**Files:** `flux_transport_driver.py` + `flux_transport_model.py`

Evolves a single pressure field `p(x,t)`:
```
dp/dt = -div(Q) + S
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `START_ON` | `"supercritical"` | Initial gradient branch |
| `chi0` | 10.0 | NL transport coefficient |
| `chi_RR` | 0.05 | Background diffusivity (always active) |
| `g_c` | 4.0 | Critical gradient (NL flux → 0 at `g_c`) |
| `g_stiff` | 3.0 | Stiff transport onset |
| `n_stiff` | 2 | Stiffness exponent |
| `power_balance` | 1.0 | Ratio ∫S dx / Q_edge |
| `heating_mode` | `"global"` | `"global"` (rescale S) or `"localized"` (add edge Gaussian) |
| `alpha` | 0.1 | Source feedback coefficient: S ∝ (1 + α·p) |
| `flux_models` | `["nl"]` | Per-region flux type: `"nl"`, `"core"`, `"linear"` |

### Initial profile
Power-law: `p = p_ped + A·(1 - (x/L)^m)`, with `m=2`, amplitude scaled so max gradient = 1.8×g_crit (supercritical) or 0.6×g_crit (subcritical).

---

## 2. Coupled Pressure-Density Solver

**Files:** `flux_transport_driver_coupled.py` + `flux_transport_model_coupled.py`

Evolves pressure and density together; temperature is derived as `T = p/n`:
```
dp/dt = -div(Q)     + S_p
dn/dt = -div(Gamma)  + S_n
```

### Additional parameters (beyond single-field)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_state_p` | `"supercritical"` | Pressure gradient initial state |
| `initial_state_n` | `"subcritical"` | Density gradient initial state |
| `power_balance_mode` | `"separate"` | `"separate"`, `"coupled_to_p"`, or `"coupled_to_n"` |
| `power_balance_p` | 1.0 | Pressure source ratio |
| `power_balance_n` | 0.8 | Density source ratio |
| `chi0_n` | 1.0 | NL particle transport coefficient |
| `chi_RR_n` | 0.01 | Background particle diffusivity |
| `g_c_n` | 2.0 | Critical density gradient |
| `V_n` | 0.0 | Convective velocity (particle pinch) |
| `m_p` | 2 | Power-law exponent for pressure |
| `m_n` | 2 | Power-law exponent for density |

### Initial profile
Power-law only: `f = f_ped + A·(1 - (x/L)^m)`, amplitude auto-scaled to target gradient.

---

## 3. Two-Species (Electron + Ion) Solver

**Files:** `flux_transport_driver_two_species.py` + `flux_transport_model_two_species.py`

The most complete solver. Evolves four fields with **log gradients** and species coupling:
```
dp_e/dt = -div(Q_e)     + S_pe + Q_ei - P_brem + P_alpha_e
dn_e/dt = -div(Gamma_e) + S_ne
dp_i/dt = -div(Q_i)     + S_pi - Q_ei + P_alpha_i
dn_i/dt = -div(Gamma_i) + S_ni
```

where:
- Heat flux driven by log temperature gradient: `κ_T = -d(ln T)/dx`
- Particle flux driven by log density gradient: `κ_n = -d(ln n)/dx`
- `Q_ei = n_e (T_i - T_e) / τ_ei` — collisional energy exchange

### Transport parameters (four independent channels)

Each channel (electron heat, electron particle, ion heat, ion particle) has its own set:

| Parameter | Electron heat | Electron particle | Ion heat | Ion particle |
|-----------|:---:|:---:|:---:|:---:|
| χ₀ | `chi0_e` = 10 | `chi0_n_e` = 1 | `chi0_i` = 3 | `chi0_n_i` = 1 |
| χ_RR | `chi_RR_e` = 0.05 | `chi_RR_n_e` = 0.01 | `chi_RR_i` = 0.05 | `chi_RR_n_i` = 0.01 |
| g_c | `g_c_e` = 4 | `g_c_n_e` = 2 | `g_c_i` = 3 | `g_c_n_i` = 2 |
| g_stiff | `g_stiff_e` = 3 | `g_stiff_n_e` = 1.5 | `g_stiff_i` = 2.5 | `g_stiff_n_i` = 1.5 |
| n_stiff | `n_stiff_e` = 2 | `n_stiff_n_e` = 2 | `n_stiff_i` = 2 | `n_stiff_n_i` = 2 |

Plus `tau_ei = 0.05` for e-i coupling strength (larger = weaker coupling).

### Physics sources (T3D-like)

| Flag | Default | Description |
|------|---------|-------------|
| `C_brem` | 0.03 | Bremsstrahlung coefficient. `P_brem = C_brem × n_e² × Z_eff × √T_e(keV)`. Set 0 to disable. Electrons only. |
| `C_alpha` | 1e-3 | Alpha heating coefficient. Uses Bosch-Hale D-T reactivity `⟨σv⟩(T_i)` and Stix critical energy formula for e/i power split. Set 0 to disable. |
| `Z_eff` | 1.0 | Effective charge (affects bremsstrahlung) |
| `T_ref_keV` | 10.0 | Maps dimensionless T=1 to physical keV (for reactivity lookup) |
| `n_ref_20` | 1.0 | Maps dimensionless n=1 to 10²⁰ m⁻³ |
| `Z_i` | 1 | Ion charge state |
| `A_i` | 2.5 | Ion mass in amu (D-T average) |
| `f_deuterium` | 0.5 | Deuterium fuel fraction |
| `f_tritium` | 0.5 | Tritium fuel fraction |

**Bremsstrahlung** (radiation loss from electrons):
```
P_brem = C_brem × 5.35e-3 × n_e² × Z_eff × √(T_e in keV)   [MW/m³]
```

**Alpha heating** (D-T fusion power):
```
P_alpha = C_alpha × 5.6e-13 × ⟨σv⟩(T_i) × n_D × n_T   [W/m³]
```
Split between electrons and ions using the Stix critical energy:
```
E_crit = 4 × 14.8 × T_e × (Σ n_i Z_i² / A_i / n_e)^(2/3)
f_ion = E_alpha / E_crit   (clamped to [0,1])
```

### Initial state controls

```python
initial_state_e = "supercritical"   # T_e log-gradient
initial_state_i = "supercritical"   # T_i log-gradient
initial_state_n = "subcritical"     # density log-gradient
```

### Profile types (selectable per species)

```python
profile_type_Te = "power_law"   # "tanh", "plateau", or "power_law"
profile_type_Ti = "power_law"
profile_type_n  = "power_law"
```

| Type | Shape | Controls |
|------|-------|----------|
| `"tanh"` | Smooth transition peaked at `x_sym` | `x_sym` (midpoint), `delta_super`/`delta_sub` (width) |
| `"plateau"` | Constant κ over `[x_start, x_end]` | `x_start`, `x_end` (gradient region bounds), `delta_ramp` (smoothness) |
| `"power_law"` | `f = f_ped + A·(1 - (x/L)^m)` | `m` (exponent), `kap_ratio_super`/`kap_ratio_sub` (amplitude as multiple of κ_crit) |

Each type can be set independently for T_e, T_i, and n.

### Power balance

```python
power_balance_pe = 1.0    # ∫S_pe dx = ratio × Q_e_edge
power_balance_pi = 1.0    # ∫S_pi dx = ratio × Q_i_edge
power_balance_ne = None   # None = disabled
power_balance_ni = None   # None = disabled
heating_mode = "global"   # "global" or "localized"
```

Set to `None` to disable power balance enforcement for a channel.

### Core/pedestal values

```python
T_e_core = 2.0;  T_e_ped = 0.5   # Electrons hotter than ions
T_i_core = 1.5;  T_i_ped = 0.4
n_core   = 2.0;  n_ped   = 0.2   # Used for supercritical density
n_ped_sub = 1.2                   # Used for subcritical density (flatter)
```

---

## Flux Model (shared across all solvers)

All three solvers use the same nonlinear saturation flux model, blended spatially via smooth windows:

**NL (nonlinear):**
```
Q_nl = χ₀ · g · max(1 - g/g_c, 0)
```
Flux increases linearly for small gradients, peaks at `g = g_c/2`, drops to zero at `g = g_c`.

**Core (stiff):**
```
Q_core = χ_core · g · (1 + (g/g_stiff)^n_stiff)
```
Rapid super-linear increase above the stiffness onset.

**Linear:**
```
Q_linear = χ_core · g
```

**Background (always added):**
```
Q_RR = χ_RR · g
```

The active model per region is selected via `flux_models = ["nl"]` (list of region labels). Multiple regions can be blended with `make_windows(x, boundaries, deltas)`.

---

## Diagnostic plots

All drivers produce numbered PDF plots saved to `two_species_plots/`, `coupled_plots/`, or `plots/`:

- **01x**: Initial profiles (T, n, p)
- **02x**: Initial gradients (κ or g) with critical threshold lines
- **02c–e** (two-species): Profile + gradient with supercritical region shading
- **03x**: Flux vs cumulative source, flux vs enforced source, source before/after power balance
- **04x**: Temperature/pressure evolution snapshots
- **05x**: Density evolution snapshots
- **06x**: Gradient heatmaps
- **07x**: Integrated quantities over time
- **08x**: Edge fluxes over time
- **09x**: Gradient evolution at tracked points
- **10x**: Max gradient over time with critical threshold
- **11x**: Gradient evolution heatmaps
- **12x**: Flux balance at final snapshots
- **13x**: Final state profiles
