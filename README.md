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

**Single-field and coupled drivers** enforce power balance inside `compute_rhs`, which is called at every RHS evaluation. With RK4 this means the source is rescaled **4 times per timestep** (once per stage), using the intermediate state at each stage. This is instantaneous perfect control — there is no physical response timescale.

**Two-species driver** offers two modes:

```python
power_balance_mode = "PI"     # "instantaneous" or "PI"

power_balance_pe = 1.0        # target ∫S_pe dx / Q_e_edge (None = disabled)
power_balance_pi = None        # None = disabled (no external ion heat source e.g. EBW)
power_balance_ne = None        # None = disabled
power_balance_ni = None        # None = disabled
heating_mode = "global"        # "global" or "localized"
```

| Mode | Behaviour |
|------|-----------|
| `"instantaneous"` | Rescales source at every RHS call (same as single-field). Instantaneous, perfect control — unphysical for studying transient dynamics. |
| `"PI"` | A PI controller adjusts a time-dependent source multiplier once per timestep. The source shape is fixed; only its amplitude changes. Physically realistic: the external heating system responds on a finite timescale. |

#### PI controller response time

The controller maintains one multiplier per active channel, updated as:
```
error(t)       = W₀ - W(t)          # W = ∫f dx, W₀ = initial value
integral_error += error * dt
source_scale    = 1 + Kp*error + Ki*integral_error   (clamped ≥ 0)
```

The response timescale is approximately `τ_response ~ 1/Kp`. With `Kp=10` and a timestep `dt=1e-6`, the controller reacts over ~0.1 time units. `Ki` eliminates steady-state offset — without it the multiplier would settle at a value slightly away from 1.0 when physics sources (bremsstrahlung, alpha heating) are active. Larger `Kp`/`Ki` → faster response but risk oscillation; smaller → slower, smoother.

The target `W₀` is the **initial** stored energy, so the controller tries to maintain the t=0 integrated pressure/density against any perturbation including time-varying physics sources.

PI controller gains (only used when `power_balance_mode = "PI"`):

```python
Kp_pe = 10.0;  Ki_pe = 50.0   # electron pressure
Kp_ne = 10.0;  Ki_ne = 50.0   # electron density
Kp_pi = 10.0;  Ki_pi = 50.0   # ion pressure
Kp_ni = 10.0;  Ki_ni = 50.0   # ion density
```

Set any channel to `None` to disable power balance enforcement entirely for that channel.

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

Plots are saved as PDFs to `plots/` (single-field), `coupled_plots/` (coupled), or `two_species_plots/` (two-species).

### Single-field driver

| Plot | Description |
|------|-------------|
| `01` | Initial pressure profile with supercritical region highlighted in red |
| `02` | Initial gradient `g = -dp/dx` with `g_crit` dashed line |
| `03` | Initial flux `Q(x)` overlaid with cumulative source `∫S dx` |
| `03b` | Same after power balance enforcement |
| `03c` | Source before and after power balance rescaling |
| `05` | Pressure evolution snapshots |
| `06` | Total integrated pressure `∫p dx` vs time |
| `07` | Gradient evolution at tracked x locations |
| `08` | Gradient heatmap `g(x,t)` |
| `09` | Power imbalance (edge flux − total heating) vs time |
| `10` | Local flux and source vs time at tracked x locations |
| `11` | Pressure heatmap `p(x,t)` |
| `12` | Max gradient and its x location vs time |
| `13` | Edge gradient and edge source vs time |
| `14` | Flux balance snapshots (Q vs ∫S dx) at final timesteps |
| `15` | Effective diffusivity `χ_eff = dQ/dg` at final state |

### Two-species driver

#### Initial state (pre-solve)

| Plot | Description |
|------|-------------|
| `01a` | T_e and T_i overlaid |
| `01b` | Density n |
| `01c` | Pressures p_e and p_i overlaid |
| `02a` | Log-gradient κ_Te and κ_Ti with κ_crit lines |
| `02b` | Log-gradient κ_n with κ_crit line |
| `02c` | T_e profile with supercritical region shaded (where κ_Te > κ_crit) |
| `02d` | T_i profile with supercritical region shaded |
| `02e` | n profile with supercritical region shaded |
| `03a–e` | Flux vs cumulative source for each channel (Q_e, Γ_e, Q_i, Γ_i) |
| `03f–i` | Raw vs scaled source for each channel |
| `03j–m` | Flux vs enforced cumulative source for each channel |

#### Evolution

| Plot | Description |
|------|-------------|
| `04a` | T_e snapshots |
| `04b` | T_i snapshots |
| `04c` | p_e snapshots |
| `04d` | p_i snapshots |
| `05a` | n_e snapshots |
| `05b` | n_i snapshots |
| `06a` | κ_Te heatmap `κ_Te(x,t)` |
| `06b` | κ_Ti heatmap |
| `06c` | κ_n heatmap |
| `07a` | Integrated p_e, p_i, n_e, n_i vs time |
| `07b` | Integrated T_e and T_i vs time |
| `08a` | Edge heat fluxes Q_e, Q_i vs time |
| `08b` | Edge particle fluxes Γ_e, Γ_i vs time |
| `09a` | κ_Te evolution at tracked x points |
| `09b` | κ_Ti evolution at tracked x points |
| `09c` | κ_n evolution at tracked x points |
| `10a` | Max κ_Te and κ_Ti vs time with κ_crit lines |
| `10b` | Max κ_n vs time with κ_crit line |
| `11a` | κ_Te heatmap (full time range) |
| `11b` | κ_Ti heatmap |

#### Final state and diagnostics

| Plot | Description |
|------|-------------|
| `12a` | Flux balance Q_e vs ∫S_pe dx at final snapshots |
| `12b` | Flux balance Γ_e vs ∫S_ne dx |
| `13a` | Effective diffusivity χ_eff,e at final state |
| `13b` | Effective diffusivity χ_eff,i at final state |
| `14` | Combined final state: T_e, T_i, n_e, Q_ei on shared axes |
| `14a` | PI controller source multiplier vs time (one line per active channel) |
| `14b` | Stored energy W = ∫f dx vs PI target W₀ for each active channel |

Plots `14a` and `14b` only appear when `power_balance_mode = "PI"`. They show whether the controller has converged (multiplier → 1) and how quickly the stored energy tracks the target.
