# Transport Regimes and End States

Guide to the steady-state outcomes of the single-field pressure solver and the parameter knobs that select between them.

---

## The flux curve: the map of everything

All dynamics are governed by the shape of Q(g), where g = -dp/dx is the local gradient:

```
Q(g) = chi0 * g * max(1 - g/g_c, 0)      [NL model]
     + chi_RR * g                          [background]
     + chi_MHD * max(g - g_MHD, 0)        [MHD cliff]
```

Key landmarks on Q(g):

| Symbol | Value (defaults) | Meaning |
|--------|-----------------|---------|
| g* | g_c/2 = 2.0 | Flux peak. dQ/dg = 0 here. |
| Q_max | chi0 * g_c/4 + chi_RR * g* | Maximum flux the NL model can carry |
| g_c | 4.0 | NL flux drops to zero (only chi_RR remains) |
| g_MHD | 0.9 * g_c = 3.6 | MHD cliff switches on |
| Q_min | Local minimum between g_c/2 and the cliff | The flux valley — lowest flux any steady state can sustain |

The profile at each x sits somewhere on this curve. The **branch** it's on determines the local transport character:

- **Rising branch** (g < g*): Diffusive. Perturbations are damped. Larger gradient = more flux out.
- **Falling branch** (g* < g < g_MHD): Antidiffusive. Perturbations are amplified. Larger gradient = *less* flux out. This is the transport barrier.
- **Cliff branch** (g > g_MHD): Strongly diffusive again. MHD modes clamp the gradient. Safety valve.

---

## End states

### 1. Subcritical equilibrium (L-mode-like)

**What it looks like:** Smooth profile, gradients everywhere below g*. No transport barrier. Pressure slowly adjusts until ∫S = Q_edge.

**How to get there:**
- `START_ON = "subcritical"` (initial max g < g*)
- Or start supercritical with `power_balance` high enough that the profile relaxes back below g*

**Key requirement:** The source must be low enough that the rising branch can carry it: ∫S < Q_max. If you pump more heat than Q_max, the gradient is forced past g* and you leave this regime.

**Knobs:**
- `power_balance` close to 1.0 with subcritical start
- Low `chi0` or high `g_c` (pushes Q_max up, easier to stay subcritical)

---

### 2. Transport barrier (H-mode-like)

**What it looks like:** Steep gradient region (pedestal) where g > g*, sitting on the falling branch. Core is relatively flat (diffusive). Sharp transition between the two regions. ∫p is elevated compared to L-mode.

**How to get there:**
- `START_ON = "supercritical"` — the initial gradient is above g*, and the antidiffusion locks it there
- Or start subcritical and push `power_balance` high enough to force g past g*

**What sustains it:** The falling branch acts as a bottleneck. As the gradient steepens, flux *decreases*, trapping heat. This positive feedback is self-reinforcing — once formed, the barrier persists even if you later reduce the source (hysteresis).

**What determines the barrier width and height:**
- **Source level vs Q_min:** The barrier steepens until Q_edge matches ∫S. If ∫S > Q_min, there's a stable equilibrium on the falling branch.
- **Fraction of domain that's supercritical:** More of the domain on the falling branch = wider barrier = more trapped pressure. The initial profile shape (`m`, `p_core`, `p_ped`) and the source shape (`sigma`) determine where the subcritical/supercritical boundary lands.
- `chi_RR`: Background diffusion leaks through the barrier. Higher chi_RR = weaker barrier, lower pedestal.
- `g_c`: Controls the width of the falling branch (g* to g_c). Larger g_c = wider gradient range for the barrier = taller pedestal.

**Knobs:**
- `power_balance` between ~Q_min/Q_edge,init and 1.0
- `chi0` (barrier strength), `g_c` (barrier width), `chi_RR` (leakage)
- Initial profile shape via `m`, `p_core`, `p_ped`

---

### 3. Oscillatory / limit cycle (ELM-like)

**What it looks like:** Periodic or quasi-periodic oscillations in ∫p and Q_edge. The gradient repeatedly steepens into the barrier, then collapses, then rebuilds.

**How to get there:**
- `power_balance` pushed below Q_min/Q_edge,init — there's no stable equilibrium on the falling branch
- The gradient steepens until it hits the MHD cliff (g > g_MHD), flux jumps up, pressure crashes, gradient relaxes below g*, flux drops, source > flux again, gradient steepens, cycle repeats

**The mechanism:** This is a relaxation oscillation between two branches:
1. Barrier builds on falling branch (slow, antidiffusive)
2. Gradient hits cliff → flux spike → pressure crash (fast, strongly diffusive)
3. Profile relaxes to low gradient, flux drops below source
4. Gradient steepens again → repeat

**What controls the oscillation:**
- `chi_MHD`: Stiffer cliff = sharper crash = more violent oscillation. Gentler cliff = smoother cycle.
- `g_MHD`: How far the gradient must steepen before the cliff triggers. Closer to g* = earlier trigger = smaller oscillations.
- `power_balance`: Further below Q_min = stronger drive for the cycle.
- `chi_RR`: Higher background diffusion damps the oscillation. Can stabilize it entirely if large enough.

**Knobs:**
- `power_balance` well below 1.0 (below the critical threshold)
- `chi_MHD` > 0, `g_MHD` < g_c
- Low `chi_RR`

---

### 4. Gradient blowup (crash)

**What it looks like:** Gradient increases without bound, NaN or overflow. The simulation fails.

**How to get there:**
- `chi_MHD = 0` (no cliff) and the gradient reaches g_c where Q_NL = 0
- Only chi_RR remains to carry heat out, but if ∫S >> chi_RR * g, the gradient must keep steepening with nothing to stop it

**The mechanism:** Without the MHD cliff, there's no safety valve. Once Q_NL → 0 at g = g_c, the only outward transport is chi_RR * g, which is weak. The gradient overshoots, profile develops extreme features, numerics blow up.

**How to avoid it:**
- Always have `chi_MHD > 0` as a safety valve
- Or ensure `chi_RR` is large enough to carry the full source: chi_RR > ∫S / (g_c * L) roughly

---

## The critical threshold: Q_min

The boundary between the stable barrier (state 2) and the oscillatory regime (state 3) is set by Q_min — the local minimum of Q(g) between g* and g_MHD + cliff onset.

You can read Q_min directly from the phase diagram plot. If ∫S > Q_min, a stable barrier exists. If ∫S < Q_min, the system oscillates or crashes.

With default parameters:
```
Q_max ≈ chi0 * g_c/4 + chi_RR * g_c/2 = 10 * 1 + 0.05 * 2 = 10.1
Q_min depends on chi_MHD and g_MHD — visible on the Q(g) curve
```

The `power_balance` parameter sets ∫S = power_balance * Q_edge,init. So the critical power_balance threshold is roughly Q_min / Q_edge,init.

---

## How the subcritical fraction matters

You're right that the fraction of the domain on each branch matters. The total edge flux Q_edge is determined by the gradient at the boundary, but the *profile shape* inside — how much is subcritical vs supercritical — determines how the system responds to perturbations.

**More subcritical (core flat, only edge is steep):**
- Small barrier region, most transport is diffusive
- Profile is robust — the diffusive core distributes heat efficiently to the barrier
- The barrier is "thin" and carries nearly all the flux
- Easier to push into oscillation (small barrier = small Q_min margin)

**More supercritical (steep over a wide region):**
- Wide barrier, lots of trapped pressure
- The barrier must carry the cumulative source from the entire supercritical region
- More prone to the feedback loop: steepening anywhere in the wide barrier reduces flux everywhere
- But also more stored energy, so oscillations (if they occur) are larger and slower

**What controls the subcritical fraction:**
- **Source shape** (`sigma` in `base_source`): Narrow core heating → most heat deposited in the subcritical core → flux must pass through the barrier. Wide heating → heat deposited directly in the barrier region.
- **Initial profile** (`m`, `p_core`): Higher `m` = flatter core, steeper edge = smaller supercritical fraction. Lower `m` = gradual slope = wider supercritical zone.
- **`g_c`**: Higher g_c = g* higher = more of the domain can be subcritical for the same profile shape.

---

## Quick reference: parameter → outcome

| Parameter | Increase → | Decrease → |
|-----------|-----------|------------|
| `power_balance` | Stronger drive, more trapped pressure, eventually oscillation | Weaker drive, subcritical equilibrium |
| `chi0` | Stronger NL transport, higher Q_max, taller barrier | Weaker barrier, easier L-mode |
| `chi_RR` | More leakage through barrier, weaker pedestal, stabilizes oscillations | Stronger barrier, sharper transition |
| `g_c` | Wider falling branch, taller barrier possible, higher Q_max | Narrower barrier, lower Q_max, easier to oscillate |
| `chi_MHD` | Stiffer cliff, sharper ELM-like crashes, but prevents blowup | Gentler safety valve, softer oscillations |
| `g_MHD` | Cliff triggers later, barrier can steepen more before crash | Earlier trigger, smaller oscillations |
| `C_alpha` | Self-heating amplifies barrier (more heat → higher T → more alpha → more heat) | Less feedback, more predictable |
| `C_brem` | Radiation loss offsets source, effectively lowers net ∫S | Less cooling, net source higher |
| Source `sigma` | Heat deposited across wider region including barrier | Heat concentrated in core, must traverse barrier |
| Profile `m` | Flatter core, steeper edge, smaller supercritical fraction | More gradual profile, wider supercritical region |
| `START_ON` | `"supercritical"`: starts on falling branch, barrier from t=0 | `"subcritical"`: starts on rising branch, may or may not form barrier |
