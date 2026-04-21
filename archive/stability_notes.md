# Stability Notes: Single-Field Transport Model

## When does the system find a steady state vs collapse or run away?

### The NL flux curve is the key

The nonlinear flux model is:

```
Q_nl = χ₀ · g · max(1 − g/g_c, 0)
```

This is a downward parabola in `g`: flux *increases* for `g < g_c/2`, *peaks* at `g = g_c/2 ≡ g_crit`, then *decreases* back to zero at `g = g_c`. With the background `chi_RR` floor, the total flux still decreases for `g > g_crit` — but the shift is tiny (`~chi_RR/chi0 · g_c ≈ 0.02` with default values), so the instability region is essentially unchanged.

At steady state, local power balance requires `dQ/dx = S(x)`, i.e. the flux at each point carries all the source deposited inward of it:

```
Q(x) = ∫₀ˣ S(x′) dx′
```

For the edge this becomes `Q_edge = ∫₀ᴸ S dx` — all power must exit at the boundary.

---

### Two coexisting branches

For a given integrated source `C = ∫₀ˣ S dx′`, solving `Q(g) = C` gives:

- **If `C < χ₀ g_c / 4`** (below peak flux): two solutions exist — a subcritical steady state (`g < g_crit`) and a supercritical one (`g > g_crit`)
- **If `C > χ₀ g_c / 4`**: no NL solution; the gradient blows up until only `chi_RR · g` carries the flux, giving `g = C / chi_RR` (potentially very large)

This is the classic S-curve / bifurcation picture. Both branches are mathematically valid steady states for intermediate source strengths, but they have very different stability properties.

---

### Stability of each branch

The subcritical branch is **stable**: if `g` is perturbed upward, `Q` increases and carries away more heat, restoring `g`. This is negative feedback — transport acts as a brake.

On the supercritical branch `dQ/dg < 0`: an upward perturbation *reduces* flux, allowing the gradient to steepen further. This is positive feedback — the supercritical branch is intrinsically unstable. Crucially, this means that as `p` rises in the supercritical regime, `Q_edge` *decreases* — less heat escapes just as more is being deposited. Transport is not a brake here; it is making things worse.

---

### What the initial power balance determines

The ratio `∫S dx / Q_edge` at `t=0` (the `power_balance` parameter, 0.8 by default) controls the global energy budget:

| Condition | What happens |
|-----------|--------------|
| `∫S dx < Q_edge` | Transport exceeds source — pressure drains, gradient falls |
| `∫S dx = Q_edge` | Exact balance — profile tends toward a steady state |
| `∫S dx > Q_edge` | Source exceeds transport — pressure builds, gradient steepens |

Combined with the starting branch, this gives four qualitatively different outcomes:

| Starting branch | Power balance | Outcome |
|-----------------|---------------|---------|
| Subcritical | `∫S < Q_edge` | Gradient falls, system settles on subcritical steady state. The "safe" regime. |
| Subcritical | `∫S > Q_edge` | Pressure builds until gradient crosses `g_crit`. Either locks onto the supercritical branch or overshoots to `g ~ g_c`. |
| Supercritical | `∫S < Q_edge` | Transport wins, gradient crashes back toward subcritical — ELM-like collapse. |
| Supercritical | `∫S > Q_edge` | Pressure keeps building. If `g` exceeds `g_c` the NL flux vanishes entirely and only `chi_RR` carries heat; the gradient must reach `g = Q_edge / chi_RR` — very steep. This is the "runaway" case. |

The runaway is not physically unbounded because the Dirichlet BC at the edge fixes `p = p_ped`, but the profile can steepen dramatically between the core and edge.

---

### Role of alpha heating and bremsstrahlung

The physics source terms add state-dependent feedback that can shift which of the above regimes applies.

**Alpha heating** (`P_alpha ∝ ⟨σv⟩(T)`) grows rapidly with temperature (roughly as `T²` to `T³` in the 5–20 keV range). This is a self-reinforcing source: higher T → more alpha power → higher T. On the supercritical branch this combines with the transport feedback to produce a **doubly destabilising** loop:

- Rising `p` → rising `T` → rising `P_alpha` → more source in
- Rising `p` → rising `g` → falling `Q_edge` → less flux out

Both effects simultaneously push `p` upward with nothing pushing back. On the subcritical branch, if alpha gain is large enough, it can push the system across `g_crit` even when the external source alone would not.

**Bremsstrahlung** (`P_brem ∝ n² √T`) grows more slowly with T than alpha heating does. It is a loss term that partially offsets alpha gain. In the 10–30 keV range typical of these simulations it is a weaker function of T than alpha, so it slows but does not stop the runaway.

Because `P_alpha − P_brem` is part of the effective source, the true steady-state power balance condition is:

```
∫S_ext dx + ∫P_alpha dx − ∫P_brem dx = Q_edge
```

If alpha gain is large enough, `S_ext` can be zero and the plasma is self-sustaining — the ignition condition. However, in this model a supercritical ignited state is not a true steady state: `Q_edge` is falling as `p` rises, so the left-hand side is growing and the right-hand side is shrinking simultaneously. The only things that eventually halt the runaway are the fixed edge BC capping how large `p` can get, and bremsstrahlung growing with T.

---

### Transport feedback summary by branch

| Branch | Transport response to rising p | Alpha response | Overall |
|--------|-------------------------------|----------------|---------|
| Subcritical (`g < g_crit`) | Q increases — restoring force | Grows with T | Stable if brem keeps up; otherwise slow drift toward g_crit |
| Supercritical (`g_crit < g < g_c`) | Q **decreases** — destabilising | Grows with T | Doubly unstable runaway; only edge BC and brem can stop it |
| Beyond NL (`g > g_c`) | Q_nl = 0; only chi_RR·g remains, which increases with g | Grows with T | New steep equilibrium possible at `g = Q_source / chi_RR`, but requires enormous gradients |

### Summary

The system avoids complete collapse or runaway when:

1. The source is in the intermediate range where two steady-state branches exist (`C < χ₀ g_c / 4` locally)
2. The system is on the **subcritical** branch, where transport provides a genuine restoring force
3. The net physics feedback (`P_alpha − P_brem`) does not tip the effective source above the threshold that pushes `g` past `g_crit`

On the supercritical branch there is no transport-based restoring force — rising `p` reduces `Q_edge`, making the imbalance worse, not better. A supercritical state with significant alpha heating in this model is therefore not a controlled steady state. It is a runaway that the fixed edge BC eventually caps. Starting subcritical with `∫S < Q_edge` is the only configuration where the transport itself acts as a stabiliser.

---

## Designing for Sustained Fusion Power

### What can be controlled

The model inputs split into two categories:

**Directly controllable:**
- `S_ext` shape and amplitude — external heating (analogous to NBI, ECRH, ICRH)
- `n_ref` / `n_ref_20` — plasma density, controlled by fuelling
- `p_ped` — edge boundary condition, controlled by ELM/divertor conditions
- `f_deuterium`, `f_tritium` — fuel mix
- `Z_eff` — impurity content (directly scales bremsstrahlung)
- Starting state — which branch and how far into it

**Not controllable** (set by plasma physics):
- `chi0`, `g_c`, `g_stiff` — transport coefficients are properties of the plasma, not inputs
- `P_alpha`, `P_brem` — these respond to whatever `p` and `n` do

The transport coefficients set the minimum heating required to reach ignition temperature. This is the confinement quality — you cannot directly tune it, but choosing a different flux model or `g_c` value in the model represents choosing a different confinement scenario.

---

### The ignition condition

The condition for sustained fusion with no external source is:

```
∫P_alpha dx = Q_edge + ∫P_brem dx     (with S_ext = 0)
```

Since `P_alpha ∝ ⟨σv⟩(T) · n²` and `P_brem ∝ n² √T`, the density cancels from this condition and it becomes purely a temperature threshold — roughly `T ~ 5–10 keV` for D-T fusion, which in model units (with `T_ref_keV = 10`) maps to `p/n_ref ≈ 0.5–1.0`.

---

### The subcritical ignition path

The double destabilisation described above only applies on the supercritical branch. On the subcritical branch transport is a restoring force, and there is a controlled path to ignition.

As you ramp up `S_ext` slowly while staying subcritical, the steady-state gradient tracks upward along the stable branch. The profile gets hotter, `P_alpha` grows, and at some point `P_alpha − P_brem` becomes large enough that you can reduce `S_ext` while the plasma maintains its own gradient through alpha self-heating. That is ignition achieved stably — transport is pushing back the whole time rather than amplifying the runaway.

Whether this is possible depends on whether the ignition temperature is reachable before the subcritical branch runs out. The subcritical branch ends at `g = g_crit`, which in steady state means the maximum total source that can be balanced subcritically is:

```
Q_max = χ₀ g_c / 4   (peak of the NL flux parabola)
```

With default parameters (`χ₀ = 10`, `g_c = 4`): `Q_max = 10`, and `g_crit = 2`. Over a domain of length 1 with `p_ped = 1`, this gives a core pressure of roughly `p_core ~ p_ped + g_crit · L ~ 3`, corresponding to `T ~ 30 keV` with default normalisation — well into the alpha-burning regime. The ignition temperature is therefore reachable subcritically with these parameters.

### Strategy to reach ignition

**1. Use edge-localised external heating and ramp it up slowly.**
Switch to `heating_mode = "localized"` to concentrate `S_ext` near the edge. An edge source heats the edge plasma first, creating a steep but potentially subcritical edge gradient and high local T, while the core remains flat and stable. Alpha heating kicks in at the edge — where T is highest — before the whole profile needs to go supercritical. This is a more controlled path than heating the core uniformly.

In the model: set `power_balance_mode = "free"` and gradually increase the edge Gaussian amplitude. Watch plot `09b` for `∫P_alpha dx` growing toward `Q_edge + ∫P_brem dx`.

**2. Set density high (`n_ref`, `n_ref_20`).**
Alpha power scales as `n²`; bremsstrahlung also scales as `n²` but only as `√T` vs the fast `⟨σv⟩(T)` growth of alpha. At sufficiently high T, increasing density always favours alpha over bremsstrahlung.

**3. Keep `Z_eff` close to 1.**
Impurities increase bremsstrahlung directly without contributing fuel ions — they are pure losses. Lower `Z_eff` reduces the ignition temperature threshold.

**4. Maintain the edge boundary condition.**
`p_ped` sets the pressure at the edge and therefore the maximum gradient the profile can sustain. A higher `p_ped` supports a larger stored energy. If the pedestal collapses (`p_ped → 0`) the gradient flattens and confinement is lost regardless of the source.

**5. Reduce external heating carefully once alpha takes over.**
When `∫P_alpha dx` is growing and approaching `Q_edge + ∫P_brem dx` in plot `09b`, reduce `S_ext`. If still subcritical at this point, transport continues to act as a brake and the transition is controlled. If already supercritical, `Q_edge` is simultaneously falling, so the ignition condition is being approached from both sides — backing off too fast risks a quench; not backing off means the runaway continues until the edge BC stops it.

---

### The L-H transition analogue: pushing the edge supercritical

There is a distinct scenario worth understanding separately: deliberately pushing the edge gradient past `g_crit` by ramping up edge heating. This is the analogue of the L-H (low-to-high confinement mode) transition.

**While subcritical at the edge**, transport is a restoring force and the edge gradient tracks the source stably. Q_edge increases with g.

**At the bifurcation point** (`g = g_crit`, `Q = Q_max`), the subcritical solution ceases to exist. If the source is pushed just past `Q_max`, the system has no subcritical steady state available and snaps discontinuously to the supercritical branch. This is not a gradual crossing — it is a sudden jump. Q_edge drops as the gradient overshoots `g_crit`.

**After the snap**, the edge is supercritical. The gradient steepens rapidly at the edge while the interior may remain subcritical, producing a sharp edge layer with a flat core — the structure of an H-mode pedestal. Local T at the edge rises quickly, and if `C_alpha > 0`, `P_alpha` spikes there. Whether this then drives core ignition or remains a localised edge effect depends on whether the heat conducts inward before the edge gradient overshoots `g_c` and collapses the NL flux entirely.

**The risk of overshooting.** If the edge source is too large when the snap occurs, the gradient shoots past `g_c` and `Q_nl → 0`. The edge then relies only on `chi_RR · g` to carry the flux, requiring an enormous gradient:

```
g_edge ~ Q_source / chi_RR
```

With `chi_RR = 0.05` and `chi0 = 10`, this is 200× larger than the NL-dominated gradient — a very narrow, very steep edge layer rather than a true H-mode pedestal.

**The target window.** There is an optimal edge source amplitude: large enough to trigger the snap to supercritical, small enough that the resulting gradient lands between `g_crit` and `g_c` rather than overshooting into the `chi_RR`-only regime. With default parameters that window is:

```
g_crit < g_edge < g_c   →   2 < g < 4
```

This is not a wide target. Ramping the edge source too aggressively skips past the H-mode window entirely.

---

### Failure modes to watch for

**Too little heating early on.**
Profile stays subcritical, T never reaches the alpha-ignition threshold, alpha heating stays negligible. The plasma never self-heats. Fix: increase `S_ext` amplitude or start closer to `g_crit`.

**Supercritical collapse when backing off external heating too fast.**
If `S_ext` is reduced before alpha heating has grown enough to take over, the gradient crashes and T drops — which reduces alpha power further. The plasma quenches. Fix: reduce heating more gradually, or check plot `09b` to confirm alpha is genuinely carrying the load before reducing `S_ext`.

**Radiative collapse at high density with low temperature.**
If density is raised while T is still low, `P_brem ∝ n² √T` can exceed `P_alpha ∝ ⟨σv⟩(T) n²`. The plasma radiates itself into collapse before reaching ignition temperature. Fix: raise density and temperature together, or use external heating to pre-heat before fuelling.

**Overshooting into the chi_RR-only regime.**
Ramping the edge source too hard skips past the H-mode window (`g_crit` to `g_c`) and drives the edge gradient past `g_c`. The NL flux collapses to zero at the edge and only `chi_RR · g` carries the heat, requiring an enormous localised gradient. Fix: ramp the edge source slowly and watch the edge gradient in plot `07` or `08`.

**Ignition instability / supercritical runaway.**
In the supercritical regime, rising `p` simultaneously increases the source (via alpha heating) and decreases the sink (via falling `Q_edge`). This is a doubly destabilising loop with no transport-based restoring force. Even if `∫P_alpha dx` exactly equals `Q_edge + ∫P_brem dx` at some moment, any upward fluctuation in `p` makes both sides move in the wrong direction. The runaway is eventually halted by the fixed edge BC capping the total pressure drop, not by the transport physics. Watch for `∫P_alpha dx` and `Q_edge` diverging in plot `09b`. In a real reactor this is managed by active control of the heating system, magnetic geometry, and impurity seeding — none of which are captured here.
