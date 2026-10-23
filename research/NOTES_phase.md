# The (W, N) phase plane of the flux-driven T3D runs

Script `exp_phase.py` (one process, about a minute; `--shelf` runs the variant of §6.3), log `phase_log.txt`
(`phase_log_shelf.txt`), cache `phase_cache.npz`. The same arrays are in `transport_equations_paper/data/t3d/phase_portrait.npz`
(README beside it). The figure script is `transport_equations_paper/scripts/make_phase_portrait.py`, which writes
`plots/fig_phase_portrait.pdf` and `plots/fig_phase_HI.pdf`. Written 23 Oct 2026. Nothing is committed and no .tex is touched.

Units: W = (3/2)∫p dV in MJ; N = ∫n_e dV in 10^20 (the plots use 10^22); T̄ = W/(3Nk) in keV, with k = 0.016022 MJ per
(10^20 keV); powers in MW.

---------------------------------------------------------------------------------------------------------------------

## 1. Reduction by multiple scales

The transport equations of the paper are, per channel,

$$
\tfrac32\,\partial_t p + \frac{1}{V'}\partial_\rho\big(V'Q(p,\nabla p, n,\dots)\big) = S_{\rm aux}+P_\alpha(n,T)-P_{\rm rad}(n,T;Z_{\rm eff}),
\qquad
\partial_t n + \frac{1}{V'}\partial_\rho(V'\Gamma) = S_n ,
$$

and the T3D runs have three well-separated time scales:

| layer | time | measured value (scenario F) |
|---|---|---|
| fast: local relaxation of each tube to its branch of Q(g), and barrier formation | τ_loc ~ 0.1–0.3 s | the barrier forms at 0.3 s in every member |
| thermal: the global energy content | τ_th = 1/abs(γ_th) at fixed N | 1.0–1.3 s (Z_eff = 0), 1.4–1.8 s (3.4, paux0), 2.3–7 s (Z_eff = 6, near the fold) |
| slow: the particle inventory behind the barrier | τ_N = N/(dN/dt) | 40 s on the hot branch (N/S_N = 21 s, the paper's "inventory time") |

So ε ≡ τ_th/τ_N = 0.03 (Z_eff = 0), 0.04 (3.4), 0.17 (6), and 0.6 on the cold reverse leg at Z_eff = 6.
**Fast layer (order ε_loc⁰).** On τ_loc each tube relaxes to the steady Cahn–Hilliard profile of its own law.
In the bulk the flux equals the integrated drive, Q(ρ) = Q_ss(ρ) = ∫_0^ρ S dV, and fronts sit where Q_ss = Q_M (Section 4).
The front law is the solvability condition at the next order. The profile is therefore a functional of the sources and
the boundary values. Two numbers summarise it for the global balance:

1. **Deposition efficiency.** Power dP deposited in cell c raises every temperature jump outside c by R_k dP, where
   R_k = ∂ΔT_k/∂Q_k is the incremental resistance of tube k. It therefore raises W by Φ_c dP, with
   Φ_c = Σ_{k≥c} R_k M_k and M_k = Σ_{c'≤k} 3 n_{c'} dV_{c'} k the heat capacity inside tube k. With η_c ≡ Φ_c/Φ_0:
   $$\delta W_{\rm steady} = \Phi_0\sum_c \eta_c\,\delta P_c .$$
   η runs from 1 on the axis to 0 at the edge. The part 1−η of an outer deposit leaves through the edge without
   heating the core.
2. **The barrier loss** L(W, N): the part of the edge flux that the stored energy controls. It is set by the
   barrier's residual (Rechester–Rosenbluth) branch.

**Thermal layer (order 1, exact integral).** Integrating the energy equation over the volume gives
dW/dt = S − Q_edge exactly. Closing it with the slaved profile of the fast layer gives the outer region's
solvability statement,

$$
Q_{\rm edge} = L(W,N) + \sum_s (1-\eta_s)\,P_s ,
$$

and with it

$$
\boxed{\;\frac{dW}{dt} = \eta_{\rm aux}P_{\rm aux} + \eta_\alpha P_\alpha(W,N) - \eta_{\rm rad}P_{\rm rad}(W,N;Z_{\rm eff}) - L(W,N) \equiv F(W,N)\;}
$$

$$
\boxed{\;\frac{dN}{dt} = S_N - \Gamma_{\rm edge}(N) \equiv \varepsilon\, G(W,N)\;}
$$

**Slow layer (T = εt).** With τ = t/τ_th the system reads W_τ = F and N_τ = εG.
- At leading order W sits on the stable root of F = 0, the slow manifold W = W*(N), and N drifts along it:
  $$W_0 = W^\ast(N(t)),\qquad \frac{dN}{dt} = S_N - \Gamma(N).$$
- At first order W lags the moving root:
  $$W = W^\ast + \varepsilon W_1,\qquad W_1\tau_N^{-1} = \frac{dW^\ast/dN\;dN/dt}{\gamma_{\rm th}},\qquad \gamma_{\rm th}=\partial_W F<0 .$$
- Near a fold, F ≈ a(N−N_f) − b(W−W_f)². With r = dN/dt, a slow passage through the saddle-node delays the jump past
  the static fold by
  $$t_d = \Omega_0\,(a\,r\,b)^{-1/3},\qquad \Omega_0 = 2.3381\ \text{(first zero of Ai)},$$
  With the natural scales a ~ W_s/(τ_th N_s), b ~ 1/(τ_th W_s) and r ~ N_s/τ_N, a r b ~ 1/(τ_th² τ_N), so
  t_d ~ Ω_0 τ_th^{2/3} τ_N^{1/3} = Ω_0 ε^{2/3} τ_N, up to factors of order one set by how steeply the fold is
  crossed. Those factors are large here, because the fold runs almost parallel to N (§3).
- A system held just past a fold (no root, F_max < 0 slightly) crosses the ghost of the fold in
  $$T_{\rm ghost} \simeq \pi/\sqrt{b\,|F_{\max}|}.$$

How good is leading order? It needs ε ≪ 1 and abs(γ_th) bounded away from 0.
- Burning members (ε = 0.03–0.04): W* is good to about 5% after about 3τ_th, and the first-order lag closes half of
  what remains (table §4).
- Near the fold (ε → O(1) as γ_th → 0) and on the cold legs (ε = 0.6): leading order is poor by construction. There
  the ODE itself, not its slow manifold, is the prediction.

### Reinterpretation of the paper's γ_th = −0.28/s

That number comes from slopes along the run path, where N rises. At fixed N the model gives (W = 1200, N = 1786):
- ∂(η_α P_α)/∂W = 0.69 MW/MJ and ∂L/∂W = 1.26 MW/MJ, so γ_th = −0.57/s (−0.75/s at the end of the Z_eff = 0 run);
- in logarithmic slopes, m = ∂ln L/∂ln W = 3.2 and n = ∂ln P_α/∂ln W = 1.6.

Along the path the same model gives −0.18/s, which matches the paper's −0.28/s: the path slope mixes the drift of
the root with N. So **saturation at fixed density is robust, not marginal** (m = 3.2 > n = 1.6, even with f_α = 1).
What looks like a slow 3.6 s relaxation is W following a root that climbs with N.

---------------------------------------------------------------------------------------------------------------------

## 2. The closures and where each number comes from

All the scenario-F constants come from **es_hkbm_F_frozen** (Z_eff = 0, frozen; the ladder's own template), at
t > 0.5 s, with the reference profile at t = 2.0 s (W = 1112 MJ, N_ref = 1740, T̄_ref = 13.30 keV).
- Bookkeeping: 3/2∫n(T_e+T_i)dV = 800.6 MJ against 800.4 MJ logged at t = 0.

**P_α** is the reference profile rescaled with the edge pinned (n_c → n_c N/N_ref, T_c → T_edge + s(T_c,ref − T_edge),
s = (T̄ − T_b)/(T̄_ref − T_b)), Bosch–Hale with the deck's fuel fractions:

$$P_\alpha = c_\alpha (N/N_{\rm ref})^2\sum_c w_c\,\langle\sigma v\rangle\big(T_{i,c}(s)\big),\qquad c_\alpha = 1.0006 .$$

c_α needs no calibration: Bosch–Hale reproduces the logged P_α to 0.06%.

**P_rad** is the deck's formula, Z_eff · 5.35×10⁻³ ∫ n² √T_e dV, on the same rescaled profile. It is 50.7 MW per
unit Z_eff at the reference (40.6 at t = 0). The model is within 1–4% of the logged P_rad on the hot and collapsing
rungs and 10% high on the cold legs.

**Deposition efficiencies** come from the incremental resistances, which are fitted along the reference path, t > 1 s:
- R_k = 0.0046, 0.012, 0.0093, 0.0066, 0.0046, 0.0032, 0.0017, 0.0017, 0.0003 keV/MW.
- η_c = 1, 0.99, 0.90, 0.76, 0.59, 0.42, 0.26, 0.15, 0.02, 0 (Φ_0 = 1.04 s).
- Weighted by each source's own deposition: **η_aux = 0.250** (EC, 57% of it at ρ_N 0.75–0.85), **η_rad = 0.524**
  (brem ∝ n²√T_e, with n peaked at 0.75) and **η_α = 0.708**.
- The secant resistances (ΔT/Q) give 0.33, 0.59 and 0.74.
- A posteriori check, not used for calibration: a pooled fit over the hot ladder members (fit1, which uses the
  members) wanted pass-through fractions of 0.7 (aux) and 0.5 (rad), that is η_aux ≈ 0.3 and η_rad ≈ 0.5.

**Barrier loss.**

$$L = C\,\mathrm{sgn}\,\theta\,|\theta|^{\mu}\,(N/N_{\rm ref})^{\nu},\qquad
\theta = \frac{\bar T - T_b}{\bar T_{\rm ref} - T_b}$$

- T_b = (T_e + T_i)/2 at the Dirichlet point = 3.07 keV (input). Heat flows inward when the core is colder than the
  pedestal, which is what es_hkbm_G_frozen shows.
- ν is not identifiable from one run (ln N moves 0.11 while ln T̄ moves 0.30). It is fixed by a profile argument:
  adding density in the run's own ramp pattern (n(4.7 s) − n(1 s), which lands at 0.65–0.8) at a fixed temperature
  profile lowers T̄ (d ln T̄/d ln N = −0.345, so d ln θ/d ln N = −0.448). Since RR depends on the temperatures and not
  on n, L at fixed T̄ rises with N: ν = −μ · d ln θ/d ln N.
- Fit on the reference: **C = 392.4 MW, μ = 2.49, ν = 1.12**, rms 12 MW.
- Band: ν = 0 gives C 395, μ 2.91 (rms 9.8); ν doubled (1.95) gives μ 2.18 (rms 14).

**Particles.**
- S_N inside face 0.9 = 79.3×10²⁰/s (input; 8.2 more lands in the Dirichlet cell).
- Γ_edge = Γ(0.9) = 27.4 (N/N_ref)^2.64. This is the hKBM-QL channel of the 0.9 tube, fitted on the reference.

**Check on the calibration run:** model dW/dt − run dW/dt = +25 ± 9 MW; Q_edge to 13 MW rms; P_α to 0.7%.

**Scenarios H/I** (pre-v1.2.6, GS2-QL, 6-point grid). Calibrated on **I only**:
- η from I's secant resistances: η_c = 1, 0.95, 0.78, 0.61, 0.35, 0.
- **η_EC = 0.669, η_QRAD = 0.351** (the 228 MW uniform sink, 93 MW of it in the Dirichlet cell), η_α = 0.702.
- **μ = 2.49 carried from F**; T_b = 4.82 keV (I's Dirichlet values); C = 196 MW at T̄_ref = 8.78 keV, rms 3.8 MW over I.
- dN/dt is held at I's measured +79.5/s. H is I plus the QRAD sink and nothing else.

---------------------------------------------------------------------------------------------------------------------

## 3. Predictions against the runs (no refit)

Each member is integrated from its own (W, N) at 0.5 s, after the barrier has formed (the model cannot represent the
imported-profile relaxation, Q_edge 86 → 500 MW in 0.3 s). The reverse legs start at 1.48 s.

| member | Z_eff | run W at t_end (MJ) | model W at t_end | run dW/dt now (MJ/s) | model fate (to 20 s) |
|---|---|---|---|---|---|
| frozen Z0 (calibration) | 0 | 1345 @ 4.68 s | 1378 (+2.5%) | +75 | burns, follows W*(N): 1643 @ 10 s, 1967 @ 20 s |
| brem 3.4 | 3.4 | 1221 @ 4.92 | 1227 (+0.5%) | +63 | burns, 1790 @ 20 s |
| brem 6 | 6 | 1020 @ 4.42 | 1008 (−1.2%) | +42 | burns, 1594 @ 20 s |
| brem 6.5 | 6.5 | 857 @ 0.74 | 850 | +55 | burns, 1535 @ 20 s |
| brem 7 | 7 | 842 @ 0.63 | 838 | +50 | burns, 1418 @ 20 s |
| brem 8 | 8 | 832 @ 0.76 | 823 | +30 | **past the fold**: lingers at 800–820 MJ to about 4 s (the ghost of the fold), then collapses, 508 @ 10 s, at the floor by 15 s |
| brem 10 | 10 | 564 @ 4.44 | 539 (−4%) | −116 | collapses to the temperature floor by about 10 s; **no re-ignition** |
| brem 12 | 12 | 433 @ 3.41 | 456 (+5%) | −157 | collapses |
| brem 15 | 15 | 191 @ 3.72 | 194 (+2%) | −72 | collapses |
| brem 20 | 20 | 181 @ 2.50 | 168 (−7%) | −114 | collapses |
| rev 20 → 3.4 | 3.4 | 550 @ 3.41 | 440 (−20%) | +65 | **re-ignites** (no cold root at this N), but slowly through the ghost; burns by 17 s |
| rev 20 → 5 | 5 | 465 @ 3.33 | 378 (−19%) | +15 | **stays cold**; the cold root deepens as N rises |
| rev 20 → 6 | 6 | 427 @ 3.28 | 345 (−19%) | −3 | **stays cold** |
| paux0 | 0 (P_aux = 0) | 1203 @ 4.52 | 1232 (+2.4%) | +71 | burns; saturates without auxiliary power |
| hkbm_F_k1_v2 (CDE on) | 0 | 1346 @ 5.34 | — (shown in the plane only) | | |

**Outcome score.**
- Every member with a decided outcome is right: burn at 0, 3.4, 6, paux0; collapse at 10, 12, 15, 20; re-ignition at
  rev 3.4; cold at rev 5 and 6.
- W is within 0.5–2.5% on the burning branch, including paux0, whose difference from Z_eff = 0 is a pure test of
  η_aux, and within 2–7% on the collapsing rungs.
- The cold legs are about 20% low (and drifting).
- z6.5, 7 and 8 are undecided in the runs (0.6–0.8 s).

**Fold and hysteresis** (base, ν = 1.12; band [ν = 0; ν doubled]):

| N (10²⁰) | 1673 (t = 0) | 1767 (~2 s) | 1850 (~3.5 s) | 1950 (~5 s) | 2100 | 2300 |
|---|---|---|---|---|---|---|
| hot root exists up to Z_eff | 7.8 [8.6; 7.3] | 7.8 [8.7; 6.9] | 7.7 [8.9; 6.6] | 7.7 [9.2; 6.4] | 7.8 [9.7; 6.0] | 8.0 [10.5; 5.7] |
| cold root exists from Z_eff | 5.5 | 4.9 | 4.5 | 4.1 | 3.6 | 3.0 |

- The bistable window (hysteresis) is Z_eff ≈ 5.5–7.8 at the starting density.
- It widens downwards as N rises: the cold basin grows, because the radiation (∝ N²) beats the alpha power of a cold
  core (∝ N²⟨σv⟩ ≪) while L ∝ N^ν.
- The predictor (`predict_endstate.md` §3, §6.5) puts the fold at 6.45–6.5 and the cold root from 5. The model has
  the same cold edge and a fold 1.3 higher.

**The z8 member decides between the two folds.** The base model says z8 is already past the fold: a ghost to about
4 s, then collapse. The predictor (6.45) says the same. The ν = 0 band (fold 8.6) says it burns.

**The fold vs the density.**
- Base ν: the fold is pinned at 7.7–8.0. The density lifts the hot branch (W* +40% from N = 1673 to 2300 at
  Z_eff = 0) but lifts the unstable branch with it.
- ν = 0: the fold moves to higher Z_eff (8.6 → 10.5), which is the predictor's direction (6.45 → 9–10).
- ν doubled: it moves down.
- So "the fold moves with the density" is exactly a statement about ν, which a Z_eff = 0 run cannot measure. The
  members near the fold (z7, z8) decide it.

**Slow passage, base model.**
- In the base model the ramp crosses no hot fold. At Z_eff = 8 the hot root exists only for N < 1602, below the
  starting 1673, so the ramp carries the member away from the fold, not through it. At 7.8 the fold is at
  N_f = 1773. It is almost parallel to the N axis (a = ∂F/∂N = 0.01–0.04 MW per 10²⁰), so a crossing there would
  take t_d = 21–30 s. The only base-model folds the ramp reaches are the appearances of cold roots (below).
- In the ν-doubled band, where the ramp does cross the fold (Z_eff 6.5 at N_f = 1912; Z_eff 7 at N_f = 1752),
  a ≈ 0.08, b ≈ 5.5×10⁻⁴ MW/MJ² and r ≈ 45/s, so **t_d ≈ 17–19 s**, about 3ε^{2/3}τ_N.
- The appearance of the cold root at Z_eff = 3.4 (N_f = 2145), 4 (1968) and 5 (1746) has t_d = 17–22 s.
- Conclusion: **the density ramp does not push any member through the thermal fold within the campaign's horizon.**
  The fold in Z_eff is thermal. A member just past it sits in the ghost: at Z_eff = 8 the net heating near
  W = 790 MJ is only −3 MW, and the bottleneck time π/√(b·abs(F_max)) is tens of seconds.
- **z10 re-ignition:** the model gives none. z10 has no cold root above the solver's temperature floor and is past
  the hot fold at every N up to 3500. The predictor's static estimate of 9.5 s would, by the formula above, be
  delayed to about 9.5 + t_d ≈ 25–30 s even if its fold were right (a, b, r as measured here).
- The run (564 MJ, −116 MJ/s at 4.44 s) is falling at the model's rate (model 582 MJ at 4 s).

---------------------------------------------------------------------------------------------------------------------

## 4. Slow manifold along the burning runs (leading and first order)

| run | t (s) | run W | W*(N(t)) | W* + lag | τ_th (s) |
|---|---|---|---|---|---|
| Z0 | 1 / 2 / 3 / 4.7 | 995 / 1114 / 1208 / 1345 | 1259 / 1312 / 1364 / 1447 | 1201 / 1256 / 1310 / 1398 | 1.0–1.1 |
| 3.4 | 1 / 2 / 3 / 4.9 | 933 / 1018 / 1093 / 1221 | 1147 / 1195 / 1243 / 1331 | 1073 / 1125 / 1175 / 1270 | 1.4–1.5 |
| 6 | 1 / 2 / 3 / 4.4 | 880 / 916 / 959 / 1020 | 1024 / 1067 / 1110 / 1169 | 919 / 964 / 1011 / 1076 | 2.3–2.5 |
| paux0 | 1 / 2 / 3 / 4.5 | 926 / 1015 / 1093 / 1203 | 1160 / 1217 / 1272 / 1353 | 1073 / 1137 / 1198 / 1287 | 1.3–1.5 |

- The early gaps are the relaxation from the imported profile (800 MJ), which decays on τ_th.
- By 4.5–5 s the runs are within 4–7% of the first-order slow manifold, and the full ODE is within 0.5–2.5%.
- **No member is near a steady state.** The burning ones ride W*(N), which climbs by about 50 MJ/s with the density.
  dN/dt > 0 everywhere in the window: S_N = 79 > Γ = 27–40. The N-nullcline (Γ = S_N) lies at N ≈ 2600 by
  extrapolation of the hot-branch Γ ∝ N^2.6, outside anything the runs reach.

---------------------------------------------------------------------------------------------------------------------

## 5. H and I

| | I (EC) | H (EC + QRAD) |
|---|---|---|
| net external heating, volume integral | +141 MW | +141 − 228 = −87 MW (+4 MW in the evolved region) |
| deposition-weighted, η_EC P_EC − η_QRAD P_QRAD | +94 MW | +94 − 80 = +14 MW |
| stable root of F at the starting N (1522) | 671 MJ | 590 MJ |
| model from t = 0, W at the run's end | 710 (run 690, +3%) | 638 at fixed dN/dt (run 400) |
| run W vs W*(N_run(t)) | 650 / 664 / 690 vs 672 / 702 / 731 (+3–6%) | 618 / 490 / 425 / 400 vs 577 / 490 / 455 / 448 (0 to +12%) |

Reading:
- The only change, the radiation, lowers H's W-nullcline below its starting point (671 → 590 MJ), so W must fall.
- The thermal model alone keeps a stable root (the root survives up to 1.48× QRAD), so it does not collapse H on its
  own.
- H's collapse is a transport event. The drive at the 0.84 tube is Q_ss = 184 MW in H against 337 in I at 0.2 s. The
  barrier is lost at 0.2–0.3 s (a/L_Ti(0.84): 5.0 → 2.5; the edge-flux burst reaches 1.2–1.9 GW). With it the
  particles go: N falls 15.2 → 12.6 ×10²² in 0.3 s.
- **Along the run's own N(t), W falls onto the W-nullcline of H and follows it down** (490 vs 490 at 0.31 s; 400
  against 448 at the end).
- So H's 40% loss of stored energy is about 12% radiation (the nullcline moves), about 24% the density lost with the
  barrier (motion along the nullcline), and about 10% the higher loss of the barrier-less state, which the model does
  not contain.
- The paper's phrase "H's net external heating is ≈ 0" is a volume-integral statement. Weighted by where the power is
  deposited, H still has +14 MW, so the W-nullcline still has a root. It lies 70 MJ below the start, not above it.

---------------------------------------------------------------------------------------------------------------------

## 6. Where the model fails

1. **Cold branch.** The model's cold roots are too low: 191 / 121 MJ at Z_eff = 5 / 6 against legs at 465 / 427
   MJ that are still drifting. The integrated legs are about 20% low at their last times.
   - The cold profile is flat-to-hollow at T ≈ 5 keV. The η's and the RR resistance are those of the hot state, but
     in the cold state the edge resistance R ∝ Q^{-2/3} is larger, so edge deposits (EC at 0.75–0.85) are more
     effective.
   - The cold-leg Q_edge (111–156 MW) ≈ the pass-through (1−η_aux)P_aux − (1−η_rad)P_rad (133–169 MW), and the core
     sits at the shelf temperature T_e(0.85) ≈ 5 keV. On the reference's own Q_edge–T_e(0.85) law, that is the
     temperature that carries the pass-through 236 MW.
   - Variant (`--shelf`, T_b = 5.02 keV from that law):
     - What it fixes: the cold legs move to 399 / 376 (−14% / −12%), and the fold at the starting N moves to 6.5,
       which is the predictor's value.
     - What it breaks: z6 is predicted to collapse (888 at 4.42 s against the run's rising 1020), and z15 / z20 sit
       at 255–270 against 181–191.
   - One floor cannot hold both the cold shelf and the hot fold. The cold state is a different branch of the
     barrier, a flat core at the shelf with no gradient, not the low end of the hot one. The base model keeps the
     Dirichlet floor because it gets every decided outcome right.
2. **Particles.** One Γ law, calibrated hot (Γ(0.9) ≈ 28, with hKBM on at the 0.9 tube). On the collapsed members
   the 0.9 tube's hKBM is off and Γ(0.9) ≈ 3, so dN/dt is 77/s against the model's ~50/s. The cold legs' N lags,
   which delays any N-driven event on them by about 50%. H's particle loss is outside the model entirely.
3. **The fold position depends on ν** (§3). It is the one exponent the calibration run cannot fix.
4. **The first 0.3–0.5 s**, the relaxation of the imported profile and barrier formation, is the fast layer. The
   model is started after it.
5. **H/I.** μ is borrowed from F, the runs are pre-fix, and the collapse itself needs the transport threshold.

---------------------------------------------------------------------------------------------------------------------

## 7. Plain-language explanations (ready to paste)

**The two variables.** A flux-driven run has two clocks. The stored energy W relaxes in a second or two to whatever
the heating and the barrier's leak allow. The particle inventory N, fed by pellets that leave only by neoclassical
diffusion through the barrier, changes on tens of seconds. So at any moment W sits on a fixed point of

$$\frac{dW}{dt} = \eta_{\rm aux}P_{\rm aux} + \eta_\alpha P_\alpha(W,N) - \eta_{\rm rad}P_{\rm rad}(W,N) - L(W,N)$$

at the current N, and the fixed point moves as N grows. The weights η say how much of each source stays in the core.
Power put near the edge leaves almost at once: only a quarter of the EC power, half the bremsstrahlung and 70% of the
alpha power count.

**Burn.** The barrier's leak rises as the cube of the core temperature (m = 3.2) while the alpha power rises as its
square or less (n = 1.6). Above some W the leak wins, so the hot fixed point is stable. It does not need the
auxiliary heating: the member without EC burns too, as predicted to 2%.

**Why nothing settles.** The hot fixed point climbs with the density, because more fuel at the same temperature means
more alpha power. The density never stops rising: the pellets add 79×10²⁰/s and the edge removes 30. The burning runs
are therefore not relaxing slowly to an end state. They relax in about a second onto a fixed point that moves up by
about 50 MJ/s. The paper's slow "−0.28/s" is that drift seen along the path; at fixed density the relaxation is
−0.6 to −0.75/s.

**The fold.** Radiation lowers the W-nullcline. Its hot branch moves down until, near Z_eff ≈ 7.8 at the starting
density (6.45 in the profile predictor), it meets the unstable branch below it and both disappear: the fold. Past
it, no hot state exists and the plasma cools until radiation balances on the cold branch or the temperature floor.
Just past the fold the net heating is only a few MW, so the plasma lingers near the vanished state for seconds (the
ghost). A member that is "still rising at 0.8 s" near the fold has not yet been tested.

**Hysteresis.** Below the fold, from Z_eff ≈ 5.5 (5 in the predictor), a cold fixed point exists as well. A collapsed
plasma restarted at Z_eff = 5 or 6 stays cold, because it would need fifteen times its alpha power to climb over the
unstable branch. At 3.4 there is no cold fixed point, and the plasma re-ignites.

**The density and the fold.** The density lifts the hot branch, but it also raises the barrier leak at a given mean
temperature, because the new particles sit behind the barrier where it is cooler. With the leak's density exponent
measured from the profile, the two effects cancel: the fold stays at Z_eff ≈ 8 and the cold basin widens instead.
If the leak did not depend on density, the fold would move up with it, as the profile predictor says. The members at
Z_eff = 7 and 8 will decide which. Either way, the density moves the fold too slowly to drive a member through it
within the runs: the slow-passage delay is 17–30 s.

**H and I.** I and H differ only in the imported radiation. Weighted by where it is deposited, it removes 80 of the
94 MW of effective EC heating, so H's W-nullcline sits 80 MJ below I's and below the start: W must fall. The thermal
balance alone would let H settle 12% lower. The collapse comes when the reduced drive through the outer tube can no
longer hold the barrier: the barrier goes, the particles follow (−17% in 0.3 s), and W slides down H's own nullcline
with the density, to 400 MJ.
