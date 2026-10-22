# The reduced model with a gyrokinetic-like flux law

Research notes, 22 Oct 2026.  Code: `gk_law.py` (law, Maxwell construction, BDF integrator, front law),
`gk_law_runs.py` (fit / table / runs / alpha / nu4test / figs).  Caches (all `.npz`, keys in section 7):
`gk_law_bump_fit.npz`, `gk_law_maxwell_table.npz`, `gk_law_runs.npz`, `gk_law_alpha.npz`, `gk_law_nu4.npz`.
Quick-look PNGs in `gk_law_figs/`.  Nothing in `reduced.py` or the clean model files was modified; the RHS is
re-implemented with the same discretisation (dx = 0.005, three binomial passes on the face gradients,
conservative hyperviscosity, Neumann axis, pinned edge, Gaussian source of width 0.25 L) because
`reduced.make_rhs` hard-codes the paper's flux function.

## 0. Question and answer

Should the reduced model use a flux law that follows the nonlinear gyrokinetic curve, and what then decides
between L-mode-like clamping, barrier, collapse and runaway, and whether the burn saturates?

**Answer.**  The Cahn-Hilliard theory of `theory.tex` carries over unchanged; only the numbers change, and they
change by orders of magnitude.  With the law Q = H B(g) + c_e g^3 + c_i g (bump of height H on the
Rechester-Rosenbluth and neoclassical floors) the Maxwell flux is Q_M = (4A/3)^{3/4} c_e^{1/4}
= 1.24 (A^3 c_e)^{1/4} = 0.805 H^{3/4} c_e^{1/4} with A = 0.562 H the bump area (exact to 2-8% for
c_e/H <= 1e-3), i.e. on the STEP line (c_e = 1.07e-4 H) **Q_M = 0.0847 H**.  A barrier can be
held only if the drive exceeds Q_M, i.e. only if **H/S < 11.8**.  The nonlinear bump has H/S ~ 4e2-6e3
(12 GW/m^2 against 2-30 MW/m^2), so Q_M/S ~ 30-500 and no barrier can be held: a supercritical profile collapses
in a time (W_0 - W_L)/(Q_M - S) (milliseconds in STEP units) onto the rising branch at the tiny gradient
H B(g) = S, g = (S/H)^{0.365} ~ 0.04-0.11 g_pk (L-mode-like clamping).  A barrier exists only for a bump
at most ~10x the drive (the quasilinear regime), and then it sits on the RR floor at g_+(S) = (S/c_e)^{1/3}
~ 9-15 g_pk (a/L_Te ~ 10-20), which is a runaway in all but name.  With alpha heating the burn always
saturates once a barrier exists, because the floor Q ~ g^3 outruns P_alpha ~ p^2 (thermal stability
condition 2 P_alpha < 3 S_tot is automatic); where no barrier exists alpha heating does not create one unless
the alpha power itself exceeds Q_M - S_ext (GW/m^2 for STEP), and the L-mode burn is fixed by the pinned edge.

## 1. The fitted bump

Data: `data/gk/q_vs_betaprime.npz`, stella Q_e + Q_i (gB) against |beta'| (20 points, 18 live).  The peak is
pinned at the data maximum, g_pk = |beta'| = 0.388 (beta_e = 0.07), Q_pk = 747 gB (= 12.1 GW/m^2 in the
published units, 2.97 GW/m^2 in the BPAR unit).  Fit of ln(Q/Q_pk) against u = ln(g/g_pk), equal weights:

| shape | form | parameters | rms ln residual |
|---|---|---|---|
| lognormal (asymmetric Gaussian in ln g) | -u^2/(2 s^2), s = s_L (u<0), s_R (u>0) | s_L = 0.528, s_R = 0.199 | 0.87 (factor 2.4) |
| general | -\|u/s\|^m per side | s_L = 0.367, m_L = 1.00; s_R = 0.289, m_R = 2.08 | 0.43 |
| **plgauss (used)** | B = g^k (g<1); exp(-u^2/(2 s_R^2)) (g>1) | **k = 2.74, s_R = 0.199** | 0.43 (factor 1.5) |

The rising branch is a power law, Q ~ |beta'|^{2.7}, not a Gaussian in ln g: the asymmetric log-Gaussian
cannot fit both the two-decade rise over a factor 4 in |beta'| and the shallow bottom (its residuals reach a
factor 7 at the lowest point and 2-4 on the transition points).  The general 4-parameter fit returns
m_L = 1.00 by itself, which is the power law; fixing m_L = 1, m_R = 2 (plgauss) loses nothing.  Residuals of
plgauss (ln data - ln fit) are within +-0.65 everywhere except the two "plateau" points at |beta'| = 0.22, 0.28
(+0.6, +0.8): the data have a flat top between 0.22 and 0.44 that the fit rounds off.  The bump area
A = int B dg = 0.562 g_pk (plgauss), 0.70 (lognormal); the Maxwell flux is 24% higher with the lognormal.

STEP line.  At the nominal point (|beta'| = 0.499, g_nom = 1.286 g_pk) the RR floor as T3D runs it is
2.27e-4 Q_pk and the NEO ion floor 1.14e-5 Q_pk.  (These are half the task's "RR/peak ~ 4e-4, NEO/peak
~ 2e-5" because those, like the "7 GW/m^2" of gyrokinetics.tex, refer to the electron flux alone, whose
peak is 411 gB = 6.7 GW/m^2 at |beta'| = 0.39: 2.76/6678 = 4.1e-4.  The bump here is Q_e + Q_i, as the
task asked, with peak 747 gB = 12.1 GW/m^2, so H/S and Q_M below are to be read against the total.)  With Q_RR ~ g^3 and Q_NEO ~ g measured from g = 0 (the single-field path,
no density-gradient offset; the paper's `make_gk_maxwell.py` scales only the temperature gradients and keeps
an offset b0 = 0.19 in |beta'|, which puts its floor 2-3x higher at g ~ 2-7 and its Q_M 40% higher, 1478
against 1028 MW/m^2):

    c_e / H = 1.07e-4,   c_i / H = 8.9e-6.

Dimensionless units used throughout: g in units of g_pk, Q in units of the integrated external drive S
(S = 1), x in units of L, p = P_PED + int g dx with P_PED = 1, time in units of g_pk L^2 / S.
For STEP (g_pk = 0.388 in beta' per r/a, B = 2.79 T, a = 2.02 m) the pressure unit is g_pk B^2/2mu0 =
1.2 MPa and the time unit 1.2 s at S = 2 MW/m^2, 0.08 s at 30 MW/m^2.

## 2. Theory: the Maxwell flux of a bump on a floor

Equal-area rule int_{g_a}^{g_b} [Q - Q_M] dg = 0 with Q(g_a) = Q(g_b) = Q_M.  For a bump of area A that is
dead by g_b, on a floor c g^n with c g_b^n = Q_M and g_a << g_b,

    A + c g_b^{n+1}/(n+1) - Q_M g_b = 0   =>   A = n/(n+1) Q_M g_b = n/(n+1) Q_M^{(n+1)/n} c^{-1/n}

    Q_M = [(n+1)/n A]^{n/(n+1)} c^{1/(n+1)}  ~  (A^n c)^{1/(n+1)};   n = 3:  Q_M = (4A/3)^{3/4} c_e^{1/4}.

With A = 0.562 H this is Q_M/H = 0.805 (c_e/H)^{1/4}.  Checked against the exact construction
(`gk_law_maxwell_table.npz`, `curve_plgauss_step_*`): exact/estimate = 1.02, 1.01, 1.02, 1.04, 1.08, 1.19, 1.53
at c_e/H = 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 (the estimate degrades as g_a and the bump's own
contribution at g_b stop being negligible).  No falling branch (floor swallows the bump) for c_e/H > 0.61.

Because Q depends on (H, c_e, c_i) only, Q_M/H is a function of c_e/H alone at fixed c_i/H (table columns
`curve_*`); Q_M/S = (H/S) f(c_e/H).  The 2-D table on (H/S, c_e/S) with c_i/S = 1e-3 agrees with
(H/S) f(c_e/S / (H/S)) to 0.6% (median) except where the ion floor dominates (c_e/H < 1e-5 and H/S < 1).
The ion floor is irrelevant on the STEP line (c_i/H = 8.9e-6: Q_M/H = 0.0847 with it and without; it would
take c_i/H = 1e-2 to move Q_M by 50%).  (The curve's nearest grid point, c_e/H = 1.12e-4, gives 0.0857;
the exact STEP value is 0.0847 and H_marg = 11.8 is used in the runs.)

| c_e/H | Q_M/H | g_a | g_b | | STEP line | H/S | Q_M/S | Q_min/S | g_-(S) | g_+(S) | regime |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1e-7 | 0.0146 | 0.21 | 52 | | | 0.1 | 0.0085 | 1.5e-4 | - | 45 | forced barrier |
| 1e-6 | 0.0258 | 0.26 | 29 | | | 1 | 0.085 | 1.5e-3 | (1.0) | 21 | forced/marginal L |
| 1e-5 | 0.0460 | 0.33 | 17 | | | 3 | 0.254 | 4.4e-3 | 0.67 | 14.6 | bistable |
| 1.07e-4 | 0.0847 | 0.41 | 9.2 | | | 10 | 0.85 | 1.5e-2 | 0.43 | 9.8 | bistable |
| 1e-3 | 0.154 | 0.51 | 5.4 | | | 11.8 | 1.00 | 1.7e-2 | 0.41 | 9.3 | marginal |
| 1e-2 | 0.302 | 0.64 | 3.1 | | | 30 | 2.54 | 4.4e-2 | 0.29 | 6.8 | L only (collapse) |
| 1e-1 | 0.694 | 0.85 | 1.9 | | | 100 | 8.5 | 0.15 | 0.19 | 4.5 | L only |
| | | | | | | 1000 | 85 | 1.5 | 0.080 | - | L only, Q_min > S |
| | | | | | | 1e4 | 850 | 15 | 0.035 | - | L only, Q_min > S |

(g_-(S) = (S/H)^{1/k} is the L-mode gradient; g_+(S) the gradient at which the residual branch carries S,
undefined when S < Q_min.)

Physical numbers on the STEP line (published units, Q_pk = 12.1 GW/m^2 for Q_e + Q_i): Q_M = 1.03 GW/m^2 (0.25 GW/m^2 in
the BPAR unit); for S = 2, 10, 30 MW/m^2: H/S = 6100, 1200, 400; Q_M/S = 510, 100, 34; Q_min/S = 9.0, 1.8, 0.6;
L-mode gradient |beta'| = 0.016, 0.029, 0.043 (a/L_Te = 0.05, 0.09, 0.14 if all gradients scale together),
i.e. 10-30x below the nominal |beta'| = 0.5 and 1.3-3.5x below the lowest scan point (0.055), so the clamped
gradient rests on the extrapolation Q ~ g^{2.7} below the data.  Q_M in the (A^3 c)^{1/4} scaling means:
halving the RR coefficient lowers Q_M by 16%; the "no T_ref^1.5" RR coefficient (35x) raises it by 2.4x.

## 3. The end-state map in the (H/S, c_e/S) plane

`gk_law_maxwell_table.npz`: `table_regime` on `H_over_S` (51 points, 0.1-1e4) x `ce_over_S` (81 points,
1e-9-10), c_i/S = 1e-3.  Regime codes:

* 0 monotone: c_e/H > 0.61, no falling branch, a single diffusive branch (L-mode-like at any drive).
* 1 forced barrier: Q_max < S (H/S < ~1): the L branch does not exist, any start goes to the residual branch.
* 2 bistable: Q_M < S < Q_max: barrier from a supercritical start, L from a subcritical one.
* 3 L only: S < Q_M: a barrier front has no rest point, the supercritical start collapses.
* 4 L only and S < Q_min: the residual branch cannot even carry the drive (STEP line for H/S > ~700).

Boundaries: Q_max = S is H/S = 1 (vertical); Q_M = S is `maxwell_boundary_*`, fitted c_e/S = 1.96 (H/S)^{-3.02}
(leading order (4A/3)^{-3} (H/S)^{-3} = 2.38 (H/S)^{-3}), running from (H/S = 1, c_e/S = 0.22) to
(H/S = 800, c_e/S = 1e-12); the STEP line c_e/S = 1.07e-4 H/S crosses it at H/S = 11.8.  "Runaway" is not a
separate region once c_e > 0 (the gradient is always bounded); it is drawn as the contour g_+(S) = g_cap with
g_cap = 10 g_pk (a/L_Te ~ 12; a choice), c_e/S = 1e-3, below which every barrier state sits at an
unphysical gradient.  On the STEP line every barrier state (H/S < 11.8) has g_+(S) >= 9.3 g_pk, so the
barrier regime of this law is the runaway regime in practice; the only "moderate-gradient" barriers are off the
line, at c_e/S >~ 0.01 with H/S < 5 (e.g. H/S = 3, c_e/S = 0.03: g_+(S) = 3.2, Q_M = 0.90 S).

Text map (rows c_e/S, columns H/S; F forced, B bistable, L collapse/L-only, l L-only with Q_min > S, . monotone):

```
             0.1   0.32   1     3.2   10    32    1e2   3e2   1e3   3e3   1e4
 1e+01       .     .      .     .     .     l     l     l     l     l     l
 1e+00       .     .      .     l     l     l     l     l     l     l     l
 1e-01       .     F      B     L     L     L     l     l     l     l     l
 1e-02       F     F      B     B     L     L     L     L     L     L     L
 1e-03       F     F      B     B     B     L     L     L     L     L     L
 1e-04       F     F      B     B     B     L     L     L     L     L     L
 1e-05       F     F      B     B     B     B     L     L     L     L     L
 1e-06       F     F      B     B     B     B     B     L     L     L     L
 1e-08       F     F      B     B     B     B     B     B     L     L     L
```

## 4. Dynamics without alpha heating (`gk_law_runs.npz`, `gk_law_runs.py runs`)

All runs: dx = 0.005 (N = 200), n_smooth = 3, conservative nu4 = 5e-5 max|Q'| (the paper's 5e-4 at
max|Q'| = 10; here max|Q'| = 3.6 H on the falling branch, so nu4 = 1.8e-4 H), BDF with a banded Jacobian,
rtol 1e-6.  Source: Gaussian of width 0.25 L normalised on the evolved cells so that the steady edge flux is
exactly S_tot; Q_ss(x) of the analytic branches uses the same discrete sum.  Supercritical start = the
analytic branches with a front at x0 (g_-(Q_ss) inside, g_+(max(Q_ss, Q_M)) outside, i.e. the Maxwell
plateau g_b where Q_ss < Q_M); subcritical start = the paper's power-law profile with max g = 0.5 g_pk.
Every run takes 0.1-5 s.  Steadiness certified by dW/dt and Q_edge = S_tot, not by t.

| case | H/S | c_e/S | S | start | Q_M/S | x_M | end state | W_end | analytic | front/collapse |
|---|---|---|---|---|---|---|---|---|---|---|
| a | 1000 | 0.107 | 1 | super, x0 = 0.3 | 84.7 | - (Q_min = 1.48 > S) | collapse to L, g_max = 0.080 | 1.0391 | W_L = 1.0396 | t_W = 0.047, front (mid-level) at 0.95 at t = 0.042; Stefan law t_wall = 0.048; (W_0-W_L)/(Q_M-S) = 0.050 |
| a | 1000 | 0.107 | 1 | sub | 84.7 | - | L, g_max = 0.080 | 1.0391 | 1.0396 | relaxes in t ~ 0.05 |
| a | 30 | 3.2e-3 | 1 | super | 2.54 | - | collapse; bulk on L, edge half-front (see below) | 1.426 | W_L = 1.143 | t_W = 2.50, front at 0.95 at t = 2.39; law 2.60; energy est. 2.65 |
| a | 30 | 3.2e-3 | 1 | sub | 2.54 | - | L, g_max = 0.289 | 1.1423 | 1.1426 | |
| b | 3 | 3.2e-4 | 1 | super | 0.254 | 0.059 | barrier filling the domain, foot x_f = 0.058, g_edge = 14.6 = g_+(S) | 8.156 | W_H = 8.198 (0.5%) | steady by t ~ 10 (residual-branch diffusion D = 3 c_e g^2 ~ 0.2) |
| b | 3 | 3.2e-4 | 1 | sub | 0.254 | 0.059 | L, g_max = 0.67 (bistable) | 1.3301 | W_L = 1.3303 | |
| b2 | 3 | 0.03 | 1 | super | 0.905 | 0.296 | barrier, foot x_f = 0.304, g_edge = 3.22 = g_+(S) | 2.482 | W_H = 2.492 (0.4%) | |
| b2 | 3 | 0.03 | 1 | sub | 0.905 | 0.296 | L (bistable) | 1.3291 | 1.3292 | |
| c | 11.8 | 1.26e-3 | 0.9 | super | 1.00 | - | collapsing: x_f = 0.84 at t = 40, still moving | 2.01 | W_L = 1.19 | law: wall at t = 39; energy est. 40; numerical lag ~1.3x |
| c | 11.8 | 1.26e-3 | 1.0 | super | 1.00 | 0.998 | marginal: front at 0.45, dW/dt/W = -3e-5 | 4.62 | | rest point degenerate: Q_ss = Q_M to 1e-4 for x > 0.6, front speed < 1e-3 |
| c | 11.8 | 1.26e-3 | 1.1 | super | 1.00 | 0.300 | barrier, x_f = 0.294, g_edge = 9.55 | 5.291 | W_H = 5.355 (1.2%) | |
| c | 11.8 | 1.26e-3 | 1.1 | sub | 1.00 | 0.300 | L (bistable) | 1.2072 | 1.2074 | |
| e | 0.3 | 3.2e-5 | 1 | sub | 0.025 | 0.008 | forced transition: residual branch everywhere (x_f = 0.018), g_edge = 31.5 | 16.51 | W_H = 16.54 (0.2%) | L branch does not exist (Q_max = 0.3 < S) |

What the runs show.

* **Collapse (a)** is the Stefan retreat of `theory.tex` eq. (frontspeed) with Q_M ~ 10^2 S: the barrier dumps
  its stored pressure at the Maxwell flux, Q_edge sits at 85-93 S while the front crosses the domain, then
  falls to S.  The collapse time agrees with the integrated front law to 2% and with (W_0 - W_L)/(Q_M - S)
  to 6%, much better than the paper's factor 1.5-2.5 because here the drain Q_M - S is so large that the
  pedestal re-equilibration lag is negligible.  When the last edge layer dissolves (t ~ 0.12) its gradient
  crosses the bump from the residual branch to the L branch and Q_edge flashes (to > 30 S on the sampled
  output, H S in principle): the final burst of the collapse.  The end state is the analytic L branch to
  0.05% in W; the
  gradient g = (S/H)^{0.365} = 0.080 g_pk, with Q_edge = S.  In STEP units: 60 ms at S = 2 MW/m^2, 4 ms
  at 30 MW/m^2 (time unit g_pk B^2 a/(2 mu0 S)); generically the pedestal stored energy per unit area
  divided by Q_M ~ 1 GW/m^2, i.e. milliseconds.
* **Barrier (b)** exists only because H/S = 3 < 11.7.  It is the paper's H branch with the cliff replaced by
  the RR floor: foot at x_M (0.304 against 0.296 for b2, 0.058 against 0.059 for b; x_f is the mid-level
  crossing (g_a + g_b)/2 of the front, whose 10-90% width is 0.05-0.15), gradient g_+(Q_ss(x)) behind it,
  W to 0.5%.  On
  the STEP line the gradient is 14.6 g_pk (|beta'| = 5.7, a/L_Te ~ 18) and the foot is at x_M = 0.06, i.e.
  the barrier is the whole profile; only off the line (b2, c_e/S = 0.03) is the barrier a moderate-gradient
  pedestal with a core.
* **Bistability** holds in every case with Q_M < S < Q_max (b, b2, c at S = 1.1): the subcritical start stays
  on L to 0.02%; the supercritical start finds the barrier.  Forward threshold Q_max = H(1 + ...), back
  threshold Q_M, hysteresis Q_max - Q_M = H (1 - 0.0857) on the STEP line.
* **Marginal (c)**.  With the source localised (Q_ss = S to 1e-4 for x > 0.6), S = Q_M leaves the front with
  no preferred position over the outer half of the domain: the numerical front creeps at < 1e-3 per time
  unit and the run is not steady at t = 40 (dW/dt/W = -3e-5).  At S = 0.9 Q_M the retreat is slow, time
  ~ x_M (g_b - g_a)/(Q_M - S) ~ 40, and the numerical front lags the law by ~1.3x (the pedestal relaxation
  lag documented in the paper); at S = 1.1 Q_M the front rests at x_M = 0.300 (numerical 0.294) with W to
  1.2%.
* **Forced transition (e)**: with H < S there is no L branch; the subcritical start climbs through the bump and
  lands on the residual branch everywhere, W to 0.2% of the analytic state, g_edge = 31 g_pk.  This is the
  "runaway" of the paper with the cliff replaced by the RR floor: bounded, but at a gradient no equilibrium
  holds.

## 5. Regularisation and numerical caveats (`gk_law_nu4.npz`, `gk_law_runs.py nu4test`)

* nu4 and the smoothing do not change the end states: collapse (a, H = 1e3) W_end = 1.0396, 1.0395, 1.0391,
  1.0383, 1.0358 for nu4 x 0.1, 0.3, 1, 3, 10 (W_L = 1.0398), g_max = 0.080 throughout; barrier (b, H = 3)
  W_end within 0.3% over the same range; n_smooth = 0 gives the same numbers as n_smooth = 3.
* The front has two scales: sqrt(nu4/Q'(g_a)) on the steep (bump) side and sqrt(nu4/(3 c_e g_b^2)) on the
  soft residual-branch side; with Q'(g_a)/Q'(g_b) ~ 20-100 the 10-90% width is dominated by the soft side
  (23 cells at H = 1e3, 8 at H = 3 for nu4 x 1; 9 and 4 cells for nu4 x 0.1).  The collapse time of (a) is
  nu4-independent for nu4 <= 1x (0.047-0.054) and lengthens for wider fronts (0.077, 0.175 at 3x, 10x).
* **Grid pinning.**  For near-marginal drives (|S - Q_M| <~ 10%) the front sticks to the grid when the
  steep side of the front is < ~2 cells (nu4 x 0.3 and below): at S = 0.95 Q_M the retreating front stops
  at x = 0.30 (nu4 x 0.1) or 0.47 (x 0.3) at t = 40 against 0.77 from the Stefan law, and at S = 1.05 Q_M
  the advancing front stays at its initial 0.30 instead of moving to x_M = 0.351; with nu4 x 1 (the
  paper's rule, used for all runs) they reach 0.75 and 0.34, i.e. follow the law to 3-5%; wider fronts
  (x 3, x 10) lag again.  It is a Peierls-type pinning of a discrete front with a vanishing driving force.
  The barrier foot of case (b) sits at 0.058 (x_M = 0.059) for nu4 x 0.3-1 and drifts to 0.08-0.11 for
  x 3-10, where the front is 17-22 cells wide and the foot is within a front width of the axis.
* **Edge half-front.**  A collapsed profile whose residual branch can carry S (Q_min < S, e.g. H = 30) keeps
  a boundary layer at the pinned edge: a half-front of width ~sqrt(nu4/(3 c_e g^2)) sitting on the residual
  branch at g_+(S) = 6.78 with Q = S.  It is steady (dW/dt ~ 1e-7) and its stored pressure scales as
  sqrt(nu4): W_end - W_L = 10%, 15%, 25% of W_L for nu4 x 0.1, 0.3, 1.  The paper's NOTES record the same
  artefact ("one-cell edge step, W 5-10% above the L branch").  It cannot exist when Q_min > S (H = 1e3: the
  collapse goes all the way to L).  A flux (Robin) edge condition would remove it; not done here.
* With H = 1e3 the regularising nu4 = 0.14 is large enough to distort the L-branch gradient near the axis,
  where D_L = Q'(g_-) is small: g is 30%, 10%, 4%, 1% below g_-(Q_ss) at x = 0.1, 0.2, 0.3, 0.4 (6%, 1% at
  H = 30; < 1% at H = 3).  W is unaffected (0.05%) because the inner profile is flat.  So the H = 1e3
  dynamics are clean for W, Q_edge, the front and the collapse time, not for the inner L-branch gradient.
* The effective drive of `reduced.py` is 0.9887 S (the axis half-cell's source is lost in the ghost cell);
  here the source is normalised on the evolved cells.  This matters only within ~1% of marginal.

## 6. Alpha heating (`gk_law_alpha.npz`, `gk_law_runs.py alpha`)

The paper's closure (`flux_transport_model_clean.alpha_heating`, `bremsstrahlung`): Bosch-Hale D-T reactivity
with n = 1e20 m^-3, T = T_ref p, T_ref = 3 keV (the pinned edge p = 1 is 3 keV, as in the paper's ignition
scan), C_brem = 0.01, C_alpha scanned.  P_alpha is quoted integrated over x in units of S_ext.
The self-consistent burning state on the H branch solves F(S_tot) = S_tot - S_ext - int P_alpha(p_H(S_tot)) = 0,
with p_H the analytic barrier profile of drive S_tot (`alpha_fixed_point`; roots stored as `*__fixed_points`).

**Access (H/S_ext = 1e3, S_ext < Q_M = 84.7, supercritical start with a front at x0 = 0.3).**

Two estimates of the burning state are stored: the "Gaussian-shape" root (H-branch profile of the external
source shape scaled to S_tot, `fixed_points`) and the self-consistent state (iterate p -> H-branch profile
of the drive S_ext + P_alpha(p), `fixed_point_sc`; the drive of a burning state is concentrated where p is
large, so its x_M and pedestal differ from the Gaussian-shape state).  Only the self-consistent one is a
real fixed point; the Gaussian-shape root overestimates existence near threshold.

| C_alpha | P_alpha(IC) | Gaussian-shape root | self-consistent state | outcome (x0 = 0.3 start / start from the state) |
|---|---|---|---|---|
| 1e-5 | 4.2 | none | none | collapse, t ~ 0.05 (as without alpha) |
| 1e-4 | 42 | none | none | collapse |
| 1.5e-4 | 63 | none | none | collapse; edge half-front survives because S_tot = 1.6 > Q_min = 1.5 |
| 2e-4 | 84 (= Q_M - S_ext) | none | none (iteration falls below Q_M) | collapse |
| 3e-4 | 127 | 172 | none (decays 172 -> 63 -> L) | transient: S_tot and Q_edge balance at 133 for t < 0.02, then the burn dies; the start from the Gaussian-shape state also collapses |
| 5e-4 | 210 | 348 | 331 | barrier held from both starts, S_tot = 330, W = 7.6, front at x_f = 0.15 |
| 1e-3 | 421 | 832 | 812 | S_tot = 810, W = 10.0, x_f = 0.07 |
| 1e-2 | 4200 | 1.03e4 | 1.045e4 | S_tot = 1.04e4, W = 21, x_f = 0.02 |

Alpha heating cannot hold the nonlinear barrier unless the alpha power of the initial barrier itself exceeds
Q_M - S_ext by a factor ~2 (P_alpha(IC) >= 210 against Q_M - S_ext = 84; 1-3 GW/m^2 in STEP terms, i.e.
10^2-10^3 times the alpha power available).  The threshold in C_alpha (between 3e-4 and 5e-4) is where the
self-consistent burning state appears, and once it exists it is reached from the x0 = 0.3 front: existence
and access coincide here because the collapse is so fast that no transient barrier survives long enough to
matter.  The self-consistent state predicts the numerical burning state to 0.2-0.4% (331 vs 330, 812 vs 810).  In the L-mode state (subcritical start) the burn is fixed by the pinned edge temperature:
C_alpha = 1e-2 gives P_alpha = 47 S_ext with g_max = 0.33 g_pk and the profile still on the L branch
(it would need S_tot > Q_max = H = 1000 S_ext to force the transition).

**Sustainment / saturation (H/S_ext = 3, S_ext > Q_M; H/S_ext = 30, S_ext < Q_M).**  Start from the barrier
of the external drive (H = 3) or from the burning fixed point (H = 30), and from the x0 = 0.3 front.

| H | C_alpha | P_alpha(IC) | predicted S_tot* | numerical S_tot (steady, both starts) | W | g_edge | T_core |
|---|---|---|---|---|---|---|---|
| 3 | 3e-5 | 21 | 31.8 | 32.3 | 21.7 | 46 | 111 keV |
| 3 | 1e-4 | 69 | 101 | 105 | 31.0 | 69 | 158 |
| 3 | 3e-4 | 207 | 283 | 295 | 42.6 | 97 | 218 |
| 3 | 1e-3 | 691 | 851 | 894 | 60.0 | 141 | 307 |
| 3 | 3e-3 | 2070 | 2281 | 2407 | 81.8 | 196 | 418 |
| 30 | 3e-6 | 1.3 (< Q_M - S_ext = 1.5) | none | burn dies: collapse to L (+ edge layer), S_tot = 1.01 | 1.43 | | 4.7 |
| 30 | 1e-5 | 6.4 / 4.2 | 7.3 | 6.9 (front at x_f = 0.12) | 6.9 | 12.9 | 32 |
| 30 | 2e-5 | 15 / 8.4 | 16.3 | 15.8 | 8.8 | 17 | 44 |
| 30 | 3e-5 | 25 / 13 | 26.1 | 25.5 | 10.2 | 20 | 51 |
| 30 | 1e-4 | 98 / 42 | 99.1 | 99.0 | 15.2 | 31 | 77 |
| 30 | 1e-3 | 1000 / 420 | 1005 | 1038 | 30.9 | 69 | 158 |

* **The burn saturates whenever a barrier exists**, at the H-branch fixed point: the Gaussian-shape root
  predicts S_tot to 1.5-5%, the self-consistent state to 0.03-0.8% (H = 3: 32.27, 104.5, 295.4, 894, 2407
  against 32.26, 104.5, 295.4, 894, 2407; H = 30, C_alpha = 1e-5: 6.96 against 6.91), from either start,
  and every state found is stable (dW/dt/W ~ 1e-10 at the end).  The
  reason is structural: on the floor Q ~ g^3 the H-branch edge flux grows as W^3 at fixed foot while
  P_alpha grows as p^{n_eff} with n_eff ~ 2 in the burning window (Bosch-Hale gives n_eff ~ 3.5-4 below
  ~7 keV, ~2 at 10-20 keV, < 1 above 50 keV), so the thermal criterion dS/dW < dQ_edge/dW of `model.tex`
  (eq. thermal_criterion) reads n_eff P_alpha < 3 S_tot and is satisfied wherever n_eff <= 3, i.e. for any
  burning volume above a few keV; the reactivity roll-over is not needed.  Every state found is stable, which
  confirms it.  No cliff is needed either: the floor is the "safety valve".
* Caveat: `sigma_v_DT` is the Bosch-Hale fit, valid 0.2-100 keV and clamped only below; the fixed points
  with T_core > 100 keV (H = 3 at C_alpha >= 3e-5, H = 30 at C_alpha >= 3e-4) extrapolate it.  The
  structural claim does not depend on this; the quoted S_tot*, W and T_core of those rows do.
* **Hysteresis lower branch.**  At H = 30 the external drive alone (S_ext = 1 < Q_M = 2.54) cannot hold a
  barrier, but a burning barrier with S_tot = 6.9 >= Q_M is self-sustained at C_alpha = 1e-5 and is reached
  from the x0 = 0.3 front (P_alpha(IC) = 4.2 > Q_M - S_ext); at C_alpha = 3e-6 (P_alpha(IC) = 1.3) no
  fixed point exists and the burn dies.  The ignition threshold in C_alpha is where the burning state's own
  drive crosses Q_M, exactly as in Section 5 of the paper.
* **What saturation means physically.**  The fixed points sit at g_edge = 13-200 g_pk and T_core = 30-400 keV:
  the burn is bounded, but by a residual branch the plasma cannot physically hold (a/L_Te of 15-250) and at
  temperatures beyond any equilibrium.  "Saturates" is a statement about the transport closure, not about
  STEP; with a physical ceiling on the gradient (ideal MHD, density limit) the end state would be a
  disruption rather than a hot fixed point.  S_tot* scales roughly linearly with C_alpha over the range.

## 7. Files and keys

All `.npz` files load with `dict(np.load(path, allow_pickle=True))`.  Units: section 1.

* `gk_law_bump_fit.npz`: `bp` (|beta'|, 20 points), `Q_gB`, `Qe_gB`, `Qi_gB`, `flag`, `saturated`, `live`;
  `rr_gB`, `neo_gB` (floors at the scan's gradients), `bp_nominal`; `g_grid` (g/g_pk, 600 points) and
  `{plgauss,lognormal,general}_B` (fitted B on it), `{kind}_params` (JSON: g_pk, Q_pk_gB, sL, sR, mL, mR,
  rms_ln; in the exponent convention B = exp(-(|ln g|/s)^m), so for the lognormal s = sqrt2 sigma),
  `{kind}_resid` (ln fit - ln data on the live points), `step_ratios` (JSON: g_nom, rr_over_peak,
  neo_over_peak, ce_over_H, ci_over_H).
* `gk_law_maxwell_table.npz`: 1-D curves on `ce_over_H` (161 points, 1e-8-1): `curve_{kind}_{ci}_{Q_M, g_a,
  g_b, Q_M_est, Q_min, A_lobe}` with kind in plgauss/lognormal/general and ci in step (c_i/H = 8.9e-6), zero,
  1e-4, 1e-2 (all for H = 1, i.e. Q_M/H etc.).  2-D tables on `H_over_S` (51) x `ce_over_S` (81), c_i/S =
  `ci_over_S` = 1e-3: `table_{Q_M, g_a, g_b, Q_min, Q_max, g_minus_S, g_plus_S, x_M, W_L, W_H, regime}`
  (regime codes in section 3; x_M, W_L, W_H for the Gaussian source with S = 1).  `maxwell_boundary_H`,
  `maxwell_boundary_ce` (the curve Q_M = S), `step_ce_over_H`, `step_ci_over_H`, `g_cap`, `A_bump`, `f_step`
  (= Q_M/H on the STEP line, exact 0.0847; `f_step_note`), `ce_mono_over_H`, `shape`, `shape_params`.
* `gk_law_runs.npz`: `x`, `case_names`; per case `<name>__<key>` with keys H, c_e, c_i, S_tot, ic, x0,
  T_end, nu4, Q_M, g_a, g_b, Q_min, Q_max, x_M, W_L, W_H, g_L, g_H, p_L, p_H (analytic branches on x), t,
  W, Q_edge, S_tot_t, P_alpha, g_max, g_edge, x_f (front mid-level crossing), p (snapshots t x x), p_end,
  g_end (face gradients), dWdt_rel, flux_balance, t_front_095, t_W_collapse, t_front_gone, g_dev_max, and
  for supercritical starts law_t, law_x_f (Stefan law), law_t_wall, t_est_energy.
* `gk_law_alpha.npz`: same keys plus `fixed_points`, `fp_scan_S`, `fp_scan_F` (Gaussian-shape F(S_tot) on a
  log grid), `fixed_point_sc`, `fixed_point_sc_converged`, `fixed_point_sc_hist`, `p_fixed_point_sc`;
  `T_REF_KEV`.  Case names `alpha_access_H1e3_Ca<C>[_fp]`, `alpha_sustain_H<H>_Ca<C>_<fp|super>`,
  `alpha_Lmode_H1e3_Ca<C>`.
* `gk_law_nu4.npz`: `keys`, `x`; per key `<case>__fac<f>_ns<n>__{H, S_tot, nu4, n_smooth, t, W, x_f, g_edge,
  W_L, W_H, x_M, law_t_wall, law_x_f_end, t_front_095, front_width_cells}`.
* `gk_law_figs/*.png`: quick looks (bump_fit, maxwell_and_map, regime_map_runs, alpha_summary, one panel
  per run).  Nothing was written into the paper's `plots/` or `sections/`.

## 8. Conclusions the paper can quote

1. With the nonlinear gyrokinetic bump the reduced model has no barrier state at any realistic drive: the
   Maxwell flux of a bump of height H on the Rechester-Rosenbluth floor is Q_M = (4A/3)^{3/4} c_e^{1/4}
   = 0.085 H for the STEP coefficients (1.0 GW/m^2 against the 12 GW/m^2 peak of Q_e + Q_i), and with
   H/S ~ 4e2-6e3 this is 30-500 times the drive.
   A supercritical profile sheds its pedestal at Q_M in a time (W_0 - W_L)/(Q_M - S), milliseconds, and the
   plasma settles on the rising branch at the gradient where the bump's foot carries the drive,
   g = (S/H)^{0.37} ~ 0.04-0.1 of the peak gradient: an L-mode-like clamp 10-30 times below the nominal
   |beta'| = 0.5 (and below the lowest scan point, so the number rests on the power law Q ~ |beta'|^{2.7}).
2. A barrier requires H/S < 12 on the STEP line, i.e. a bump no more than ten times the drive: the quasilinear
   regime.  It then sits on the residual branch at g_+(S) = (S/c_e)^{1/3} >= 9 g_pk (a/L_Te >~ 12), which is
   the paper's runaway with the cliff replaced by a g^3 floor; only a floor 100 times stiffer (c_e/S ~ 0.03)
   gives a pedestal with a core (g ~ 3 g_pk).  The four end states are selected by two numbers: S/H (below 1
   the L branch does not exist, forced transition; above 1 the L branch is linearly stable from a subcritical
   start) and S/Q_M (above 1 a supercritical start keeps its barrier, below 1 it collapses), with Q_M/S =
   0.805 (H/S)^{3/4} (c_e/S)^{1/4}; the boundary Q_M = S is c_e/S = 2 (H/S)^{-3} in the (H/S, c_e/S) plane.
3. Where a barrier exists the burn saturates at a thermal fixed point on the H branch, because the floor
   Q ~ g^3 outruns P_alpha ~ p^{n_eff} with n_eff ~ 2 in the burning window (n_eff P_alpha < 3 S_tot): no
   cliff is needed and there is no runaway in W.  The fixed point is predicted by the analytic branch to a few per cent, is reached with hysteresis
   (a burning barrier survives below the external-drive threshold once the alpha power of the barrier itself
   exceeds Q_M - S_ext), but it lies at gradients and temperatures the plasma cannot hold.
4. Where no barrier exists alpha heating does not create one: the nonlinear barrier would need an alpha power
   of order Q_M ~ 1 GW/m^2 through the surface, and the L-mode burn is fixed by the edge temperature.  The
   Cahn-Hilliard theory (spinodal, Maxwell rule, Stefan law, bistability) needs no change; the front law
   predicts the collapse time to 2% when Q_M >> S and to ~30% near marginal, where the discrete front also
   pins unless the steep side of the front is resolved by >= 2 cells.

Shape sensitivity: the asymmetric log-Gaussian fit (rms factor 2.4) gives Q_M 24% higher (0.106 H); the
offset path of `make_gk_maxwell.py` gives 1.48 GW/m^2 instead of 1.03.  Neither changes any conclusion.
