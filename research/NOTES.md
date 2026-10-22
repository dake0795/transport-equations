# How the reduced model saturates: Cahn–Hilliard structure and the Maxwell flux

Research notes, 22 Oct 2026.  Scripts in this directory; all runs use the
paper's discretisation (dx = 0.005, nu4 = 5e-4, n_smooth = 3 unless stated)
but a stiff BDF integrator (`reduced.py`) so that runs reach t ~ 10.  The
paper's explicit runs stop at t = 0.2, which is far too short to see the
steady states (the chi_RR time is L^2/chi_RR = 20, and front motion near
threshold takes t ~ 3).

## 1. The equation is Cahn–Hilliard

Differentiating the pressure equation in x,

    g_t = d_xx [ Q(g) - nu4 g_xx ] - S_x ,        g = -p_x.

This is the Cahn–Hilliard equation with "chemical potential"
mu = Q(g) - nu4 g_xx and bulk potential Phi(g) = int_0^g Q.  Without a
source it is a gradient flow of F = int [Phi(g) + nu4 g_x^2 / 2] dx.
Consequences (all verified numerically below):

* **Spinodal interval** = falling branch, Q'(g) < 0, i.e. g_* < g < g_c
  (2 < g < 4 for the paper's parameters).  A uniform gradient there
  decomposes into a staircase.
* **Coexisting phases** g_a (rising branch) and g_b (cliff branch) are fixed
  by the equal-area (Maxwell) rule on Q(g):

      Q(g_a) = Q(g_b) = Q_M,      int_{g_a}^{g_b} [Q(g) - Q_M] dg = 0.

  For the paper's parameters: **Q_M = 5.85, g_a = 0.71, g_b = 6.37**
  (`maxwell.py`).
* A uniform state with g_a < g < g_* or g_c < g < g_b is metastable (binodal
  but not spinodal): it does not decompose spontaneously.

## 2. Q_min in the paper is wrong for the paper's own parameters

Eq. (Qmin) evaluates Q at g_MHD and gives 3.78.  But with chi_MHD = 2 the
cliff slope is smaller than the falling-branch slope at g_MHD
(Q_NL' = chi0 (1 - 2 g_MHD/g_c) = -8), so Q keeps falling to g_c, where
**Q_min = Q(g_c) = (chi_RR + chi_MHD) g_c - chi_MHD g_MHD = 1.0**.  The
paper's formula holds only if chi_MHD > chi0 (2 g_MHD/g_c - 1) = 8.
Either way Q_min is *not* the barrier threshold (next item).

## 3. The barrier threshold is Q_M, not Q_min; the foot is at Q_ss(x_M) = Q_M

With a source, steady state requires Q(g) - nu4 g_xx = Q_ss(x) =
int_0^x S.  A front between the two phases is stationary only where the
local flux equals the Maxwell flux.  Hence:

* **Existence**: a barrier steady state exists iff S_tot = Q_ss(L) >= Q_M.
* **Location**: the foot sits at x_M with Q_ss(x_M) = Q_M; the pedestal is
  [x_M, L] with g = g_cliff(Q_ss(x)) on the cliff branch, and the core has
  g = g_-(Q_ss(x)).  The pedestal *spatial width* L - x_M is set by the
  source profile and Q_M, not by the cliff.  The cliff fixes only the
  gradient inside the pedestal.
* **Front motion**: if Q_ss(x_f) > Q_M the front advances inward (barrier
  widens from the edge); if S_tot < Q_M it retreats to the wall and the
  barrier collapses.

`scan_pb.py` (physics sources off, supercritical start as in the paper,
pb = S_tot / Q_edge(0) with Q_edge(0) = 3.78, predicted threshold
pb_c = Q_M / 3.78 = 1.55):

| pb | S_tot | t = 10 outcome | x_foot (num) | x_M (pred) |
|---|---|---|---|---|
| 0.4, 1.0, 1.4, 1.5 | <= 5.67 | collapsed | – | – |
| 1.6 | 6.05 | barrier, still creeping | 0.50 | 0.38 |
| 1.7 | 6.43 | steady barrier | 0.309 | 0.300 |
| 2.0 | 7.56 | steady barrier | 0.214 | 0.214 |
| 2.5 | 9.45 | steady barrier | 0.153 | 0.155 |

**Every run in the paper (pb <= 1.2, i.e. S_tot <= 4.5 < Q_M) collapses
at long times, including those labelled "steady barrier".**  The paper's
own t = 0.05 collapse data already show the Maxwell state: g_edge = 6.2–6.3
and Q_edge = 5.8–5.97 (g_b = 6.37, Q_M = 5.85).  The gradient is not
"settling near g_MHD"; it sits at the Maxwell value on the cliff branch,
and W decays because Q_edge = Q_M > S_tot, not "at a rate set by chi_RR".

## 4. Bistability and hysteresis (`exp_hysteresis.py`, `exp_bifurcation.py`)

| S_tot | subcritical start | supercritical start |
|---|---|---|
| < Q_M = 5.85 | L-mode | L-mode (barrier collapses) |
| Q_M < S_tot < Q_max = 10.1 | L-mode, g = g_-(Q_ss) | barrier with foot at x_M |
| > Q_max | forced transition to the same barrier state | barrier |

The L branch exists while S_tot < Q_max (the peak of Q) and is linearly
stable, so a subcritical start never forms a barrier below Q_max: the
"power threshold" for the forward transition is Q_max, the back-transition
is at Q_M, and the hysteresis width is Q_max - Q_M.  The analytic steady
states (L: g_-(Q_ss(x)); H: Maxwell front at x_M) reproduce the numerical W
to ~1% on both branches (`bifurcation.npy`, `figs/bifurcation.png`).

## 5. Regularisation (`exp_nu4.py`)

Front width (10–90 %) = 0.070, 0.040, 0.020 for nu4 = 2e-3, 5e-4, 1.25e-4:
exactly sqrt(nu4).  Foot position (0.204–0.234) and W (4.39) are
independent of nu4 and of the binomial smoothing (n_smooth = 0 or 3).  So
the barrier *thickness* in the model is a regularisation length
~ sqrt(nu4 / |Q'|); only its position and height are physical.

Boundary artefact: the paper's hyperviscosity stencil is not
flux-conservative at the Dirichlet edge, so "collapsed" states keep a
one-cell edge layer with g ~ 5 carrying Q(g_edge) = 2.3 while S_tot = 5.7;
the missing flux leaks through the hyperviscous boundary term.  With a
conservative form (`nu4_conservative=True`) the edge layer is a half-front
of width ~ sqrt(nu4) sitting on the cliff branch with Q(g_edge) = S_tot.
Either way the collapsed bulk is L-mode with g = g_-(Q_ss).

## 6. Source-free staircase (`exp_spinodal.py`, periodic domain)

g0 = 3 (spinodal) breaks into 12 steps by t = 0.01 with plateaus at
0.71 / 6.29 (predicted 0.706 / 6.37) and high-phase fraction 0.412 (lever
rule 0.405).  Coarsening stops at 9 steps: 1D Cahn–Hilliard coarsening is
logarithmically slow.  g0 = 1.5 and g0 = 5 (metastable) stay uniform.
With a source the staircase is not coarsened but *swept*: steps at small x
sit where Q_ss < Q_M and dissolve, the outermost one advances to x_M.
This is the "steps merge into one barrier at the edge" of the ASIPP talk.

## 7. Self-heating (`exp_physics.py`, conservative nu4)

* With C_alpha = 4e-6 the alpha power of the initial profile (2.66) already
  exceeds the pb = 0.4 target, so the paper's collapse run has **zero
  external heating**: it is a burn dying away (S_tot 2.6 -> 0.64, W -> 1.02).
  The pb = 1.0 run also collapses (W -> 1.14).  pb = 1.6 and 2.0 give
  steady barriers with S_tot = 6.16 and 9.03.
* The "ignition runaway" (C_alpha x 50) has S_ext = 0 and S_tot(0) = 133,
  thirteen times Q_max, so it is a forced transition, not a marginal case.
  Integrated to t = 10 it is **not a runaway**: W saturates at 25.9 by
  t ~ 2.5 with Q_edge = 157 on the cliff branch (g_edge = 80) and the
  barrier filling the domain (x_foot = 0.04).  The alpha power saturates
  because the Bosch–Hale reactivity rolls over above ~60 keV (T = 10 p keV).
  C_alpha x 10: W -> 9.86, S_tot -> 43.
* The paper's bound int P_alpha <~ chi_MHD g_MHD is not a limit of the
  model: the cliff branch carries any flux at a steeper gradient.  What the
  paper's t = 0.2 run shows is the thermal excursion from the low to the
  high (reactivity-limited) fixed point, the standard burn-instability
  picture, not an unbounded runaway.

## 8. Front speed (`scan_pb_nophys_ns3.npz`)

A lift-the-core estimate, v = (Q_ss(x_f) - Q_M) / [x_f (g_b - g_a)],
reaches the right endpoint and the right timescale within a factor of 2
(too fast), and misses an initial outward retreat (0.55 -> 0.65) while the
falling-branch part of the initial profile reorganises.

## What this means for the paper

1. Replace Q_min by the Maxwell flux Q_M as the collapse threshold, with the
   equal-area rule as a one-line result; fix the Q_min formula.
2. Barrier foot at Q_ss(x_M) = Q_M; pedestal width set by the source
   profile, gradient by the cliff branch; thickness is numerical.
3. Burn vs collapse: bistable for Q_M < S_tot < Q_max, forward threshold
   Q_max, back-transition Q_M, hysteresis.  Existence-vs-access is this
   bistability.
4. The barrier/collapse/runaway figures need long runs; the current
   "steady barrier" and "ignition runaway" cases are transients.
5. Connects to Lebedev & Diamond (1997) and Malkov & Diamond (2008) on
   Maxwell-rule selection of transport-barrier fronts, and to Radko (2003)
   / Balmforth et al. on staircases.

## 9. What the initial condition predicts (added later on 22 Oct)

Two tests of the front rule as a predictor from the IC (physics off,
conservative nu4, S_tot = 8 in the bistable window, x_M = 0.196):

* **Nucleation radius.**  A subcritical base profile plus one local
  supercritical patch.  Patches holding ~0.5 units of pressure drop
  (high-phase slab ~0.09 after local phase separation, i.e. wider than the
  front, ~0.04) nucleate the full barrier (W = 4.39, foot at 0.193) iff they
  sit at x > x_M; at x = 0.12 and 0.17 they dissolve.  Patches holding
  ~0.1 units (slab ~0.02, thinner than the front) never nucleate anywhere:
  a critical-nucleus condition.  Large patches that extend past x_M
  nucleate even when centred inside it.
* **Transient barrier lifetime** when S < Q_M, from the supercritical IC:
  measured t_gone = 0.18, 0.33, 1.05, 2.65 for S = 1.51, 3.78, 5.29, 5.67;
  the estimate (W_0 - W_L)/(Q_M - S) gives 0.25, 0.45, 1.51, 4.50.  Right
  divergence ~1/(Q_M - S), overestimates by 1.4-1.7 because part of the
  stored energy is released after the barrier is gone.

So from an IC one can predict: (i) the end state, given S and whether a
sufficiently large supercritical patch lies beyond x_M; (ii) the final
profile, independent of the IC's details; (iii) the order of magnitude of
the transient barrier's lifetime when it must collapse; (iv) with
self-heating, whether the transient lives long enough for P_alpha to lift
S above Q_M (the ignition access condition).
