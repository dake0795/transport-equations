# Front lag (exp_front_lag.py; output front_lag.out), 25 Oct 2026

**Why it never finished before.** Two attempts (23 Oct on cpu-r-29, 9 h then OOM; 25 Oct job 39093208, cancelled after 1 h) hung in
law (b): driven by a flux above Q_M, dx_f/dt = -(Q - Q_M)/(x_f [g]) carries the predicted front to x_f = 0 in finite time, and RK45 at
rtol 1e-8 chatters about the singularity indefinitely. Fix: terminal events at x_f = 0.02 and 0.98. The script now runs in seconds.

**Second fix.** The "measured front flux" used Q(g) alone. Inside the front Q(g) is near the bump top (about 10), and the conservative
hyperviscous flux nu4 p_xxx carries the rest. The total face flux is now used, read 3 cells inside the front.

**Result** (drive 9.45 -> 7.56 and back; Q_M = 5.85):

| transition | measured half-time | law (a): Q_ss, Maxwell jump | law (b): measured flux, Maxwell jump | law (c): measured flux and jump |
|---|---|---|---|---|
| step down (front out, 0.153 -> 0.209) | 0.105 | 0.040 | 0.34 | 0.33 |
| step up (front in, 0.209 -> 0.153) | 0.060 | 0.030 | 0.42 | 0.41 |

- **Law (a) reproduces the paper's numbers** (sec:front: 0.11 vs 0.04, 0.06 vs 0.03): the law is 2-2.6x too fast.
- **Laws (b) and (c) cannot decompose the remainder.** The flux 3 cells inside the front is essentially the local core flux, 0.89-0.93 Q_M throughout,
  so these laws are 3-7x too slow and, after the step up, predict the wrong direction.
- **What decomposing the lag would need:** the jump condition V[g] = [p_t], with the measured core and pedestal heating rates on each side of the front,
  not a flux read at a fixed offset. That has not been done.
- **Paper:** unchanged. Its text already says what this supports: the law gives the direction, the end point and the order of magnitude of the time,
  the lag is partly the pedestal's relaxation, and the remainder is "presumably the readjustment inside the front".

# The lag decomposed with the jump condition (exp_front_lag_jump.py; output front_lag_jump.out, front_lag_jump.npz), 25 Oct 2026 late

Checked adversarially: `check_front_lag_{measured,sharp,coresource,pedsource,nsmooth}.py`, output `check_front_lag.out`. A rerun
of exp_front_lag_jump.py reproduces front_lag_jump.out line for line. **This supersedes the two last bullets above**: the
decomposition is now done, and no remainder is left for "the readjustment inside the front".

**Result.** The fronts lag the Stefan law because the pedestal's own energy changes, which eq. (lift) leaves out. The Stefan
law is the *frozen*-pedestal limit (p_t = 0 at the pedestal top, tau_p -> infinity), not the quasi-steady one. Neither a
delay in the pedestal's relaxation nor the front interior causes the lag: a pedestal that relaxed instantly (tau_p = 0)
would make the front slower still (1.25 / 1.63 of the measured half-times at chi_MHD = 2).

- **Jump condition** V[g] = p_t^+ - p_t^- holds (median residual 0.1 %; max 4 % at grid-scale stick-slip jumps). It is the
  continuity of p, so it only confirms the diagnostics. Of the law's three assumptions, read at the half-time: the flux
  through the front is Q_M (|mu_G - Q_M| <= 0.07 of the drive Q_ss - Q_M); the core is lifted uniformly (non-uniform
  heating 3 % of the core's at chi_MHD = 2, 6-9 % at 20); the pedestal is **not** frozen: p_t^+/p_t^- = 0.47-0.50 at
  chi_MHD = 2 (0.28-0.32 at 20).
- **Two pedestal effects.** (i) With its flux held at Q_M at a moving front, a quasi-steady pedestal adds
  Gamma = S(x_f)(L - x_f)/(chi_RR + chi_MHD) to the latent heat g_b - g_a (p_t^+ = -Gamma V). Gamma/[g] = 1.06 / 2.07 at
  chi_MHD = 2 (down / up), 0.22 / 0.44 at 20. (ii) The drive step changes the pedestal's equilibrium height by
  eta0 = +0.25 / -0.14 (chi_MHD = 2), against [g]|dx_M| = 0.34; eta0/([g]|dx_M|) = 0.74 / 0.40 at chi_MHD = 2,
  0.11 / 0.06 at 20. Both reach the front by diffusion in tau_p = (L - x_f)^2/3(chi_RR + chi_MHD) = 0.10-0.12 at
  chi_MHD = 2 and 0.01 at 20. This slab has a fixed flux at the front, so its lowest mode is 4 l^2/pi^2 D (0.12), and
  l^2/3D is the one-pole match to tanh(kl)/kl. The paper's l^2/pi^2 D (0.03; 0.003 at chi_MHD = 20) is the mode with p
  fixed at both ends and does not apply.
- **The Stefan law is recovered when both Gamma/[g] and eta0/([g]|dx_M|) are small**: only for a stiff cliff, and the
  second only matters when the drive change reaches the pedestal.

| chi_MHD, step | measured half-time | Stefan law | ratio | sharp two-phase model / measured | two-region ODE (tau_p = l^2/3D) / measured |
|---|---|---|---|---|---|
| 2, down 9.45 -> 7.56 | 0.0966 | 0.0372 | 2.60 | 1.06 | 1.11 |
| 2, up | 0.0598 | 0.0310 | 1.93 | 1.00 | 0.91 |
| 20, down 11.77 -> 9.88 | 0.0311 | 0.0200 | 1.55 | 1.01 | 0.94 |
| 20, up | 0.0231 | 0.0170 | 1.36 | 1.03 | 0.95 |

- The sharp two-phase model (pedestal and core as diffusion equations, front speed from p continuity, nothing fitted) was
  written twice independently; the check's version gives 0.1028 / 0.0596 / 0.0315 / 0.0239. Over the whole transient it
  tracks the front within 2-5 % of the distance travelled (Stefan law 13-31 %).
- **Two-region ODE** (corrected front law): x_f'[g] = -eta/tau_p - (Q_ss(x_f) - Q_M)/x_f, eta' = Gamma x_f' - eta/tau_p,
  eta(0) the old pedestal's excess height for the new source. Other tau_p: 4l^2/pi^2 D gives 0.99 / 0.82 / 0.93 / 0.94;
  the paper's l^2/pi^2 D 1.32 / 1.44 / 0.95 / 1.01. No single tau_p fits all four; the one-pole pedestal is up to ~13 %
  off tanh(kl)/kl at the relevant frequency.
- **Pedestal share of the lag.** Sharp-model budget (pedestal / core / remainder, fractions of the measured lag): 1.07 / 0.03 /
  -0.11, 0.95 / 0.05 / +0.01, 0.82 / 0.22 / -0.04, 0.82 / 0.31 / -0.13. The measured pedestal share (p_t^+ put back into the
  law) is 1.13 / 1.02 / 0.83 / 0.75. The split of the pedestal share into front-motion and source-change parts depends on
  the order of the steps; the total does not. The measured-term "accounting" is not an independent check (its last row is
  an identity); only its split carries information. At chi_MHD = 20 the lag is only 0.011 / 0.006, so the core's share
  is about 8 % of each half-time (0.0024 / 0.0019), not a large effect.
- **Two new tests the mechanism passed** (check_front_lag_coresource.py, _pedsource.py). Drive removed only near the axis,
  pedestal source unchanged (eta0 = 0; a different case, rest point x_M 0.2375): lag over the law drops to 1.86 / 1.45
  (chi_MHD = 2) and 1.40 / 1.30 (20); the sharp model gives the half-times to 3-6 %. 1.89 of drive added only inside the
  pedestal (Q_ss at the front unchanged, so the Stefan law predicts no motion): the front moves out by 0.0064 and back
  (sharp model 0.0072; chi_MHD = 20: 0.0036 against 0.0044). Qualitative only: about 1.3 cells.
- **Discretisation uncertainty about 7 %.** dx/2 and nu4/4 move the half-times by up to 4 % (chi_MHD = 2) and 6 % (20); the
  3-pass gradient filter off (n_smooth = 0) moves them by +7 % / -1.5 % (chi_MHD = 2) and -6 % / +2 % (20). Without the
  filter the -0.11 remainder at chi_MHD = 2 down disappears (sharp/measured 0.99), while at chi_MHD = 20 down it becomes
  1.08. No front-interior lag beyond this. The remainder at chi_MHD = 2 down is about 6 % of the half-time.
- **The paper's numbers.** "0.11 against 0.04" is 0.10 against 0.04 with dense output and a sub-cell marker (0.097 against
  0.038). "1.5-2" for the stiff cliff came from exp_front_stiff.py at dt = 0.01 with a whole-cell marker (0.040 / 0.020,
  0.030 / 0.020); the ratios are 1.55 and 1.36. Marker choice (x_G, x_mid, x_foot) changes nothing (0.0004).
- **No effect on the half-times from:** the measured jump [g] (5.65-5.68 against 5.660), the measured front flux, the core's
  relaxation time (4e-4 to 1.4e-3).

**Not understood.** The rest-point offset x_G - x_M (-0.003 to +0.002), which at nu4/4 grows at S_lo instead of shrinking
(x_G 0.2135, x_M about 0.2167); the gradient filter is ruled out (n_smooth = 0: -0.0021 -> -0.0015; dx/2, nu4/4:
-0.0030 -> -0.0027). Not checked: whether the T3D runs show the same pedestal-energy lag.

**Paper:** sec:front, the Rankine-Hugoniot parenthesis and the sec:mms front-width paragraph changed accordingly (same day).
