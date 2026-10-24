# Grid pinning (24 Oct 2026): why a coarse-grid barrier survives below the Maxwell flux

Script: exp_grid_pinning.py. Outputs: grid_pinning_coarse.out (N = 6, 10, 20), grid_pinning_N50.out, and grid_pinning_resolved.out
(N = 50-400, Slurm 39097492, pending).
Protocol: build a barrier at S = Q_M + 3.6, hold S_test for T = 20, read the edge-face gradient and W. Settings: n_smooth = 0,
conservative nu4 = 5e-4, physics off. Model: Q_M = 5.85, upper-branch minimum Q_min = 1.0 (at g = g_c = 4).

| N (dx) | S = 0.7 | S = 2.0 | S = 3.5 | S = 5.0 | S = 6.5 |
|---|---|---|---|---|---|
| 6 (0.20) | lost, W 1.02 | edge step, W 1.76 | 1.86 | 1.96 | 2.06 |
| 10 (0.11) | lost, 1.03 | 1.82 | 1.96 | 2.10 | 2.25 |
| 20 (0.053) | lost, 1.03 | 1.69 | 2.00 | 2.16 | 2.34 |
| 50 (0.020) | lost, 1.03 | 1.19 | 1.29 | 3.76 (PINNED: no retreat in T = 20) | 4.09 |
| 100 (0.010) | lost, 1.03 | 1.164 | 1.261 | 1.390 | 4.09 |
| 200 (0.005) | lost, 1.03 | 1.154 | 1.255 | 1.381 | 4.05 |

**Reading.**
- For Q_min < S < Q_M the front retreats until it reaches the boundary cell, and then it stops.
- A one-cell edge step survives, with its gradient on the upper branch, carrying flux S. It survives because a single cell has no room for a front. The step goes only when S < Q_min, the fold.
- Its width is dx. On a resolved grid it is a negligible remnant: W is 1.19 at N = 50, against an L-branch W of about 1.03. The paper already notes this "one-cell edge step".
- On a coarse grid the remnant IS the barrier: W is 1.76 at N = 6, against 1.96 for a "real" barrier at S = 5.
- **Consequence for T3D H/I** (5 tubes, spacing 0.17): a barrier one tube wide persists whenever the edge tube's heating stays above its upper-branch minimum. That is I (0.70-0.81 gB against Q_min 0.49). H drops below Q_min and loses it. Q_M is the correct threshold only once the front is resolved.
- Pending: the resolved runs (N up to 400) should show the step's W excess shrinking about in proportion to dx. Also pending: the evolved-state scan of I's edge tube (does the gB law move?) and the finer-grid T3D run of I (pre-registered).

**Resolved grids (24 Oct 11:40, grid_pinning_N100_200.out).**
- The remnant's W excess at S = 2 goes 0.16 (N = 50), 0.13 (100), 0.12 (200). It converges, not to zero: with g ≈ 4.5 it is a
  step about 0.03 wide, the hyperviscous front width (w ≈ 0.01-0.03). So the remnant's width is max(dx, w), the regularisation
  scale, as sec:solver says of barrier widths in general. On a resolved grid it carries ~4 % of a real barrier's W excess,
  so the barrier is lost; on T3D's 0.17 grid the remnant is the whole edge tube.
- **Lattice pinning near Q_M.** At S = 5 (0.85 Q_M), N = 50 does not retreat at all (W 3.76), while N = 100 and 200 retreat
  fully (W 1.39). This is front propagation failure on a discrete lattice: a front is pinned for driving inside a window
  around the Maxwell point, and the window widens as dx/w grows (Keener 1987 SIAM J. Appl. Math. 47, 556, TO VERIFY;
  Erneux & Nicolis 1993, TO VERIFY). On the coarsest grids the window reaches down to the fold, which is the H/I case.
- **Threshold rule:**
  - dx << w: Q_M decides.
  - dx >> w: anything between the fold Q_min and Q_max is pinned, and a barrier, once made, survives down to Q_min.
  - In between: the window is between those two limits.
- **Next:** measure the pinning window [S_lo(dx), Q_M] directly by bisection in S per N. That would give the paper a figure:
  the window width against dx/w.

**Pinning window, measured (24 Oct 12:47, exp_pinning_window.py; pinning_window.out, pinning_window_nu4_0.out).**
Lowest S at which a built barrier keeps > 1/2 of its W excess:

| dx/w | 20 | 11 | 5.3 | 3.4 | 2.0 | 1.4 | 1.0 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| S_lo/Q_M (nu4 = 5e-4) | 0.32 | 0.30 | 0.34 | 0.43 | 0.66 | 0.82 | 0.94 | 1.00 |

- With nu4 = 0: S_lo/Q_M = 0.23 (N = 10), 0.19 (50) and 0.20 (200). That is the fold (Q_min/Q_M = 0.17) at EVERY resolution. The front is one cell wide however
  fine the grid, so refining never brings Q_M back.
- Q_M needs dx <= w/2 with an explicit front scale w. A solver with no front scale (T3D) is pinned down to the barrier tube's fold.
- Paper: sec:solver paragraph and fig_pinning (a586d4b).
- Implication for the finer-grid T3D run of SPP3 without radiation (10 tubes, pre-registered 11:03): this model predicts the barrier holds, i.e.
  reading A or C, not B. That conclusion came AFTER the registration and is recorded as an addendum, not an edit.
