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
| 50 (0.020) | lost, 1.03 | 1.19 | 1.29 | 3.76 (still retreating, S near Q_M) | 4.09 |

**Reading.**
- For Q_min < S < Q_M the front retreats until it reaches the boundary cell, and then it stops.
- A one-cell edge step survives, with its gradient on the upper branch, carrying flux S. It survives because a single cell has no room for a front. The step goes only when S < Q_min, the fold.
- Its width is dx. On a resolved grid it is a negligible remnant: W is 1.19 at N = 50, against an L-branch W of about 1.03. The paper already notes this "one-cell edge step".
- On a coarse grid the remnant IS the barrier: W is 1.76 at N = 6, against 1.96 for a "real" barrier at S = 5.
- **Consequence for T3D H/I** (5 tubes, spacing 0.17): a barrier one tube wide persists whenever the edge tube's heating stays above its upper-branch minimum. That is I (0.70-0.81 gB against Q_min 0.49). H drops below Q_min and loses it. Q_M is the correct threshold only once the front is resolved.
- Pending: the resolved runs (N up to 400) should show the step's W excess shrinking about in proportion to dx. Also pending: the evolved-state scan of I's edge tube (does the gB law move?) and the finer-grid T3D run of I (pre-registered).
