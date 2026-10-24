"""Pinning window against grid spacing: the lowest source S_lo(N) at which a barrier, once built, survives.

Resolved grids should give S_lo -> Q_M (Maxwell selection); coarse grids S_lo -> Q_min (the upper branch's minimum,
the fold), the 'grid pinning' seen in T3D scenarios H/I (NOTES_grid_pinning.md).  Same protocol as exp_grid_pinning.py:
build at S_hi = Q_M + 3.6 (T = 10), hold S (T = 20).  'Kept' = the end state retains more than half of the W excess
over the L state that a barrier at S_ref = Q_M + 0.65 has on the same grid: (W - W_L) / (W_ref - W_L) > 0.5, with
W_L the end state at S = 0.7 (below the fold).  Bisection in S on [Q_min, Q_M + 0.65] to 0.05.
    python3 exp_pinning_window.py N1 N2 ...  (default 6 10 20 30 50 70 100 200)     -> pinning_window.out
    NU4=0 python3 exp_pinning_window.py 10 50 200                                 -> pinning_window_nu4_0.out
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import reduced as R
from maxwell import maxwell

import os
NU4 = float(os.environ.get("NU4", R.PAPER["nu4"]))     # NU4=0: no regularisation but the grid (as in T3D)
P = dict(R.PAPER, n_smooth=0, nu4_conservative=True, nu4=NU4)
QM = maxwell(dict(chi0=P["chi0"], chi_RR=P["chi_RR"], g_c=P["g_c"], g_MHD=P["g_MHD"], chi_MHD=P["chi_MHD"]))["Q_M"]
Q_MIN, S_REF, TOL = 1.0, QM + 0.65, 0.05
W_FRONT = 0.01          # front width (nu4/|Q'|)^(1/2), tab:theory
Ns = [int(a) for a in sys.argv[1:]] or [6, 10, 20, 30, 50, 70, 100, 200]

print(f"Q_M = {QM:.3f}, Q_min = {Q_MIN}, S_ref = {S_REF:.2f}, nu4 = {NU4} (front width w = {W_FRONT} at nu4 = 5e-4)", flush=True)
print(" N    dx     dx/w   S_lo    S_lo/Q_M   W_L    W_ref", flush=True)
for N in Ns:
    R.grid = lambda N=N: np.linspace(0.0, R.L, N)
    x = R.grid(); dx = x[1] - x[0]; S0 = np.trapezoid(R.base_source(x), x)
    p_bar = R.run(None, params=P, g_tgt=1.8, T_end=10.0, n_out=3, physics=False, source_scale=(QM + 3.6) / S0)["p"][-1]
    W = lambda S: np.trapezoid(R.run(None, params=P, T_end=20.0, n_out=3, physics=False,
                                     source_scale=S / S0, p0=p_bar)["p"][-1], x)
    W_L, W_ref = W(0.7), W(S_REF)
    kept = lambda S: (W(S) - W_L) / (W_ref - W_L) > 0.5
    lo, hi = Q_MIN, S_REF
    if kept(lo + 1e-3):
        S_lo = lo
    else:
        while hi - lo > TOL:
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if kept(mid) else (mid, hi)
        S_lo = hi
    print(f"{N:4d}  {dx:.4f}  {dx / W_FRONT:6.1f}  {S_lo:5.2f}   {S_lo / QM:5.2f}    {W_L:.3f}  {W_ref:.3f}", flush=True)
