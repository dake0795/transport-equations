"""Grid pinning: does a barrier survive below the Maxwell flux on a coarse grid?

Hypothesis (from T3D scenarios H/I, 24 Oct 2026): on a grid whose spacing exceeds the front width, the barrier is one
cell wide and cannot move as a front, so it persists while the edge flux stays above the upper branch's minimum
(the fold, Q_min) rather than above the Maxwell flux Q_M.  Resolved grids should lose the barrier for S < Q_M.

Protocol, per grid of N cells on [0, 1]: build a barrier at S_hi = Q_M + 3.6 (T = 10), then hold the source at S_test
(edge flux = S_test in steady state) for T = 20 and read the edge face gradient: upper branch if g_edge > g_c = 4.
No smoothing (n_smooth = 0) on every grid; conservative hyperviscosity nu4 = 5e-4 (front width ~0.01).
    python3 exp_grid_pinning.py N1 N2 ...      (default 200 50 20 10 6)
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import reduced as R
from maxwell import maxwell

P = dict(R.PAPER, n_smooth=0, nu4_conservative=True)
QM = maxwell(dict(chi0=P["chi0"], chi_RR=P["chi_RR"], g_c=P["g_c"], g_MHD=P["g_MHD"], chi_MHD=P["chi_MHD"]))["Q_M"]
Q_MIN = 1.0                                   # upper-branch minimum of the model flux (g = g_c = 4)
S_TESTS = [0.7, 2.0, 3.5, 5.0, 6.5]          # < Q_min, three in (Q_min, Q_M), > Q_M
Ns = [int(a) for a in sys.argv[1:]] or [200, 50, 20, 10, 6]

print(f"Q_M = {QM:.3f}, Q_min(upper) = {Q_MIN}; S_tests {S_TESTS}", flush=True)
for N in Ns:
    R.grid = lambda N=N: np.linspace(0.0, R.L, N)
    x = R.grid(); S0 = np.trapezoid(R.base_source(x), x)
    pre = R.run(None, params=P, g_tgt=1.8, T_end=10.0, n_out=3, physics=False, source_scale=(QM + 3.6) / S0)
    p_bar = pre["p"][-1]; dx = x[1] - x[0]
    g0 = -(p_bar[-1] - p_bar[-2]) / dx
    row = []
    for S in S_TESTS:
        r = R.run(None, params=P, T_end=20.0, n_out=3, physics=False, source_scale=S / S0, p0=p_bar)
        p = r["p"][-1]; g_e = -(p[-1] - p[-2]) / dx; W = np.trapezoid(p, x)
        row.append(f"S={S}: g_edge {g_e:5.2f} {'BARRIER' if g_e > P['g_c'] else 'lost   '} W {W:.3f}")
    print(f"N={N:4d} dx={dx:.3f}  (barrier built: g_edge {g0:.2f})\n   " + "\n   ".join(row), flush=True)
