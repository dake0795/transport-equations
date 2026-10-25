import sys, numpy as np
sys.path.insert(0, "/rds/project/rds-aSo1XX0UOlw/ir-kenn3/TRANSPORT_EQUATIONS/transport-equations/research")
import exp_front_lag_jump as E
R = E.R
for chi, dxr, nu4 in ((2.0, R.DX, 5e-4), (2.0, R.DX/2, 1.25e-4), (20.0, R.DX, 5e-4), (20.0, R.DX/2, 1.25e-4)):
    for ns in (3, 0):
        c = E.Case(chi, nu4=nu4, dx=dxr); c.P["n_smooth"] = ns
        runs = E.reduced_runs(c); row = []
        for (a, b), run in runs.items():
            D = E.diagnose(c, run, half_window=0.07 if nu4 >= 5e-4 else 0.05)
            row.append(f"{'down' if b < a else 'up'}: x_G end {D['xG'][-1]:.4f} (x_M {c.xM(b):.4f}, off {D['xG'][-1]-c.xM(b):+.4f}), half {E.half_time(D['t'], D['xG']):.4f}")
        print(f"chi {chi:4.0f} dx {c.dx:.5f} nu4 {nu4:.2e} n_smooth {ns}: " + "; ".join(row), flush=True)
