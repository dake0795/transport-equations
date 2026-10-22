"""Regularisation dependence: front width should scale as sqrt(nu4); foot
position and W should not depend on nu4 or on the binomial smoothing."""
import numpy as np, reduced as R
from maxwell import maxwell
mx = maxwell()
for n_smooth in [3, 0]:
    for nu4 in [2e-3, 5e-4, 1.25e-4]:
        params = dict(R.PAPER, nu4=nu4, n_smooth=n_smooth, nu4_conservative=True)
        try:
            res = R.run(2.0, params=params, T_end=6.0, n_out=61, physics=False)
        except Exception as e:
            print(f"n_smooth={n_smooth} nu4={nu4:.2e}: FAILED {e}", flush=True); continue
        d = R.diagnostics(res); x=res["x"]; dx=x[1]-x[0]
        g = -(res["p"][-1][1:]-res["p"][-1][:-1])/dx; xf=0.5*(x[1:]+x[:-1])
        # front width: distance over which g rises from 10% to 90% of (g_b - g_a)
        lo, hi = mx["g_a"]+0.1*(mx["g_b"]-mx["g_a"]), mx["g_a"]+0.9*(mx["g_b"]-mx["g_a"])
        i_lo = np.argmax(g > lo); i_hi = np.argmax(g > hi)
        print(f"n_smooth={n_smooth} nu4={nu4:.2e}: x_foot={d['x_foot'][-1]:.3f} width(10-90%)={xf[i_hi]-xf[i_lo]:.4f} "
              f"W={d['W'][-1]:.3f} Qe={d['Q_edge'][-1]:.3f} gmax={d['g_max'][-1]:.2f} dW/dt={(d['W'][-1]-d['W'][-3])/(d['t'][-1]-d['t'][-3]):+.1e}", flush=True)
