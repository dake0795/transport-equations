"""Does the non-conservative hyperviscosity stencil leak flux at the edge?"""
import numpy as np, reduced as R
from maxwell import maxwell
mx = maxwell()
for cons in [False, True]:
    params = dict(R.PAPER, nu4_conservative=cons)
    for pb in [1.0, 1.5, 2.0]:
        res = R.run(pb, params=params, T_end=10.0, n_out=101, physics=False)
        d = R.diagnostics(res); x=res["x"]; dx=x[1]-x[0]
        g = -(res["p"][-1][1:]-res["p"][-1][:-1])/dx
        print(f"conservative={cons!s:5} pb={pb}: S_tot={res['S_ext_tot']:.3f} Qe(end)={d['Q_edge'][-1]:.3f} "
              f"g_edge={g[-1]:.2f} g[-4:]={np.round(g[-4:],2)} W(end)={d['W'][-1]:.3f} dW/dt={(d['W'][-1]-d['W'][-3])/(d['t'][-1]-d['t'][-3]):+.1e} x_foot={d['x_foot'][-1]:.3f}", flush=True)
