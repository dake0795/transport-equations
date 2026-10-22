"""Bistability: subcritical start.  Prediction: no barrier for S_tot < Qmax
(L-mode exists and is linearly stable); forced transition for S_tot > Qmax;
barrier survives down to S_tot = Q_M when started supercritical."""
import numpy as np, reduced as R
from maxwell import maxwell
mx = maxwell(); x = R.grid()
S0 = np.trapezoid(R.base_source(x), x)
params = dict(R.PAPER, nu4_conservative=True)
print(f"Q_M={mx['Q_M']:.3f}  Qmax={mx['Qmax']:.3f}")
for S_tot in [4.0, 6.0, 8.0, 9.5, 10.5, 12.0]:
    row = []
    for g_tgt, lab in [(0.5, "sub"), (1.8, "super")]:
        res = R.run(None, params=params, g_tgt=g_tgt, T_end=10.0, n_out=101,
                    physics=False, source_scale=S_tot / S0)
        d = R.diagnostics(res)
        row.append(f"{lab}: W={d['W'][-1]:.3f} gmax={d['g_max'][-1]:.2f} x_foot={d['x_foot'][-1]:.3f} dW/dt={(d['W'][-1]-d['W'][-3])/(d['t'][-1]-d['t'][-3]):+.1e}")
    print(f"S_tot={S_tot:5.2f} | " + " | ".join(row), flush=True)
