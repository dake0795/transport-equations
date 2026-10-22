"""Self-consistent alpha heating + bremsstrahlung: the paper's actual runs,
integrated to long times, plus the ignition case (C_alpha x50)."""
import numpy as np, reduced as R
from maxwell import maxwell
mx = maxwell()
base = dict(R.PAPER, nu4_conservative=True)
cases = [("paper collapse pb=0.4", dict(base), 0.4),
         ("paper pb=1.0", dict(base), 1.0),
         ("pb=1.6", dict(base), 1.6),
         ("pb=2.0", dict(base), 2.0),
         ("ignition C_alpha x50 pb=1.0", dict(base, C_alpha=2e-4), 1.0),
         ("ignition C_alpha x10 pb=1.0", dict(base, C_alpha=4e-5), 1.0),
         ("ignition C_alpha x50 pb=0.4", dict(base, C_alpha=2e-4), 0.4)]
for lab, params, pb in cases:
    try:
        res = R.run(pb, params=params, T_end=10.0, n_out=201, physics=True)
    except Exception as e:
        print(f"{lab}: FAILED {e}", flush=True); continue
    d = R.diagnostics(res); x=res["x"]
    Pa = [np.trapezoid(__import__('flux_transport_model_clean').alpha_heating(pk, params), x) for pk in res["p"][::50]]
    print(f"{lab}: S_ext={res['S_ext_tot']:.3f} S_tot: {d['S_tot'][0]:.3f}->{d['S_tot'][-1]:.3f} | "
          f"W: {d['W'][0]:.3f}->{d['W'][-1]:.3f} (t=2.5: {d['W'][50]:.3f}, t=5: {d['W'][100]:.3f}) dW/dt={(d['W'][-1]-d['W'][-3])/(d['t'][-1]-d['t'][-3]):+.1e} | "
          f"Qe={d['Q_edge'][-1]:.3f} gmax={d['g_max'][-1]:.2f} x_foot={d['x_foot'][-1]:.3f} | int P_alpha at t=0,2.5,5,7.5,10: {np.round(Pa,3)}", flush=True)
