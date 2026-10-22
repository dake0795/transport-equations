"""Long-time scan of the drive across the predicted Maxwell threshold."""
import sys, time, numpy as np
import reduced as R
from maxwell import maxwell
physics = len(sys.argv) > 1 and sys.argv[1] == "physics"
n_smooth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
mx = maxwell()
params = dict(R.PAPER, n_smooth=n_smooth)
pbs = [0.4, 1.0, 1.4, 1.5, 1.6, 1.7, 2.0, 2.5]
out = {}
for pb in pbs:
    t0 = time.time()
    res = R.run(pb, params=params, T_end=10.0, n_out=401, physics=physics)
    d = R.diagnostics(res)
    x = res["x"]; dx = x[1]-x[0]
    g_end = -(res["p"][-1][1:]-res["p"][-1][:-1])/dx
    xf = 0.5*(x[1:]+x[:-1])
    # predicted foot: int_0^x S_ext = Q_M
    cumS = np.concatenate([[0], np.cumsum(0.5*(res["S_ext"][1:]+res["S_ext"][:-1])*dx)])
    xM = np.interp(mx["Q_M"], cumS, x) if res["S_ext_tot"] > mx["Q_M"] else np.nan
    print(f"pb={pb:4.2f} S_ext={res['S_ext_tot']:.3f} S_tot(end)={d['S_tot'][-1]:.3f} | "
          f"W: {d['W'][0]:.3f}->{d['W'][-1]:.3f} dW/dt(end)={(d['W'][-1]-d['W'][-5])/(d['t'][-1]-d['t'][-5]):+.2e} | "
          f"Qe(end)={d['Q_edge'][-1]:.3f} g_edge={g_end[-1]:.2f} gmax={d['g_max'][-1]:.2f} | "
          f"x_foot(end)={d['x_foot'][-1]:.3f} pred x_M={xM:.3f} | {time.time()-t0:.0f}s", flush=True)
    out[f"pb{pb}"] = dict(t=d["t"], W=d["W"], Qe=d["Q_edge"], S=d["S_tot"], xfoot=d["x_foot"],
                          gmax=d["g_max"], p_end=res["p"][-1], g_end=g_end, xM=xM, S_ext=res["S_ext"],
                          p=res["p"][::20])
np.savez(f"scan_pb_{'phys' if physics else 'nophys'}_ns{n_smooth}.npz", x=x, xf=xf, **{k+"__"+kk: vv for k,v in out.items() for kk,vv in v.items()})
