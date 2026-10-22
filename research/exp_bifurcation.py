"""Bifurcation diagram W_ss(S_tot) from sub- and supercritical starts, with the
analytic steady states: L branch g = g_-(Q_ss(x)) everywhere; H branch
g = g_b(Q_ss(x)) for x > x_M (where Q_ss(x_M) = Q_M), g_-(Q_ss) inside."""
import numpy as np, reduced as R
from scipy.optimize import brentq
from maxwell import maxwell, Q, landmarks
mx = maxwell(); lm = landmarks()
x = R.grid(); dx = x[1]-x[0]
S0 = np.trapezoid(R.base_source(x), x)
params = dict(R.PAPER, nu4_conservative=True)

def g_minus(q):
    return brentq(lambda g: Q(g)-q, 0, lm["gstar"]) if q < lm["Qmax"] else np.nan
def g_cliff(q):
    return brentq(lambda g: Q(g)-q, lm["gmin"], 1e3)

def analytic_W(S_tot):
    S = S_tot/S0 * R.base_source(x)
    Qss = np.concatenate([[0], np.cumsum(0.5*(S[1:]+S[:-1])*dx)])
    gL = np.array([g_minus(q) for q in Qss])
    WL = np.nan
    if not np.isnan(gL).any():
        pL = R.P_PED + np.concatenate([np.cumsum((gL[::-1][1:]+gL[::-1][:-1])*0.5*dx)[::-1], [0]])
        WL = np.trapezoid(pL, x)
    WH = np.nan
    if S_tot > mx["Q_M"]:
        gH = np.where(Qss >= mx["Q_M"], [g_cliff(max(q, lm["Qmin"]*1.001)) for q in Qss],
                      [g_minus(q) if q < lm["Qmax"] else np.nan for q in Qss])
        pH = R.P_PED + np.concatenate([np.cumsum((gH[::-1][1:]+gH[::-1][:-1])*0.5*dx)[::-1], [0]])
        WH = np.trapezoid(pH, x)
    return WL, WH

rows = []
for S_tot in [3, 4, 5, 5.5, 6, 6.5, 7, 8, 9, 9.5, 10, 10.5, 11, 12, 13]:
    WL, WH = analytic_W(S_tot)
    num = {}
    for g_tgt, lab in [(0.5, "sub"), (1.8, "super")]:
        res = R.run(None, params=params, g_tgt=g_tgt, T_end=12.0, n_out=61, physics=False, source_scale=S_tot/S0)
        d = R.diagnostics(res)
        num[lab] = (d["W"][-1], d["x_foot"][-1], (d["W"][-1]-d["W"][-3])/(d["t"][-1]-d["t"][-3]))
    print(f"S_tot={S_tot:5.2f} | analytic W_L={WL:.3f} W_H={WH:.3f} | sub: W={num['sub'][0]:.3f} xf={num['sub'][1]:.3f} dW/dt={num['sub'][2]:+.1e} | super: W={num['super'][0]:.3f} xf={num['super'][1]:.3f} dW/dt={num['super'][2]:+.1e}", flush=True)
    rows.append([S_tot, WL, WH, num['sub'][0], num['sub'][1], num['super'][0], num['super'][1]])
np.save("bifurcation.npy", np.array(rows))
