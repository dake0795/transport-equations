"""Thermal (burn) stability of the ignited steady state of the reduced model: measure the linear growth/decay
rate of a global perturbation and compare with the criterion gamma_th = dS/dW - dQ_edge/dW (paper eq. Wstab).

Uses the ignition run of the paper (C_alpha = 2e-5, T_ref = 3 keV, int S_ext = 4.94, supercritical IC) run to a
steady state, then rescales the pressure profile by (1 + eps) and integrates the relaxation.
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import reduced as R, flux_transport_model_clean as m

x = R.grid(); dx = x[1] - x[0]
S_EXT_SCALE = 4.463319757225091
prm = dict(R.PAPER, nu4_conservative=True, C_alpha=2.0e-5, T_ref_keV=3.0)
base = R.run(None, params=prm, T_end=12.0, n_out=25, physics=True, source_scale=S_EXT_SCALE)
p0 = base["p"][-1]; d0 = R.diagnostics(base)
W0 = d0["W"][-1]; S0 = d0["S_tot"][-1]; Q0 = d0["Q_edge"][-1]
print(f"steady state: W={W0:.3f} S={S0:.3f} Q_edge={Q0:.3f} dW/dt~{(d0['W'][-1]-d0['W'][-2])/(d0['t'][-1]-d0['t'][-2]):.2e}")
# dS/dW by rescaling the profile uniformly (the heating response to a global change of stored energy)
def S_of(p):
    return np.trapezoid(R.base_source(x) * S_EXT_SCALE / np.trapezoid(R.base_source(x), x) + R.physics_source(p, prm), x)
eps = 0.02
dSdW = (S_of(p0 * (1 + eps)) - S_of(p0 * (1 - eps))) / (2 * eps * W0)
print(f"dS/dW (uniform rescaling) = {dSdW:.3f};  2 P_alpha/W = {2*np.trapezoid(m.alpha_heating(p0, prm), x)/W0:.3f}")
# relaxation of a global perturbation
out = {}
for sgn in (+1, -1):
    p1 = p0 * (1 + sgn * eps); p1[-1] = p0[-1]
    res = R.run(None, params=prm, T_end=3.0, n_out=301, physics=True, source_scale=S_EXT_SCALE, p0=p1)
    d = R.diagnostics(res); dW = d["W"] - W0; t = d["t"]
    sel = (t > 0.05) & (np.abs(dW) > 1e-4 * W0)
    slope, icpt = np.polyfit(t[sel], np.log(np.abs(dW[sel])), 1) if sel.sum() > 5 else (np.nan, np.nan)
    out[sgn] = dict(t=t, dW=dW, Qe=d["Q_edge"], S=d["S_tot"], gamma=slope)
    print(f"perturbation {sgn:+d}{eps}: dW(0)={dW[0]:+.4f} -> dW(end)={dW[-1]:+.4f}; fitted rate gamma = {slope:+.3f} (negative = stable)")
    # the loss response: dQ_edge/dW from the early transient
    k = np.argmax(t > 0.02); print(f"   early dQ_edge/dW ~ {(d['Q_edge'][k]-Q0)/dW[k]:.2f},  dS/dW ~ {(d['S_tot'][k]-S0)/dW[k]:.2f}")
np.savez(pathlib.Path(__file__).resolve().parent / "thermal.npz", W0=W0, S0=S0, dSdW=dSdW,
         **{f"{'p' if s>0 else 'm'}_{k}": np.asarray(v) for s, dd in out.items() for k, v in dd.items()})
