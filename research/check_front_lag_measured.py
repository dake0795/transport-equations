"""Independent re-measurement of the reduced-model half-times with the paper's own R.run (default tolerances),
dense output, sub-cell foot (g = g*) and mid ((g_a+g_b)/2) crossings.  Also n_smooth = 0 variant."""
import sys, numpy as np
RES = "/rds/project/rds-aSo1XX0UOlw/ir-kenn3/TRANSPORT_EQUATIONS/transport-equations/research"
sys.path.insert(0, RES); sys.path.insert(0, RES + "/..")
import reduced as R
from maxwell import maxwell

def cross(g, xf, lev):
    j = np.where(g > lev)[0][0]
    return xf[j-1] + (lev - g[j-1]) / (g[j] - g[j-1]) * (xf[j] - xf[j-1])

def half(t, a):
    f = np.abs(a - a[0]) - 0.5 * abs(a[-1] - a[0]); k = np.argmax(f > 0)
    return t[k-1] + (t[k]-t[k-1]) * (-f[k-1]) / (f[k]-f[k-1])

x = R.grid(); dx = x[1]-x[0]; xf = 0.5*(x[1:]+x[:-1]); S0 = np.trapezoid(R.base_source(x), x)
for n_smooth in (3, 0):
  for chi in (2.0, 20.0):
    P = dict(R.PAPER, nu4_conservative=True, chi_MHD=chi, n_smooth=n_smooth)
    mx = maxwell(dict(chi0=10.0, chi_RR=0.05, g_c=4.0, g_MHD=3.6, chi_MHD=chi))
    QM, ga, gb = mx["Q_M"], mx["g_a"], mx["g_b"]
    S_lo, S_hi = QM + 1.71, QM + 3.60
    st = {}
    st[S_hi] = R.run(None, params=P, g_tgt=1.8, T_end=10.0, n_out=3, physics=False, source_scale=S_hi/S0)["p"][-1]
    st[S_lo] = R.run(None, params=P, T_end=10.0, n_out=3, physics=False, source_scale=S_lo/S0, p0=st[S_hi])["p"][-1]
    for a, b in ((S_hi, S_lo), (S_lo, S_hi)):
        r1 = R.run(None, params=P, T_end=0.4, n_out=4001, physics=False, source_scale=b/S0, p0=st[a])
        r2 = R.run(None, params=P, T_end=3.0, n_out=3, physics=False, source_scale=b/S0, p0=st[a])
        t = r1["t"]; p = np.vstack([r1["p"], r2["p"][-1:]]); t = np.append(t, 3.0)
        G = -(p[:, 1:] - p[:, :-1]) / dx
        foot = np.array([cross(g, xf, 2.0) for g in G]); mid = np.array([cross(g, xf, 0.5*(ga+gb)) for g in G])
        print(f"n_smooth={n_smooth} chi={chi:4.0f} S {a:.2f}->{b:.2f}: foot {foot[0]:.4f}->{foot[-1]:.4f} half {half(t, foot):.4f}; "
              f"mid {mid[0]:.4f}->{mid[-1]:.4f} half {half(t, mid):.4f}", flush=True)
