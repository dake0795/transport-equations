"""Does the lag of the front behind the Stefan law come from the pedestal's diffusive relaxation?
Test: repeat the step experiment of the paper (S 9.45 -> 7.56 and back) with a much stiffer cliff, chi_MHD = 20
instead of 2, so that the pedestal relaxes ten times faster ((L - x_M)^2 / pi^2 chi).  If the attribution is right
the measured half-time approaches the law's.  Everything else as in the paper (conservative nu4, n_smooth 3).
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import reduced as R
from maxwell import maxwell, branches, landmarks
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

x = R.grid(); S0 = np.trapezoid(R.base_source(x), x)
for chi in (2.0, 20.0):
    P = dict(R.PAPER, nu4_conservative=True, chi_MHD=chi)
    Pm = dict(chi0=P["chi0"], chi_RR=P["chi_RR"], g_c=P["g_c"], g_MHD=P["g_MHD"], chi_MHD=chi)
    mx = maxwell(Pm); lm = landmarks(Pm)
    Qss = lambda xx, S: (S / S0) * np.trapezoid(R.base_source(np.linspace(0, xx, 400)), np.linspace(0, xx, 400))
    S_lo, S_hi = mx["Q_M"] + 1.71, mx["Q_M"] + 3.60     # the same offsets above Q_M as the paper's 7.56 and 9.45
    xM = {S: brentq(lambda xx: Qss(xx, S) - mx["Q_M"], 1e-3, 1.0) for S in (S_hi, S_lo)}
    print(f"\nchi_MHD={chi}: S_lo={S_lo:.2f} S_hi={S_hi:.2f} Q_M={mx['Q_M']:.3f} g_a={mx['g_a']:.3f} g_b={mx['g_b']:.3f} x_M(9.45)={xM[S_hi]:.3f} x_M(7.56)={xM[S_lo]:.3f}; "
          f"pedestal time (L-x_M)^2/pi^2 chi = {(1-xM[S_lo])**2/np.pi**2/(chi+P['chi_RR']):.4f}")
    pre = R.run(None, params=P, g_tgt=1.8, T_end=10.0, n_out=11, physics=False, source_scale=S_hi / S0)
    for S_from, S_to, p0 in ((S_hi, S_lo, pre["p"][-1]), (S_lo, S_hi, None)):
        if p0 is None:
            pre2 = R.run(None, params=P, T_end=10.0, n_out=11, physics=False, source_scale=S_from / S0, p0=pre["p"][-1]); p0 = pre2["p"][-1]
        post = R.run(None, params=P, T_end=4.0, n_out=401, physics=False, source_scale=S_to / S0, p0=p0)
        d = R.diagnostics(post); t = d["t"]; xf = d["x_foot"]
        ga, gb = mx["g_a"], mx["g_b"]
        sol = solve_ivp(lambda tt, y: [-(Qss(float(y[0]), S_to) - mx["Q_M"]) / (float(y[0]) * (gb - ga))], (0, t[-1]), [xf[0]], t_eval=t, rtol=1e-8)
        law = sol.y[0]
        half = lambda a: t[np.argmax(np.abs(a - a[0]) > 0.5 * abs(a[-1] - a[0]))]
        print(f"  S {S_from} -> {S_to}: x_f {xf[0]:.3f} -> {xf[-1]:.3f} (law -> {law[-1]:.3f}); half-time measured {half(xf):.3f}, law {half(law):.3f}, ratio {half(xf)/half(law):.2f}")
