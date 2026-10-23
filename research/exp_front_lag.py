"""Where does the front's lag behind the Stefan law come from?  Decompose the law's ingredients with the simulation's
own values.  The step experiment of the paper (chi_MHD = 2: S 9.45 -> 7.56 and back).  At each output time measure
(i) the actual flux through the front, Q_front(t) (the face flux at the front face), against the law's Q_ss(x_f),
(ii) the gradients a few cells inside and outside the front, g_in(t), g_out(t), against g_a, g_b.
Then integrate the law with (a) Q_ss and (g_b - g_a) [the paper], (b) the measured Q_front(t) and (g_b - g_a),
(c) the measured Q_front(t) and the measured jump g_out - g_in, and compare the half-times with the measured front.
Also print the pointwise ratio of the measured speed to the law's speed built from measured ingredients.
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import reduced as R
import flux_transport_model_clean as m
from maxwell import maxwell
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

x = R.grid(); dx = x[1] - x[0]; S0 = np.trapezoid(R.base_source(x), x)
P = dict(R.PAPER, nu4_conservative=True)
Pm = dict(chi0=P["chi0"], chi_RR=P["chi_RR"], g_c=P["g_c"], g_MHD=P["g_MHD"], chi_MHD=P["chi_MHD"])
mx = maxwell(Pm); QM, ga, gb = mx["Q_M"], mx["g_a"], mx["g_b"]; gstar = P["g_c"] / 2
Qss = lambda xx, S: (S / S0) * np.trapezoid(R.base_source(np.linspace(0, xx, 400)), np.linspace(0, xx, 400))
S_lo, S_hi = QM + 1.71, QM + 3.60
print(f"Q_M={QM:.3f} g_a={ga:.3f} g_b={gb:.3f}; D_L = Q'(g_a) = {P['chi0']*(1-2*ga/P['g_c']):.2f}, D_ped = {P['chi_RR']+P['chi_MHD']:.2f}")

pre = R.run(None, params=P, g_tgt=1.8, T_end=10.0, n_out=11, physics=False, source_scale=S_hi / S0)
state = {S_hi: pre["p"][-1]}
state[S_lo] = R.run(None, params=P, T_end=10.0, n_out=11, physics=False, source_scale=S_lo / S0, p0=state[S_hi])["p"][-1]
K = 3   # cells inside/outside the front face at which the coexisting gradients are read
for S_from, S_to in ((S_hi, S_lo), (S_lo, S_hi)):
    post = R.run(None, params=P, T_end=1.0, n_out=201, physics=False, source_scale=S_to / S0, p0=state[S_from])
    t = post["t"]; xf, Qfr, gin, gout = [], [], [], []
    for pk in post["p"]:
        Qf, xface = m.compute_face_flux(pk, dx, x, R.P_PED, P)
        g = -(pk[1:] - pk[:-1]) / dx
        i = np.where(g > gstar)[0][0]
        xf.append(xface[i]); Qfr.append(Qf[i]); gin.append(g[max(i - K, 0)]); gout.append(g[min(i + K, len(g) - 1)])
    xf, Qfr, gin, gout = map(np.array, (xf, Qfr, gin, gout))
    law = lambda Qfun, jump: solve_ivp(lambda tt, y: [-(Qfun(tt, float(y[0])) - QM) / (float(y[0]) * jump(tt))], (0, t[-1]), [xf[0]], t_eval=t, rtol=1e-8).y[0]
    Qmeas = lambda tt, xx: np.interp(tt, t, Qfr); jmeas = lambda tt: np.interp(tt, t, gout - gin)
    la = law(lambda tt, xx: Qss(xx, S_to), lambda tt: gb - ga)
    lb = law(Qmeas, lambda tt: gb - ga)
    lc = law(Qmeas, jmeas)
    half = lambda a: t[np.argmax(np.abs(a - a[0]) > 0.5 * abs(a[-1] - a[0]))]
    print(f"\nS {S_from:.2f} -> {S_to:.2f}: x_f {xf[0]:.3f} -> {xf[-1]:.3f}; half-times: measured {half(xf):.3f}, law(a) Q_ss & Maxwell jump {half(la):.3f}, "
          f"(b) measured Q_front & Maxwell jump {half(lb):.3f}, (c) measured Q_front & measured jump {half(lc):.3f}")
    print("   t      x_f    Q_ss(x_f)  Q_front   g_in   g_out   v_meas   v_law(a)  v_law(c)")
    v = np.gradient(xf, t)
    for j in range(0, 61, 10):
        print(f"  {t[j]:.3f}  {xf[j]:.3f}  {Qss(xf[j], S_to):7.3f}  {Qfr[j]:7.3f}  {gin[j]:5.2f}  {gout[j]:5.2f}  {v[j]:+7.3f}  {-(Qss(xf[j], S_to)-QM)/(xf[j]*(gb-ga)):+7.3f}  {-(Qfr[j]-QM)/(xf[j]*(gout[j]-gin[j])):+7.3f}")
np.savez("front_lag.npz", note="see exp_front_lag.py")
