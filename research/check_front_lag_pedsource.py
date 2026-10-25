"""Second out-of-sample test: add 1.89 of drive ONLY inside the pedestal (Gaussian at x=0.6, width 0.1).  Q_ss(x_f) is
unchanged near the front, so the Stefan law predicts no front motion at all; the two-phase mechanism predicts a
transient excursion driven by p_t^+ and a return to x_M.  Compare the excursion in the reduced model and the sharp model."""
import sys, numpy as np
sys.path.insert(0, "/rds/project/rds-aSo1XX0UOlw/ir-kenn3/TRANSPORT_EQUATIONS/transport-equations/research")
import importlib.util
spec = importlib.util.spec_from_file_location("cc", "/rds/project/rds-aSo1XX0UOlw/ir-kenn3/TRANSPORT_EQUATIONS/transport-equations/research/check_front_lag_coresource.py")
src = open(spec.origin).read().split("for chi in (2.0, 20.0):\n    Pm")[0]
exec(src)
from scipy.special import erf
def ped_src(Sint, add):
    c, w = 0.6, 0.1
    A = add/(w*np.sqrt(np.pi)/2*(erf((1-c)/w) - erf((0-c)/w)))
    s = lambda z: Sint/S0*5*np.exp(-np.asarray(z, float)**2/0.0625) + A*np.exp(-(np.asarray(z, float)-c)**2/w**2)
    Fc = lambda z: Sint/S0*Fb(z) + A*w*np.sqrt(np.pi)/2*(erf((np.asarray(z, float)-c)/w) - erf(-c/w))
    return s, Fc
for chi in (2.0, 20.0):
    Pm = dict(chi0=10.0, chi_RR=0.05, g_c=4.0, g_MHD=3.6, chi_MHD=chi)
    mx = maxwell(Pm); QM, ga, gb = mx["Q_M"], mx["g_a"], mx["g_b"]
    P = dict(R.PAPER, nu4_conservative=True, chi_MHD=chi, C_alpha=0.0, C_brem=0.0)
    S_hi = QM + 3.60
    hi = make_src(S_hi); hp = ped_src(S_hi, 1.89)
    st_hi = R.run(None, params=P, g_tgt=1.8, T_end=10.0, n_out=3, physics=False, source_scale=S_hi/S0)["p"][-1]
    t, Pt = reduced(P, hp[0], st_hi)
    G = -(Pt[:, 1:]-Pt[:, :-1])/dx
    mid = np.array([cross(g, 0.5*(ga+gb)) for g in G])
    ts, xs = sharp(Pm, QM, ga, gb, hi, hp)
    j = np.argmax(np.abs(mid - mid[0])); k = np.argmax(np.abs(xs - xs[0]))
    print(f"chi={chi:4.0f} pedestal-only +1.89: reduced x_mid {mid[0]:.4f}, max excursion {mid[j]-mid[0]:+.4f} at t={t[j]:.3f}, end {mid[-1]-mid[0]:+.4f};  "
          f"sharp: max excursion {xs[k]-xs[0]:+.4f} at t={ts[k]:.3f}, end {xs[-1]-xs[0]:+.4f};  Stefan law: 0", flush=True)
