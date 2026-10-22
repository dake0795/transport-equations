"""Maxwell (equal-area) construction for the reduced flux function.

g_t = d_xx[ Q(g) - nu4 g_xx ] - S_x  is Cahn-Hilliard with mu = Q(g) - nu4 g_xx
and bulk potential Phi(g) = int_0^g Q.  Two uniform phases g_a < g_b coexist
across a stationary front iff Q(g_a) = Q(g_b) = Q_M and
int_{g_a}^{g_b} (Q(g) - Q_M) dg = 0.
"""
import numpy as np
from scipy.optimize import brentq

DEFAULT = dict(chi0=10.0, chi_RR=0.05, g_c=4.0, g_MHD=3.6, chi_MHD=2.0)


def Q(g, P=DEFAULT):
    g = np.asarray(g, float)
    return (P["chi0"] * g * np.maximum(1 - g / P["g_c"], 0)
            + P["chi_RR"] * g
            + P["chi_MHD"] * np.maximum(g - P["g_MHD"], 0))


def landmarks(P=DEFAULT):
    gs = np.linspace(0, 4 * P["g_c"], 400001)
    q = Q(gs, P)
    gstar = P["g_c"] / 2
    Qmax = Q(gstar, P)
    # local minimum beyond the peak
    i_pk = np.argmax(q[gs < P["g_c"]])
    rest = slice(i_pk, None)
    i_min = i_pk + np.argmin(q[rest])
    return dict(gstar=gstar, Qmax=Qmax, gmin=gs[i_min], Qmin=q[i_min])


def branches(mu, P=DEFAULT):
    """g_a on rising branch, g_b on the high-g (cliff) branch, both with Q = mu."""
    lm = landmarks(P)
    ga = brentq(lambda g: Q(g, P) - mu, 0.0, lm["gstar"])
    g_hi = max(50 * P["g_c"], 2 * lm["Qmax"] / max(P["chi_RR"], 1e-9))
    gb = brentq(lambda g: Q(g, P) - mu, lm["gmin"], g_hi)
    return ga, gb


def maxwell(P=DEFAULT):
    lm = landmarks(P)
    if lm["Qmin"] >= lm["Qmax"]:
        return None

    def area(mu):
        ga, gb = branches(mu, P)
        gg = np.linspace(ga, gb, 20001)
        return np.trapezoid(Q(gg, P) - mu, gg)

    lo, hi = lm["Qmin"] * (1 + 1e-6), lm["Qmax"] * (1 - 1e-6)
    if area(lo) * area(hi) > 0:
        return None            # no coexistence: high-g phase cannot be balanced
    mu = brentq(area, lo, hi)
    ga, gb = branches(mu, P)
    return dict(Q_M=mu, g_a=ga, g_b=gb, **lm)


if __name__ == "__main__":
    r = maxwell()
    print("default parameters:")
    for k, v in r.items():
        print(f"  {k:6s} = {v:.4f}")
    print("\nscan chi_MHD:")
    for cm in [0.5, 1, 2, 5, 10, 50]:
        P = dict(DEFAULT, chi_MHD=cm)
        r = maxwell(P)
        print(f"  chi_MHD={cm:5.1f}: " + (f"Q_M={r['Q_M']:.3f}  Qmin={r['Qmin']:.3f}  g_a={r['g_a']:.3f} g_b={r['g_b']:.3f}" if r else "no coexistence (runaway)"))
    print("\nscan g_MHD/g_c:")
    for f in [0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        P = dict(DEFAULT, g_MHD=f * DEFAULT["g_c"])
        r = maxwell(P)
        print(f"  g_MHD={f:.2f}g_c: " + (f"Q_M={r['Q_M']:.3f}  Qmin={r['Qmin']:.3f}  g_a={r['g_a']:.3f} g_b={r['g_b']:.3f}" if r else "no coexistence (runaway)"))
    print("\nno cliff, chi_RR only:")
    for rr in [0.05, 0.5, 1.0, 2.0]:
        P = dict(DEFAULT, chi_MHD=0.0, chi_RR=rr)
        r = maxwell(P)
        print(f"  chi_RR={rr}: " + (f"Q_M={r['Q_M']:.3f} g_b={r['g_b']:.2f}" if r else "no coexistence (runaway)"))
