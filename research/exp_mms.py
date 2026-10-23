"""Multiple scales / matched asymptotics for the flux-driven reduced model: numerical verification (NOTES_mms.md).

Three small parameters, each dialled at fixed everything else:

  delta = front width / L     (nu4)                    sharp interface: Maxwell rest condition, Stefan law
  mu    = thermal margin      (C_brem -> the fold)     the global thermal mode is slow: amplitude equation for W
  eps   = tau_th / tau_N      (D_n, at a fixed path)   the density is slower still: slow manifold W*(N), fold passage

Stages (``python3 exp_mms.py <stage>``; each caches to mms_<stage>.npz and is skipped if the cache exists; one
process, this node only):
  hot     the hot root at C_brem = 0.01 (BDF from the paper's supercritical start, Newton polish)      mms_hot.npz
  mualpha the same with the margin dialled by C_alpha at C_brem = 1.0                               mms_mualpha.npz
  mu      continuation in C_brem (natural) then in W (bordered) through the thermal fold: thermal mode, next modes,
          steady-family tangent, the rate from the W balance closed on that family, from its Rayleigh quotient,
          and the margin mu                                                                           mms_mu.npz
  mutraj  perturbation runs at four margins: the late rate, the trajectory slopes (the T3D method) and the
          amplitude lost or gained in the first transient against the adjoint projection          mms_mutraj.npz
  delta   steady barriers at fixed source for nu4 = 2e-5 ... 5e-3 on a fine grid: front position (foot, mid,
          Gibbs) and W against the sharp-interface limit                                           mms_delta.npz
  epsstat the frozen-density branch W*(s), gamma(s) along the density path and its fold s_f       mms_epsstat.npz
  epsdyn  time-dependent runs at D_n spanning two decades: slow-manifold lag and the delayed jump   mms_epsdyn.npz
  t3d     the margin mu and the ratio eps for the T3D runs from data/t3d                            mms_t3d.npz
  figs    quick-look PNGs in mms_figs/
  tables  print what NOTES_mms.md quotes
"""
import pathlib
import sys
import time

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse.linalg import splu
from scipy.special import ai_zeros

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mms_core as C   # noqa: E402

PAPER = HERE.parent.parent / "transport_equations_paper"
FIGS = HERE / "mms_figs"
OMEGA0 = float(-ai_zeros(1)[0][0])          # 2.338107410459767, first zero of Ai(-z)
NGRID = 1000                                 # h = 1e-3
NU4 = 2e-3                                   # front width ~ 0.02-0.03 = 20+ cells: grid-converged to 1e-4 in gamma
                                             # (nu4 = 5e-4 needs N = 2000; N = 500 pins the front, see NOTES_mms.md)


def model(N=None, **kw):
    M = C.Model(N=NGRID if N is None else N)
    M.P["nu4"] = NU4
    M.P.update(kw)
    return M


def cache(name):
    return HERE / f"mms_{name}.npz"


def set_cbrem(M, c):
    M.P["C_brem"] = c


# =============================================================================================== hot root
def do_hot():
    f = cache("hot")
    if f.exists():
        print("mms_hot.npz exists"); return
    M = model()
    p0 = 1.0 + 1.8 * (1 - M.x ** 2)           # max gradient 3.6 = 1.8 g_star at the edge: the paper's supercritical start
    sol = C.integrate(M, p0, 10.0, np.linspace(0, 10, 11))
    p, ok, err = C.newton(M, sol.y[:, -1])
    print(f"hot root: W = {M.W(p):.5f}, S = {M.S_tot(p):.5f}, Q_edge = {M.Q_edge(p):.5f}, newton ok={ok} |r|={err:.1e}")
    np.savez(f, p=p, x=M.x)


# =============================================================================================== mu
def thermal_record(M, p, b=None):
    """Everything about the thermal mode at a steady state p; b = d rhs / d c for the steady-family tangent."""
    md = C.modes(M, p, k=6)
    lam, v, u, Js = md["lam"], md["v"], md["u"], md["Js"]
    gam = lam[0]
    dSv, dQv = C.directional(M, p, v)
    mu = -gam / dQv
    rec = dict(W=M.W(p), S=M.S_tot(p), Qe=M.Q_edge(p), gamma=gam, lam2=lam[1], lam3=lam[2], dSdW_v=dSv,
               dQdW_v=dQv, mu=mu, asym=md["asym"], **{f"front_{k}": val for k, val in M.fronts(p).items()})
    rec["P_alpha"] = float(np.sum(C.P_alpha(p, M.n, M.P)) * M.h)
    rec["P_brem"] = float(np.sum(C.P_brem(p, M.n, M.P)) * M.h)
    if b is not None:
        lu = splu(Js)
        t = lu.solve(b)
        t = t / (np.sum(t) * M.h)               # steady-family tangent, unit Delta W
        dSt, dQt = C.directional(M, p, t)
        Jt = Js @ t
        rec.update(gamma_int=dSt - dQt, dSdW_t=dSt, dQdW_t=dQt,
                   gamma_ray=float(t @ Jt / (t @ t)),
                   gamma_pg=float(u @ Jt / (u @ t)),
                   tv_angle=float(np.sqrt(max(0.0, 1 - (t @ v) ** 2 / ((t @ t) * (v @ v))))),
                   proj_ratio=float(np.sum(b) / ((u @ b) / (u @ v))))   # (int dN/dc) / (its thermal projection)
        rec["t"] = t
    # uniform rescaling of the excess pressure (the paper's / predictor's proxy)
    dSu, dQu = C.directional(M, p, p - C.P_PED)
    rec.update(dSdW_uni=dSu, dQdW_uni=dQu, gamma_uni=dSu - dQu)
    rec["v"], rec["u"] = v, u
    return rec


def dcdrhs(M, p, dc=1e-6):
    c = M.P["C_brem"]
    set_cbrem(M, c + dc); rp = M.source(p)
    set_cbrem(M, c - dc); rm = M.source(p)
    set_cbrem(M, c)
    return (rp - rm) / (2 * dc)


def do_mu():
    f = cache("mu")
    if f.exists():
        print("mms_mu.npz exists"); return
    M = model()
    p = np.load(cache("hot"))["p"]
    rows, profiles = [], []

    def record(p, c, tag):
        set_cbrem(M, c)
        rec = thermal_record(M, p, b=dcdrhs(M, p))
        rec["C_brem"] = c
        rows.append(rec); profiles.append(p.copy())
        print(f"  [{tag}] C={c:.5f} W={rec['W']:.4f} gam={rec['gamma']:+.4f} lam2={rec['lam2']:+.2f} mu={rec['mu']:.4f} "
              f"g_int={rec['gamma_int']:+.4f} g_ray={rec['gamma_ray']:+.4f} g_uni={rec['gamma_uni']:+.3f} "
              f"x_mid={rec['front_mid']:.3f} angle={rec['tv_angle']:.2e}", flush=True)

    t0 = time.time()
    for c in [0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2]:
        set_cbrem(M, c)
        p, ok, err = C.newton(M, p)
        if not ok:
            print(f"  newton failed at C={c} |r|={err:.1e}")
        record(p, c, "C")
    c = 1.2
    W = M.W(p)
    dW = 0.04
    passed = 0
    while passed < 6 and W > 2.5:
        Wn = W - dW
        pn, cn, ok = C.newton_bordered(M, p, c, set_cbrem, Wn)
        if not ok:
            dW /= 2
            if dW < 1e-3:
                print("  bordered continuation stalled"); break
            continue
        p, c, W = pn, cn, Wn
        record(p, c, "W")
        if len(rows) > 2 and rows[-1]["C_brem"] < rows[-2]["C_brem"]:
            passed += 1
        if rows[-1]["mu"] < 0.05:
            dW = min(dW, 0.01)
    print(f"  {time.time() - t0:.0f} s")
    keys = [k for k in rows[0] if k not in ("t", "v", "u")]
    out = {k: np.array([r[k] for r in rows]) for k in keys}
    out["profiles"] = np.array(profiles)
    out["t_vec"] = np.array([r["t"] for r in rows]); out["v_vec"] = np.array([r["v"] for r in rows])
    out["u_vec"] = np.array([r["u"] for r in rows]); out["x"] = M.x
    np.savez(f, **out)


# =============================================================================================== mu via C_alpha
def set_calpha(M, c):
    M.P["C_alpha"] = c * 1e-5                     # c in units of 1e-5


def do_mualpha():
    """The same test with the margin dialled by C_alpha (at C_brem = 1.0): the family's generating perturbation is
    the alpha power, concentrated in the core, so its projection factor Pi differs from the sink family's."""
    f = cache("mualpha")
    if f.exists():
        print("mms_mualpha.npz exists"); return
    zmu = np.load(cache("mu"))
    M = model(C_brem=1.0)
    i0 = int(np.argmin(np.abs(zmu["C_brem"] - 1.0)))
    p, c = zmu["profiles"][i0].copy(), 2.0
    rows = []

    def record(p, c):
        set_calpha(M, c)
        dc = 1e-6
        set_calpha(M, c + dc); sp = M.source(p); set_calpha(M, c - dc); sm = M.source(p); set_calpha(M, c)
        rec = thermal_record(M, p, b=(sp - sm) / (2 * dc))
        rec["C_alpha"] = c * 1e-5
        rows.append(rec)
        print(f"  C_alpha={c:.5f}e-5 W={rec['W']:.4f} gam={rec['gamma']:+.4f} lam2={rec['lam2']:+.2f} mu={rec['mu']:.4f} "
              f"int/g={rec['gamma_int'] / rec['gamma']:.4f} Pi={rec['proj_ratio'] * M.h:.4f} "
              f"ray/g-1={rec['gamma_ray'] / rec['gamma'] - 1:+.2e} uni={rec['gamma_uni']:+.3f}", flush=True)

    record(p, c)
    W, dW, passed = M.W(p), -0.04, 0
    while passed < 3 and W > 2.5:
        pn, cn, ok = C.newton_bordered(M, p, c, set_calpha, W + dW)
        if not ok:
            dW /= 2
            if abs(dW) < 1e-3: break
            continue
        p, c, W = pn, cn, W + dW
        record(p, c)
        if rows[-1]["gamma"] > 0:
            passed += 1
        if rows[-1]["mu"] < 0.15:
            dW = max(dW, -0.01)
    keys = [k for k in rows[0] if k not in ("t", "v", "u")]
    np.savez(f, **{k: np.array([r[k] for r in rows]) for k in keys})


# =============================================================================================== mutraj
def do_mutraj():
    f = cache("mutraj")
    if f.exists():
        print("mms_mutraj.npz exists"); return
    z = np.load(cache("mu"))
    M = model()
    out = {}
    stable = np.where(z["gamma"] < 0)[0]
    picks = []
    for target in (0.5, 0.3, 0.15, 0.05):
        i = stable[np.argmin(np.abs(z["mu"][stable] - target))]
        if i not in picks:
            picks.append(int(i))
    for j, i in enumerate(picks):
        c = float(z["C_brem"][i]); set_cbrem(M, c)
        p_star = z["profiles"][i]; gam = float(z["gamma"][i]); lam2 = float(z["lam2"][i])
        W_star = M.W(p_star)
        amp = 2e-3
        dp0 = amp * (p_star - C.P_PED)                     # a uniform rescaling of the excess pressure
        T_end = min(10.0 / abs(gam), 120.0)
        t_eval = np.concatenate([np.linspace(0, 2.0 / abs(lam2) * 5, 60), np.linspace(10.0 / abs(lam2), T_end, 300)])
        t_eval = np.unique(t_eval)
        t0 = time.time()
        sol = C.integrate(M, p_star + dp0, T_end, t_eval, rtol=1e-8, atol=1e-11)
        W = np.array([M.W(q) for q in sol.y.T]); S = np.array([M.S_tot(q) for q in sol.y.T])
        Qe = np.array([M.Q_edge(q) for q in sol.y.T])
        dW = W - W_star
        late = sol.t > 4.0 / abs(lam2)
        late &= np.abs(dW) > 1e-4 * abs(dW[0])
        A = np.polyfit(sol.t[late], np.log(np.abs(dW[late])), 1)
        gfit, A0 = A[0], np.sign(dW[late][0]) * np.exp(A[1])
        # trajectory slopes (the T3D method): regress S and Q_edge on W over the late window
        sS = np.polyfit(W[late], S[late], 1)[0]; sQ = np.polyfit(W[late], Qe[late], 1)[0]
        u = z["u_vec"][i]
        A_pred = float(np.sum(u * dp0) * M.h)              # adjoint projection: slow amplitude in units of Delta W
        dW0 = float(np.sum(dp0) * M.h)
        print(f"  mu={z['mu'][i]:.3f} C={c:.5f} gam={gam:+.4f} fit={gfit:+.4f} ({gfit / gam - 1:+.1e}); "
              f"slopes traj {sS:.4f},{sQ:.4f} vs eigvec {z['dSdW_v'][i]:.4f},{z['dQdW_v'][i]:.4f}; "
              f"dW0={dW0:.3e} A_fit={A0:.3e} A_adjoint={A_pred:.3e} ({A0 / A_pred - 1:+.1e}); {time.time() - t0:.0f} s",
              flush=True)
        for k, val in dict(t=sol.t, W=W, S=S, Qe=Qe).items():
            out[f"r{j}_{k}"] = val
        for k, val in dict(i=i, mu=z["mu"][i], C=c, gamma=gam, lam2=lam2, gfit=gfit, sS=sS, sQ=sQ,
                           dSv=z["dSdW_v"][i], dQv=z["dQdW_v"][i], dW0=dW0, A0=A0, A_pred=A_pred, W_star=W_star,
                           g_int=z["gamma_int"][i], g_uni=z["gamma_uni"][i]).items():
            out[f"r{j}_{k}"] = val
    out["n"] = len(picks)
    np.savez(f, **out)


# =============================================================================================== delta
S_DELTA = 7.56                    # the paper's front test (Section 4.4): int S_ext = 7.56, x_M = 0.214, no alpha, no sink


def heteroclinic(P):
    """The leading-order inner solution nu4 g'' = Q(g) - Q_M, in the stretched variable xi = x / sqrt(nu4):
    xi(g) = int dg / sqrt(2 (Omega(g) - Omega(g_a))), Omega = Phi - Q_M g.  Returns the offsets of the foot (g = g_c/2)
    and the mid-gradient level from the Gibbs (equal-area) dividing surface, and the width (g_b - g_a) / max g'."""
    MX = C.maxwell(P)
    QM, ga, gb = MX["Q_M"], MX["g_a"], MX["g_b"]
    # Omega - Omega_a vanishes quadratically at g_a and g_b: integrate in a variable that removes the log ends
    gg = np.linspace(ga, gb, 200001)[1:-1]
    om = C.Phi(gg, P) - QM * gg - (C.Phi(ga, P) - QM * ga)
    om = np.maximum(om, 1e-300)
    dxi = 1.0 / np.sqrt(2 * om)
    xi = np.concatenate([[0.0], np.cumsum(0.5 * (dxi[1:] + dxi[:-1]) * np.diff(gg))])
    # the g-average of xi(g) is the Gibbs surface (the log ends are integrable; the end cells add < 1e-6)
    xi_G = np.trapezoid(xi, gg) / (gg[-1] - gg[0])
    xi_foot = np.interp(P["g_c"] / 2, gg, xi) - xi_G
    xi_mid = np.interp(0.5 * (ga + gb), gg, xi) - xi_G
    width = (gb - ga) / np.sqrt(2 * om.max())        # in units of sqrt(nu4)
    return dict(xi_foot=float(xi_foot), xi_mid=float(xi_mid), width=float(width), Q_M=QM, g_a=ga, g_b=gb)


def sharp_fixed_source(P, S_tot=S_DELTA):
    """Sharp-interface steady state for the fixed Gaussian source: x_M, W and the pedestal height."""
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.special import erf
    A = S_tot / C.BASE_INT
    Qss = lambda x: A * 5.0 * 0.25 * np.sqrt(np.pi) / 2 * erf(x / 0.25)
    MX = C.maxwell(P)
    xM = brentq(lambda x: Qss(x) - MX["Q_M"], 1e-6, 1.0, xtol=1e-14)
    gm = lambda x: float(C.g_minus(Qss(x), P))
    gp = lambda x: float(C.g_plus(Qss(x), P))
    # W = int_0^L p = L p_ped + int_0^L x g(x) dx
    W = C.L * C.P_PED + quad(lambda x: x * gm(x), 0, xM, limit=400, epsabs=1e-13)[0] \
        + quad(lambda x: x * gp(x), xM, C.L, limit=400, epsabs=1e-13)[0]
    return dict(x_M=xM, W=W, Qss=Qss)


def gibbs_local(M, p, half=12.0, w=None):
    """Gibbs dividing surface of the front from the local excess: on a window around the mid crossing, find x_G with
    int g dx = int_a^{x_G} g_-(J) + int_{x_G}^b g_+(J), J the steady flux through each face (= the outer mu)."""
    from scipy.optimize import brentq
    J, g = M.flux(p)
    xf = M.xf
    xm = M.fronts(p)["mid"]
    a, b = xm - half * w, xm + half * w
    sel = (xf >= a) & (xf <= b)
    xs, gs, Js = xf[sel], g[sel], J[sel]
    tot = np.trapezoid(gs, xs)
    gmo, gpo = C.g_minus(Js, M.P), C.g_plus(Js, M.P)
    cm = np.concatenate([[0.0], np.cumsum(0.5 * (gmo[1:] + gmo[:-1]) * np.diff(xs))])
    cp = np.concatenate([[0.0], np.cumsum(0.5 * (gpo[1:] + gpo[:-1]) * np.diff(xs))])
    f = lambda xg: np.interp(xg, xs, cm) + (cp[-1] - np.interp(xg, xs, cp)) - tot
    return float(brentq(f, xs[1], xs[-2], xtol=1e-14))


NU4_LIST = [2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3, 3.2e-3, 6.4e-3]


def do_delta():
    f = cache("delta")
    if f.exists():
        print("mms_delta.npz exists"); return
    P0 = dict(C.PRM, C_alpha=0.0, C_brem=0.0)
    het = heteroclinic(P0)
    sh = sharp_fixed_source(P0)
    print(f"inner solution: xi_foot = {het['xi_foot']:+.4f}, xi_mid = {het['xi_mid']:+.4f}, width = {het['width']:.4f} "
          f"(units sqrt(nu4)); sharp: x_M = {sh['x_M']:.6f}, W = {sh['W']:.6f}")
    rows = []
    p_prev, x_prev = None, None
    for nu in sorted(NU4_LIST, reverse=True):
        w = het["width"] * np.sqrt(nu)
        for N in sorted({max(1000, int(np.ceil(16 / w / 1000)) * 1000), 2 * max(1000, int(np.ceil(16 / w / 1000)) * 1000)}):
            M = C.Model(N=N, P=P0, S_tot=S_DELTA)
            M.P["nu4"] = nu
            if p_prev is None:
                p0 = 1.0 + 1.8 * (1 - M.x ** 2)
                sol = C.integrate(M, p0, 6.0, np.array([0.0, 6.0]))
                p0 = sol.y[:, -1]
            else:
                p0 = np.interp(M.x, x_prev, p_prev)
            p, ok, err = C.newton(M, p0, maxit=200)
            if not ok:
                print(f"   nu4={nu:.2e} N={N}: newton not converged |r|={err:.1e}")
            fr = M.fronts(p)
            xG = gibbs_local(M, p, w=w)
            md = C.modes(M, p, k=3)
            rows.append(dict(nu4=nu, N=N, w=w, delta=w / C.L, x_foot=fr["foot"], x_mid=fr["mid"], x_G=xG, W=M.W(p),
                             Qe=M.Q_edge(p), lam1=md["lam"][0], ok=ok))
            print(f"   nu4={nu:.2e} N={N:5d} w={w:.4f} (w/h={w * N:.0f}) x_foot-x_M={fr['foot'] - sh['x_M']:+.3e} "
                  f"x_mid-x_M={fr['mid'] - sh['x_M']:+.3e} x_G-x_M={xG - sh['x_M']:+.3e} W-W_sh={M.W(p) - sh['W']:+.3e} "
                  f"lam1={md['lam'][0]:.2f}", flush=True)
        p_prev, x_prev = p, M.x
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out.update(x_M=sh["x_M"], W_sharp=sh["W"], **{f"het_{k}": v for k, v in het.items()})
    np.savez(f, **out)


# =============================================================================================== fold location vs delta
def sharp_fold_branch(P, N=4000, xfs=None):
    """Sharp-interface steady branch with self-consistent sources, parametrised by the front position x_f: a Maxwell
    front at x_f, g = g_-(Q_ss) inside, g_+(Q_ss) outside, and C_brem chosen so that Q_ss(x_f) = Q_M (Picard on p).
    The fold is the maximum of C_brem(x_f)."""
    h = C.L / N
    x = (np.arange(N) + 0.5) * h
    S_ext = C.base_source(x) * (C.S_EXT_TOT / C.BASE_INT)
    QM = C.maxwell(P)["Q_M"]
    n = np.ones(N)
    rows = []
    p = 1.0 + 3.0 * (1 - x ** 2)
    for xf in xfs:
        core = x < xf
        k = int(np.searchsorted(x, xf))
        frac = (xf - (x[k - 1] + h / 2)) / h            # part of cell k-1 ... handled by face accumulation below
        for it in range(3000):
            Pa = C.P_alpha(p, n, P); Pb1 = C.P_brem(p, n, dict(P, C_brem=1.0))
            # Q_ss at faces: cumulative cell sums; the front sits at x_f between faces: interpolate linearly
            cf = np.concatenate([[0.0], np.cumsum(S_ext + Pa) * h]); cb = np.concatenate([[0.0], np.cumsum(Pb1) * h])
            xfa = np.arange(N + 1) * h
            A_f, B_f = np.interp(xf, xfa, cf), np.interp(xf, xfa, cb)
            Cb = (A_f - QM) / B_f
            Qss = cf - Cb * cb                           # at faces
            xm = 0.5 * (xfa[1:] + xfa[:-1])
            gf = np.where(xfa < xf, C.g_minus(Qss, P), C.g_plus(Qss, P))
            # p at cell centres from the edge: p_i = 1 + int_{x_i}^L g
            cum = np.concatenate([[0.0], np.cumsum((0.5 * (gf[1:] + gf[:-1]) * h)[::-1])])[::-1]   # int_{x_face_j}^L
            pf = 1.0 + cum                               # at faces
            # step in g at x_f: integrate exactly piecewise by splitting the cell containing x_f
            pn = 0.5 * (pf[1:] + pf[:-1])
            err = np.max(np.abs(pn - p))
            p = 0.5 * p + 0.5 * pn
            if err < 1e-11:
                break
        W = float(np.sum(p) * h)
        rows.append((xf, Cb, W, err))
    return np.array(rows), x


def do_foldd():
    f = cache("foldd")
    if f.exists():
        print("mms_foldd.npz exists"); return
    P = dict(C.PRM)
    xfs = np.linspace(0.30, 0.40, 51)
    sh, _ = sharp_fold_branch(P, N=8000, xfs=xfs)
    k = int(np.argmax(sh[:, 1])); kk = slice(k - 3, k + 4)
    cfit = np.polyfit(sh[kk, 0], sh[kk, 1], 2); xf_f = -cfit[1] / (2 * cfit[0]); C_sh = np.polyval(cfit, xf_f)
    W_sh = np.interp(xf_f, sh[:, 0], sh[:, 2])
    print(f"  sharp fold: C_brem = {C_sh:.6f} at x_f = {xf_f:.4f}, W = {W_sh:.5f} (Picard residual {sh[:, 3].max():.1e})")
    zmu = np.load(cache("mu"))
    kf = int(np.argmax(zmu["C_brem"]))
    p_f, x_f0, W_f0 = zmu["profiles"][kf], zmu["x"], float(zmu["W"][kf])
    rows = []
    for nu, N in ((4e-3, 1000), (2e-3, 1000), (1e-3, 1500), (5e-4, 2000), (2.5e-4, 3000), (1.25e-4, 4000)):
        M = C.Model(N=N); M.P["nu4"] = nu
        p = np.interp(M.x, x_f0, p_f); c = float(zmu["C_brem"][kf])
        p, c, ok = C.newton_bordered(M, p, c, set_cbrem, M.W(p), maxit=200)
        Ws, Cs = [], []
        W0 = M.W(p)
        for dW in np.linspace(-0.08, 0.08, 9):
            pn, cn, ok = C.newton_bordered(M, p, c, set_cbrem, W0 + dW, maxit=200)
            if ok:
                Ws.append(M.W(pn)); Cs.append(cn)
        Ws, Cs = np.array(Ws), np.array(Cs)
        cf2 = np.polyfit(Ws, Cs, 2); Wf = -cf2[1] / (2 * cf2[0]); Cf = np.polyval(cf2, Wf)
        w = 1.4758 * np.sqrt(nu)
        rows.append((nu, N, w, Cf, Wf))
        print(f"  nu4={nu:.2e} N={N} w={w:.4f} (w/h={w * N:.0f}): C_fold = {Cf:.6f} (sharp {C_sh:.6f}, diff {Cf - C_sh:+.2e}), "
              f"W_fold = {Wf:.5f} (sharp {W_sh:.5f}); {len(Ws)} points", flush=True)
        x_f0, p_f = M.x, p
    np.savez(f, rows=np.array(rows), C_sharp=C_sh, W_sharp=W_sh, xf_sharp=xf_f, sharp_branch=sh)


# =============================================================================================== eps
# Density path: n(x, 0) = 1, pellet-like source s_N uniform inside x < X_N, n(L) = 1 held; n_t = D_n (n_xx + s_N).
# The path n(x; s), s = D_n t, is fixed; D_n sets only how fast it is traversed.  Base: C_brem = EPS_CB (mu = 0.49).
EPS_CB = 1.0
X_N = 0.6
A_N = 1.0                                     # s_N amplitude (n_ss(0) = 1 + A_N X_N (L - X_N / 2) ~ 1.42)


def density(M, A=A_N):
    sN = A * (M.x < X_N).astype(float)
    return C.Density(M, sN)


def set_s(dens):
    def f(M, s):
        M.n = dens(s)
    return f


def do_epsstat():
    f = cache("epsstat")
    if f.exists():
        print("mms_epsstat.npz exists"); return
    zmu = np.load(cache("mu"))
    M = model(C_brem=EPS_CB)
    i0 = int(np.argmin(np.abs(zmu["C_brem"] - EPS_CB)))
    p = zmu["profiles"][i0].copy()
    dens = density(M)
    setter = set_s(dens)
    rows, profs = [], []

    def rec(p, s):
        setter(M, s)
        md = C.modes(M, p, k=3)
        dSv, dQv = C.directional(M, p, md["v"])
        r = dict(s=s, W=M.W(p), gamma=md["lam"][0], lam2=md["lam"][1], mu=-md["lam"][0] / dQv,
                 N=float(np.sum(M.n) * M.h), n0=float(M.n[0]), x_mid=M.fronts(p)["mid"],
                 P_alpha=float(np.sum(C.P_alpha(p, M.n, M.P)) * M.h), P_brem=float(np.sum(C.P_brem(p, M.n, M.P)) * M.h))
        rows.append(r); profs.append(p.copy())
        print(f"  s={s:.5f} N={r['N']:.4f} W={r['W']:.5f} gam={r['gamma']:+.4f} lam2={r['lam2']:+.2f} mu={r['mu']:.4f} "
              f"Pa={r['P_alpha']:.3f} Pb={r['P_brem']:.3f} x_mid={r['x_mid']:.3f}", flush=True)
        return r

    s, ds = 0.0, 0.02
    setter(M, s); p, ok, _ = C.newton(M, p); r = rec(p, s)
    # natural continuation in s while the margin is comfortable, then bordered in W through the fold
    while r["mu"] > 0.15 and s < 3.0:
        s_try = s + ds
        setter(M, s_try)
        pn, ok, _ = C.newton(M, p)
        if not ok or C.modes(M, pn, k=1)["lam"][0] > 0:
            ds /= 2
            if ds < 1e-4: break
            continue
        p, s = pn, s_try
        r = rec(p, s)
    if r["mu"] > 0.15:
        print("  no fold along this density path"); return
    W = r["W"]; dW = np.sign(rows[-1]["W"] - rows[-2]["W"]) * 0.01
    passed = 0
    while passed < 5:
        pn, sn, ok = C.newton_bordered(M, p, s, setter, W + dW)
        if not ok:
            dW /= 2
            if abs(dW) < 1e-4: break
            continue
        p, s, W = pn, sn, W + dW
        r = rec(p, s)
        if r["gamma"] > 0:
            passed += 1
    out = {k: np.array([q[k] for q in rows]) for k in rows[0]}
    # fold: maximum of s(W) by a parabola through the three points around it
    k = int(np.argmax(out["s"]))
    kk = slice(max(k - 2, 0), k + 3)
    cf = np.polyfit(out["W"][kk], out["s"][kk], 2)
    W_f = -cf[1] / (2 * cf[0]); s_f = np.polyval(cf, W_f)
    # normal form: gamma^2 = 4 A B (s_f - s) on the stable side near the fold
    st = (out["gamma"] < 0) & (out["mu"] < 0.2)
    AB = -np.polyfit(out["s"][st], out["gamma"][st] ** 2, 1)[0] / 4
    print(f"  fold: s_f = {s_f:.6f}, W_f = {W_f:.5f}; AB = {AB:.5f} (from {st.sum()} points)")
    out.update(s_f=s_f, W_f=W_f, AB=AB, profiles=np.array(profs), lam1=dens.lam1, x=M.x)
    np.savez(f, **out)


EPS_DN = [1.5, 0.75, 0.375, 0.1875, 0.09375, 0.046875, 0.0234375, 0.01171875, 0.005859375, 0.0029296875,
          0.00146484375, 0.000732421875, 0.0003662109375]


def do_epsdyn():
    f = cache("epsdyn")
    z = np.load(cache("epsstat"))
    done = dict(np.load(f)) if f.exists() else {}
    M = model(C_brem=EPS_CB)
    dens = density(M)
    s_f, W_f = float(z["s_f"]), float(z["W_f"])
    nf = (z["gamma"] < 0) & (z["mu"] < 0.12)              # the normal-form region: gamma^2 = 4 AB (s_f - s)
    AB = -np.polyfit(z["s"][nf], z["gamma"][nf] ** 2, 1)[0] / 4
    print(f"  AB = {AB:.4f} from {nf.sum()} points with mu < 0.12 (epsstat: {float(z['AB']):.4f})")
    p0 = z["profiles"][0]
    gam0 = float(z["gamma"][0])
    W_thr = W_f - 0.6                          # past the fold: the jump has happened (cold root W ~ 1.3)
    stab = z["gamma"] < 0
    s_tab, W_tab, g_tab = z["s"][stab], z["W"][stab], z["gamma"][stab]
    o = np.argsort(s_tab); s_tab, W_tab, g_tab = s_tab[o], W_tab[o], g_tab[o]
    for D in EPS_DN:
        tag = f"D{D:.6g}"
        if f"{tag}_t" in done:
            continue
        eps = D * dens.lam1 / abs(gam0)
        T_end = (s_f + 3.0) / D
        ev = lambda t, p: M.W(p) - W_thr
        ev.terminal = True
        t0 = time.time()
        sol = C.integrate(M, p0, T_end, np.linspace(0, T_end, 2001), n_of_t=lambda t: dens(D * t),
                          rtol=1e-8, atol=1e-10, events=ev)
        W = np.array([M.W(q) for q in sol.y.T])
        tj = float(sol.t_events[0][0]) if len(sol.t_events[0]) else np.nan
        s = D * sol.t
        ds_meas = D * tj - s_f
        ds_pred = OMEGA0 * D ** (2 / 3) * AB ** (-1 / 3)
        print(f"  D_n={D:.5g} eps={eps:.4f}: jump at s={D * tj:.5f}, delay {ds_meas:.5f} vs {ds_pred:.5f} "
              f"(ratio {ds_meas / ds_pred:.3f}); {time.time() - t0:.0f} s, {sol.nfev} rhs", flush=True)
        done.update({f"{tag}_t": sol.t, f"{tag}_W": W, f"{tag}_s": s, f"{tag}_tj": tj, f"{tag}_eps": eps,
                     f"{tag}_D": D, f"{tag}_dsm": ds_meas, f"{tag}_dsp": ds_pred})
        np.savez(f, **done)                    # incremental: a long run can be resumed


# =============================================================================================== t3d
A_MINOR = 2.4194
AREA_GRID = np.array([11.647, 33.0372, 53.1469, 71.1147, 85.3053, 96.4534, 106.1902, 116.5375, 123.9938, 129.1894])
GRHO_GRID = np.array([0.6896, 0.6364, 0.6499, 0.7087, 0.7705, 0.8403, 0.9171, 1.0597, 1.2693, 1.4468])
DV10 = A_MINOR ** 3 * AREA_GRID / GRHO_GRID * 0.1       # m^3 per point, scenario F 10-point grid (predict_zeff_ladder.py)
T3D_RUNS = [("hkbm_F_k1_v2", "hkbm_F_k1_v2.npz"), ("cde_v127_F_k1", "cde_v127_F_k1.npz"),
            ("es_frozen_Z0", "endstate/es_hkbm_F_frozen.npz"), ("es_brem_Z3.4", "endstate/es_hkbm_F_frozen_brem.npz"),
            ("es_brem_Z6", "endstate/es_hkbm_F_frozen_brem_z6.npz"), ("es_brem_Z6.5", "endstate/es_hkbm_F_frozen_brem_z6.5.npz"),
            ("es_brem_Z7", "endstate/es_hkbm_F_frozen_brem_z7.npz"), ("es_brem_Z8", "endstate/es_hkbm_F_frozen_brem_z8.npz"),
            ("es_paux0", "endstate/es_hkbm_F_frozen_paux0.npz"), ("es_frozen_nt21", "endstate/es_hkbm_F_frozen_nt21.npz"),
            ("HI_I_EConly", "HI/I_FS_low.npz"), ("HI_H_QRAD", "HI/H_FS_low_QRAD.npz")]


def t3d_one(d, t_min=0.3):
    t, W = d["t"], d["Wtot_MJ"]
    if "Psource_net_MW" in d.files:
        S, Qe = d["Psource_net_MW"], d["Ptransport_MW"]
    else:                                               # the HI extraction (scenarios H, I)
        S, Qe = d["S_net_MW"], d["Q_edge_MW"]
    ne = d["n_e"]
    if "dV_m3" in d.files:
        Ne = ne @ d["dV_m3"]
    elif ne.shape[1] == len(DV10):
        Ne = ne @ DV10                                  # 1e20 electrons
    else:
        Ne = ne.mean(axis=1)                            # 21-point grid: use the mean (a rate is all that is used)
    sel = (t > t_min) & np.isfinite(S) & np.isfinite(Qe)
    if sel.sum() < 20:
        return None
    t, W, S, Qe, Ne = t[sel], W[sel], S[sel], Qe[sel], Ne[sel]
    sS = np.polyfit(W, S, 1)[0]; sQ = np.polyfit(W, Qe, 1)[0]
    # two-variable fit S = a + b W + c ln Ne (fixed-density thermal slope b); same for Q_edge
    X = np.column_stack([np.ones_like(W), W, np.log(Ne)])
    cS, rS, *_ = np.linalg.lstsq(X, S, rcond=None); cQ, rQ, *_ = np.linalg.lstsq(X, Qe, rcond=None)
    cond = np.linalg.cond(X / X.std(axis=0, where=np.ones_like(X, bool)).clip(1e-30) if False else
                          np.column_stack([np.ones_like(W), (W - W.mean()) / W.std(), (np.log(Ne) - np.log(Ne).mean()) / max(np.log(Ne).std(), 1e-12)]))
    gam = sS - sQ
    mu = 1 - sS / sQ
    gam_p = cS[1] - cQ[1]
    mu_p = 1 - cS[1] / cQ[1]
    dlnN = np.polyfit(t, np.log(Ne), 1)[0]
    dlnW = np.polyfit(t, np.log(W), 1)[0]
    # instantaneous fixed point of the amplitude equation dW/dt = gamma (W - W*): W* = W + (S - Q_edge)/|gamma|
    last = t > t[-1] - 0.3
    gap = float(np.mean(S[last] - Qe[last])); Wl = float(np.mean(W[last]))
    out = dict(t_end=float(t[-1]), W_end=float(W[-1]), dSdW=sS, dQdW=sQ, gamma=gam, mu=mu, tau_th=1 / abs(gam),
               dSdW_p=cS[1], dQdW_p=cQ[1], gamma_p=gam_p, mu_p=mu_p, cond=cond, dlnN_dt=dlnN, dlnW_dt=dlnW,
               gap_end=gap, W_inst=np.nan, eps_N=np.nan, eps_W=np.nan, tauE=float(np.mean(W[last] / Qe[last])))
    if gam < 0:
        Wst = Wl + gap / abs(gam)
        out.update(W_inst=Wst, eps_N=dlnN / abs(gam), eps_W=(gap / abs(gam)) / Wst)
    return out


def do_t3d():
    f = cache("t3d")
    if f.exists():
        print("mms_t3d.npz exists"); return
    base = PAPER / "data" / "t3d"
    out = {}
    print(f"{'run':16s} {'t_end':>5s} {'W':>6s} {'dS/dW':>6s} {'dQ/dW':>6s} {'mu':>6s} {'tau_th':>6s} | {'mu(fixed n)':>11s} "
          f"{'cond':>5s} | {'dlnN/dt':>7s} {'eps_N':>6s} {'W*_inst':>7s} {'eps_W':>6s} tau_E")
    for name, rel in T3D_RUNS:
        p = base / rel
        if not p.exists():
            print(f"{name:16s} missing"); continue
        r = t3d_one(np.load(p))
        if r is None:
            print(f"{name:16s} too short"); continue
        print(f"{name:16s} {r['t_end']:5.2f} {r['W_end']:6.0f} {r['dSdW']:6.3f} {r['dQdW']:6.3f} {r['mu']:6.3f} {r['tau_th']:6.2f} | "
              f"{r['mu_p']:11.3f} {r['cond']:5.1f} | {r['dlnN_dt']:7.4f} {r['eps_N']:6.3f} {r['W_inst']:7.0f} {r['eps_W']:6.3f} {r['tauE']:.2f}")
        for k, v in r.items():
            out[f"{name}__{k}"] = v
    # scenario I (EC only, burning, 5 radial points) if the power keys exist
    hi = base / "HI" / "I_FS_low.npz"
    if hi.exists():
        d = np.load(hi)
        print("HI/I_FS_low keys with power:", [k for k in d.files if "MW" in k and "Q_" not in k])
    np.savez(f, **out)


# =============================================================================================== tables
def slope(x, y):
    x, y = np.log(np.asarray(x)), np.log(np.abs(np.asarray(y)))
    return float(np.polyfit(x, y, 1)[0])


def eps_lag(zs, zd, s_tests=(0.15, 0.25)):
    """Slow-manifold test: W(t) - W*(s) against the O(eps) correction D (dW*/ds) / gamma(s)."""
    st = zs["gamma"] < 0
    s_tab, W_tab, g_tab = zs["s"][st], zs["W"][st], zs["gamma"][st]
    o = np.argsort(s_tab); s_tab, W_tab, g_tab = s_tab[o], W_tab[o], g_tab[o]
    from scipy.interpolate import CubicSpline
    Ws = CubicSpline(s_tab, W_tab); gs = CubicSpline(s_tab, g_tab)
    Ds = sorted({float(zd[k]) for k in zd.files if k.endswith("_D")}, reverse=True)
    out = []
    for D in Ds:
        tag = f"D{D:.6g}"
        s, W = zd[f"{tag}_s"], zd[f"{tag}_W"]
        eps = float(zd[f"{tag}_eps"])
        for s0 in s_tests:
            Wd = float(np.interp(s0, s, W))
            lag = Wd - float(Ws(s0))
            lag1 = D * float(Ws(s0, 1)) / float(gs(s0))
            out.append((D, eps, s0, lag, lag1, lag - lag1))
    return np.array(out)


def do_tables():
    lines = []
    P = lambda *a: lines.append(" ".join(str(x) for x in a))
    # ---------------- mu
    z = np.load(cache("mu")); h = 1.0 / NGRID
    st = z["gamma"] < 0
    P("== mu: thermal rate along C_brem -> fold (N = %d, nu4 = %g, sig = %g) ==" % (NGRID, NU4, C.PRM["sig"]))
    P("C_brem  W  mu  gamma  lam2  |gamma/lam2|  g_int/g  Pi(proj)  |g_ray/g-1|  g_uni  angle(t,v)")
    for i in np.where(st)[0]:
        P(f"{z['C_brem'][i]:.5f} {z['W'][i]:.4f} {z['mu'][i]:.4f} {z['gamma'][i]:+.4f} {z['lam2'][i]:+.2f} "
          f"{abs(z['gamma'][i] / z['lam2'][i]):.4f} {z['gamma_int'][i] / z['gamma'][i]:.4f} {z['proj_ratio'][i] * h:.4f} "
          f"{abs(z['gamma_ray'][i] / z['gamma'][i] - 1):.2e} {z['gamma_uni'][i]:+.3f} {z['tv_angle'][i]:.2e}")
    near = st & (z["mu"] < 0.3)
    P(f"slopes (mu < 0.3, {near.sum()} pts): |gamma/lam2| ~ mu^{slope(z['mu'][near], z['gamma'][near] / z['lam2'][near]):.2f}; "
      f"Rayleigh rel. error ~ mu^{slope(z['mu'][near], z['gamma_ray'][near] / z['gamma'][near] - 1):.2f}; "
      f"W-balance-on-family rel. error ~ mu^{slope(z['mu'][near], z['gamma_int'][near] / z['gamma'][near] - 1):.2f} "
      f"(-> Pi - 1 = {z['proj_ratio'][near][-1] * h - 1:.3f}); uniform-rescaling rel. error ~ "
      f"mu^{slope(z['mu'][near], z['gamma_uni'][near] / z['gamma'][near] - 1):.2f}")
    kf = int(np.argmax(z["C_brem"]))
    P(f"fold (continuation): C_brem = {z['C_brem'][kf]:.5f}, W = {z['W'][kf]:.4f}")
    za = np.load(cache("mualpha")); sa = za["gamma"] < 0
    P("\n== mu via C_alpha (C_brem = 1.0): fold at C_alpha = %.4e ==" % za["C_alpha"][np.argmin(za["C_alpha"])])
    P("C_alpha  W  mu  gamma  lam2  g_int/g  Pi  |g_ray/g-1|  g_uni")
    for i in np.where(sa)[0]:
        P(f"{za['C_alpha'][i]:.5e} {za['W'][i]:.4f} {za['mu'][i]:.4f} {za['gamma'][i]:+.4f} {za['lam2'][i]:+.2f} "
          f"{za['gamma_int'][i] / za['gamma'][i]:.4f} {za['proj_ratio'][i] * h:.4f} {abs(za['gamma_ray'][i] / za['gamma'][i] - 1):.2e} {za['gamma_uni'][i]:+.3f}")
    na = sa & (za["mu"] < 0.3)
    P(f"slopes (mu < 0.3, {na.sum()} pts): |gamma/lam2| ~ mu^{slope(za['mu'][na], za['gamma'][na] / za['lam2'][na]):.2f}; "
      f"Rayleigh rel. error ~ mu^{slope(za['mu'][na], za['gamma_ray'][na] / za['gamma'][na] - 1):.2f} "
      f"(mu < 0.15: ^{slope(za['mu'][sa & (za['mu'] < 0.15)], za['gamma_ray'][sa & (za['mu'] < 0.15)] / za['gamma'][sa & (za['mu'] < 0.15)] - 1):.2f}); "
      f"W-balance-on-family rel. error -> Pi - 1 = {za['proj_ratio'][na][-1] * h - 1:.3f}")
    zt = np.load(cache("mutraj"))
    P("\n== mutraj: perturbation runs (2e-3 uniform rescaling of p - p_ped) ==")
    P("mu  gamma(eig)  gamma(fit, late)  dS/dW traj, eig  dQ/dW traj, eig  dW0  A_fit  A_adjoint  A_fit/dW0")
    for j in range(int(zt["n"])):
        g = lambda k: float(zt[f"r{j}_{k}"])
        P(f"{g('mu'):.3f} {g('gamma'):+.4f} {g('gfit'):+.4f} {g('sS'):.4f},{g('dSv'):.4f} {g('sQ'):.4f},{g('dQv'):.4f} "
          f"{g('dW0'):.3e} {g('A0'):.3e} {g('A_pred'):.3e} {g('A0') / g('dW0'):.3f}")
    # ---------------- delta
    zd = np.load(cache("delta"))
    P("\n== delta: steady front, fixed source 7.56, no alpha (finest grid per nu4) ==")
    P(f"inner solution: xi_foot = {float(zd['het_xi_foot']):+.4f}, xi_mid = {float(zd['het_xi_mid']):+.4f}, "
      f"width = {float(zd['het_width']):.4f} sqrt(nu4); sharp x_M = {float(zd['x_M']):.6f}, W = {float(zd['W_sharp']):.6f}")
    nus = sorted(set(zd["nu4"]))
    rows = []
    for nu in nus:
        k = np.where(zd["nu4"] == nu)[0]; k = k[np.argmax(zd["N"][k])]
        xM = float(zd["x_M"]); sq = np.sqrt(nu)
        rows.append((zd["delta"][k], zd["x_G"][k] - xM, zd["x_foot"][k] - xM, zd["x_foot"][k] - xM - sq * float(zd["het_xi_foot"]),
                     zd["x_mid"][k] - xM - sq * float(zd["het_xi_mid"]), zd["W"][k] - float(zd["W_sharp"]), zd["N"][k]))
    rows = np.array(rows)
    P("delta  x_G-x_M  x_foot-x_M  x_foot-x_M-sqrt(nu4)xi_foot  x_mid-x_M-sqrt(nu4)xi_mid  W-W_sharp  N")
    for r in rows:
        P(" ".join(f"{v:+.3e}" for v in r[:-1]), int(r[-1]))
    sm = rows[:, 0] < 0.07
    P(f"slopes in delta (delta < 0.07): x_G {slope(rows[sm, 0], rows[sm, 1]):.2f}; x_foot {slope(rows[sm, 0], rows[sm, 2]):.2f}; "
      f"x_foot after the O(delta) inner offset {slope(rows[sm, 0], rows[sm, 3]):.2f}; x_mid after offset "
      f"{slope(rows[sm, 0], rows[sm, 4]):.2f}; W {slope(rows[sm, 0], rows[sm, 5]):.2f}")
    zf = np.load(cache("foldd"))
    P("\n== fold location against delta (C_brem fold; sharp-interface fold with self-consistent sources) ==")
    P(f"sharp: C_fold = {float(zf['C_sharp']):.6f}, W_fold = {float(zf['W_sharp']):.5f}, x_f = {float(zf['xf_sharp']):.4f}")
    fr = zf["rows"]
    for r in fr:
        P(f"nu4={r[0]:.2e} N={int(r[1])} delta={r[2]:.4f} C_fold={r[3]:.6f} diff={r[3] - float(zf['C_sharp']):+.3e} W_fold={r[4]:.5f}")
    P(f"slope of |C_fold - C_sharp| in delta: {slope(fr[:, 2], fr[:, 3] - float(zf['C_sharp'])):.2f}")
    # ---------------- eps
    zs, zy = np.load(cache("epsstat")), np.load(cache("epsdyn"))
    P("\n== eps: density path, fold at s_f = %.6f (W_f = %.5f); AB from gamma^2 = 4 AB (s_f - s) ==" % (float(zs["s_f"]), float(zs["W_f"])))
    Ds = sorted({float(zy[k]) for k in zy.files if k.endswith("_D")}, reverse=True)
    er = []
    for D in Ds:
        tag = f"D{D:.6g}"
        er.append((float(zy[f"{tag}_eps"]), D, float(zy[f"{tag}_dsm"]), float(zy[f"{tag}_dsp"])))
    er = np.array(er)
    P("eps  D_n  delay_meas(s)  delay_Airy(s)  ratio  meas-Airy")
    for r in er:
        P(f"{r[0]:.4f} {r[1]:.5g} {r[2]:.5f} {r[3]:.5f} {r[2] / r[3]:.3f} {r[2] - r[3]:+.2e}")
    sm = er[:, 0] < 0.06
    P(f"slopes (eps < 0.06): delay ~ eps^{slope(er[sm, 0], er[sm, 2]):.3f} (theory 2/3); |meas - Airy| ~ eps^{slope(er[sm, 0], er[sm, 2] - er[sm, 3]):.2f}"
      f"; |meas - Airy| / (eps ln(1/eps)) = " + ", ".join(f"{abs(a - b) / (e * np.log(1 / e)):.3f}" for e, _, a, b in er[sm]))
    lag = eps_lag(zs, zy)
    P("eps  s  W-W*(s) (leading-order error)  O(eps) prediction D W*'/gamma  remainder")
    for r in lag:
        P(f"{r[1]:.4f} {r[2]:.2f} {r[3]:+.3e} {r[4]:+.3e} {r[5]:+.3e}")
    for s0 in sorted(set(lag[:, 2])):
        q = (lag[:, 2] == s0) & (lag[:, 1] < 0.06)
        P(f"s = {s0}: leading-order error ~ eps^{slope(lag[q, 1], lag[q, 3]):.2f}, after the O(eps) term ~ eps^{slope(lag[q, 1], lag[q, 5]):.2f}")
    # ---------------- t3d
    zt3 = np.load(cache("t3d"))
    P("\n== T3D: margin and ratio from the runs (t > 0.3 s) ==")
    names = sorted({k.split("__")[0] for k in zt3.files})
    P("run  t_end  W  dS/dW  dQ/dW  mu_traj  tau_th  mu_fixed-n(cond)  dlnN/dt  eps_N  eps_W  W*_inst  tau_E")
    for nm in names:
        g = lambda k: float(zt3[f"{nm}__{k}"])
        P(f"{nm} {g('t_end'):.2f} {g('W_end'):.0f} {g('dSdW'):.3f} {g('dQdW'):.3f} {g('mu'):.3f} {g('tau_th'):.2f} "
          f"{g('mu_p'):.2f}({g('cond'):.0f}) {g('dlnN_dt'):.4f} {g('eps_N'):.3f} {g('eps_W'):.3f} {g('W_inst'):.0f} {g('tauE'):.2f}")
    txt = "\n".join(lines)
    (HERE / "mms_tables.txt").write_text(txt + "\n")
    print(txt)


# =============================================================================================== figs (quick look)
def do_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.special import airy
    FIGS.mkdir(exist_ok=True)
    plt.rcdefaults(); plt.rcParams.update({"font.size": 9})
    # 1. thermal mode (right = left: self-adjoint) and the steady-family tangent
    z = np.load(cache("mu"))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    st = np.where(z["gamma"] < 0)[0]
    for a_, i in zip(ax, (st[0], st[-1])):
        a_.plot(z["x"], z["v_vec"][i], label="right eigvec v")
        a_.plot(z["x"], z["u_vec"][i] * np.sum(z["v_vec"][i] ** 2) / np.sum(z["u_vec"][i] * z["v_vec"][i]), "--", label="left eigvec u (scaled)")
        a_.plot(z["x"], z["t_vec"][i], ":", label="steady-family tangent t")
        a_.set_title(f"C_brem={z['C_brem'][i]:.4f}, mu={z['mu'][i]:.3f}, gamma={z['gamma'][i]:+.3f}")
        a_.set_xlabel("x"); a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "thermal_mode.png", dpi=130); plt.close(fig)
    # 2. fold passage in inner variables against the Riccati solution u = -Ai'(-tau)/Ai(-tau)
    zs, zy = np.load(cache("epsstat")), np.load(cache("epsdyn"))
    s_f, W_f = float(zs["s_f"]), float(zs["W_f"])
    nf = (zs["gamma"] < 0) & (zs["mu"] < 0.12)
    AB = -np.polyfit(zs["s"][nf], zs["gamma"][nf] ** 2, 1)[0] / 4
    near = np.abs(zs["W"] - W_f) < 0.06
    A = -np.polyfit(zs["W"][near], zs["gamma"][near], 1)[0] / 2      # gamma = -2 A (W - W_f)
    B = AB / A
    fig, ax = plt.subplots(figsize=(5, 3.4))
    tau = np.linspace(-6, 2.3, 400)
    ai, aip, _, _ = airy(-tau)
    ax.plot(tau, -aip / ai, "k", lw=2, label="Riccati: -Ai'(-tau)/Ai(-tau)")
    for k in zy.files:
        if not k.endswith("_D"):
            continue
        tg = k[:-2]; D = float(zy[k]); eps = float(zy[f"{tg}_eps"])
        if eps > 0.06:
            continue
        c = (A * B * D) ** (-1 / 3); kk = 1 / (A * c)
        t = zy[f"{tg}_t"]; tt = (t - s_f / D) / c
        ax.plot(tt, (zy[f"{tg}_W"] - W_f) / kk, lw=1, label=f"eps={eps:.3g}")
    ax.set_xlim(-6, 3); ax.set_ylim(-4, 3); ax.axvline(OMEGA0, color="grey", ls=":")
    ax.set_xlabel("tau = (t - t_f)(ABr)^{1/3}"); ax.set_ylabel("u = (W - W_f)/k"); ax.legend(fontsize=6)
    ax.set_title(f"A={A:.3f}, B={B:.3f}, AB={AB:.3f}")
    fig.tight_layout(); fig.savefig(FIGS / "fold_inner.png", dpi=130); plt.close(fig)
    # 3. T3D: W(t) and the instantaneous fixed point W + (S - Q_edge)/|gamma|
    base = PAPER / "data" / "t3d"
    zt = np.load(cache("t3d"))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for nm, rel, col in (("hkbm_F_k1_v2", "hkbm_F_k1_v2.npz", "C0"), ("es_frozen_Z0", "endstate/es_hkbm_F_frozen.npz", "C1"),
                         ("es_brem_Z6", "endstate/es_hkbm_F_frozen_brem_z6.npz", "C3"), ("cde_v127_F_k1", "cde_v127_F_k1.npz", "C2")):
        d = np.load(base / rel)
        g = abs(float(zt[f"{nm}__gamma"]))
        t = d["t"]; m = t > 0.3
        Wi = d["Wtot_MJ"] + (d["Psource_net_MW"] - d["Ptransport_MW"]) / g
        ax.plot(t, d["Wtot_MJ"], color=col, label=nm)
        ax.plot(t[m], np.convolve(Wi[m], np.ones(15) / 15, mode="same"), color=col, ls="--", lw=0.8)
    ax.set_xlabel("t [s]"); ax.set_ylabel("W [MJ]  (dashed: W + (S - Q_edge)/|gamma_th|)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "t3d_inst_fixed_point.png", dpi=130); plt.close(fig)
    print("wrote", ", ".join(p.name for p in sorted(FIGS.glob("*.png"))), f"(A = {A:.4f}, B = {B:.4f})")


# =============================================================================================== main
STAGES = dict(hot=do_hot, mu=do_mu, mutraj=do_mutraj, delta=do_delta, epsstat=do_epsstat, epsdyn=do_epsdyn, t3d=do_t3d, foldd=do_foldd, tables=do_tables, figs=do_figs, mualpha=do_mualpha)

if __name__ == "__main__":
    for st in sys.argv[1:]:
        STAGES[st]()
