"""The thermal fold, the hysteresis window and the edge-flux response of the reduced model with a radiation sink.

Three questions (see NOTES_fold.md), all in the paper's reduced model with its production parameters
(reduced.PAPER with nu4_conservative=True, C_alpha = 2e-5, T_ref = 3 keV, int S_ext = 4.94: the ignition / radiation
scan of Section 5, scripts/make_brem_scan.py):

  1. fold:   does the burn end at a thermal fold (saddle-node of the W balance) or at the transport threshold S = Q_M?
  2. edge:   does moving a fraction f of the external source to x in [0.75, 0.85] open a hysteresis window in C_brem?
  3. step:   how much of a sink comes out of the edge flux, quasi-statically and in the true steady state?

Stages (``python3 exp_fold.py <stage>``; every stage caches to an .npz and is skipped if the cache exists):
  scan    fine quasi-static continuation of the burning state in C_brem (up) and of the collapsed state (down), f = 0,
          with the thermal rate from the Jacobian of the discretised model at every fixed point   -> fold_scan.npz
  newton  Newton continuation of the discretised steady state parametrised by W through the fold (C_brem is the
          unknown), giving the unstable middle branch and the fold point                           -> fold_newton.npz
  sharp   the sharp-interface (Maxwell-front) branch with self-consistent sources, continued in the front position,
          with the branch slopes of the quasi-steady (Stefan) family                               -> fold_sharp.npz
  edge    the C_brem scan up and down for f = 0 ... 1 (core -> edge deposition)                    -> fold_edge.npz
  highS   supplementary: the same with int S_ext = 7 > Q_M, where the H branch exists without the burn, to show the
          thermal (T3D-like) hysteresis on a single transport branch                               -> fold_highS.npz
  step    the time response of Q_edge and W to a step of the sink at the burning state             -> fold_step.npz
          (its 0.8 -> 1.0 case started on the cold branch; see extra)
  sharp_edge  the sharp-interface fold C_fold(f) for f = 0 ... 0.75                               -> fold_sharp_edge.npz
  extra   the f at which the burn is lost at C_brem = 0.01 (bisection), and sink steps on the barrier branch at
          C_brem = 0.8 and 1.2 (its part (b) is superseded by highS_core)                         -> fold_extra.npz
  highS_core  int S_ext = 7, f = 0.57: bisect the core-burn loss and the end of the edge barrier  -> fold_highS_core.npz
  figs    quick-look PNGs in fold_figs/
  tables  print the tables that NOTES_fold.md quotes
  all     everything in order

Conventions.  x = reduced.grid() (200 points, dx = 1/199), the paper's discretisation (three binomial passes on the
face gradients, conservative hyperviscosity nu4 = 5e-4, Neumann axis, p(L) = 1).  The drive S is the *injected* drive,
the interior-cell sum dx sum_{i=1}^{N-2} [S_ext + P_alpha - P_brem]_i, which equals Q_edge (the flux through the last
face) exactly in a steady state; the paper's trapezoid diagnostic over-counts it by the axis half-cell (0.099 at the
burning state, 0.056 of it external: the injected external drive is 4.888, not 4.944).  W is the paper's trapezoid.
The sink is the model's own closure, P_brem = C_brem n^2 Z_eff sqrt(T) with n = 1 fixed and T = T_ref p, i.e.
C_brem sqrt(3 p): the n^2 sqrt(T) form of T3D's bremsstrahlung (p^2 / T^{3/2} if written in p and T at fixed n = p/T).
"""
import pathlib
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse import diags

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import reduced as R                        # noqa: E402
import flux_transport_model_clean as m     # noqa: E402
from maxwell import maxwell, Q as Qfun, landmarks   # noqa: E402

FIGS = HERE / "fold_figs"
x = R.grid()
dx = x[1] - x[0]
N = len(x)
L = R.L
P_PED = R.P_PED
S_EXT_TRAP = 4.944392605799654          # the paper's int S_ext (trapezoid), make_brem_scan.S_EXT_SCALE * int base_source
BASE = dict(R.PAPER, nu4_conservative=True, C_alpha=2.0e-5, T_ref_keV=3.0, C_brem=0.01)
MX = maxwell()                           # Q_M = 5.85, g_a = 0.71, g_b = 6.37
LM = landmarks()
Q_M, G_A, G_B = MX["Q_M"], MX["g_a"], MX["g_b"]
G_STAR = BASE["g_c"] / 2
G_MID = 0.5 * (G_A + G_B)
CHI_PED = BASE["chi_RR"] + BASE["chi_MHD"]


# ----------------------------------------------------------------------------------------------- sources and drive
def source_profile(f=0.0, S_tot=S_EXT_TRAP, x_lo=0.75, x_hi=0.85):
    """External source: (1 - f) of S_tot in the paper's core Gaussian, f of it in a top hat on [x_lo, x_hi].
    Each part is normalised on the trapezoid rule (the paper's convention), so f = 0 reproduces the paper exactly."""
    core = R.base_source(x)
    core = core / np.trapezoid(core, x)
    edge = ((x >= x_lo) & (x <= x_hi)).astype(float)
    edge = edge / np.trapezoid(edge, x)
    return S_tot * ((1.0 - f) * core + f * edge)


def phys(p, prm):
    return m.alpha_heating(p, prm) - m.bremsstrahlung(p, prm)


def dphys_dp(p, prm, eps=1e-6):
    return (phys(p * (1 + eps), prm) - phys(p * (1 - eps), prm)) / (2 * eps * p)


def injected(arr):
    """Interior-cell sum: what the scheme actually injects (the BC cells are held)."""
    return float(np.sum(arr[1:-1]) * dx)


def Qss_faces(p, prm, S_ext):
    """Cumulative injected drive at the faces, Q_ss(x_{k+1/2}) = dx sum_{i<=k} S_i, k = 0..N-2 (face k between cells k, k+1)."""
    s = S_ext + phys(p, prm)
    s = s.copy(); s[0] = 0.0; s[-1] = 0.0
    return np.cumsum(s[:-1]) * dx


def face_g(p):
    pb = p.copy(); pb[0] = pb[1]; pb[-1] = P_PED
    return -(pb[1:] - pb[:-1]) / dx


def diag(p, prm, S_ext):
    """Everything NOTES_fold.md tabulates for one profile."""
    Qf, xf = m.compute_face_flux(p, dx, x, P_PED, prm)
    g = face_g(p)
    ph = phys(p, prm)
    Pa = m.alpha_heating(p, prm); Pb = m.bremsstrahlung(p, prm)
    Qss = Qss_faces(p, prm, S_ext)
    sup = np.where(g > G_STAR)[0]
    x_f = xf[sup[0]] if len(sup) else np.nan
    mid = np.where(g > G_MID)[0]
    x_mid = xf[mid[0]] if len(mid) else np.nan
    # the sink and heating inside / outside the front (by the g > g_star foot)
    inside = x < (x_f if np.isfinite(x_f) else L)
    S_loc = (S_ext + ph)
    k = int(np.argmax(Qss))
    d = dict(W=float(np.trapezoid(p, x)), S=injected(S_ext + ph), S_ext=injected(S_ext),
             P_alpha=injected(Pa), P_brem=injected(Pb), Q_edge=float(Qf[-1]),
             x_f=float(x_f), x_mid=float(x_mid), g_max=float(g.max()), g_edge=float(g[-1]),
             p_core=float(p[1]), T_core_keV=float(p[1] * prm["T_ref_keV"]),
             Qss_max=float(Qss[k]), x_Qss_max=float(xf[k]), Qss_L=float(Qss[-1]),
             P_brem_in=float(np.sum(Pb[1:-1][inside[1:-1]]) * dx), P_brem_out=float(np.sum(Pb[1:-1][~inside[1:-1]]) * dx),
             P_alpha_in=float(np.sum(Pa[1:-1][inside[1:-1]]) * dx),
             S_at_front=float(np.interp(x_f, x, S_loc)) if np.isfinite(x_f) else np.nan,
             p_min=float(p.min()))
    return d


DIAG_KEYS = ["W", "S", "S_ext", "P_alpha", "P_brem", "Q_edge", "x_f", "x_mid", "g_max", "g_edge", "p_core", "T_core_keV",
             "Qss_max", "x_Qss_max", "Qss_L", "P_brem_in", "P_brem_out", "P_alpha_in", "S_at_front", "p_min"]


# ----------------------------------------------------------------------------------------------- time stepping
def integrate(prm, S_ext, p0, T_end, n_out=2, rtol=1e-7, atol=1e-9):
    rhs = R.make_rhs(x, prm, S_ext)
    bw = 2 + prm.get("n_smooth", 0)
    sp = diags([np.ones(N - k) for k in range(-bw, bw + 1)], list(range(-bw, bw + 1)), shape=(N, N))
    p0 = p0.copy(); p0[0] = p0[1]; p0[-1] = P_PED
    sol = solve_ivp(rhs, (0.0, T_end), p0, method="BDF", t_eval=np.linspace(0, T_end, n_out), jac_sparsity=sp,
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    P = sol.y.T.copy(); P[:, 0] = P[:, 1]; P[:, -1] = P_PED
    return sol.t, P


def to_steady(prm, S_ext, p0, T_chunk=6.0, T_max=60.0, tol=2e-5, verbose=False):
    """Integrate in chunks until |dW/dt| < tol and |S - Q_edge| < 10 tol (certified steady), or T_max.
    Returns (p, diag, info).  Near a fold the thermal rate is small and this takes long; the caller reads info['steady']."""
    p = p0.copy(); t_tot = 0.0
    while True:
        t, P = integrate(prm, S_ext, p, T_chunk, n_out=4)
        p = P[-1]; t_tot += T_chunk
        W = np.trapezoid(P, x, axis=1)
        dWdt = (W[-1] - W[-2]) / (t[-1] - t[-2])
        d = diag(p, prm, S_ext)
        gap = d["S"] - d["Q_edge"]
        if verbose:
            print(f"      t={t_tot:5.1f} W={d['W']:.5f} dW/dt={dWdt:+.2e} S-Qe={gap:+.2e} x_f={d['x_f']:.3f}", flush=True)
        if abs(dWdt) < tol and abs(gap) < 10 * tol:
            return p, d, dict(t=t_tot, dWdt=dWdt, steady=True)
        if t_tot >= T_max:
            return p, d, dict(t=t_tot, dWdt=dWdt, steady=False)


def initial_profile():
    return R.initial_profile(x, 1.8, BASE)


# ----------------------------------------------------------------------------------------------- Jacobian / thermal mode
def jacobian(prm, S_ext, p, eps=1e-6):
    """Finite-difference Jacobian of the interior RHS, d rhs_i / d p_j for i, j = 1..N-2 (central differences)."""
    rhs = R.make_rhs(x, prm, S_ext)
    n = N - 2
    J = np.zeros((n, n))
    for j in range(1, N - 1):
        h = eps * max(1.0, abs(p[j]))
        pp = p.copy(); pp[j] += h
        pm = p.copy(); pm[j] -= h
        J[:, j - 1] = (rhs(0.0, pp)[1:-1] - rhs(0.0, pm)[1:-1]) / (2 * h)
    return J


def thermal_mode(prm, S_ext, p, J=None):
    """Leading eigenvalue of the Jacobian (the thermal rate gamma_th) and the split of gamma into the heating slope
    dS/dW and the loss slope dQ_edge/dW along its eigenvector (normalised to unit change of W).  Also the paper's
    uniform-rescaling proxies.  The split is exact: dx sum_i (J v)_i = delta S - delta Q_edge for any v."""
    if J is None:
        J = jacobian(prm, S_ext, p)
    lam, V = np.linalg.eig(J)
    k = int(np.argmax(lam.real))
    gam = float(lam[k].real)
    v = V[:, k].real
    v = v / (np.sum(v) * dx)                      # delta W_int = 1
    vf = np.zeros(N); vf[1:-1] = v; vf[0] = vf[1]
    dSdW = float(np.sum((dphys_dp(p, prm) * vf)[1:-1]) * dx)
    h = 1e-4
    Qe = lambda pp: m.compute_face_flux(pp, dx, x, P_PED, prm)[0][-1]
    dQdW = float((Qe(p + h * vf) - Qe(p - h * vf)) / (2 * h))
    # uniform rescaling (the paper's proxy, exp_thermal.py): S(lambda p), Q_edge(lambda p)
    W = np.trapezoid(p, x); e = 0.01
    pu, pd = p * (1 + e), p * (1 - e); pu[-1] = pd[-1] = P_PED
    dSdW_u = (injected(S_ext + phys(pu, prm)) - injected(S_ext + phys(pd, prm))) / (2 * e * W)
    dQdW_u = (Qe(pu) - Qe(pd)) / (2 * e * W)
    # second eigenvalue, to show the gap to the next mode
    lam2 = float(np.sort(lam.real)[-2])
    return dict(gamma=gam, dSdW=dSdW, dQdW=dQdW, gamma_check=dSdW - dQdW, dSdW_uni=dSdW_u, dQdW_uni=dQdW_u,
                lam2=lam2, v=vf)


# ----------------------------------------------------------------------------------------------- stage: scan (f = 0)
CB_UP = ([0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 2.0])
CB_DN = [2.0, 1.6, 1.5, 1.4, 1.3, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01, 0.0]


def scan_branch(prm0, S_ext, cb_list, p0, label, bisect=True, cb_tol=0.005, want_mode=True, verbose=True):
    """Quasi-static continuation: run each C_brem to a steady state from the previous one.  If the burn is lost between
    two rungs, bisect the end to cb_tol (keeping the last burning state as the start)."""
    rows, modes = [], []
    p = p0.copy()
    burning = None
    last_burn = None
    for cb in cb_list:
        prm = dict(prm0, C_brem=cb)
        p, d, info = to_steady(prm, S_ext, p)
        mode = thermal_mode(prm, S_ext, p) if want_mode else dict(gamma=np.nan, dSdW=np.nan, dQdW=np.nan, dSdW_uni=np.nan, dQdW_uni=np.nan, lam2=np.nan)
        is_burn = d["x_f"] < 0.95 and np.isfinite(d["x_f"]) and d["W"] > 2.0
        rows.append(dict(C_brem=cb, steady=info["steady"], t=info["t"], dWdt=info["dWdt"], burn=is_burn, **d,
                         **{k: mode[k] for k in ("gamma", "dSdW", "dQdW", "dSdW_uni", "dQdW_uni", "lam2")}))
        if verbose:
            print(f"  {label:4s} C_brem={cb:6.3f} {'burn' if is_burn else 'cold'} W={d['W']:.4f} S={d['S']:.4f} Qe={d['Q_edge']:.4f} "
                  f"Pa={d['P_alpha']:.3f} Pb={d['P_brem']:.3f} x_f={d['x_f']:.3f} Qss_max={d['Qss_max']:.3f}@{d['x_Qss_max']:.2f} "
                  f"gam={mode['gamma']:+.3f} dS/dW={mode['dSdW']:.2f} dQ/dW={mode['dQdW']:.2f} "
                  f"[{'steady' if info['steady'] else 'NOT steady'} t={info['t']:.0f}]", flush=True)
        if is_burn:
            last_burn = (cb, p.copy())
        if burning is None:
            burning = is_burn
        elif bisect and burning and not is_burn and last_burn is not None:
            # bisect between last_burn[0] and cb
            lo, hi = last_burn[0], cb
            p_lo = last_burn[1]
            while hi - lo > cb_tol:
                mid = 0.5 * (lo + hi)
                prm = dict(prm0, C_brem=mid)
                pm_, dm, im = to_steady(prm, S_ext, p_lo, T_max=120.0)
                modem = thermal_mode(prm, S_ext, pm_) if want_mode else dict(gamma=np.nan, dSdW=np.nan, dQdW=np.nan, dSdW_uni=np.nan, dQdW_uni=np.nan, lam2=np.nan)
                isb = dm["x_f"] < 0.95 and np.isfinite(dm["x_f"]) and dm["W"] > 2.0
                rows.append(dict(C_brem=mid, steady=im["steady"], t=im["t"], dWdt=im["dWdt"], burn=isb, **dm,
                                 **{k: modem[k] for k in ("gamma", "dSdW", "dQdW", "dSdW_uni", "dQdW_uni", "lam2")}))
                if verbose:
                    print(f"  {label:4s} C_brem={mid:6.4f} {'burn' if isb else 'cold'} W={dm['W']:.4f} S={dm['S']:.4f} Qe={dm['Q_edge']:.4f} "
                          f"Pa={dm['P_alpha']:.3f} Pb={dm['P_brem']:.3f} x_f={dm['x_f']:.3f} Qss_max={dm['Qss_max']:.3f}@{dm['x_Qss_max']:.2f} "
                          f"gam={modem['gamma']:+.3f} dS/dW={modem['dSdW']:.2f} dQ/dW={modem['dQdW']:.2f} "
                          f"[{'steady' if im['steady'] else 'NOT steady'} t={im['t']:.0f}]  (bisection)", flush=True)
                if isb:
                    lo, p_lo = mid, pm_.copy(); last_burn = (mid, pm_.copy())
                else:
                    hi = mid
            burning = False
            bisect = False
    rows.sort(key=lambda r: r["C_brem"])
    return rows, last_burn


def rows_to_npz(rows, prefix):
    out = {}
    keys = [k for k in rows[0].keys()]
    for k in keys:
        out[f"{prefix}{k}"] = np.array([r[k] for r in rows], dtype=float)
    return out


def do_scan():
    cache = HERE / "fold_scan.npz"
    if cache.exists():
        print("fold_scan.npz exists"); return
    S_ext = source_profile(0.0)
    print("scan up from the ignition run (supercritical IC) ...")
    t0 = time.time()
    up, last_burn = scan_branch(BASE, S_ext, CB_UP, initial_profile(), "up")
    print(f"  ... {time.time()-t0:.0f} s; last burning C_brem = {last_burn[0]:.4f}")
    # the trajectory of the loss: start from the last burning state and step C_brem up by 0.05
    cb_from, p_from = last_burn
    cb_to = cb_from + 0.05
    t, P = integrate(dict(BASE, C_brem=cb_to), S_ext, p_from, 6.0, n_out=601)
    traj = {k: [] for k in DIAG_KEYS}
    for pk in P:
        d = diag(pk, dict(BASE, C_brem=cb_to), S_ext)
        for k in DIAG_KEYS: traj[k].append(d[k])
    print("scan down from the collapsed state ...")
    dn, _ = scan_branch(BASE, S_ext, CB_DN, up[-1]["p"] if "p" in up[-1] else P[-1], "down", bisect=False)
    out = dict(**rows_to_npz(up, "up_"), **rows_to_npz(dn, "dn_"), p_last_burn=p_from, cb_last_burn=cb_from,
               traj_t=t, traj_cb_from=cb_from, traj_cb_to=cb_to, traj_P=P, Q_M=Q_M, x=x,
               **{f"traj_{k}": np.array(v) for k, v in traj.items()})
    np.savez(cache, **out)
    print("wrote fold_scan.npz")


# ----------------------------------------------------------------------------------------------- stage: newton (continuation in W)
def steady_residual(prm, S_ext, p):
    return R.make_rhs(x, prm, S_ext)(0.0, p)[1:-1]


def newton_W(prm0, S_ext, p_start, C_start, W_target, n_iter=40, tol=1e-9, verbose=False):
    """Solve rhs(p; C) = 0 (interior) and W_trap(p) = W_target for (p_interior, C) by Newton with a bordered Jacobian.
    Returns (p, C, converged, |res|)."""
    p = p_start.copy(); C = float(C_start)
    wvec = np.full(N, dx); wvec[0] = wvec[-1] = dx / 2            # trapezoid weights
    for it in range(n_iter):
        prm = dict(prm0, C_brem=C)
        res = steady_residual(prm, S_ext, p)
        p_bc = p.copy(); p_bc[0] = p_bc[1]; p_bc[-1] = P_PED
        Wres = float(np.sum(wvec * p_bc)) - W_target
        nrm = max(np.max(np.abs(res)) / 1e3, abs(Wres))           # residual scale: rhs entries are O(1e3) stiff
        if verbose:
            print(f"      it={it} C={C:.5f} |res|={np.max(np.abs(res)):.2e} Wres={Wres:+.2e}")
        if np.max(np.abs(res)) < 1e-6 and abs(Wres) < tol:
            return p_bc, C, True, np.max(np.abs(res))
        J = jacobian(prm, S_ext, p_bc)
        dC = -m.bremsstrahlung(p_bc, dict(prm, C_brem=1.0))[1:-1]   # d rhs / d C_brem
        dW = wvec[1:-1].copy(); dW[0] += wvec[0]                   # p_0 = p_1 carries the axis half-cell
        A = np.zeros((N - 1, N - 1))
        A[:-1, :-1] = J; A[:-1, -1] = dC; A[-1, :-1] = dW
        rhs_vec = -np.concatenate([res, [Wres]])
        try:
            du = np.linalg.solve(A, rhs_vec)
        except np.linalg.LinAlgError:
            return p_bc, C, False, np.max(np.abs(res))
        # damped step (keep C positive, limit the step)
        lam = 1.0
        if C + lam * du[-1] <= 0:
            lam = 0.5 * C / abs(du[-1])
        step = np.max(np.abs(du[:-1]));
        if step > 0.5:
            lam = min(lam, 0.5 / step)
        p[1:-1] += lam * du[:-1]; C += lam * du[-1]
        p[0] = p[1]; p[-1] = P_PED
    return p, C, False, np.max(np.abs(res))


def do_newton():
    cache = HERE / "fold_newton.npz"
    if cache.exists():
        print("fold_newton.npz exists"); return
    z = np.load(HERE / "fold_scan.npz")
    S_ext = source_profile(0.0)
    p = z["p_last_burn"]; C = float(z["cb_last_burn"])
    print(f"Newton continuation in W from the last burning BDF state (C_brem = {C:.4f}, W = {np.trapezoid(p, x):.4f})")
    # polish the starting state at its own W
    W0 = float(np.trapezoid(p, x))
    p, C, ok, r = newton_W(BASE, S_ext, p, C, W0)
    print(f"  polish: converged={ok} C={C:.5f} |res|={r:.1e}")
    rows = []
    def record(p, C):
        prm = dict(BASE, C_brem=C)
        d = diag(p, prm, S_ext)
        mode = thermal_mode(prm, S_ext, p)
        rows.append(dict(C_brem=C, **d, **{k: mode[k] for k in ("gamma", "dSdW", "dQdW", "dSdW_uni", "dQdW_uni", "lam2")}))
        print(f"  W={d['W']:.4f} C_brem={C:.5f} S={d['S']:.4f} Qe={d['Q_edge']:.4f} Pa={d['P_alpha']:.3f} Pb={d['P_brem']:.3f} "
              f"x_f={d['x_f']:.3f} Qss_max={d['Qss_max']:.3f} gam={mode['gamma']:+.4f} dS/dW={mode['dSdW']:.3f} dQ/dW={mode['dQdW']:.3f}", flush=True)
    record(p, C)
    # go up in W first (towards the paper's burning state) a little, then down through the fold
    profiles = {}
    for direction, Ws in (("up", np.arange(W0 + 0.1, 5.9, 0.1)), ("down", np.arange(W0 - 0.05, 1.0, -0.05))):
        pp, CC = p.copy(), C
        for Wt in Ws:
            pn, Cn, ok, r = newton_W(BASE, S_ext, pp, CC, float(Wt))
            if not ok:
                # retry with a half step
                pn, Cn, ok, r = newton_W(BASE, S_ext, pp, CC, float(0.5 * (Wt + np.trapezoid(pp, x))))
                if ok:
                    pn, Cn, ok, r = newton_W(BASE, S_ext, pn, Cn, float(Wt))
            if not ok:
                print(f"  Newton failed at W={Wt:.3f} (|res|={r:.1e}); stopping the {direction} branch", flush=True)
                break
            pp, CC = pn, Cn
            record(pp, CC)
            profiles[f"p_W{Wt:.3f}"] = pp.copy()
    rows.sort(key=lambda r: r["W"])
    out = rows_to_npz(rows, "nw_")
    out.update(profiles); out["x"] = x; out["Q_M"] = Q_M
    np.savez(cache, **out)
    print("wrote fold_newton.npz")


# ----------------------------------------------------------------------------------------------- stage: sharp (Maxwell-front branch)
def g_minus(q):
    q = np.asarray(q, float)
    out = np.zeros_like(q)
    ok = (q > 0) & (q < LM["Qmax"])
    c0, cr, gc = BASE["chi0"], BASE["chi_RR"], BASE["g_c"]
    out[ok] = gc / (2 * c0) * ((c0 + cr) - np.sqrt((c0 + cr) ** 2 - 4 * c0 * q[ok] / gc))
    out[q >= LM["Qmax"]] = np.nan
    return out


def g_plus(q):
    q = np.asarray(q, float)
    out = (q + BASE["chi_MHD"] * BASE["g_MHD"]) / CHI_PED
    out[q < LM["Qmin"]] = np.nan
    return out


def sharp_profile(x_f, prm, S_ext, p_init=None, relax=0.5, n_iter=400, tol=1e-10):
    """Transport-quasi-steady profile with a Maxwell front at x_f and the core lifted uniformly at pdot (the Stefan
    family): mu(x) = Q_ss(x) - pdot min(x, x_f), pdot = (Q_ss(x_f) - Q_M)/x_f, g = g_-(mu) inside, g_+(mu) outside,
    p = p_ped + int_x^L g, with Q_ss from the profile's own sources (Picard iteration).  Returns (p, pdot, info)."""
    p = p_init.copy() if p_init is not None else np.full(N, 2.0)
    core = x < x_f
    for it in range(n_iter):
        s = S_ext + phys(p, prm)
        Qss = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * dx)])     # on the nodes
        Qf_at = float(np.interp(x_f, x, Qss))
        pdot = (Qf_at - Q_M) / x_f
        mu = Qss - pdot * np.minimum(x, x_f)
        g = np.where(core, g_minus(np.maximum(mu, 0.0)), g_plus(mu))
        if np.any(np.isnan(g)):
            return p, pdot, dict(converged=False, reason="branch undefined (mu < Q_min in the pedestal or mu > Q_max in the core)", it=it)
        pn = P_PED + np.concatenate([np.cumsum((g[::-1][1:] + g[::-1][:-1]) * 0.5 * dx)[::-1], [0.0]])
        err = np.max(np.abs(pn - p))
        p = (1 - relax) * p + relax * pn
        if err < tol:
            return p, pdot, dict(converged=True, it=it, mu=mu, Qss=Qss, g=g)
    return p, pdot, dict(converged=False, reason="no convergence", it=it, mu=mu, Qss=Qss, g=g)


def sharp_diag(x_f, prm, S_ext, p_init=None):
    p, pdot, info = sharp_profile(x_f, prm, S_ext, p_init)
    if not info["converged"]:
        return None, p
    s = S_ext + phys(p, prm)
    W = float(np.trapezoid(p, x)); S = float(info["Qss"][-1]); Qe = S - pdot * x_f
    return dict(x_f=x_f, W=W, S=S, Q_edge=Qe, pdot=pdot, dWdt=pdot * x_f, P_alpha=float(np.trapezoid(m.alpha_heating(p, prm), x)),
                P_brem=float(np.trapezoid(m.bremsstrahlung(p, prm), x)), S_at_front=float(np.interp(x_f, x, s)),
                Qss_max=float(np.max(info["Qss"])), p_core=float(p[0]), mu_min_ped=float(np.min(info["mu"][x >= x_f])),
                int_dSdp_core=float(np.trapezoid((dphys_dp(p, prm) * (x < x_f)), x)),
                int_dSdp_ped=float(np.trapezoid((dphys_dp(p, prm) * (x >= x_f)), x))), p


def do_sharp():
    cache = HERE / "fold_sharp.npz"
    if cache.exists():
        print("fold_sharp.npz exists"); return
    S_ext = source_profile(0.0)
    # (i) the fixed-point branch: for each front position find C_brem with pdot = 0
    xfs = np.concatenate([np.arange(0.12, 0.30, 0.01), np.arange(0.30, 0.95, 0.025)])
    rows = []
    p_guess = None
    print("sharp-interface fixed-point branch: C_brem(x_f) with pdot = 0")
    for xf_ in xfs:
        def F(C):
            d, _ = sharp_diag(xf_, dict(BASE, C_brem=C), S_ext, p_guess)
            return np.nan if d is None else d["pdot"]
        # bracket: pdot decreases with C
        Cs = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
        vals = np.array([F(C) for C in Cs])
        okv = np.isfinite(vals)
        if okv.sum() < 2 or not (np.any(vals[okv] > 0) and np.any(vals[okv] < 0)):
            print(f"  x_f={xf_:.3f}: no root (pdot values {np.round(vals, 3)})"); continue
        i = int(np.argmax(okv & (vals < 0)))                      # first negative
        j = i - 1
        while j >= 0 and not okv[j]: j -= 1
        if j < 0 or vals[j] <= 0:
            print(f"  x_f={xf_:.3f}: bracket failed"); continue
        C = brentq(F, Cs[j], Cs[i], xtol=1e-6)
        d, p_guess = sharp_diag(xf_, dict(BASE, C_brem=C), S_ext, p_guess)
        # slopes of the quasi-steady family at fixed C: d/dx_f of W, S, Q_edge, pdot x_f
        h = 0.004
        dp_, _ = sharp_diag(xf_ + h, dict(BASE, C_brem=C), S_ext, p_guess)
        dm_, _ = sharp_diag(xf_ - h, dict(BASE, C_brem=C), S_ext, p_guess)
        if dp_ is not None and dm_ is not None:
            dW = (dp_["W"] - dm_["W"]) / (2 * h)
            d["dSdW"] = (dp_["S"] - dm_["S"]) / (2 * h) / dW
            d["dQdW"] = (dp_["Q_edge"] - dm_["Q_edge"]) / (2 * h) / dW
            d["gamma"] = (dp_["dWdt"] - dm_["dWdt"]) / (2 * h) / dW
            d["dW_dxf"] = dW
            d["lift"] = -(dp_["p_core"] - dm_["p_core"]) / (2 * h)          # core lift per unit inward front displacement
            d["dPa_dxf"] = (dp_["P_alpha"] - dm_["P_alpha"]) / (2 * h)
            d["dPb_dxf"] = (dp_["P_brem"] - dm_["P_brem"]) / (2 * h)
            d["dQssf_dxf"] = (dp_["dWdt"] - dm_["dWdt"]) / (2 * h)          # d/dx_f of Q_ss(x_f; x_f) - Q_M
        else:
            d["dSdW"] = d["dQdW"] = d["gamma"] = d["dW_dxf"] = d["lift"] = d["dPa_dxf"] = d["dPb_dxf"] = d["dQssf_dxf"] = np.nan
        d["C_brem"] = C
        # the leading-order analytic pieces at this point: Lambda = (g_b - g_a) + S(x_f)(L - x_f)/chi_ped is the
        # core lift per unit inward front displacement without the thermal feedback; the fold criterion reads
        # S(x_f) = lift * int_0^{x_f} dS/dp dx
        d["Lambda"] = (G_B - G_A) + d["S_at_front"] * (L - xf_) / CHI_PED
        d["fold_lhs"] = d["S_at_front"]
        d["fold_rhs_Lambda"] = d["Lambda"] * d["int_dSdp_core"]
        d["fold_rhs_lift"] = d["lift"] * d["int_dSdp_core"]
        rows.append(d)
        print(f"  x_f={xf_:.3f} C_brem={C:.4f} W={d['W']:.4f} S={d['S']:.4f} Pa={d['P_alpha']:.3f} Pb={d['P_brem']:.3f} "
              f"S(x_f)={d['S_at_front']:.2f} mu_min_ped={d['mu_min_ped']:.2f} gam={d['gamma']:+.3f} dS/dW={d['dSdW']:.2f} dQ/dW={d['dQdW']:.2f} "
              f"int dS/dp core={d['int_dSdp_core']:.2f}", flush=True)
    out = rows_to_npz(rows, "sh_")
    # (ii) the quasi-steady family at fixed C for three values of C: pdot(x_f), W(x_f)
    for C in (0.01, 0.8, 1.2):
        fam = []
        pg = None
        for xf_ in np.arange(0.10, 0.95, 0.01):
            d, pg = sharp_diag(xf_, dict(BASE, C_brem=C), S_ext, pg)
            if d is None: continue
            fam.append([xf_, d["W"], d["S"], d["Q_edge"], d["pdot"], d["dWdt"], d["P_alpha"], d["P_brem"], d["S_at_front"], d["mu_min_ped"]])
        out[f"fam_C{C:g}"] = np.array(fam)
    out["x"] = x; out["Q_M"] = Q_M
    np.savez(cache, **out)
    print("wrote fold_sharp.npz")


# ----------------------------------------------------------------------------------------------- stage: nu4 (front pinning test)
def do_nu4():
    """Does the thermal eigenvalue of the discretised model soften towards zero at the end of the burn once the front
    is wider than the grid?  Repeat the tail of the scan (C_brem >= 1.0) with nu4 x 0.25, 1, 4, 16 (front widths
    (nu4/|Q'|)^(1/2) = 0.7, 1.4, 2.8, 5.6 cells), recording the leading eigenvalue and the end of the burn."""
    cache = HERE / "fold_nu4.npz"
    if cache.exists():
        print("fold_nu4.npz exists"); return
    S_ext = source_profile(0.0)
    z = np.load(HERE / "fold_scan.npz")
    out = {"x": x, "Q_M": Q_M}
    for fac in (0.25, 1.0, 4.0, 16.0):
        prm0 = dict(BASE, nu4=BASE["nu4"] * fac)
        print(f"\n=== nu4 x {fac:g} = {prm0['nu4']:.2e}, front width ~ {np.sqrt(prm0['nu4'] / 10) / dx:.1f} cells ===")
        p0, d0, _ = to_steady(dict(prm0, C_brem=0.01), S_ext, initial_profile())
        rows, last_burn = scan_branch(prm0, S_ext, [0.01, 0.5, 1.0, 1.1, 1.2, 1.25, 1.3, 1.325, 1.35, 1.4], p0, f"nu{fac:g}", cb_tol=0.005)
        out.update(rows_to_npz(rows, f"nu{fac:g}_"))
        out[f"nu{fac:g}_cb_last_burn"] = last_burn[0] if last_burn else np.nan
        out[f"nu{fac:g}_factor"] = fac
    np.savez(cache, **out)
    print("wrote fold_nu4.npz")


# ----------------------------------------------------------------------------------------------- stage: edge (f scan)
F_LIST = [0.0, 0.25, 0.5, 0.57, 0.75, 1.0]
CB_UP_EDGE = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0]
CB_DN_EDGE = [2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01, 0.0]


def do_edge(S_tot=S_EXT_TRAP, cache_name="fold_edge.npz", f_list=F_LIST, cb_up=CB_UP_EDGE, cb_dn=CB_DN_EDGE):
    cache = HERE / cache_name
    if cache.exists():
        print(f"{cache_name} exists"); return
    out = {"f_list": np.array(f_list), "S_tot": S_tot, "Q_M": Q_M, "x": x}
    # continuation in f at C_brem = 0.01 from the f = 0 burning state (so that the burning root is followed, not re-ignited)
    z = np.load(HERE / "fold_scan.npz") if (HERE / "fold_scan.npz").exists() and S_tot == S_EXT_TRAP else None
    if z is not None:
        i0 = int(np.argmin(np.abs(z["up_C_brem"] - 0.01)))
        p_burn = None
    p_prev = None
    for f in f_list:
        S_ext = source_profile(f, S_tot)
        print(f"\n=== f = {f:.2f}: int S_ext = {np.trapezoid(S_ext, x):.3f} (injected {injected(S_ext):.3f}); core part {(1-f)*S_tot:.2f}, edge part {f*S_tot:.2f}")
        # burning root at C_brem = 0.01: continuation from the previous f (or the ignition run for f = 0)
        p0 = initial_profile() if p_prev is None else p_prev
        p_b, d_b, info = to_steady(dict(BASE, C_brem=0.01), S_ext, p0)
        if not (d_b["x_f"] < 0.95 and d_b["W"] > 2.0):
            # try the supercritical IC too (the ignition test)
            p_b2, d_b2, info2 = to_steady(dict(BASE, C_brem=0.01), S_ext, initial_profile())
            if d_b2["x_f"] < 0.95 and d_b2["W"] > 2.0:
                p_b, d_b, info = p_b2, d_b2, info2
        burn0 = d_b["x_f"] < 0.95 and d_b["W"] > 2.0
        print(f"  C_brem=0.01 from continuation: {'burn' if burn0 else 'no burning root'} W={d_b['W']:.4f} S={d_b['S']:.4f} x_f={d_b['x_f']:.3f} Qss_max={d_b['Qss_max']:.3f}@{d_b['x_Qss_max']:.2f}")
        out[f"f{f:.2f}_p_C0.01"] = p_b
        if burn0:
            p_prev = p_b
            up, last_burn = scan_branch(BASE, S_ext, cb_up, p_b, f"up{f:.2f}", cb_tol=0.01)
            out.update(rows_to_npz(up, f"f{f:.2f}_up_"))
            out[f"f{f:.2f}_cb_last_burn"] = last_burn[0] if last_burn else np.nan
            p_cold = None
            for r in up[::-1]:
                pass
            # the collapsed state: run the last rung (cold) once more to be sure, then go down
            prm = dict(BASE, C_brem=cb_up[-1])
            p_c, d_c, _ = to_steady(prm, S_ext, last_burn[1] if last_burn else p_b)
        else:
            up = []
            p_c, d_c, _ = to_steady(dict(BASE, C_brem=cb_up[-1]), S_ext, p_b)
        dn, _ = scan_branch(BASE, S_ext, cb_dn, p_c, f"dn{f:.2f}", bisect=False)
        out.update(rows_to_npz(dn, f"f{f:.2f}_dn_"))
        out[f"f{f:.2f}_burn0"] = burn0
        # the analytic L branch and H branch (external drive only) for reference: x_M of the external source alone
        Sx = np.concatenate([[0.0], np.cumsum(0.5 * (S_ext[1:] + S_ext[:-1]) * dx)])
        out[f"f{f:.2f}_Qss_ext"] = Sx
    np.savez(cache, **out)
    print(f"wrote {cache_name}")


def do_highS():
    """Supplementary: int S_ext = 7 > Q_M so that the barrier exists on the external drive alone, and the sink scan
    probes the thermal roots on the H branch (the T3D structure, Q_M below the drive)."""
    do_edge(S_tot=7.0, cache_name="fold_highS.npz", f_list=[0.0, 0.57], cb_up=[0.01, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            cb_dn=[5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.01])


# ----------------------------------------------------------------------------------------------- stage: step (edge-flux response)
def do_step():
    cache = HERE / "fold_step.npz"
    if cache.exists():
        print("fold_step.npz exists"); return
    S_ext = source_profile(0.0)
    z = np.load(HERE / "fold_scan.npz")
    out = {"x": x, "Q_M": Q_M}
    for cb0, cb1 in ((0.01, 0.21), (0.8, 1.0), (0.01, 0.06)):
        prm0 = dict(BASE, C_brem=cb0)
        i0 = int(np.argmin(np.abs(z["up_C_brem"] - cb0)))
        # rebuild the steady state at cb0 (from the ignition run, quasi-statically)
        p0, d0, _ = to_steady(prm0, S_ext, initial_profile())
        if cb0 > 0.01:
            p0, d0, _ = to_steady(prm0, S_ext, p0)
        prm1 = dict(BASE, C_brem=cb1)
        t, P = integrate(prm1, S_ext, p0, 4.0, n_out=2001)
        rows = {k: [] for k in DIAG_KEYS}
        for pk in P:
            d = diag(pk, prm1, S_ext)
            for k in DIAG_KEYS: rows[k].append(d[k])
        d_sink = diag(p0, prm1, S_ext)       # the sink of the new C on the OLD profile
        tag = f"step_{cb0:g}_{cb1:g}_"
        out[tag + "t"] = t
        for k in DIAG_KEYS: out[tag + k] = np.array(rows[k])
        out[tag + "d0"] = np.array([d0[k] for k in DIAG_KEYS])
        out[tag + "dsink"] = np.array([d_sink[k] for k in DIAG_KEYS])
        dPb = d_sink["P_brem"] - d0["P_brem"]
        print(f"step C_brem {cb0} -> {cb1}: delta P_brem on the old profile = {dPb:.4f} (inside the front {d_sink['P_brem_in']-d0['P_brem_in']:.4f}, outside {d_sink['P_brem_out']-d0['P_brem_out']:.4f})")
        for tt in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0):
            i = int(np.argmin(np.abs(t - tt)))
            print(f"   t={tt:5.2f}: dQ_edge={rows['Q_edge'][i]-d0['Q_edge']:+.4f} ({(rows['Q_edge'][i]-d0['Q_edge'])/dPb:+.3f} of the sink)  "
                  f"dW={rows['W'][i]-d0['W']:+.4f}  dS={rows['S'][i]-d0['S']:+.4f}  dP_alpha={rows['P_alpha'][i]-d0['P_alpha']:+.4f}  x_f={rows['x_f'][i]:.3f}")
    np.savez(cache, **out)
    print("wrote fold_step.npz")


# ----------------------------------------------------------------------------------------------- stage: sharp_edge
def sharp_fixed_branch(S_ext, xfs, Cs=(0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)):
    """C_brem(x_f) on the sharp-interface fixed-point branch (pdot = 0) for a given external source; no slopes."""
    rows = []
    pg = None
    Cs = np.array(Cs)
    for xf_ in xfs:
        def F(C):
            d, _ = sharp_diag(xf_, dict(BASE, C_brem=C), S_ext, pg)
            return np.nan if d is None else d["pdot"]
        vals = np.array([F(C) for C in Cs])
        okv = np.isfinite(vals)
        if okv.sum() < 1 or not np.any(vals[okv] < 0) or not np.any(vals[okv] > 0):
            rows.append([xf_, np.nan, np.nan, np.nan, np.nan, vals[0] if okv[0] else np.nan]); continue
        i = int(np.argmax(okv & (vals < 0)))
        j = i - 1
        while j >= 0 and not okv[j]: j -= 1
        if j < 0 or vals[j] <= 0:
            rows.append([xf_, np.nan, np.nan, np.nan, np.nan, np.nan]); continue
        C = brentq(F, Cs[j], Cs[i], xtol=1e-6)
        d, pg = sharp_diag(xf_, dict(BASE, C_brem=C), S_ext, pg)
        rows.append([xf_, C, d["W"], d["S"], d["P_alpha"], vals[0]])
    return np.array(rows)          # columns: x_f, C_brem, W, S, P_alpha, pdot at C=0


def do_sharp_edge():
    """Q2 in the sharp-interface model: C_fold(f) = max_x_f C_brem(x_f) on the fixed-point branch, the front position
    at the fold, and x_M at C_brem = 0 (where the burning root sits with no sink), for f = 0 ... 0.75."""
    cache = HERE / "fold_sharp_edge.npz"
    if cache.exists():
        print("fold_sharp_edge.npz exists"); return
    xfs = np.concatenate([np.arange(0.10, 0.30, 0.01), np.arange(0.30, 0.95, 0.0125)])
    out = {"x": x, "Q_M": Q_M}
    flist = [0.0, 0.1, 0.25, 0.4, 0.5, 0.57, 0.6, 0.65, 0.7, 0.75]
    summ = []
    for f in flist:
        br = sharp_fixed_branch(source_profile(f), xfs)
        out[f"br_f{f:.2f}"] = br
        ok = np.isfinite(br[:, 1])
        if ok.sum() == 0:
            print(f"  f={f:.2f}: no fixed point on the sharp branch for C_brem in [0, 3]"); summ.append([f] + [np.nan] * 5); continue
        k = int(np.nanargmax(np.where(ok, br[:, 1], -np.inf)))
        # parabolic refinement of the maximum
        if 0 < k < len(br) - 1 and ok[k - 1] and ok[k + 1]:
            xa, xb, xc = br[k - 1:k + 2, 0]; ya, yb, yc = br[k - 1:k + 2, 1]
            den = (xa - xb) * (xa - xc) * (xb - xc)
            A = (xc * (yb - ya) + xb * (ya - yc) + xa * (yc - yb)) / den
            B = (xc ** 2 * (ya - yb) + xb ** 2 * (yc - ya) + xa ** 2 * (yb - yc)) / den
            xv = -B / (2 * A); Cv = A * xv ** 2 + B * xv + (ya - A * xa ** 2 - B * xa)
        else:
            xv, Cv = br[k, 0], br[k, 1]
        # x_M of the burning root at the smallest C on the stable side (x_f < x_fold): extrapolate C -> 0
        st = ok & (br[:, 0] < xv)
        xM0 = float(np.interp(0.0, br[st, 1], br[st, 0])) if st.sum() >= 2 and br[st, 1].min() < 0.2 else np.nan
        summ.append([f, Cv, xv, br[k, 2], br[k, 3], xM0])
        print(f"  f={f:.2f}: sharp fold C_brem={Cv:.4f} at x_f={xv:.3f} (W={br[k,2]:.3f}, S={br[k,3]:.3f}); "
              f"burning-root x_f at C->0 ~ {xM0:.3f}; fixed points at {ok.sum()} of {len(br)} front positions", flush=True)
    out["summary"] = np.array(summ)
    np.savez(cache, **out)
    print("wrote fold_sharp_edge.npz")


# ----------------------------------------------------------------------------------------------- stage: extra (BDF checks)
def do_extra():
    """(a) the f at which the discretised burning root at C_brem = 0.01 disappears (bisection in f between 0.57 and
    0.75, continuing the burning state); (b) the unbisected core-burn loss of fold_highS f = 0.57; (c) small sink
    steps on the barrier branch at C_brem = 0.8 and 1.2 (replacing the 0.8 -> 1.0 step of fold_step, which started
    from the cold branch)."""
    cache = HERE / "fold_extra.npz"
    if cache.exists():
        print("fold_extra.npz exists"); return
    out = {"x": x, "Q_M": Q_M}
    isb = lambda d: d["x_f"] < 0.95 and np.isfinite(d["x_f"]) and d["W"] > 2.0
    # (a)
    prm = dict(BASE, C_brem=0.01)
    p, d, _ = to_steady(prm, source_profile(0.0), initial_profile())
    for f in (0.25, 0.5, 0.57):
        p, d, _ = to_steady(prm, source_profile(f), p)
    lo, hi, p_lo = 0.57, 0.75, p.copy()
    while hi - lo > 0.005:
        mid = 0.5 * (lo + hi)
        pm_, dm, im = to_steady(prm, source_profile(mid), p_lo, T_max=120.0)
        print(f"  f={mid:.4f} C_brem=0.01: {'burn' if isb(dm) else 'cold'} W={dm['W']:.4f} S={dm['S']:.4f} x_f={dm['x_f']:.3f} "
              f"[{'steady' if im['steady'] else 'NOT steady'}]", flush=True)
        if isb(dm): lo, p_lo = mid, pm_.copy()
        else: hi = mid
    out["f_c_lo"], out["f_c_hi"] = lo, hi
    print(f"  burning root at C_brem = 0.01 exists up to f = {lo:.4f} (lost by {hi:.4f})")
    # (b) highS f = 0.57 -- SUPERSEDED by do_highS_core: from initial_profile() this lands on the edge-barrier state
    #     (x_f ~ 0.8), not the core burn, so hs57_* in fold_extra.npz bisect nothing; kept only so the cache matches
    S7 = source_profile(0.57, 7.0)
    p, d, _ = to_steady(dict(BASE, C_brem=0.01), S7, initial_profile())
    p, d, _ = to_steady(dict(BASE, C_brem=0.5), S7, p)
    lo, hi, p_lo = 0.5, 0.625, p.copy()
    rows = []
    while hi - lo > 0.004:
        mid = 0.5 * (lo + hi)
        pm_, dm, im = to_steady(dict(BASE, C_brem=mid), S7, p_lo, T_max=120.0)
        core = dm["x_f"] < 0.6
        rows.append([mid, dm["W"], dm["S"], dm["x_f"], dm["Qss_max"], dm["x_Qss_max"], dm["P_alpha"], dm["P_brem"], float(core)])
        print(f"  highS f=0.57 C_brem={mid:.4f}: x_f={dm['x_f']:.3f} W={dm['W']:.4f} S={dm['S']:.4f} Qss_max={dm['Qss_max']:.3f}@{dm['x_Qss_max']:.2f} "
              f"Pa={dm['P_alpha']:.3f} Pb={dm['P_brem']:.3f} [{'steady' if im['steady'] else 'NOT steady'}]", flush=True)
        if core: lo, p_lo = mid, pm_.copy()
        else: hi = mid
    out["hs57_rows"] = np.array(rows); out["hs57_lo"], out["hs57_hi"] = lo, hi
    # the edge-barrier state's own end (Q_ss at the deposition vs Q_M): one more bisection from C = 0.7
    # (already in fold_highS: burn at 0.7188, cold at 0.7266)
    # (c) steps on the barrier branch
    S_ext = source_profile(0.0)
    p, d, _ = to_steady(dict(BASE, C_brem=0.01), S_ext, initial_profile())
    for cb in (0.4, 0.8):
        p, d, _ = to_steady(dict(BASE, C_brem=cb), S_ext, p)
    states = {0.8: p.copy()}
    for cb in (1.0, 1.2):
        p, d, _ = to_steady(dict(BASE, C_brem=cb), S_ext, p)
    states[1.2] = p.copy()
    for cb0, cb1 in ((0.8, 0.85), (1.2, 1.22)):
        p0 = states[cb0]; prm0 = dict(BASE, C_brem=cb0); prm1 = dict(BASE, C_brem=cb1)
        p0, d0, _ = to_steady(prm0, S_ext, p0)
        assert isb(d0), f"start state at {cb0} is not burning"
        mode = thermal_mode(prm0, S_ext, p0)
        t, P = integrate(prm1, S_ext, p0, 4.0, n_out=2001)
        rr = {k: [] for k in DIAG_KEYS}
        for pk in P:
            dd = diag(pk, prm1, S_ext)
            for k in DIAG_KEYS: rr[k].append(dd[k])
        ds = diag(p0, prm1, S_ext)
        tag = f"step_{cb0:g}_{cb1:g}_"
        out[tag + "t"] = t
        for k in DIAG_KEYS: out[tag + k] = np.array(rr[k])
        out[tag + "d0"] = np.array([d0[k] for k in DIAG_KEYS]); out[tag + "dsink"] = np.array([ds[k] for k in DIAG_KEYS])
        out[tag + "mode"] = np.array([mode["gamma"], mode["dSdW"], mode["dQdW"]])
        dPb = ds["P_brem"] - d0["P_brem"]
        print(f"step C_brem {cb0} -> {cb1} on the barrier branch (x_f={d0['x_f']:.3f}, W={d0['W']:.3f}, tau_E={d0['W']/d0['Q_edge']:.3f}): "
              f"delta P_brem = {dPb:.4f}, outside the front {(ds['P_brem_out']-d0['P_brem_out'])/dPb:.3f} of it; "
              f"gamma={mode['gamma']:+.3f}, predicted steady dQe/dPb = {-mode['dQdW']/(mode['dQdW']-mode['dSdW']):+.3f}")
        for tt in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0):
            i = int(np.argmin(np.abs(t - tt)))
            print(f"   t={tt:5.2f}: dQ_edge/dPb={(rr['Q_edge'][i]-d0['Q_edge'])/dPb:+.3f}  dW={rr['W'][i]-d0['W']:+.4f}  "
                  f"dP_alpha={rr['P_alpha'][i]-d0['P_alpha']:+.4f}  x_f={rr['x_f'][i]:.3f}")
    np.savez(cache, **out)
    print("wrote fold_extra.npz")


def do_highS_core():
    """Replaces part (b) of do_extra, which started from initial_profile() and so landed on the edge-barrier state
    (x_f ~ 0.8) rather than the core burn: at int S_ext = 7, f = 0.57 the two coexist.  Here the core burn is reached
    by continuation in f at C_brem = 0.01 from the f = 0 burn (as in fold_highS), raised to C = 0.5, and its loss is
    bisected; then the edge-barrier state is followed from there to its own end."""
    cache = HERE / "fold_highS_core.npz"
    if cache.exists():
        print("fold_highS_core.npz exists"); return
    S7 = source_profile(0.57, 7.0)
    p, d, _ = to_steady(dict(BASE, C_brem=0.01), source_profile(0.0, 7.0), initial_profile())
    for f in (0.25, 0.5, 0.57):
        p, d, _ = to_steady(dict(BASE, C_brem=0.01), source_profile(f, 7.0), p)
    print(f"  C=0.01 f=0.57 S=7 by continuation: x_f={d['x_f']:.3f} W={d['W']:.4f}")
    p, d, _ = to_steady(dict(BASE, C_brem=0.5), S7, p)
    print(f"  C=0.5: x_f={d['x_f']:.3f} W={d['W']:.4f} S={d['S']:.4f} Qss_max={d['Qss_max']:.3f}@{d['x_Qss_max']:.2f}")
    assert d["x_f"] < 0.6
    lo, hi, p_lo = 0.5, 0.625, p.copy()
    rows = [[0.5, d["W"], d["S"], d["x_f"], d["Qss_max"], d["x_Qss_max"], d["P_alpha"], d["P_brem"], 1.0]]
    while hi - lo > 0.004:
        mid = 0.5 * (lo + hi)
        pm_, dm, im = to_steady(dict(BASE, C_brem=mid), S7, p_lo, T_max=120.0)
        core = dm["x_f"] < 0.6
        rows.append([mid, dm["W"], dm["S"], dm["x_f"], dm["Qss_max"], dm["x_Qss_max"], dm["P_alpha"], dm["P_brem"], float(core)])
        print(f"  C_brem={mid:.4f}: x_f={dm['x_f']:.3f} W={dm['W']:.4f} S={dm['S']:.4f} Qss_max={dm['Qss_max']:.3f}@{dm['x_Qss_max']:.2f} "
              f"Pa={dm['P_alpha']:.3f} Pb={dm['P_brem']:.3f} [{'steady' if im['steady'] else 'NOT steady'}]", flush=True)
        if core: lo, p_lo = mid, pm_.copy()
        else: hi = mid
    print(f"  core burn lost between C_brem = {lo:.4f} and {hi:.4f}")
    # the edge-barrier state (front in the top hat): fold_highS labels it 'cold' once W < 2 (at C ~ 0.73), but it
    # continues; follow it from the state just past the core-burn loss and bisect its own end (front leaves x ~ 0.85)
    p_e, d_e, _ = to_steady(dict(BASE, C_brem=hi), S7, p_lo)
    erows = []
    for C in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0):
        pn, dn_, _ = to_steady(dict(BASE, C_brem=C), S7, p_e)
        erows.append([C, dn_["W"], dn_["S"], dn_["x_f"], dn_["Qss_max"], dn_["x_Qss_max"], dn_["P_alpha"], dn_["P_brem"]])
        print(f"  edge branch C_brem={C:.3f}: x_f={dn_['x_f']:.3f} W={dn_['W']:.4f} S={dn_['S']:.4f} Qss_max={dn_['Qss_max']:.3f}@{dn_['x_Qss_max']:.2f}", flush=True)
        if dn_["x_f"] > 0.95:
            break
        p_e = pn
    lo2, hi2 = erows[-2][0], erows[-1][0]
    while hi2 - lo2 > 0.004:
        mid = 0.5 * (lo2 + hi2)
        pn, dn_, _ = to_steady(dict(BASE, C_brem=mid), S7, p_e, T_max=120.0)
        erows.append([mid, dn_["W"], dn_["S"], dn_["x_f"], dn_["Qss_max"], dn_["x_Qss_max"], dn_["P_alpha"], dn_["P_brem"]])
        print(f"  edge branch C_brem={mid:.4f}: x_f={dn_['x_f']:.3f} W={dn_['W']:.4f} S={dn_['S']:.4f} Qss_max={dn_['Qss_max']:.3f}@{dn_['x_Qss_max']:.2f}", flush=True)
        if dn_["x_f"] < 0.95: lo2, p_e = mid, pn
        else: hi2 = mid
    np.savez(cache, rows=np.array(rows), lo=lo, hi=hi, erows=np.array(erows), lo2=lo2, hi2=hi2, Q_M=Q_M)
    print(f"  edge barrier lost between C_brem = {lo2:.4f} and {hi2:.4f}; wrote fold_highS_core.npz")


# ----------------------------------------------------------------------------------------------- figures and tables
def do_figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    if (HERE / "fold_scan.npz").exists():
        z = np.load(HERE / "fold_scan.npz")
        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
        ub = z["up_burn"] > 0.5
        ax[0, 0].plot(z["up_C_brem"], z["up_W"], "o-", ms=3, label="up (BDF)")
        ax[0, 0].plot(z["dn_C_brem"], z["dn_W"], "s--", ms=3, label="down (BDF)")
        if (HERE / "fold_newton.npz").exists():
            n = np.load(HERE / "fold_newton.npz")
            ax[0, 0].plot(n["nw_C_brem"], n["nw_W"], "k-", lw=0.8, label="Newton branch")
            ax[1, 0].plot(n["nw_W"], n["nw_gamma"], "k-", lw=0.8, label="Newton")
            ax[1, 1].plot(n["nw_W"], n["nw_S"], "r-", lw=0.8, label="S (Newton)")
        if (HERE / "fold_sharp.npz").exists():
            s = np.load(HERE / "fold_sharp.npz")
            ax[0, 0].plot(s["sh_C_brem"], s["sh_W"], "g:", lw=1.2, label="sharp-interface branch")
            ax[1, 0].plot(s["sh_W"], s["sh_gamma"], "g:", lw=1.2, label="sharp (Stefan family)")
        ax[0, 0].set_xlabel("C_brem"); ax[0, 0].set_ylabel("W"); ax[0, 0].legend(fontsize=7)
        ax[0, 1].plot(z["up_C_brem"][ub], z["up_S"][ub], "o-", ms=3, label="S")
        ax[0, 1].plot(z["up_C_brem"][ub], z["up_Qss_max"][ub], "^-", ms=3, label="max Q_ss")
        ax[0, 1].plot(z["up_C_brem"][ub], z["up_P_alpha"][ub], "v-", ms=3, label="P_alpha")
        ax[0, 1].plot(z["up_C_brem"][ub], z["up_P_brem"][ub], "d-", ms=3, label="P_brem")
        ax[0, 1].axhline(Q_M, color="grey", ls="--", lw=0.7); ax[0, 1].set_xlabel("C_brem"); ax[0, 1].legend(fontsize=7)
        ax[1, 0].plot(z["up_W"][ub], z["up_gamma"][ub], "o", ms=3, label="Jacobian eigenvalue (BDF states)")
        ax[1, 0].axhline(0, color="grey", lw=0.5); ax[1, 0].set_xlabel("W"); ax[1, 0].set_ylabel("gamma_th"); ax[1, 0].legend(fontsize=7)
        ax[1, 1].plot(z["traj_t"], z["traj_W"], "k", label="W"); ax2 = ax[1, 1].twinx()
        ax2.plot(z["traj_t"], z["traj_Q_edge"], "b--", label="Q_edge"); ax2.plot(z["traj_t"], z["traj_S"], "r:", label="S")
        ax2.axhline(Q_M, color="grey", lw=0.5); ax[1, 1].set_xlabel("t"); ax[1, 1].set_title(f"loss trajectory C_brem {float(z['traj_cb_from']):.3f} -> {float(z['traj_cb_to']):.3f}", fontsize=8)
        ax2.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIGS / "fold_scan.png", dpi=120); plt.close(fig)
    if (HERE / "fold_nu4.npz").exists():
        z = np.load(HERE / "fold_nu4.npz")
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        for fac, col in ((0.25, "C3"), (1.0, "k"), (4.0, "C0"), (16.0, "C2")):
            k = f"nu{fac:g}_"
            b = z[k + "burn"] > 0.5
            ax[0].plot(z[k + "C_brem"][b], z[k + "W"][b], "o-", color=col, ms=3, label=f"nu4 x {fac:g}")
            ax[1].plot(z[k + "C_brem"][b], z[k + "gamma"][b], "o-", color=col, ms=3, label=f"nu4 x {fac:g}")
            ax[2].plot(z[k + "C_brem"][b], z[k + "S"][b], "o-", color=col, ms=3, label=f"nu4 x {fac:g}")
        if (HERE / "fold_sharp.npz").exists():
            s = np.load(HERE / "fold_sharp.npz")
            ax[0].plot(s["sh_C_brem"], s["sh_W"], "g:", lw=1.2, label="sharp-interface branch")
            ax[1].plot(s["sh_C_brem"], s["sh_gamma"], "g:", lw=1.2, label="sharp (Stefan family)")
            ax[2].plot(s["sh_C_brem"], s["sh_S"], "g:", lw=1.2)
        ax[1].axhline(0, color="grey", lw=0.5); ax[2].axhline(Q_M, color="grey", ls="--", lw=0.7)
        ax[0].set_xlabel("C_brem"); ax[0].set_ylabel("W"); ax[1].set_xlabel("C_brem"); ax[1].set_ylabel("leading eigenvalue")
        ax[2].set_xlabel("C_brem"); ax[2].set_ylabel("S (= Q_edge)"); ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
        ax[0].set_xlim(0.9, 1.45); ax[1].set_xlim(0.9, 1.45); ax[2].set_xlim(0.9, 1.45); ax[0].set_ylim(3.5, 4.8); ax[2].set_ylim(5, 9)
        fig.tight_layout(); fig.savefig(FIGS / "fold_nu4.png", dpi=120); plt.close(fig)
    if (HERE / "fold_sharp.npz").exists():
        s = np.load(HERE / "fold_sharp.npz")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for C, col in ((0.01, "C0"), (0.8, "C1"), (1.2, "C3")):
            fam = s[f"fam_C{C:g}"]
            ax[0].plot(fam[:, 0], Q_M + fam[:, 4] * fam[:, 0], color=col, label=f"C_brem = {C:g}")
        ax[0].axhline(Q_M, color="grey", ls="--", lw=0.7); ax[0].set_xlabel("front position x_f"); ax[0].set_ylabel("drive inside the front, Q_ss(x_f; x_f)")
        ax[0].legend(fontsize=7)
        ax[1].plot(s["sh_x_f"], s["sh_C_brem"], "k-", label="C_brem(x_f) on the fixed-point branch")
        ax[1].set_xlabel("x_f"); ax[1].set_ylabel("C_brem"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIGS / "fold_family.png", dpi=120); plt.close(fig)
    if (HERE / "fold_edge.npz").exists():
        z = np.load(HERE / "fold_edge.npz")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for f in z["f_list"]:
            k = f"f{f:.2f}_"
            if k + "up_C_brem" in z.files:
                ax[0].plot(z[k + "up_C_brem"], z[k + "up_W"], "o-", ms=3, label=f"f={f:.2f} up")
            ax[0].plot(z[k + "dn_C_brem"], z[k + "dn_W"], "s--", ms=2, label=f"f={f:.2f} down")
            ax[1].plot(x, z[k + "Qss_ext"], label=f"f={f:.2f}")
        ax[1].axhline(Q_M, color="grey", ls="--"); ax[1].set_xlabel("x"); ax[1].set_ylabel("Q_ss of S_ext alone")
        ax[0].set_xlabel("C_brem"); ax[0].set_ylabel("W"); ax[0].legend(fontsize=6); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIGS / "fold_edge.png", dpi=120); plt.close(fig)
    if (HERE / "fold_step.npz").exists():
        z = np.load(HERE / "fold_step.npz")
        tags = sorted({k[:-1] for k in z.files if k.startswith("step_") and k.endswith("_t")})
        fig, ax = plt.subplots(1, len(tags), figsize=(5 * len(tags), 4))
        for a, tag in zip(np.atleast_1d(ax), tags):
            t = z[tag + "t"]; d0 = dict(zip(DIAG_KEYS, z[tag + "d0"])); ds = dict(zip(DIAG_KEYS, z[tag + "dsink"]))
            dPb = ds["P_brem"] - d0["P_brem"]
            a.plot(t, (z[tag + "Q_edge"] - d0["Q_edge"]) / dPb, label="dQ_edge / dP_brem")
            a.plot(t, (z[tag + "S"] - z[tag + "Q_edge"]) / dPb, label="dW/dt / dP_brem")
            a.plot(t, (z[tag + "W"] - d0["W"]) / d0["W"], label="dW/W")
            a.axhline(-(ds["P_brem_out"] - d0["P_brem_out"]) / dPb, color="grey", ls=":", label="-sink outside front")
            a.set_xscale("log"); a.set_xlabel("t"); a.set_title(tag, fontsize=8); a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIGS / "fold_step.png", dpi=120); plt.close(fig)
    print("figures in", FIGS)


def do_tables():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    if (HERE / "fold_scan.npz").exists():
        z = np.load(HERE / "fold_scan.npz")
        print("\n### fold_scan.npz: f = 0, up (burning branch, BDF steady states) ###")
        print("C_brem   W      S      Q_edge  P_alpha P_brem  x_f    x_mid  Qss_max@x    S(x_f)  gamma    dS/dW  dQ/dW  (uni: dS/dW dQ/dW)  lam2   steady")
        for i in range(len(z["up_C_brem"])):
            print(f"{z['up_C_brem'][i]:6.3f} {z['up_W'][i]:7.4f} {z['up_S'][i]:7.4f} {z['up_Q_edge'][i]:7.4f} {z['up_P_alpha'][i]:7.3f} {z['up_P_brem'][i]:7.3f} "
                  f"{z['up_x_f'][i]:6.3f} {z['up_x_mid'][i]:6.3f} {z['up_Qss_max'][i]:7.3f}@{z['up_x_Qss_max'][i]:4.2f} {z['up_S_at_front'][i]:7.2f} "
                  f"{z['up_gamma'][i]:+8.4f} {z['up_dSdW'][i]:6.2f} {z['up_dQdW'][i]:6.2f}  ({z['up_dSdW_uni'][i]:5.2f} {z['up_dQdW_uni'][i]:5.2f})  {z['up_lam2'][i]:7.2f}  {'y' if z['up_steady'][i] else 'N'}")
        print("\n### down ###")
        for i in range(len(z["dn_C_brem"])):
            print(f"{z['dn_C_brem'][i]:6.3f} {z['dn_W'][i]:7.4f} {z['dn_S'][i]:7.4f} {z['dn_Q_edge'][i]:7.4f} {z['dn_P_alpha'][i]:7.3f} {z['dn_P_brem'][i]:7.3f} "
                  f"{z['dn_x_f'][i]:6.3f} gmax={z['dn_g_max'][i]:5.2f} gamma={z['dn_gamma'][i]:+8.4f} p_min={z['dn_p_min'][i]:.3f}")
    if (HERE / "fold_newton.npz").exists():
        n = np.load(HERE / "fold_newton.npz")
        print("\n### fold_newton.npz: the discretised steady branch continued in W (C_brem the unknown) ###")
        print("   W     C_brem    S      P_alpha P_brem  x_f   Qss_max  gamma    dS/dW  dQ/dW")
        for i in range(len(n["nw_W"])):
            print(f"{n['nw_W'][i]:7.4f} {n['nw_C_brem'][i]:8.5f} {n['nw_S'][i]:7.4f} {n['nw_P_alpha'][i]:7.3f} {n['nw_P_brem'][i]:7.3f} {n['nw_x_f'][i]:6.3f} {n['nw_Qss_max'][i]:7.3f} "
                  f"{n['nw_gamma'][i]:+8.4f} {n['nw_dSdW'][i]:6.3f} {n['nw_dQdW'][i]:6.3f}")
        k = int(np.argmax(n["nw_C_brem"]))
        print(f"fold (max C_brem on the branch): C_brem = {n['nw_C_brem'][k]:.4f} at W = {n['nw_W'][k]:.4f}, S = {n['nw_S'][k]:.4f} (Q_M = {Q_M:.3f}), "
              f"P_alpha = {n['nw_P_alpha'][k]:.3f}, P_brem = {n['nw_P_brem'][k]:.3f}, gamma = {n['nw_gamma'][k]:+.4f}")
    if (HERE / "fold_sharp.npz").exists():
        s = np.load(HERE / "fold_sharp.npz")
        print("\n### fold_sharp.npz: the sharp-interface branch (Stefan family) ###")
        print("  x_f   C_brem    W      S     P_alpha P_brem  S(x_f) mu_min_ped  gamma   dS/dW  dQ/dW  int dS/dp (core) Lambda")
        for i in range(len(s["sh_x_f"])):
            print(f"{s['sh_x_f'][i]:6.3f} {s['sh_C_brem'][i]:8.4f} {s['sh_W'][i]:7.4f} {s['sh_S'][i]:7.3f} {s['sh_P_alpha'][i]:7.3f} {s['sh_P_brem'][i]:7.3f} "
                  f"{s['sh_S_at_front'][i]:6.2f} {s['sh_mu_min_ped'][i]:7.3f} {s['sh_gamma'][i]:+8.3f} {s['sh_dSdW'][i]:6.2f} {s['sh_dQdW'][i]:6.2f}  {s['sh_int_dSdp_core'][i]:6.2f}  {s['sh_Lambda'][i]:6.2f}")
        k = int(np.argmax(s["sh_C_brem"]))
        print(f"sharp fold: C_brem = {s['sh_C_brem'][k]:.4f} at x_f = {s['sh_x_f'][k]:.3f}, W = {s['sh_W'][k]:.4f}, S = {s['sh_S'][k]:.3f}")
    for name in ("fold_edge.npz", "fold_highS.npz"):
        if (HERE / name).exists():
            z = np.load(HERE / name)
            print(f"\n### {name}: int S_ext = {float(z['S_tot']):.2f} ###")
            for f in z["f_list"]:
                k = f"f{f:.2f}_"
                print(f"-- f = {f:.2f}: burning root at C_brem = 0.01: {bool(z[k + 'burn0'])}")
                if k + "up_C_brem" in z.files:
                    print("   up:   C_brem   W      S     Q_edge  P_alpha P_brem  x_f   Qss_max@x   gamma   dS/dW dQ/dW steady")
                    for i in range(len(z[k + "up_C_brem"])):
                        print(f"        {z[k+'up_C_brem'][i]:6.3f} {z[k+'up_W'][i]:7.4f} {z[k+'up_S'][i]:7.3f} {z[k+'up_Q_edge'][i]:7.3f} {z[k+'up_P_alpha'][i]:7.3f} {z[k+'up_P_brem'][i]:7.3f} "
                              f"{z[k+'up_x_f'][i]:6.3f} {z[k+'up_Qss_max'][i]:7.3f}@{z[k+'up_x_Qss_max'][i]:4.2f} {z[k+'up_gamma'][i]:+8.4f} {z[k+'up_dSdW'][i]:6.2f} {z[k+'up_dQdW'][i]:6.2f} {'y' if z[k+'up_steady'][i] else 'N'}")
                    print(f"   last burning C_brem (bisected): {float(z[k + 'cb_last_burn']):.3f}")
                print("   down: C_brem   W      S     Q_edge  P_alpha P_brem  x_f   g_max  burn")
                for i in range(len(z[k + "dn_C_brem"])):
                    print(f"        {z[k+'dn_C_brem'][i]:6.3f} {z[k+'dn_W'][i]:7.4f} {z[k+'dn_S'][i]:7.3f} {z[k+'dn_Q_edge'][i]:7.3f} {z[k+'dn_P_alpha'][i]:7.3f} {z[k+'dn_P_brem'][i]:7.3f} "
                          f"{z[k+'dn_x_f'][i]:6.3f} {z[k+'dn_g_max'][i]:6.2f} {'burn' if z[k+'dn_burn'][i] else 'cold'}")
    if (HERE / "fold_sharp.npz").exists():
        s = np.load(HERE / "fold_sharp.npz")
        xf = s["sh_x_f"]
        def cross(y):
            i = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]
            return [float(xf[j] - y[j] * (xf[j + 1] - xf[j]) / (y[j + 1] - y[j])) for j in i]
        print("\n### fold criterion on the sharp branch (f = 0): zero crossings in x_f ###")
        print(f"  gamma (Stefan family) = 0           : {np.round(cross(s['sh_gamma']), 3)}")
        print(f"  dQ_ss(x_f)/dx_f = 0                 : {np.round(cross(s['sh_dQssf_dxf']), 3)}")
        print(f"  S(x_f) = lift * int_core dS/dp      : {np.round(cross(s['sh_fold_lhs'] - s['sh_fold_rhs_lift']), 3)}")
        print(f"  S(x_f) = Lambda * int_core dS/dp    : {np.round(cross(s['sh_fold_lhs'] - s['sh_fold_rhs_Lambda']), 3)}")
        print(f"  max C_brem on the fixed-point branch: x_f = {xf[int(np.argmax(s['sh_C_brem']))]:.3f}")
    if (HERE / "fold_sharp_edge.npz").exists():
        z = np.load(HERE / "fold_sharp_edge.npz")
        print("\n### fold_sharp_edge.npz: sharp-interface fold vs edge fraction f (int S_ext = 4.94) ###")
        print("   f    C_fold   x_f(fold)  W(fold)  S(fold)  x_f of the burn at C->0")
        for r in z["summary"]:
            print(f"  {r[0]:.2f}  {r[1]:7.4f}  {r[2]:7.3f}  {r[3]:7.3f}  {r[4]:7.3f}  {r[5]:7.3f}")
    if (HERE / "fold_extra.npz").exists():
        z = np.load(HERE / "fold_extra.npz")
        print(f"\n### fold_extra.npz: discrete burning root at C_brem = 0.01 exists up to f = {float(z['f_c_lo']):.4f} (lost by {float(z['f_c_hi']):.4f}) ###")
        for tag in ("step_0.8_0.85_", "step_1.2_1.22_"):
            t = z[tag + "t"]; d0 = dict(zip(DIAG_KEYS, z[tag + "d0"])); ds = dict(zip(DIAG_KEYS, z[tag + "dsink"]))
            gam, dS, dQ = z[tag + "mode"]; dPb = ds["P_brem"] - d0["P_brem"]
            G = dQ / (dQ - dS)
            print(f"  {tag}: tau_E = {d0['W']/d0['Q_edge']:.3f}, outside-front share {(ds['P_brem_out']-d0['P_brem_out'])/dPb:.3f}, gamma = {gam:+.3f}, G = {G:.3f}")
            for tt in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 4.0):
                i = int(np.argmin(np.abs(t - tt)))
                print(f"     t={tt:4.2f} (t/tau_E={tt/(d0['W']/d0['Q_edge']):.2f}): measured {(z[tag+'Q_edge'][i]-d0['Q_edge'])/dPb:+.3f}   one-mode {-G*(1-np.exp(gam*tt)):+.3f}")
    if (HERE / "fold_highS_core.npz").exists():
        z = np.load(HERE / "fold_highS_core.npz")
        print(f"\n### fold_highS_core.npz (int S_ext = 7, f = 0.57): core burn lost between {float(z['lo']):.4f} and {float(z['hi']):.4f}; "
              f"edge barrier lost between {float(z['lo2']):.4f} and {float(z['hi2']):.4f} ###")
        print("   C_brem   W      S      x_f   Qss_max@x   P_alpha P_brem")
        for r in np.concatenate([z["rows"][:, :8], z["erows"]]):
            print(f"  {r[0]:7.4f} {r[1]:7.4f} {r[2]:7.3f} {r[3]:6.3f} {r[4]:7.3f}@{r[5]:.2f} {r[6]:7.3f} {r[7]:7.3f}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("scan", "all"): do_scan()
    if what in ("newton", "all"): do_newton()
    if what in ("sharp", "all"): do_sharp()
    if what in ("nu4", "all"): do_nu4()
    if what in ("edge", "all"): do_edge()
    if what in ("highS", "all"): do_highS()
    if what in ("step", "all"): do_step()
    if what in ("sharp_edge", "all"): do_sharp_edge()
    if what in ("extra", "all"): do_extra()
    if what in ("highS_core", "all"): do_highS_core()
    if what in ("figs", "all"): do_figs()
    if what in ("tables", "all"): do_tables()
