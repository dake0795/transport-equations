"""Reduced model with a gyrokinetic-like flux law (bump on a residual floor).

Dimensionless law (g in units of the gradient at the bump peak, Q in units of the
integrated external drive S unless stated otherwise, x in units of L):

    Q(g) = H B(g) + c_e g^3 + c_i g ,

with B(g) a bump of unit height fitted to the shape of the nonlinear stella scan
(``fit_bump``), H the bump height, c_e g^3 the Rechester--Rosenbluth electron floor and
c_i g the neoclassical ion floor.  The pressure equation and its discretisation are the
paper's (``reduced.py``: dx = 0.005, binomial smoothing of the face gradients, conservative
hyperviscosity nu4, Neumann axis, pinned edge, Gaussian external source, Bosch--Hale alpha
heating and bremsstrahlung from ``flux_transport_model_clean``), only the flux function is
replaced.  ``reduced.make_rhs`` hard-codes the paper's flux, so the RHS is re-implemented
here with a flux callable; nothing in ``reduced.py`` or the clean model files is modified.

Contents
  load_gk, fit_bump          the stella data and the bump fit (lognormal / generalised)
  Law                        Q(g), dQ/dg, landmarks, branch inverses, Maxwell construction
  step_ratios                c_e/H and c_i/H of the STEP surface (RR and NEO at the nominal point)
  source, Qss, x_M           Gaussian drive and the integrated drive
  analytic_L, analytic_H     steady profiles on the two branches
  make_rhs, run, diagnostics the stiff BDF integration and its diagnostics
  front_law                  dx_f/dt = -(Q_ss(x_f) - Q_M) / [x_f (g_b - g_a)]
"""
import json
import pathlib
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares
from scipy.sparse import diags

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import flux_transport_model_clean as m  # noqa: E402
import reduced as R  # noqa: E402

DATA = HERE.parent.parent / "transport_equations_paper" / "data" / "gk" / "q_vs_betaprime.npz"
L, DX, P_PED = R.L, R.DX, R.P_PED
BP_NOMINAL = 0.4994          # |beta'| of the nominal STEP point (beta_e = 0.09)


# ----------------------------------------------------------------------------- data and fit
def load_gk():
    d = dict(np.load(DATA, allow_pickle=True))
    bp = np.abs(d["beta_prime"]); o = np.argsort(bp)
    Q = (d["Q_e"] + d["Q_i"])[o]
    return dict(bp=bp[o], Q=Q, Qe=d["Q_e"][o], Qi=d["Q_i"][o], flag=d["flag"][o].astype(str),
                saturated=d["saturated"][o].astype(bool), live=Q > 0.5,
                rr=float(d["Q_RR_gB_at_scan"]), neo=float(d["Q_NEO_i_gB_at_scan"]),
                Q_gB_MW=float(d["Q_gB_MW_m2"]))


def fit_bump(kind="lognormal", data=None):
    """Fit ln(Q/Q_pk) against u = ln(|beta'|/g_pk) with the peak pinned at the data maximum.

    lognormal : -u^2 / (2 sigma^2), sigma = sigma_L (u<0), sigma_R (u>0)      (2 shape parameters)
    general   : -|u/sigma|^m per side                                           (4 shape parameters)
    plgauss   : -|u|/sigma_L (u<0, i.e. B = g^k with k = 1/sigma_L), -u^2/(2 sigma_R^2) (u>0)  (2 parameters)
    Note: in the exponent form -|u/sigma|^m the lognormal sigma is sigma_m = sqrt(2) sigma_gauss.
    Returns the shape dict used by ``Law`` plus the fit residuals.
    """
    d = data or load_gk()
    i_pk = int(np.argmax(d["Q"])); g_pk, Q_pk = d["bp"][i_pk], d["Q"][i_pk]
    u = np.log(d["bp"][d["live"]] / g_pk); y = np.log(d["Q"][d["live"]] / Q_pk)
    if kind == "lognormal":
        def model(p, u):
            s = np.where(u < 0, p[0], p[1]); return -u ** 2 / (2 * s ** 2)
        r = least_squares(lambda p: model(p, u) - y, [0.6, 0.2], bounds=([0.01, 0.01], [5, 5]))
        shape = dict(kind=kind, sL=float(np.sqrt(2) * r.x[0]), sR=float(np.sqrt(2) * r.x[1]), mL=2.0, mR=2.0,
                     sigma_gauss=(float(r.x[0]), float(r.x[1])))
    elif kind == "general":
        def model(p, u):
            return -np.where(u < 0, (np.abs(u) / p[0]) ** p[2], (np.abs(u) / p[1]) ** p[3])
        r = least_squares(lambda p: model(p, u) - y, [0.6, 0.3, 2.0, 2.0],
                          bounds=([0.01, 0.01, 0.3, 0.3], [5, 5, 8, 8]))
        shape = dict(kind=kind, sL=float(r.x[0]), sR=float(r.x[1]), mL=float(r.x[2]), mR=float(r.x[3]))
    elif kind == "plgauss":
        # power-law rise B = g^(1/sL) joined at the peak to a log-Gaussian fall exp(-u^2 / 2 sR^2)
        def model(p, u):
            return -np.where(u < 0, np.abs(u) / p[0], u ** 2 / (2 * p[1] ** 2))
        r = least_squares(lambda p: model(p, u) - y, [0.4, 0.3], bounds=([0.01, 0.01], [5, 5]))
        shape = dict(kind=kind, sL=float(r.x[0]), sR=float(np.sqrt(2) * r.x[1]), mL=1.0, mR=2.0,
                     k_rise=float(1 / r.x[0]), sigma_gauss=(None, float(r.x[1])))
    else:
        raise ValueError(kind)
    res = model(r.x, u) - y
    shape.update(g_pk=float(g_pk), Q_pk_gB=float(Q_pk), rms_ln=float(np.sqrt(np.mean(res ** 2))),
                 u=u, resid=res, bp=d["bp"][d["live"]], Q=d["Q"][d["live"]], flag=d["flag"][d["live"]])
    return shape


def step_ratios(shape, data=None):
    """c_e/H, c_i/H, g_nom from the RR and NEO floors at the nominal point (ratios to the data peak)."""
    d = data or load_gk()
    g_nom = BP_NOMINAL / shape["g_pk"]
    rr, neo = d["rr"] / shape["Q_pk_gB"], d["neo"] / shape["Q_pk_gB"]
    return dict(g_nom=g_nom, rr_over_peak=rr, neo_over_peak=neo,
                ce_over_H=rr / g_nom ** 3, ci_over_H=neo / g_nom)


# ----------------------------------------------------------------------------- the law
def bump(g, shape):
    """Unit-height bump B(g) (g in units of g_pk); zero for g <= 0."""
    g = np.asarray(g, float)
    out = np.zeros_like(g)
    pos = g > 0
    u = np.log(g[pos])
    s = np.where(u < 0, shape["sL"], shape["sR"]); mm = np.where(u < 0, shape["mL"], shape["mR"])
    out[pos] = np.exp(-(np.abs(u) / s) ** mm)
    return out


def dbump(g, shape):
    g = np.asarray(g, float)
    out = np.zeros_like(g)
    pos = g > 0
    u = np.log(g[pos])
    s = np.where(u < 0, shape["sL"], shape["sR"]); mm = np.where(u < 0, shape["mL"], shape["mR"])
    B = np.exp(-(np.abs(u) / s) ** mm)
    out[pos] = -B * mm * np.abs(u) ** (mm - 1) * np.sign(u) / (s ** mm * g[pos])
    return out


class Law:
    """Q(g) = H B(g) + c_e g^3 + c_i g, made odd in g so that negative gradients are diffusive."""

    def __init__(self, H, c_e, c_i, shape):
        self.H, self.c_e, self.c_i, self.shape = float(H), float(c_e), float(c_i), shape
        self._lm = None
        self._mx = None
        # cumulative integral of the bump on a fine log grid, for the equal-area rule
        gg = np.concatenate([[0.0], np.logspace(-6, 3, 90001)])
        self._gI = gg
        self._IB = np.concatenate([[0.0], np.cumsum(0.5 * (bump(gg[1:], shape) + bump(gg[:-1], shape)) * np.diff(gg))])

    def Q(self, g):
        g = np.asarray(g, float); a = np.abs(g)
        return np.sign(g) * (self.H * bump(a, self.shape) + self.c_e * a ** 3 + self.c_i * a)

    def dQ(self, g):
        g = np.asarray(g, float); a = np.abs(g)
        return self.H * dbump(a, self.shape) + 3 * self.c_e * a ** 2 + self.c_i

    def IB(self, g):
        return np.interp(g, self._gI, self._IB)

    def landmarks(self):
        if self._lm is None:
            g = np.logspace(-4, 2.5, 60001); q = self.Q(g)
            # the bump peak is the first local maximum of Q(g); the floor alone is monotone
            loc_max = np.where((q[1:-1] > q[:-2]) & (q[1:-1] >= q[2:]))[0] + 1
            if len(loc_max) == 0:
                i_pk = int(np.argmax(q)); self._lm = dict(g_max=g[i_pk], Q_max=q[i_pk], g_min=g[i_pk], Q_min=q[i_pk], monotone=True)
                return self._lm
            i_pk = int(loc_max[0])
            i_min = i_pk + int(np.argmin(q[i_pk:]))
            monotone = (i_min == i_pk) or (q[i_min] >= q[i_pk] * (1 - 1e-9))
            self._lm = dict(g_max=g[i_pk], Q_max=q[i_pk], g_min=g[i_min], Q_min=q[i_min], monotone=bool(monotone))
        return self._lm

    def g_minus(self, q):
        """Rising-branch inverse (0 < q < Q_max)."""
        lm = self.landmarks()
        if q <= 0: return 0.0
        if q >= lm["Q_max"]: return np.nan
        return brentq(lambda g: self.Q(g) - q, 1e-14, lm["g_max"], xtol=1e-14, rtol=1e-12)

    def g_plus(self, q):
        """Residual-branch inverse beyond the local minimum (q > Q_min)."""
        lm = self.landmarks()
        if lm["monotone"] or q <= lm["Q_min"]: return np.nan
        hi = max(10.0, 2 * (q / max(self.c_e, 1e-300)) ** (1 / 3), 2 * q / max(self.c_i, 1e-300) if self.c_e == 0 else 0)
        return brentq(lambda g: self.Q(g) - q, lm["g_min"], hi, xtol=1e-13, rtol=1e-12)

    def area(self, mu):
        ga, gb = self.g_minus(mu), self.g_plus(mu)
        return (self.H * (self.IB(gb) - self.IB(ga)) + self.c_e * (gb ** 4 - ga ** 4) / 4
                + self.c_i * (gb ** 2 - ga ** 2) / 2 - mu * (gb - ga))

    def maxwell(self):
        """Equal-area rule.  Returns dict(Q_M, g_a, g_b, A_lobe, Q_M_est, ...) or None if no coexistence."""
        if self._mx is not None: return self._mx
        lm = self.landmarks()
        if lm["monotone"] or lm["Q_min"] >= lm["Q_max"]:
            self._mx = None; return None
        lo, hi = lm["Q_min"] * (1 + 1e-9) + 1e-300, lm["Q_max"] * (1 - 1e-9)
        if self.area(lo) * self.area(hi) > 0:
            self._mx = None; return None
        mu = brentq(self.area, lo, hi, xtol=1e-14, rtol=1e-12)
        ga, gb = self.g_minus(mu), self.g_plus(mu)
        A_lobe = self.H * (self.IB(gb) - self.IB(ga))          # bump area between the two phases
        A_full = self.H * self.IB(1e3)                          # whole bump area
        # leading-order scaling for a tall narrow bump on a cubic floor: Q_M = (4A/3)^(3/4) c_e^(1/4)
        est = (4 * A_full / 3) ** 0.75 * self.c_e ** 0.25 if self.c_e > 0 else np.nan
        self._mx = dict(Q_M=mu, g_a=ga, g_b=gb, A_lobe=A_lobe, A_full=A_full, Q_M_est=est, **lm)
        return self._mx


# ----------------------------------------------------------------------------- drive
def source(x, S_tot=1.0):
    """Gaussian drive normalised on the evolved cells 1..N-2 (cells 0 and N-1 are held by the BCs), so that the
    steady edge flux equals S_tot exactly.  (reduced.py normalises by the trapezoid rule, which loses the axis
    half-cell: its steady Q_edge is 0.9887 x the nominal drive.)"""
    b = R.base_source(x); dx = x[1] - x[0]
    return S_tot * b / (dx * b[1:-1].sum())


def Qss(x, S):
    """Integrated drive at the nodes, consistent with the scheme: the steady flux through face j is
    dx sum_{k=1..j} S_k (cell 0 is a ghost), and the node value is the mean of its two faces,
    Q_ss(x_i) = dx (sum_{k=1..i-1} S_k + S_i/2).  Q_ss(L) = S_tot up to dx S(L)/2 ~ 0."""
    dx = x[1] - x[0]
    c = np.cumsum(S) * dx                    # dx sum_{k=0..i} S_k
    q = c - dx * S[0] - 0.5 * dx * S         # dx (sum_{k=1..i-1} S_k + S_i/2)
    q[0] = 0.0
    return q


def x_M(x, S, Q_M):
    q = Qss(x, S)
    if Q_M >= q[-1]: return np.nan
    return float(np.interp(Q_M, q, x))


def profile_from_g(x, g_nodes):
    """p(x) = P_PED + int_x^L g, with g given at the nodes."""
    dx = np.diff(x)
    drop = np.concatenate([np.cumsum((0.5 * (g_nodes[::-1][1:] + g_nodes[::-1][:-1]) * dx[::-1]))[::-1], [0.0]])
    return P_PED + drop


def analytic_L(law, x, S):
    q = Qss(x, S)
    g = np.array([law.g_minus(qi) for qi in q])
    if np.isnan(g).any(): return None
    p = profile_from_g(x, g)
    return dict(g=g, p=p, W=np.trapezoid(p, x))


def analytic_H(law, x, S, x_f=None):
    """Barrier profile: g_-(Q_ss) inside x_f, g_+(max(Q_ss, Q_M)) outside (x_f = x_M by default)."""
    mx = law.maxwell()
    if mx is None: return None
    q = Qss(x, S)
    if x_f is None:
        x_f = x_M(x, S, mx["Q_M"])
        if np.isnan(x_f): return None
    g = np.empty_like(x)
    for i, (xi, qi) in enumerate(zip(x, q)):
        if xi < x_f:
            g[i] = law.g_minus(qi) if qi < mx["Q_max"] else law.g_plus(qi)
        else:
            g[i] = law.g_plus(max(qi, mx["Q_M"]))
    if np.isnan(g).any(): return None
    p = profile_from_g(x, g)
    return dict(g=g, p=p, W=np.trapezoid(p, x), x_f=x_f)


def power_law_profile(x, g_max, m_exp=R.M_EXP):
    """The paper's initial profile: p = P_PED + A [1 - (x/L)^m] scaled to a maximum gradient g_max."""
    shape = 1.0 - (x / L) ** m_exp
    dx = x[1] - x[0]
    return P_PED + g_max / np.max(-np.gradient(shape, dx)) * shape


# ----------------------------------------------------------------------------- integration
def face_gradients(p, dx, n_smooth):
    p_bc = p.copy(); p_bc[0] = p_bc[1]; p_bc[-1] = P_PED
    g = -(p_bc[1:] - p_bc[:-1]) / dx
    for _ in range(n_smooth):
        gs = g.copy()
        gs[1:-1] = 0.25 * g[:-2] + 0.5 * g[1:-1] + 0.25 * g[2:]
        g = gs
    return g, p_bc


def make_rhs(x, law, S_ext, nu4, n_smooth, phys=None):
    """f(t, p): same discretisation as reduced.make_rhs with nu4_conservative=True, flux = law.Q."""
    dx = x[1] - x[0]
    Qf = law.Q

    def rhs(t, p):
        g_face, p_bc = face_gradients(p, dx, n_smooth)
        Q_face = Qf(g_face)
        dQdx = np.zeros_like(p_bc)
        dQdx[1:-1] = (Q_face[1:] - Q_face[:-1]) / dx
        dQdx[0] = Q_face[0] / dx
        dQdx[-1] = -Q_face[-1] / dx
        d4 = np.zeros_like(p_bc)
        if nu4 > 0:
            F = np.zeros(len(p_bc) - 1)
            F[1:-1] = (p_bc[3:] - 3 * p_bc[2:-1] + 3 * p_bc[1:-2] - p_bc[:-3]) / dx ** 3
            d4[1:-1] = (F[1:] - F[:-1]) / dx
        f = -dQdx - nu4 * d4 + S_ext
        if phys is not None:
            f = f + R.physics_source(p_bc, phys)
        f[0] = 0.0; f[-1] = 0.0
        return f
    return rhs


def default_nu4(law):
    """nu4 scaled with the steepest slope of the law: paper value 5e-4 at max|Q'| = 10 gives a 4-cell front."""
    g = np.linspace(0.3, 3.0, 3001)
    return 5e-5 * np.max(np.abs(law.dQ(g)))


def initial_condition(law, x, S, kind="super", x0=0.3, g_sub=0.5):
    """'super': analytic branches with a front at x0 (g_b plateau where Q_ss < Q_M);
    'sub': the paper's power-law profile with maximum gradient g_sub (< 1, rising branch);
    'L' / 'H': the analytic steady states; 'flat': p = P_PED."""
    if kind == "super":
        pr = analytic_H(law, x, S, x_f=x0)
        if pr is None: raise RuntimeError("no residual branch for this law")
        return pr["p"]
    if kind == "sub":
        return power_law_profile(x, g_sub)
    if kind == "L":
        return analytic_L(law, x, S)["p"]
    if kind == "H":
        return analytic_H(law, x, S)["p"]
    if kind == "flat":
        return np.full_like(x, P_PED)
    raise ValueError(kind)


def run(law, S_tot=1.0, ic="super", x0=0.3, g_sub=0.5, p0=None, nu4=None, n_smooth=3, T_end=1.0,
        n_out=201, phys=None, rtol=1e-6, atol=1e-9, dx=DX, dense_early=True):
    """Integrate p_t = -Q(g)_x - nu4 p_xxxx + S_ext + P_alpha - P_brem to T_end with BDF.

    phys: None (no alpha/brems) or a params dict with C_alpha, C_brem, T_ref_keV, ... (see
    flux_transport_model_clean.alpha_heating).  Returns dict with x, t, p(t, x), S_ext, ...
    """
    x = R.grid(dx)
    S_ext = source(x, S_tot)
    if nu4 is None: nu4 = default_nu4(law)
    if p0 is None: p0 = initial_condition(law, x, S_ext, ic, x0, g_sub)
    p0 = p0.copy(); p0[0] = p0[1]; p0[-1] = P_PED
    rhs = make_rhs(x, law, S_ext, nu4, n_smooth, phys)
    N = len(x); bw = 2 + n_smooth
    sparsity = diags([np.ones(N - abs(k)) for k in range(-bw, bw + 1)], list(range(-bw, bw + 1)), shape=(N, N))
    if dense_early:   # log-spaced early samples so that fast fronts are resolved
        t_eval = np.unique(np.clip(np.concatenate([[0.0], np.logspace(np.log10(T_end) - 4, np.log10(T_end), n_out // 2),
                                                   np.linspace(0.0, T_end, n_out // 2 + 1)]), 0.0, T_end))
    else:
        t_eval = np.linspace(0.0, T_end, n_out)
    sol = solve_ivp(rhs, (0.0, T_end), p0, method="BDF", t_eval=t_eval, jac_sparsity=sparsity, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    P = sol.y.T.copy(); P[:, 0] = P[:, 1]; P[:, -1] = P_PED
    return dict(x=x, t=sol.t, p=P, S_ext=S_ext, S_tot=S_tot, nu4=nu4, n_smooth=n_smooth, phys=phys,
                nfev=sol.nfev, njev=sol.njev, nlu=sol.nlu, law=dict(H=law.H, c_e=law.c_e, c_i=law.c_i, shape=law.shape["kind"]))


def diagnostics(res, law, g_thr=None):
    """W(t), Q_edge(t), S_tot(t) (external + alpha - brems), max g, edge g, front position (first face with
    g > g_thr; default the mid-level (g_a + g_b)/2 of the Maxwell front, which is what the sharp-interface
    theory calls x_f; 1.0 if no construction exists), and nu4 |g_xx| / Q_ss in the steep part of the profile."""
    x, P, S_ext, nu4, ns = res["x"], res["p"], res["S_ext"], res["nu4"], res["n_smooth"]
    if g_thr is None:
        mx = law.maxwell()
        g_thr = 0.5 * (mx["g_a"] + mx["g_b"]) if mx else 1.0
    dx = x[1] - x[0]; xf = 0.5 * (x[1:] + x[:-1])
    Qs = Qss(x, S_ext)
    out = dict(t=res["t"], W=[], Q_edge=[], S_tot=[], P_alpha=[], g_max=[], g_edge=[], x_f=[], nu4_corr=[])
    for p in P:
        g, p_bc = face_gradients(p, dx, ns)
        out["W"].append(np.trapezoid(p, x))
        out["Q_edge"].append(float(law.Q(g[-1])))
        Pa = R.physics_source(p_bc, res["phys"]) if res["phys"] else np.zeros_like(p)
        out["S_tot"].append(res["S_tot"] + np.trapezoid(Pa, x))
        out["P_alpha"].append(np.trapezoid(m.alpha_heating(p_bc, res["phys"]), x) if res["phys"] else 0.0)
        out["g_max"].append(g.max()); out["g_edge"].append(g[-1])
        sup = np.where(g > g_thr)[0]
        out["x_f"].append(xf[sup[0]] if len(sup) else np.nan)
        gxx = np.gradient(np.gradient(g, xf), xf)
        steep = g > g_thr
        out["nu4_corr"].append(np.max(nu4 * np.abs(gxx[steep]) / np.maximum(np.interp(xf[steep], x, Qs), 1e-12)) if steep.any() else 0.0)
    return {k: np.asarray(v) for k, v in out.items()}


def front_law(law, x, S, x0, t_end, mx=None):
    """Integrate the Stefan law from x_f(0) = x0; stops at the wall."""
    mx = mx or law.maxwell()
    q = Qss(x, S); ga, gb, QM = mx["g_a"], mx["g_b"], mx["Q_M"]

    def f(t, y):
        xf = min(max(y[0], 1e-6), L)
        return [-(np.interp(xf, x, q) - QM) / (xf * (gb - ga))]

    def hit_wall(t, y): return y[0] - L
    hit_wall.terminal = True
    sol = solve_ivp(f, (0, t_end), [x0], events=hit_wall, dense_output=True, rtol=1e-8, atol=1e-10, max_step=t_end / 200)
    t_wall = sol.t_events[0][0] if len(sol.t_events[0]) else np.nan
    return dict(t=sol.t, x_f=sol.y[0], t_wall=t_wall, sol=sol)


def steady_check(d, n=5):
    """Relative dW/dt over the last n samples and |Q_edge - S_tot|/S_tot at the end."""
    t, W = d["t"][-n:], d["W"][-n:]
    slope = np.polyfit(t, W, 1)[0]
    return dict(dWdt_rel=slope / W[-1], flux_balance=abs(d["Q_edge"][-1] - d["S_tot"][-1]) / abs(d["S_tot"][-1]))


def shape_summary(shape):
    keys = ["kind", "g_pk", "Q_pk_gB", "sL", "sR", "mL", "mR", "rms_ln"]
    return json.dumps({k: (round(shape[k], 4) if isinstance(shape[k], float) else shape[k]) for k in keys})


if __name__ == "__main__":
    for kind in ("lognormal", "general", "plgauss"):
        sh = fit_bump(kind)
        print(shape_summary(sh))
    sh = fit_bump("lognormal")
    st = step_ratios(sh)
    print("STEP ratios:", {k: f"{v:.3e}" for k, v in st.items()})
    for H in (1, 3, 10, 100, 1000, 1e4):
        law = Law(H, st["ce_over_H"] * H, st["ci_over_H"] * H, sh)
        mx = law.maxwell()
        print(f"H={H:8.0f}: Qmax={mx['Q_max']:.4g} Qmin={mx['Q_min']:.4g} Q_M={mx['Q_M']:.4g} "
              f"(est {mx['Q_M_est']:.4g}) g_a={mx['g_a']:.3f} g_b={mx['g_b']:.3f} Q_M/H={mx['Q_M']/H:.4f}")
