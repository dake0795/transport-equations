"""Why does the front lag the Stefan law?  Decomposition with the jump condition V [g] = [p_t].

Experiment: the paper's drive step (sec:front, Figure fig:front left).  A steady barrier at S_hi = Q_M + 3.60 has its
integrated drive stepped to S_lo = Q_M + 1.71, and back; physics off, conservative nu4, n_smooth 3, dx = 1/199.
Done for chi_MHD = 2 (the paper) and chi_MHD = 20 (the stiffer cliff the paper uses as a check).

1. MEASURED (reduced model).  At every output time:
   * front position: the Gibbs (equal-area) surface x_G of g, with the outer solutions g_-(mu), g_+(mu) built from the
     smooth total flux mu = Q(g) + nu4 p_xxx (the face flux, hyperviscous part included);  also x_mid and x_foot;
   * V = dx_G/dt;  [g] = g_+(mu_G) - g_-(mu_G);  mu_G = mu(x_G), the flux through the front;
   * p_t^-, p_t^+: the OUTER heating rates extrapolated to x_G from either side (p_t from the model's own right-hand
     side, carried through the front with d_t g_outer = mu_t / Q'(g_outer));
   * the regions' heating rates H_core = int_0^x_G p_t, H_ped = int_x_G^L p_t (Eulerian: d/dt of the stored energy in
     a fixed region; the stored energies W_in, W_out of the moving regions differ by -/+ p(x_G) V).
   Tests: (i) energy balance of each region, H_core = Q_ss(x_G) - mu_G and H_ped = S_tot - Q_ss(x_G) + mu_G - Q_edge;
   (ii) the jump condition V [g] = p_t^+ - p_t^-;  (iii) the three Stefan-law assumptions: mu_G = Q_M (Maxwell flux),
   H_core = x_G p_t^- (the core lifted uniformly), H_ped = 0 and p_t^+ = 0 (the pedestal quasi-steady).
2. ACCOUNTING.  The front law with the measured terms put back one at a time,
       dx/dt = [ p_t^+ - (Q_ss(x) - Q_M)/x - (Q_M - mu_G)/x + C/x ] / [g],   C = H_core - x p_t^- ,
   integrated from x_G(0); the half-time after each step gives the fraction of the lag each term carries.
3. MODELS (predictive, no measured input):
   (a) the Stefan law (paper eq. frontspeed);
   (b) two-region ODE: core lifted uniformly (its own relaxation x_f^2/pi^2 Q'(g_a) ~ 4e-4 is instantaneous), pedestal
       lumped: its top heating rate p_t^+ = -eta/tau_p, with eta = H - H_qs(x_f) the excess of the pedestal height over
       its quasi-static value (flux Q_M at the front, source S inside), d eta/dt = Gamma(x_f) dx_f/dt - eta/tau_p,
       Gamma = -dH_qs/dx_f - g_b = S(x_f) int_x_f^L dx/Q'(g_+) (= S(x_f) l/D for a linear cliff), eta(0) = the
       old pedestal's excess for the new source, and dx_f/dt [g] = p_t^+ - (Q_ss - Q_M)/x_f.
       tau_p = l^2/3D (one pole matched to the exact tanh(kl)/kl response of a slab with fixed flux at the front and
       fixed p at the edge); also 4 l^2/pi^2 D (its lowest mode), l^2/pi^2 D (the paper's estimate) and 0 (quasi-static);
   (c) sharp-interface two-phase Stefan problem: pedestal diffusion PDE on the moving slab, flux Q_M at the front,
       core lumped;  (d) the same with the core a PDE too.  (c, d) are the exact zero-width limit of the two regions.
4. RESOLUTION: the reduced model at dx/2, and at nu4/4 on dx/2 (front width halved), for the front-interior part;
   both cliffs.  (The sharp models are converged: Np = 100 and 200 agree to 1e-4 in the half-time.)
Light python, seconds per run.  Output: front_lag_jump.out, front_lag_jump.npz.
"""
import sys, pathlib, time
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import brentq
from scipy.special import erf

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import reduced as R                       # noqa: E402
import flux_transport_model_clean as m    # noqa: E402
from maxwell import maxwell, Q as Qlaw    # noqa: E402

L = 1.0
T_POST = 3.0
T_EVAL = np.unique(np.concatenate([np.linspace(0, 0.4, 801), np.linspace(0.4, 1.0, 241), np.linspace(1.0, T_POST, 201)]))


def half_time(t, a, a_end=None):
    """First time |a - a0| reaches half of |a_end - a0| (linear interpolation between samples)."""
    a = np.asarray(a, float); ok = np.isfinite(a); t, a = t[ok], a[ok]
    a_end = a[-1] if a_end is None else a_end
    f = np.abs(a - a[0]) - 0.5 * abs(a_end - a[0])
    k = np.argmax(f > 0)
    if k == 0:
        return np.nan
    return t[k - 1] + (t[k] - t[k - 1]) * (-f[k - 1]) / (f[k] - f[k - 1])


class Case:
    """Parameters, Maxwell state, branches and the discrete steady flux Q_ss for one cliff and one resolution."""

    def __init__(self, chi_MHD, nu4=5e-4, dx=R.DX):
        self.chi_MHD, self.nu4 = chi_MHD, nu4
        self.P = dict(R.PAPER, nu4_conservative=True, chi_MHD=chi_MHD, nu4=nu4, C_alpha=0.0, C_brem=0.0)
        self.Pm = dict(chi0=10.0, chi_RR=0.05, g_c=4.0, g_MHD=3.6, chi_MHD=chi_MHD)
        mx = maxwell(self.Pm); self.QM, self.ga, self.gb = mx["Q_M"], mx["g_a"], mx["g_b"]; self.gmin = mx["gmin"]
        self.dxreq = dx
        self.x = np.linspace(0.0, L, int(round(L / dx))); self.dx = self.x[1] - self.x[0]
        self.xf = 0.5 * (self.x[1:] + self.x[:-1])
        self.S0 = np.trapezoid(R.base_source(self.x), self.x)
        self.S_lo, self.S_hi = self.QM + 1.71, self.QM + 3.60
        self.x0 = self.dx / 2         # the discrete model's axis: cell 0 is a ghost (p0 = p1), so no flux crosses dx/2
        # branch tables: g_-(mu) on [0, g*], g_+(mu) on [g_min, 60]
        gl = np.linspace(0, 2.0, 40001); gh = np.linspace(self.gmin, 60, 200001)
        self.gl, self.ql = gl, Qlaw(gl, self.Pm); self.gh, self.qh = gh, Qlaw(gh, self.Pm)
        self.Dped = self.Pm["chi_RR"] + chi_MHD

    # outer branches and their slopes Q'(g)
    def g_lo(self, mu): return np.interp(mu, self.ql, self.gl)
    def g_hi(self, mu): return np.interp(mu, self.qh, self.gh)
    def dQ(self, g):
        e = 1e-6
        return (Qlaw(g + e, self.Pm) - Qlaw(g - e, self.Pm)) / (2 * e)

    # continuous source of the discrete model and its integral from the discrete axis x0
    def S(self, x, Sint): return (Sint / self.S0) * R.base_source(np.asarray(x, float))
    def F(self, x): return 5.0 * 0.25 * np.sqrt(np.pi) / 2 * erf(np.asarray(x, float) / 0.25)
    def Qss(self, x, Sint): return (Sint / self.S0) * (self.F(x) - self.F(self.x0))
    def xM(self, Sint): return brentq(lambda xx: self.Qss(xx, Sint) - self.QM, self.x0 + 1e-4, L)

    def grid_patch(self):
        R.grid = lambda dx=self.dxreq: np.linspace(0.0, L, int(round(L / dx)))


# ------------------------------------------------------------------------------------------- reduced model, measured
def total_face_flux(c, pk):
    Qf, _ = m.compute_face_flux(pk, c.dx, c.x, R.P_PED, c.P)
    pb = pk.copy(); pb[0] = pb[1]; pb[-1] = R.P_PED
    F = np.zeros(len(pb) - 1); F[1:-1] = (pb[3:] - 3 * pb[2:-1] + 3 * pb[1:-2] - pb[:-3]) / c.dx ** 3
    return Qf + c.P["nu4"] * F


def reduced_runs(c):
    c.grid_patch()
    pre = R.run(None, params=c.P, g_tgt=1.8, T_end=10.0, n_out=3, physics=False, source_scale=c.S_hi / c.S0)
    st = {c.S_hi: pre["p"][-1]}
    st[c.S_lo] = R.run(None, params=c.P, T_end=10.0, n_out=3, physics=False, source_scale=c.S_lo / c.S0, p0=st[c.S_hi])["p"][-1]
    out = {}
    from scipy.sparse import diags
    for S_from, S_to in ((c.S_hi, c.S_lo), (c.S_lo, c.S_hi)):
        # R.run's integration (BDF, banded sparsity), but on the non-uniform output times T_EVAL and at tighter tolerance
        S_ext = (S_to / c.S0) * R.base_source(c.x); rhs = R.make_rhs(c.x, c.P, S_ext); N = len(c.x)
        bw = 2 + c.P.get("n_smooth", 0)
        sp = diags([np.ones(N - k) for k in range(-bw, bw + 1)], list(range(-bw, bw + 1)), shape=(N, N))
        p0 = st[S_from].copy(); p0[0] = p0[1]; p0[-1] = R.P_PED
        s = solve_ivp(rhs, (0, T_POST), p0, method="BDF", t_eval=T_EVAL, jac_sparsity=sp, rtol=1e-8, atol=1e-10)
        Pt = s.y.T.copy(); Pt[:, 0] = Pt[:, 1]; Pt[:, -1] = R.P_PED
        out[(S_from, S_to)] = dict(t=s.t, p=Pt, S_ext=S_ext, rhs=rhs, S_from=S_from, S_to=S_to)
    R.grid = lambda dx=R.DX: np.linspace(0.0, L, int(L / dx))
    return out


def diagnose(c, run, half_window=0.07):
    t, Pt, rhs, S_to = run["t"], run["p"], run["rhs"], run["S_to"]
    x, xf, dx = c.x, c.xf, c.dx
    nt = len(t)
    MU = np.array([total_face_flux(c, pk) for pk in Pt])               # (nt, nfaces)
    PT = np.array([rhs(0.0, pk) for pk in Pt]); PT[:, 0] = PT[:, 1]    # Eulerian p_t on cells (axis ghost = cell 1)
    MUt = np.gradient(MU, t, axis=0)
    gmid = 0.5 * (c.ga + c.gb); gstar = 2.0
    Qss_face = np.concatenate([[0.0], np.cumsum(run["S_ext"][1:-1] * dx)])[:len(xf)]   # discrete Q_ss on faces
    S_tot = np.sum(run["S_ext"][1:-1]) * dx
    keys = "xG xmid xfoot muG gL gR ptm ptp Hcore Hped Hcore_id Hped_id Qedge Win Wout pG QssG".split()
    D = {k: np.full(nt, np.nan) for k in keys}
    for k in range(nt):
        pk = Pt[k]; g = -(pk[1:] - pk[:-1]) / dx; mu = MU[k]
        def cross(lev):
            j = np.where(g > lev)[0][0]
            return xf[j - 1] + (lev - g[j - 1]) / (g[j] - g[j - 1]) * dx
        xm = cross(gmid); D["xmid"][k] = xm; D["xfoot"][k] = cross(gstar)
        i1 = int(np.argmin(np.abs(x - (xm - half_window)))); i2 = int(np.argmin(np.abs(x - (xm + half_window))))
        xs = np.linspace(x[i1], x[i2], 4001)
        mus = np.interp(xs, xf, mu); mut = np.interp(xs, xf, MUt[k])
        gL, gR = c.g_lo(mus), c.g_hi(mus)
        IL = cumulative_trapezoid(gL, xs, initial=0.0); IR = cumulative_trapezoid(gR, xs, initial=0.0)
        resid = IL + (IR[-1] - IR) - (pk[i1] - pk[i2])                # int_x1^xG gL + int_xG^x2 gR - int g
        j = np.where(np.diff(np.sign(resid)) != 0)[0][0]
        xG = xs[j] + (xs[j + 1] - xs[j]) * resid[j] / (resid[j] - resid[j + 1]); D["xG"][k] = xG
        D["muG"][k] = np.interp(xG, xf, mu); D["gL"][k] = c.g_lo(D["muG"][k]); D["gR"][k] = c.g_hi(D["muG"][k])
        # outer p_t carried from x1 (core side) and x2 (pedestal side) to xG with d_t g = mu_t / Q'(g)
        gtL = mut / c.dQ(gL); gtR = mut / c.dQ(gR)
        CL = cumulative_trapezoid(gtL, xs, initial=0.0); CR = cumulative_trapezoid(gtR, xs, initial=0.0)
        D["ptm"][k] = PT[k, i1] - np.interp(xG, xs, CL)              # p_t(x) = p_t(x1) - int_x1^x g_t
        D["ptp"][k] = PT[k, i2] + (CR[-1] - np.interp(xG, xs, CR))   # p_t(x) = p_t(x2) + int_x^x2 g_t
        # regions: cells 1..N-2 are active; cell i spans [x_i - dx/2, x_i + dx/2]
        w = np.clip((xG - (x - dx / 2)) / dx, 0, 1); w[0] = 0; w[-1] = 0
        act = np.ones_like(x); act[0] = 0; act[-1] = 0
        D["Hcore"][k] = np.sum(PT[k] * w) * dx; D["Hped"][k] = np.sum(PT[k] * (act - w)) * dx
        D["Win"][k] = np.sum(pk * w) * dx; D["Wout"][k] = np.sum(pk * (act - w)) * dx
        QssG = np.interp(xG, xf, Qss_face); D["QssG"][k] = QssG
        D["Qedge"][k] = mu[-1]
        D["Hcore_id"][k] = QssG - D["muG"][k]
        D["Hped_id"][k] = S_tot - QssG + D["muG"][k] - mu[-1]
        D["pG"][k] = np.interp(xG, x, pk)
    D["t"] = t; D["V"] = np.gradient(D["xG"], t)
    D["jump"] = D["gR"] - D["gL"]
    D["C"] = D["Hcore"] - (D["xG"] - c.x0) * D["ptm"]
    D["dWin"] = np.gradient(D["Win"], t); D["dWout"] = np.gradient(D["Wout"], t)
    D["S_to"] = S_to; D["S_from"] = run["S_from"]
    return D


# ---------------------------------------------------------------------------------------------- front laws and models
def integrate(rhs, y0, t, events_x=True):
    hit = lambda tt, y: min(y[0] - 0.03, 0.97 - y[0]); hit.terminal = True
    s = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, rtol=1e-9, atol=1e-11, method="LSODA", events=hit if events_x else None)
    out = np.full((len(y0), len(t)), np.nan); out[:, :s.y.shape[1]] = s.y
    return out


def stefan(c, S_to, xf0, t, extra=None, jump=None):
    """dx/dt = [extra(t) - (Q_ss(x) - Q_M)/(x - x0)] / jump(t);  extra = 0, jump = g_b - g_a is the paper's law."""
    ex = (lambda tt, xx: 0.0) if extra is None else extra
    jp = (lambda tt: c.gb - c.ga) if jump is None else jump
    return integrate(lambda tt, y: [(ex(tt, y[0]) - (c.Qss(y[0], S_to) - c.QM) / (y[0] - c.x0)) / jp(tt)], [xf0], t)[0]


def Hqs(c, xf, Sint):
    """Quasi-steady pedestal height above p_ped: flux Q_M at the front, source S inside, p_t = 0."""
    xs = np.linspace(xf, L, 2001)
    return np.trapezoid(c.g_hi(c.QM + c.Qss(xs, Sint) - c.Qss(xf, Sint)), xs)


def Hss(c, xf, Sint):
    xs = np.linspace(xf, L, 2001)
    return np.trapezoid(c.g_hi(c.Qss(xs, Sint)), xs)


def ped_gain(c, xf, Sint):
    """Gamma = S(x_f) int dx/Q'(g_+) over the quasi-steady pedestal, and its harmonic-mean diffusivity D = l / int dx/Q'."""
    xs = np.linspace(xf, L, 2001)
    gq = c.g_hi(c.QM + c.Qss(xs, Sint) - c.Qss(xf, Sint))
    I = np.trapezoid(1.0 / c.dQ(gq), xs)
    return c.S(xf, Sint) * I, (L - xf) / I


def two_region_ode(c, S_from, S_to, t, tau_rule="l2/3D", source_change=True):
    """Lumped core (uniform lift) + lumped pedestal (one pole).  y = [x_f, eta].  Returns (x_f(t), p_t^+(t), eta0).
    source_change=False starts the pedestal on its quasi-steady profile for the new source (eta0 = 0)."""
    xf0 = c.xM(S_from)
    eta0 = Hss(c, xf0, S_from) - Hqs(c, xf0, S_to) if source_change else 0.0
    fac = {"l2/3D": 1 / 3, "4l2/pi2D": 4 / np.pi ** 2, "l2/pi2D": 1 / np.pi ** 2, "0": 0.0}[tau_rule]

    def parts(xx, eta):
        Gam, D = ped_gain(c, xx, S_to)
        tau = fac * (L - xx) ** 2 / D
        pdot = (c.Qss(xx, S_to) - c.QM) / (xx - c.x0)
        if tau == 0.0:     # quasi-static pedestal: p_t^+ = -Gamma xdot
            v = -pdot / (c.gb - c.ga + Gam)
            return v, 0.0, -Gam * v
        v = (-eta / tau - pdot) / (c.gb - c.ga)
        return v, Gam * v - eta / tau, -eta / tau

    rhs = lambda tt, y: list(parts(y[0], y[1])[:2])
    if fac == 0.0:        # instantaneous initial readjustment of the quasi-static pedestal: x jumps by -eta0/([g]+Gamma)
        Gam, _ = ped_gain(c, xf0, S_to)
        y = integrate(rhs, [xf0 - eta0 / (c.gb - c.ga + Gam), 0.0], t); y[0, 0] = xf0
    else:
        y = integrate(rhs, [xf0, eta0], t)
    ptp = np.array([parts(a, b)[2] if np.isfinite(a) else np.nan for a, b in y.T])
    return y[0], ptp, eta0


def sharp(c, S_from, S_to, t, core="lumped", Nc=60, Np=200, T_relax=2.0, ped0="old"):
    """Sharp-interface two-phase problem: pedestal (and optionally core) diffusion on moving slabs (Landau
    coordinates, vertex-centred finite volumes), flux Q_M on both sides of the front, p continuous across it (the jump
    condition), p(L) = p_ped, no flux through the axis x0.  First relaxed at S_from for T_relax, then stepped to S_to.
    ped0 = "qs_new" instead starts the pedestal on its quasi-steady profile for the NEW source (flux Q_M at the old
    front), which removes the part of the pedestal's response that is due to the source change inside it.
    Returns dict(x=x_f(t), ptp=p_t^+(t), ptm=p_t^-(t), Hped=int_ped p_t)."""
    QM, ga, gb, x0 = c.QM, c.ga, c.gb, c.x0
    lump = core == "lumped"

    def unpack(y):
        if lump:
            return None, y[0], y[1:1 + Np - 1], y[-1]
        return y[:Nc], y[Nc], y[Nc + 1:Nc + Np], y[-1]

    def make(Sint, diag=False):
        def rhs(tt, y):
            u, Pf, v, xx = unpack(y)
            l = L - xx; hp = l / Np
            vv = np.concatenate([[Pf], v, [R.P_PED]])
            zp = xx + l * np.arange(Np + 1) / Np
            Qp = Qlaw(-(vv[1:] - vv[:-1]) / hp, c.Pm)
            ptp = (QM - Qp[0]) / (hp / 2) + c.S(xx, Sint)
            ptv = (Qp[:-1] - Qp[1:]) / hp + c.S(zp[1:-1], Sint)
            if lump:
                ptm = (c.Qss(xx, Sint) - QM) / (xx - x0)
            else:
                hc = (xx - x0) / Nc
                uu = np.concatenate([u, [Pf]])
                xc = x0 + (xx - x0) * np.arange(Nc + 1) / Nc
                Qc = Qlaw(np.maximum(-(uu[1:] - uu[:-1]) / hc, 0.0), c.Pm)
                ptm = (Qc[-1] - QM) / (hc / 2) + c.S(xx, Sint)
                ptu = np.empty(Nc)
                ptu[0] = -Qc[0] / (hc / 2) + c.S(x0, Sint)
                ptu[1:] = (Qc[:-1] - Qc[1:]) / hc + c.S(xc[1:-1], Sint)
            V = (ptp - ptm) / (gb - ga)
            if diag:
                return ptp, ptm, hp * (0.5 * ptp + np.sum(ptv))
            dv = ptv + (vv[2:] - vv[:-2]) / (2 * hp) * (1 - np.arange(1, Np) / Np) * V
            dPf = ptm - ga * V
            if lump:
                return np.concatenate([[dPf], dv, [V]])
            du = ptu.copy()
            du[1:] += (uu[2:] - uu[:-2]) / (2 * hc) * (np.arange(1, Nc) / Nc) * V
            return np.concatenate([du, [dPf], dv, [V]])
        return rhs

    # analytic steady state at S_from, then relax the discrete one
    xx = c.xM(S_from)
    zs = np.linspace(xx, L, 4001)
    mu_ped = c.Qss(zs, S_from) if ped0 == "old" else c.QM + c.Qss(zs, S_to) - c.Qss(xx, S_to)
    Ip = cumulative_trapezoid(c.g_hi(mu_ped)[::-1], -zs[::-1], initial=0.0)[::-1]   # int_z^L g
    zp = xx + (L - xx) * np.arange(Np + 1) / Np
    v_all = R.P_PED + np.interp(zp, zs, Ip)
    y0 = [v_all[0]] + list(v_all[1:-1]) + [xx]
    if not lump:
        xs = np.linspace(x0, xx, 4001)
        Ic = cumulative_trapezoid(c.g_lo(c.Qss(xs, S_from))[::-1], -xs[::-1], initial=0.0)[::-1]
        xc = x0 + (xx - x0) * np.arange(Nc + 1) / Nc
        u = v_all[0] + np.interp(xc[:-1], xs, Ic)
        y0 = list(u) + y0
    y0 = np.array(y0)
    if ped0 == "old":
        y0 = solve_ivp(make(S_from), (0, T_relax), y0, method="BDF", rtol=1e-8, atol=1e-10).y[:, -1]
    s = solve_ivp(make(S_to), (t[0], t[-1]), y0, method="BDF", t_eval=t, rtol=1e-8, atol=1e-10)
    out = {k: np.full(len(t), np.nan) for k in ("x", "ptp", "ptm", "Hped")}
    out["x"][:s.y.shape[1]] = s.y[-1]
    dg = make(S_to, diag=True)
    for k in range(s.y.shape[1]):
        out["ptp"][k], out["ptm"][k], out["Hped"][k] = dg(t[k], s.y[:, k])
    return out


# ----------------------------------------------------------------------------------------------------------- main
def main():
    lines = []
    def say(s=""):
        print(s, flush=True); lines.append(s)
    save = {}; summary = []
    for chi in (2.0, 20.0):
        c = Case(chi)
        say(f"\n==================== chi_MHD = {chi}: Q_M = {c.QM:.4f}, g_a = {c.ga:.4f}, g_b = {c.gb:.4f}, [g] = {c.gb - c.ga:.3f}; "
            f"S_lo = {c.S_lo:.3f}, S_hi = {c.S_hi:.3f}; x_M(S_hi) = {c.xM(c.S_hi):.4f}, x_M(S_lo) = {c.xM(c.S_lo):.4f} "
            f"(discrete axis x0 = dx/2 = {c.x0:.4f})")
        t0 = time.time(); runs = reduced_runs(c); say(f"   reduced-model runs: {time.time() - t0:.1f} s")
        for (S_from, S_to), run in runs.items():
            tag = "down (front out)" if S_to < S_from else "up (front in)"
            D = diagnose(c, run); t = D["t"]
            say(f"\n--- chi_MHD = {chi}, step {tag}: S {S_from:.2f} -> {S_to:.2f}")
            say(f"   x_G {D['xG'][0]:.4f} -> {D['xG'][-1]:.4f} (x_M {c.xM(S_from):.4f} -> {c.xM(S_to):.4f});  x_foot {D['xfoot'][0]:.4f} -> {D['xfoot'][-1]:.4f};  "
                f"x_mid {D['xmid'][0]:.4f} -> {D['xmid'][-1]:.4f}")
            # 1. tests
            j0 = np.searchsorted(t, 0.01)
            e1 = np.nanmax(np.abs(D["Hcore"] - D["Hcore_id"])[j0:]); e2 = np.nanmax(np.abs(D["Hped"] - D["Hped_id"])[j0:])
            e3 = np.nanmax(np.abs(D["V"] * D["jump"] - (D["ptp"] - D["ptm"]))[j0:])
            jm = (t > 0.01) & (t < 0.3)
            e4 = np.nanmedian(np.abs(D["V"] * D["jump"] - (D["ptp"] - D["ptm"]))[jm] / np.abs(D["ptp"] - D["ptm"])[jm])
            say(f"   energy balance of each region (t > 0.01, max abs residual): core {e1:.2e}, pedestal {e2:.2e};  jump condition V[g] - [p_t]: "
                f"max {e3:.2e} (lattice jerks; scale of [p_t] {np.nanmax(np.abs(D['ptp'] - D['ptm'])):.2f}), median relative {e4:.3f} (0.01 < t < 0.3)")
            jh = np.argmin(np.abs(t - half_time(t, D["xG"])))
            say(f"   at the half-time: p_t^+/p_t^- = {D['ptp'][jh] / D['ptm'][jh]:.2f}, H_ped/H_core = {D['Hped'][jh] / D['Hcore'][jh]:.2f}, "
                f"(mu_G - Q_M)/(Q_ss - Q_M) = {(D['muG'][jh] - c.QM) / (D['QssG'][jh] - c.QM):+.3f}, C/H_core = {D['C'][jh] / D['Hcore'][jh]:+.3f}")
            say("     t      x_G     V      [g]   mu_G-Q_M  p_t^-  x p_t^-  H_core  Qss-Q_M  p_t^+   H_ped   sigma   C      dW_in   dW_out  Q_edge")
            for tt in (0.002, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0):
                j = np.argmin(np.abs(t - tt))
                sig = D["ptp"][j] * (L - D["xG"][j]) / D["Hped"][j] if abs(D["Hped"][j]) > 1e-3 else np.nan
                say(f"   {t[j]:.3f}  {D['xG'][j]:.4f} {D['V'][j]:+.3f}  {D['jump'][j]:.3f}  {D['muG'][j] - c.QM:+.4f}  {D['ptm'][j]:+6.2f}  "
                    f"{(D['xG'][j] - c.x0) * D['ptm'][j]:+.3f}  {D['Hcore'][j]:+.3f}  {D['QssG'][j] - c.QM:+.3f}  {D['ptp'][j]:+6.2f}  {D['Hped'][j]:+.3f}  "
                    f"{sig:5.2f}  {D['C'][j]:+.4f}  {D['dWin'][j]:+.3f}  {D['dWout'][j]:+.3f}  {D['Qedge'][j]:.3f}")
            # 2. accounting with the measured terms
            ip = lambda arr: (lambda tt, xx=None: float(np.interp(tt, t, arr)))
            xf0 = D["xG"][0]; xend = D["xG"][-1]
            th = {}
            th["measured x_G"] = half_time(t, D["xG"])
            th["measured x_foot"] = half_time(t, D["xfoot"])
            th["measured x_mid"] = half_time(t, D["xmid"])
            law = stefan(c, S_to, xf0, t)
            th["law"] = half_time(t, law)
            l_ped = stefan(c, S_to, xf0, t, extra=lambda tt, xx: ip(D["ptp"])(tt))
            th["+ped"] = half_time(t, l_ped)
            l_pf = stefan(c, S_to, xf0, t, extra=lambda tt, xx: ip(D["ptp"])(tt) - (c.QM - ip(D["muG"])(tt)) / (xx - c.x0))
            th["+ped+front flux"] = half_time(t, l_pf)
            l_pfc = stefan(c, S_to, xf0, t, extra=lambda tt, xx: ip(D["ptp"])(tt) - (c.QM - ip(D["muG"])(tt)) / (xx - c.x0) + ip(D["C"])(tt) / (xx - c.x0))
            th["+ped+front flux+core"] = half_time(t, l_pfc)
            l_all = stefan(c, S_to, xf0, t, extra=lambda tt, xx: ip(D["ptp"])(tt) - (c.QM - ip(D["muG"])(tt)) / (xx - c.x0) + ip(D["C"])(tt) / (xx - c.x0),
                           jump=ip(D["jump"]))
            th["+all (+[g])"] = half_time(t, l_all)
            l_fc = stefan(c, S_to, xf0, t, extra=lambda tt, xx: -(c.QM - ip(D["muG"])(tt)) / (xx - c.x0) + ip(D["C"])(tt) / (xx - c.x0))
            th["front flux+core only"] = half_time(t, l_fc)
            say(f"   law end point {law[-1]:.4f} vs x_G end {xend:.4f};  all-terms law end {l_all[-1]:.4f}")
            lag = th["measured x_G"] - th["law"]
            say("   half-times (accounting; each row adds the measured term to the law):")
            for k, v in th.items():
                fr = (v - th["law"]) / lag if k not in ("measured x_G", "measured x_foot", "measured x_mid", "law") else np.nan
                say(f"     {k:24s} {v:.4f}" + (f"   (ratio to law {v / th['law']:.2f})" if k.startswith("measured") else "")
                    + (f"   fraction of lag {fr:+.2f}" if np.isfinite(fr) else ""))
            # 3. models (all start from the sharp-interface rest point x_M(S_from))
            xs0 = c.xM(S_from); mods = {}; mptp = {}
            mods["Stefan law (paper)"] = stefan(c, S_to, xs0, t)
            for rule in ("l2/3D", "4l2/pi2D", "l2/pi2D", "0"):
                mods[f"ODE tau_p={rule}"], mptp[f"ODE tau_p={rule}"], eta0 = two_region_ode(c, S_from, S_to, t, rule)
            mods["ODE tau_p=l2/3D, eta0=0"], mptp["ODE tau_p=l2/3D, eta0=0"], _ = two_region_ode(c, S_from, S_to, t, "l2/3D", source_change=False)
            for name, kw in (("sharp: PDE ped (qs new S at t=0) + lumped core", dict(core="lumped", ped0="qs_new")),
                             ("sharp: PDE ped + lumped core", dict(core="lumped")),
                             ("sharp: PDE ped + PDE core", dict(core="pde"))):
                r = sharp(c, S_from, S_to, t, **kw); mods[name] = r["x"]; mptp[name] = r["ptp"]
                if kw.get("core") == "pde":
                    sharp_full = r
            Gam0, D0 = ped_gain(c, c.xM(S_to), S_to)
            tauf = (c.xM(S_to) - c.x0) * (c.gb - c.ga) / c.S(c.xM(S_to), S_to)
            say(f"   models from x_M(S_from) = {xs0:.4f}.  Pedestal at x_M(S_to): Gamma = S l/D = {Gam0:.3f} (Gamma/[g] = {Gam0 / (c.gb - c.ga):.3f}), "
                f"D_ped = {D0:.3f}, tau_p = l^2/3D = {(L - c.xM(S_to)) ** 2 / 3 / D0:.4f} (paper's l^2/pi^2 D = {(L - c.xM(S_to)) ** 2 / np.pi ** 2 / D0:.4f}); "
                f"tau_f = {tauf:.4f};  eta(0) = {eta0:+.4f} against [g] |dx_M| = {(c.gb - c.ga) * abs(c.xM(S_to) - xs0):.4f}; "
                f"core time x_M^2/pi^2 Q'(g_a) = {(c.xM(S_to)) ** 2 / np.pi ** 2 / c.dQ(np.array([c.ga]))[0]:.5f}")
            tmeas = th["measured x_G"]; tS = half_time(t, mods["Stefan law (paper)"])
            th_mod = {k: half_time(t, v) for k, v in mods.items()}
            for k, v in th_mod.items():
                say(f"     {k:48s} half-time {v:.4f}   model/measured {v / tmeas:.2f}   fraction of lag {(v - tS) / (tmeas - tS):+.2f}   end {mods[k][-1]:.4f}")
            a = th_mod["sharp: PDE ped (qs new S at t=0) + lumped core"]; b = th_mod["sharp: PDE ped + lumped core"]; cc = th_mod["sharp: PDE ped + PDE core"]
            say(f"   LAG BUDGET (sharp-interface models; measured lag {tmeas - tS:.4f} = {tmeas / tS:.2f}x law):  pedestal readjustment to the moving front "
                f"{(a - tS) / (tmeas - tS):+.2f};  pedestal response to the source change {(b - a) / (tmeas - tS):+.2f};  core {(cc - b) / (tmeas - tS):+.2f};  "
                f"remainder (front interior, discreteness) {(tmeas - cc) / (tmeas - tS):+.2f}")
            summary.append((chi, "down" if S_to < S_from else "up", tmeas, tS, th_mod["ODE tau_p=l2/3D"], th_mod["ODE tau_p=4l2/pi2D"],
                            th_mod["ODE tau_p=l2/pi2D"], cc, (a - tS) / (tmeas - tS), (b - a) / (tmeas - tS), (cc - b) / (tmeas - tS), (tmeas - cc) / (tmeas - tS),
                            (th["+ped"] - th["law"]) / lag, (th["+ped+front flux"] - th["+ped"]) / lag, (th["+ped+front flux+core"] - th["+ped+front flux"]) / lag))
            say("   heating rates, measured (reduced model, at x_G) vs sharp two-phase model (PDE ped + PDE core, at its own front):")
            say("     t      x_G     x_sh    p_t^- meas  sharp   p_t^+ meas  sharp   ODE(l2/3D)   H_ped meas  sharp")
            for tt in (0.005, 0.02, 0.05, 0.1, 0.2, 0.4):
                j = np.argmin(np.abs(t - tt))
                say(f"     {t[j]:.3f}  {D['xG'][j]:.4f}  {sharp_full['x'][j]:.4f}  {D['ptm'][j]:+7.3f}  {sharp_full['ptm'][j]:+7.3f}   {D['ptp'][j]:+7.3f}  "
                    f"{sharp_full['ptp'][j]:+7.3f}   {mptp['ODE tau_p=l2/3D'][j]:+7.3f}      {D['Hped'][j]:+7.3f}  {sharp_full['Hped'][j]:+7.3f}")
            key = f"chi{int(chi)}_{'down' if S_to < S_from else 'up'}"
            save.update({f"{key}_{k}": v for k, v in D.items() if isinstance(v, np.ndarray)})
            save.update({f"{key}_model_{k}": v for k, v in mods.items()})
            save[f"{key}_law"] = law
    say("\n==================== SUMMARY (half-times; fractions of the measured lag t_meas - t_Stefan)")
    say("   chi  step  t_meas  t_Stefan  ratio | ODE l2/3D  4l2/pi2D  l2/pi2D | sharp 2-phase | budget: ped(front motion) ped(source) core remainder "
        "| accounting: ped  front-flux  core")
    for r in summary:
        say(f"   {r[0]:4.0f} {r[1]:5s} {r[2]:.4f}  {r[3]:.4f}  {r[2] / r[3]:5.2f} | {r[4] / r[2]:5.2f}  {r[5] / r[2]:7.2f}  {r[6] / r[2]:7.2f}   | {r[7] / r[2]:5.2f}        "
            f"|   {r[8]:+.2f}  {r[9]:+.2f}  {r[10]:+.2f}  {r[11]:+.2f}        |   {r[12]:+.2f}  {r[13]:+.2f}  {r[14]:+.2f}")
    say("   (ODE and sharp columns: model half-time / measured half-time)")
    # 4. resolution / front width
    say("\n==================== resolution and front width: measured half-times of x_G")
    for chi, dxr, nu4 in ((2.0, R.DX, 5e-4), (2.0, R.DX / 2, 5e-4), (2.0, R.DX / 2, 1.25e-4), (20.0, R.DX, 5e-4), (20.0, R.DX / 2, 5e-4), (20.0, R.DX / 2, 1.25e-4)):
        c = Case(chi, nu4=nu4, dx=dxr)
        t0 = time.time(); runs = reduced_runs(c)
        row = []
        for (S_from, S_to), run in runs.items():
            D = diagnose(c, run, half_window=0.07 if nu4 >= 5e-4 else 0.05)
            row.append(f"{'down' if S_to < S_from else 'up'}: x_G {D['xG'][0]:.4f}->{D['xG'][-1]:.4f}, half {half_time(D['t'], D['xG']):.4f}, "
                       f"law {half_time(D['t'], stefan(c, S_to, D['xG'][0], D['t'])):.4f}, max|mu_G-Q_M| {np.nanmax(np.abs(D['muG'] - c.QM)):.3f}")
        say(f"   chi_MHD = {chi:4.0f}, dx = {c.dx:.5f}, nu4 = {nu4:.2e} ({time.time() - t0:.0f} s): " + ";  ".join(row))
    np.savez(HERE / "front_lag_jump.npz", **save)
    (HERE / "front_lag_jump.out").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
