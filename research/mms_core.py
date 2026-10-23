"""Core numerics for the multiple-scales / matched-asymptotics study (exp_mms.py, NOTES_mms.md).

A clean cell-centred discretisation of the reduced model, written so that the global energy identity holds exactly:

    p_t = -d_x J + S(x, p, n),      J = Q(g) + nu4 p_xxx = Q(g) - nu4 g_xx,      g = -p_x,
    S   = S_ext(x) + P_alpha(T, n) - P_brem(T, n),   T = T_ref p / n,
    n_t = D_n [ n_xx + s_N(x) ]      (one-way coupled: the density does not see p),

on x in [0, L]: cells i = 0..N-1 at x_i = (i + 1/2) h, faces k = 0..N at x = k h.  Face 0 is the axis (J_0 = 0:
g_0 = 0 and the hyperviscous flux is zero); face N is the edge, p(L) = p_ped through a linear ghost, and the
hyperviscous flux through it is zero, so J_N = Q(g_N) exactly.  Then

    dW/dt = h sum_i p_i' = h sum_i S_i - Q(g_N) = S_tot - Q_edge        (exact, discrete)

which is the integral identity of the amplitude equation.  No binomial smoothing (the paper's production scheme smooths
the face gradients three times; here the front is resolved by the grid instead: w / h >= 5).

Physics closures: the paper's (flux_transport_model_clean): Bosch-Hale D-T, P_alpha = C_alpha 5.6e-13 <sv> n_D n_T with
n_D = n_T = n/2 (in 1e20 m^-3), P_brem = C_brem Z_eff n^2 sqrt(T).  At n = 1 this is the paper's model exactly.
"""
import pathlib
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded, eigh_tridiagonal
from scipy.sparse import diags
from scipy.sparse.linalg import eigs, splu

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import flux_transport_model_clean as m   # noqa: E402

L = 1.0
P_PED = 1.0

# the paper's flux law and the ignition / radiation-scan physics (make_brem_scan.py, NOTES_fold.md)
PRM = dict(chi0=10.0, chi_RR=0.05, g_c=4.0, g_MHD=3.6, chi_MHD=2.0, nu4=5e-4, sig=0.05,
           C_alpha=2e-5, C_brem=0.01, Z_eff=1.0, T_ref_keV=3.0)
S_EXT_TOT = 4.944392605799654          # int S_ext of the ignition / radiation scan (trapezoid on [0, 1])


def base_source(x):
    return 5.0 * np.exp(-(x ** 2) / 0.0625)


BASE_INT = 5.0 * 0.25 * np.sqrt(np.pi) / 2 * 0.9999999845827421   # int_0^1 base_source = 5 * 0.25 sqrt(pi)/2 erf(4)


# ------------------------------------------------------------------------------------------------ flux law
# The paper's law has two kinks (max(1 - g/g_c, 0) and max(g - g_MHD, 0)).  A kink inside a resolved front makes the
# discrete linearisation jump whenever a face gradient crosses it (the thermal eigenvalue changes by tens of per
# cent, at N = 2000 still), and the inner expansion needs Q in C^2.  Here both kinks are rounded by a softplus of
# width sig (in g; PRM['sig'] = 0.05, i.e. 1.25% of g_c): Q_M moves by 1e-3, g_a, g_b by < 1e-3.  sig = 0 restores the paper.
def _sp(z, s):
    return s * np.logaddexp(0.0, z / s) if s > 0 else np.maximum(z, 0.0)


def _sig(z, s):
    return 0.5 * (1 + np.tanh(z / (2 * s))) if s > 0 else (z > 0).astype(float)


def Q(g, P=PRM):
    g = np.asarray(g, float)
    s = P.get("sig", 0.0)
    return (P["chi0"] * g * _sp(P["g_c"] - g, s) / P["g_c"] + P["chi_RR"] * g
            + P["chi_MHD"] * _sp(g - P["g_MHD"], s))


def dQ(g, P=PRM):
    g = np.asarray(g, float)
    s = P.get("sig", 0.0)
    return (P["chi0"] * (_sp(P["g_c"] - g, s) - g * _sig(P["g_c"] - g, s)) / P["g_c"] + P["chi_RR"]
            + P["chi_MHD"] * _sig(g - P["g_MHD"], s))


_TAB = {}


def _table(P):
    key = tuple(sorted((k, P[k]) for k in ("chi0", "chi_RR", "g_c", "g_MHD", "chi_MHD", "sig") if k in P))
    if key not in _TAB:
        gg = np.linspace(0.0, 40.0, 400001)
        qq = Q(gg, P)
        ph = np.concatenate([[0.0], np.cumsum(0.5 * (qq[1:] + qq[:-1]) * np.diff(gg))])
        ipk = int(np.argmax(qq[gg < P["g_c"] + 1]))
        imin = ipk + int(np.argmin(qq[ipk:np.searchsorted(gg, 3 * P["g_c"])]))
        _TAB[key] = dict(g=gg, q=qq, phi=ph, ipk=ipk, imin=imin)
    return _TAB[key]


def Phi(g, P=PRM):
    """int_0^g Q (the bulk free energy), by quadrature on a fine table."""
    T = _table(P)
    return np.interp(g, T["g"], T["phi"])


def g_minus(q, P=PRM):
    T = _table(P)
    i = T["ipk"]
    return np.interp(q, T["q"][:i + 1], T["g"][:i + 1])


def g_plus(q, P=PRM):
    T = _table(P)
    i = T["imin"]
    return np.interp(q, T["q"][i:], T["g"][i:])


def maxwell(P=PRM):
    from scipy.optimize import brentq
    T = _table(P)
    Qmax, Qmin = float(T["q"][T["ipk"]]), float(T["q"][T["imin"]])

    def area(mu):
        ga, gb = g_minus(mu, P), g_plus(mu, P)
        return Phi(gb, P) - Phi(ga, P) - mu * (gb - ga)

    QM = brentq(area, Qmin + 1e-9, Qmax - 1e-9, xtol=1e-13)
    return dict(Q_M=QM, g_a=float(g_minus(QM, P)), g_b=float(g_plus(QM, P)), Qmax=Qmax, Qmin=Qmin)


# ------------------------------------------------------------------------------------------------ physics sources
_PA = 5.6e-13 * 0.25 * 1e40


def sigv(T):
    return m.sigma_v_DT(T)


def S_phys(p, n, P=PRM):
    T = P["T_ref_keV"] * np.maximum(p, 1e-8) / n
    Pa = P["C_alpha"] * _PA * sigv(T) * n ** 2
    Pb = P["C_brem"] * P["Z_eff"] * n ** 2 * np.sqrt(np.maximum(T, 1e-10))
    return Pa - Pb


def P_alpha(p, n, P=PRM):
    T = P["T_ref_keV"] * np.maximum(p, 1e-8) / n
    return P["C_alpha"] * _PA * sigv(T) * n ** 2


def P_brem(p, n, P=PRM):
    T = P["T_ref_keV"] * np.maximum(p, 1e-8) / n
    return P["C_brem"] * P["Z_eff"] * n ** 2 * np.sqrt(np.maximum(T, 1e-10))


def dS_dp(p, n, P=PRM, e=1e-6):
    return (S_phys(p * (1 + e), n, P) - S_phys(p * (1 - e), n, P)) / (2 * e * p)


# ------------------------------------------------------------------------------------------------ the model on a grid
class Model:
    def __init__(self, N=500, P=PRM, S_ext=None, S_tot=S_EXT_TOT):
        self.N = N
        self.h = L / N
        self.x = (np.arange(N) + 0.5) * self.h
        self.xf = np.arange(N + 1) * self.h
        self.P = dict(P)
        if S_ext is None:
            S_ext = base_source(self.x) * (S_tot / BASE_INT)
        self.S_ext = S_ext
        self.n = np.ones(N)                       # frozen density (set by the caller)
        self.MX = maxwell(self.P)

    # face gradients g_k, k = 0..N
    def g_faces(self, p):
        h = self.h
        g = np.empty(self.N + 1)
        g[0] = 0.0
        g[1:-1] = -(p[1:] - p[:-1]) / h
        g[-1] = -(P_PED - p[-1]) / (h / 2)
        return g

    def hv_faces(self, p):
        """nu4 p_xxx on faces (zero on the two boundary faces)."""
        h, N = self.h, self.N
        pe = np.empty(N + 2)                      # p_{-1} .. p_N
        pe[1:-1] = p
        pe[0] = p[0]                              # even ghost at the axis
        pe[-1] = 2 * P_PED - p[-1]                # linear ghost at the edge
        F = np.zeros(N + 1)
        # face k (1..N-1) between cells k-1, k: (p_{k+1} - 3 p_k + 3 p_{k-1} - p_{k-2}) / h^3, pe index = cell + 1
        k = np.arange(1, N)
        F[1:-1] = (pe[k + 2] - 3 * pe[k + 1] + 3 * pe[k] - pe[k - 1]) / h ** 3
        return self.P["nu4"] * F

    def flux(self, p):
        g = self.g_faces(p)
        J = Q(g, self.P) + self.hv_faces(p)
        J[0] = 0.0
        return J, g

    def source(self, p, n=None):
        n = self.n if n is None else n
        return self.S_ext + S_phys(p, n, self.P)

    def rhs(self, p, n=None):
        J, _ = self.flux(p)
        return -(J[1:] - J[:-1]) / self.h + self.source(p, n)

    # banded FD Jacobian (bandwidth 2), returned as a scipy sparse matrix and in LAPACK band form
    def jac(self, p, n=None, eps=1e-7):
        N = self.N
        f0 = None
        Jb = np.zeros((5, N))                     # ab[2 + i - j, j] = J[i, j]
        for c in range(5):
            idx = np.arange(c, N, 5)
            hstep = eps * np.maximum(1.0, np.abs(p[idx]))
            pp = p.copy(); pp[idx] += hstep
            pm = p.copy(); pm[idx] -= hstep
            df = (self.rhs(pp, n) - self.rhs(pm, n))
            for j, hj in zip(idx, hstep):
                lo, hi = max(0, j - 2), min(N, j + 3)
                Jb[2 + np.arange(lo, hi) - j, j] = df[lo:hi] / (2 * hj)
        return Jb

    def jac_analytic(self, p, n=None):
        """Exact banded Jacobian (one-sided Q' at the kinks, as dQ defines it)."""
        N, h, nu = self.N, self.h, self.P["nu4"]
        n = self.n if n is None else n
        g = self.g_faces(p)
        a = dQ(g, self.P)                         # dQ/dg on faces
        Jb = np.zeros((5, N))
        def add(i, j, val):
            Jb[2 + i - j, j] += val
        rows = np.arange(N)
        # Q part: pdot_i = -(Q_{i+1} - Q_i)/h; Q_k depends on p_k (coef -a_k/h) and p_{k-1} (+a_k/h), k = 1..N-1;
        # Q_N depends on p_{N-1} with coef +2 a_N / h; Q_0 = 0
        diag = np.zeros(N); up = np.zeros(N - 1); lo = np.zeros(N - 1)
        # face i+1 (k = i+1) contributes -1/h * dQ_{i+1}
        k = np.arange(1, N)                       # interior faces
        # dQ_k/dp_k = -a_k/h, dQ_k/dp_{k-1} = a_k/h
        # row i = k-1 (face k is its right face): -(1/h) dQ_k
        diag[k - 1] += -(1 / h) * (a[k] / h)       # d/dp_{k-1}
        up[k - 1] += -(1 / h) * (-a[k] / h)        # d/dp_k   (i = k-1, j = k)
        # row i = k (face k is its left face): +(1/h) dQ_k
        diag[k] += (1 / h) * (-a[k] / h)
        lo[k - 1] += (1 / h) * (a[k] / h)          # i = k, j = k-1
        diag[N - 1] += -(1 / h) * (2 * a[N] / h)   # edge face
        # hyperviscous part: F_k = nu (pe[k+2] - 3 pe[k+1] + 3 pe[k] - pe[k-1]) / h^3 with pe = [p0, p, 2 - p_{N-1}]
        key = (N, h, nu)
        if getattr(self, "_hvkey", None) != key:
            self._hvkey, self._hvJ = key, self._hv_band(N, h, nu)
        Jd = self._hvJ
        Jb[2, :] += diag
        Jb[1, 1:] += up                            # (i, i+1) -> ab[1, i+1]
        Jb[3, :-1] += lo                           # (i+1, i) -> ab[3, i]
        Jb += Jd
        Jb[2, :] += dS_dp(p, n, self.P)
        return Jb

    @staticmethod
    def _hv_band(N, h, nu):
        """The (constant) hyperviscous block in band form."""
        def dF(kf):
            d = {}
            for off, cf in ((1, 1.0), (0, -3.0), (-1, 3.0), (-2, -1.0)):   # cell index kf + off
                c = kf + off
                if c < 0:
                    c = 0
                if c > N - 1:                     # ghost 2 p_ped - p_{N-1}
                    d[N - 1] = d.get(N - 1, 0.0) - cf
                    continue
                d[c] = d.get(c, 0.0) + cf
            return {kk: nu * vv / h ** 3 for kk, vv in d.items()}
        Jd = np.zeros((5, N))
        for i in range(N):
            for kf, sgn in ((i + 1, -1.0 / h), (i, 1.0 / h)):
                if 1 <= kf <= N - 1:
                    for j, cf in dF(kf).items():
                        Jd[2 + i - j, j] += sgn * cf
        return Jd

    def band_to_sparse(self, Jb):
        N = self.N
        offs = [2, 1, 0, -1, -2]                  # row r of ab holds diagonal (j - i) = 2 - r
        data = []
        for r, off in enumerate(offs):
            # diagonal 'off' (j - i = off): entries J[i, i+off] = ab[2 - off, i + off]
            if off >= 0:
                data.append(Jb[r, off:])
            else:
                data.append(Jb[r, :N + off])
        return diags(data, offs, shape=(N, N), format="csc")

    # diagnostics
    def W(self, p):
        return float(np.sum(p) * self.h)

    def S_tot(self, p, n=None):
        return float(np.sum(self.source(p, n)) * self.h)

    def Q_edge(self, p):
        g = self.g_faces(p)
        return float(Q(g[-1], self.P))

    def fronts(self, p):
        """Front position by three definitions: the foot (first face with g > g_star), the mid-gradient crossing, and
        the Gibbs dividing surface of the local excess (needs a steady state: the outer solution is g_-+(J))."""
        g = self.g_faces(p)
        xf = self.xf
        gstar = self.P["g_c"] / 2
        gmid = 0.5 * (self.MX["g_a"] + self.MX["g_b"])
        out = {}
        for name, lev in (("foot", gstar), ("mid", gmid)):
            k = np.where(g > lev)[0]
            if len(k) == 0 or k[0] == 0:
                out[name] = np.nan
                continue
            k = k[0]
            out[name] = float(xf[k - 1] + (lev - g[k - 1]) / (g[k] - g[k - 1]) * self.h)
        return out


# ------------------------------------------------------------------------------------------------ steady states
def res_floor(M, p):
    """Roundoff floor of the residual: the stiff nu4 p / h^4 terms cancel to ~1e-16 relative."""
    return 3e-6 + 5e-15 * M.P["nu4"] * float(np.max(np.abs(p))) / M.h ** 4


def newton(M, p0, n=None, tol=None, maxit=60, verbose=False, dptol=1e-11):
    p = p0.copy()
    mx = np.inf
    tol = res_floor(M, p0) if tol is None else tol
    for it in range(maxit):
        r = M.rhs(p, n)
        err = np.max(np.abs(r))
        if verbose:
            print(f"    newton it {it}: |r| = {err:.3e}")
        if err < tol or mx < dptol or (mx < 1e-9 and err < 30 * tol):
            return p, True, err
        Jb = M.jac_analytic(p, n)
        dp = solve_banded((2, 2), Jb, -r)
        lam = 1.0
        mx = np.max(np.abs(dp))
        if mx > 0.3:
            lam = 0.3 / mx
        p = p + lam * dp
    return p, False, err


def newton_bordered(M, p0, c0, set_c, target_W, n=None, tol=None, maxit=60, dc=1e-6, dptol=1e-11):
    """Solve rhs(p; c) = 0 and W(p) = target_W for (p, c): continuation in W through a fold.  set_c(M, c) applies the
    parameter (it may change M.P or M.S_ext)."""
    p, c = p0.copy(), float(c0)
    N, h = M.N, M.h
    mx = np.inf
    tol = res_floor(M, p0) if tol is None else tol
    for it in range(maxit):
        set_c(M, c)
        r = M.rhs(p, n)
        rW = M.W(p) - target_W
        err = max(np.max(np.abs(r)), abs(rW))
        if (err < tol or mx < dptol or (mx < 1e-9 and err < 30 * tol)) and it > 0:
            return p, c, True
        Jb = M.jac_analytic(p, n)
        Js = M.band_to_sparse(Jb)
        set_c(M, c + dc); rp = M.source(p, n)       # the parameter enters only the source: no stiff cancellation
        set_c(M, c - dc); rm = M.source(p, n)
        set_c(M, c)
        b = (rp - rm) / (2 * dc)
        # bordered system [[J, b], [h 1^T, 0]] [dp, dcc] = [-r, -rW] by block elimination
        lu = splu(Js)
        y1 = lu.solve(-r)
        y2 = lu.solve(b)
        wv = np.full(N, h)
        dcc = (rW + wv @ y1) / (wv @ y2)
        dp = y1 - dcc * y2
        lam = 1.0
        mx = np.max(np.abs(dp))
        if mx > 0.3:
            lam = 0.3 / mx
        p = p + lam * dp
        c = c + lam * dcc
    set_c(M, c)
    return p, c, False


def modes(M, p, n=None, k=6):
    """Eigenvalues of the Jacobian nearest zero (shift-invert), right and left eigenvectors of the leading one."""
    Jb = M.jac_analytic(p, n)
    Js = M.band_to_sparse(Jb)
    lam, V = eigs(Js, k=k, sigma=0.0, which="LM")
    order = np.argsort(-lam.real)
    lam, V = lam[order], V[:, order]
    lamT, U = eigs(Js.T.tocsc(), k=k, sigma=0.0, which="LM")
    oT = np.argsort(-lamT.real)
    lamT, U = lamT[oT], U[:, oT]
    v = V[:, 0].real; v = v / (np.sum(v) * M.h)
    u = U[:, 0].real; u = u / (np.sum(u * v) * M.h)
    return dict(lam=lam.real, v=v, u=u, Js=Js, Jb=Jb, asym=float(abs(Js - Js.T).max() / abs(Js).max()))


def directional(M, p, d, n=None, e=1e-5):
    """Slopes of S_tot and Q_edge along the direction d normalised to unit Delta W."""
    d = d / (np.sum(d) * M.h)
    Sp, Sm = M.S_tot(p + e * d, n), M.S_tot(p - e * d, n)
    Qp, Qm = M.Q_edge(p + e * d), M.Q_edge(p - e * d)
    return (Sp - Sm) / (2 * e), (Qp - Qm) / (2 * e)


# ------------------------------------------------------------------------------------------------ time integration
def integrate(M, p0, T_end, t_eval, n_of_t=None, rtol=1e-8, atol=1e-10, max_step=np.inf, events=None):
    N = M.N
    sp = diags([np.ones(N - abs(k)) for k in range(-2, 3)], list(range(-2, 3)), shape=(N, N))
    if n_of_t is None:
        fun = lambda t, p: M.rhs(p)
        jac = lambda t, p: M.band_to_sparse(M.jac_analytic(p))
    else:
        fun = lambda t, p: M.rhs(p, n_of_t(t))
        jac = lambda t, p: M.band_to_sparse(M.jac_analytic(p, n_of_t(t)))
    sol = solve_ivp(fun, (0.0, T_end), p0, method="BDF", t_eval=t_eval, jac=jac, rtol=rtol, atol=atol,
                    max_step=max_step, events=events)
    if sol.status < 0:
        raise RuntimeError(sol.message)
    return sol


# ------------------------------------------------------------------------------------------------ density (one-way)
class Density:
    """n_t = D_n (n_xx + s_N(x)), n_x(0) = 0, n(L) = 1, on the same cells; solved exactly by the eigen-expansion of
    the discrete Laplacian, so n(x, t) = n(x; s) with s = D_n t: the density path is fixed and D_n sets only its rate."""

    def __init__(self, M, s_N, n0=None):
        N, h = M.N, M.h
        dmain = np.full(N, -2.0) / h ** 2
        dmain[0] = -1.0 / h ** 2                  # Neumann at the axis
        dmain[-1] = -3.0 / h ** 2                 # Dirichlet n(L) = 1 through a linear ghost
        doff = np.ones(N - 1) / h ** 2
        self.lam, self.V = eigh_tridiagonal(dmain, doff)    # symmetric: A = V diag(lam) V^T
        rhsb = np.zeros(N); rhsb[-1] = 2.0 / h ** 2         # boundary contribution of n(L) = 1
        A = diags([doff, dmain, doff], [-1, 0, 1]).toarray()
        self.n_ss = np.linalg.solve(A, -(s_N + rhsb))
        self.n0 = np.ones(N) if n0 is None else n0
        self.c0 = self.V.T @ (self.n0 - self.n_ss)
        self.lam1 = float(-self.lam.max())                  # slowest decay rate per unit D_n

    def __call__(self, s):
        return self.n_ss + self.V @ (np.exp(self.lam * s) * self.c0)

    def dnds(self, s):
        return self.V @ (self.lam * np.exp(self.lam * s) * self.c0)
