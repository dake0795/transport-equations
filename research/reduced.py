"""Stiff (BDF) integrator for the single-field reduced model, for long-time
steady-state studies.  Reuses the flux function and physics sources of
``flux_transport_model_clean`` and the paper's discretisation/boundary
treatment, but integrates implicitly so that runs to t ~ O(10) (the chi_RR
time L^2/chi_RR = 20) are cheap.

    p_t = -d_x Q(g) - nu4 p_xxxx + S_ext(x) + P_alpha(p) - P_brem(p),  g = -p_x
    p_x(0) = 0,  p(L) = p_ped.
"""
import os
import pathlib
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import flux_transport_model_clean as m  # noqa: E402

L = 1.0
DX = 0.005

PAPER = {
    "chi0": 10.0, "chi_core": 10.0, "chi_RR": 0.05, "g_c": 4.0,
    "g_stiff": 3, "n_stiff": 2, "boundaries": [], "deltas": [],
    "flux_models": ["nl"], "nu4": 5e-4, "n_smooth": 3,
    "lf_dissipation": 0, "g_MHD": 3.6, "chi_MHD": 2.0,
    "C_brem": 0.01, "Z_eff": 1.0, "C_alpha": 4e-6,
    "f_deuterium": 0.5, "f_tritium": 0.5,
    "T_ref_keV": 10.0, "n_ref": 1.0, "n_ref_20": 1.0,
}
P_CORE, P_PED, M_EXP = 2.0, 1.0, 2


def grid(dx=DX):
    return np.linspace(0.0, L, int(L / dx))


def base_source(x):
    return 5.0 * np.exp(-(x ** 2) / ((0.25 * L) ** 2))


def initial_profile(x, g_tgt_over_gstar=1.8, params=PAPER):
    dx = x[1] - x[0]
    shape = 1.0 - (x / L) ** M_EXP
    scale = (g_tgt_over_gstar * params["g_c"] / 2) / np.max(-np.gradient(shape, dx))
    return P_PED + scale * (P_CORE - P_PED) * shape


def physics_source(p, params):
    return m.alpha_heating(p, params) - m.bremsstrahlung(p, params)


def make_rhs(x, params, S_ext):
    """Return f(t, p) for the interior unknowns with BCs folded in."""
    dx = x[1] - x[0]
    nu4 = params.get("nu4", 0.0)
    n_smooth = params.get("n_smooth", 0)
    conservative = params.get("nu4_conservative", False)
    x_face = 0.5 * (x[1:] + x[:-1])

    def rhs(t, p):
        p_bc = p.copy()
        p_bc[0] = p_bc[1]
        p_bc[-1] = P_PED
        g_face = -(p_bc[1:] - p_bc[:-1]) / dx
        for _ in range(n_smooth):
            g_s = g_face.copy()
            g_s[1:-1] = 0.25 * g_face[:-2] + 0.5 * g_face[1:-1] + 0.25 * g_face[2:]
            g_face = g_s
        Q_face = m.flux_function(g_face, x_face, params)
        dQdx = np.zeros_like(p_bc)
        dQdx[1:-1] = (Q_face[1:] - Q_face[:-1]) / dx
        dQdx[0] = Q_face[0] / dx
        dQdx[-1] = -Q_face[-1] / dx
        d4 = np.zeros_like(p_bc)
        if nu4 > 0 and conservative:
            # hyperviscous flux F = p_xxx on faces, zero on the boundary faces,
            # so that hyperviscosity redistributes but never transports
            # pressure across x=0 or x=L.
            F = np.zeros(len(p_bc) - 1)
            F[1:-1] = (p_bc[3:] - 3 * p_bc[2:-1] + 3 * p_bc[1:-2] - p_bc[:-3]) / dx ** 3
            d4[1:-1] = (F[1:] - F[:-1]) / dx
        elif nu4 > 0:
            d4[2:-2] = (p_bc[:-4] - 4 * p_bc[1:-3] + 6 * p_bc[2:-2]
                        - 4 * p_bc[3:-1] + p_bc[4:]) / dx ** 4
        f = -dQdx - nu4 * d4 + S_ext + physics_source(p_bc, params)
        f[0] = 0.0      # ghost-like boundary cells held by BCs
        f[-1] = 0.0
        return f

    return rhs


def run(pb, params=PAPER, g_tgt=1.8, T_end=5.0, n_out=201, p0=None,
        physics=True, source_scale=None, rtol=1e-6, atol=1e-8):
    """Integrate to T_end.  ``pb`` sets int S_ext = pb*Q_edge(0) - int S_phys(0)
    exactly as in the paper (``source_scale`` overrides with an explicit
    multiplier of base_source).  Returns dict with x, t, p(t,x), S_tot."""
    params = dict(params)
    if not physics:
        params["C_alpha"] = 0.0
        params["C_brem"] = 0.0
    x = grid()
    dx = x[1] - x[0]
    if p0 is None:
        p0 = initial_profile(x, g_tgt, params)
    p0 = p0.copy()
    p0[0] = p0[1]
    p0[-1] = P_PED
    Q_face0, _ = m.compute_face_flux(p0, dx, x, P_PED, params)
    Q_edge0 = Q_face0[-1]
    S0 = np.trapezoid(base_source(x), x)
    Sphys0 = np.trapezoid(physics_source(p0, params), x)
    if source_scale is None:
        source_scale = max((pb * Q_edge0 - Sphys0) / S0, 0.0)
    S_ext = source_scale * base_source(x)
    rhs = make_rhs(x, params, S_ext)
    N = len(x)
    bw = 2 + params.get("n_smooth", 0)
    sparsity = diags([np.ones(N - k) for k in range(-bw, bw + 1)],
                     list(range(-bw, bw + 1)), shape=(N, N))
    t_eval = np.linspace(0.0, T_end, n_out)
    sol = solve_ivp(rhs, (0.0, T_end), p0, method="BDF", t_eval=t_eval,
                    jac_sparsity=sparsity, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    P = sol.y.T.copy()
    P[:, 0] = P[:, 1]          # Neumann core cell, as in solving_loop
    P[:, -1] = P_PED
    return dict(x=x, t=sol.t, p=P, S_ext=S_ext, params=params,
                S_ext_tot=np.trapezoid(S_ext, x), Q_edge0=Q_edge0,
                source_scale=source_scale)


def diagnostics(res):
    """W(t), Q_edge(t), S_tot(t) (ext + physics), max g, foot position."""
    x, p, params = res["x"], res["p"], res["params"]
    dx = x[1] - x[0]
    gstar = params["g_c"] / 2
    W, Qe, St, gmax, xfoot = [], [], [], [], []
    for pk in p:
        Qf, xf = m.compute_face_flux(pk, dx, x, P_PED, params)
        g = -(pk[1:] - pk[:-1]) / dx
        W.append(np.trapezoid(pk, x))
        Qe.append(Qf[-1])
        St.append(res["S_ext_tot"] + np.trapezoid(physics_source(pk, params), x))
        gmax.append(g.max())
        sup = np.where(g > gstar)[0]
        xfoot.append(xf[sup[0]] if len(sup) else np.nan)
    return dict(t=res["t"], W=np.array(W), Q_edge=np.array(Qe),
                S_tot=np.array(St), g_max=np.array(gmax), x_foot=np.array(xfoot))
