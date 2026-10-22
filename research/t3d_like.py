"""Two-channel reduced model with T3D's channel structure, no cliff.

Question: does a flux-driven steady state exist when the turbulent branch of
both channels is stabilised by the TOTAL pressure gradient (alpha), the only
high-gradient transport is a Rechester--Rosenbluth-like g_e^3 branch for the
electrons and a weak linear (neoclassical) branch for the ions, and the two
channels are coupled collisionally?  And does the burn saturate without a
temperature-dependent sink?

Fields p_e(x,t), p_i(x,t) on x in [0,1] with a fixed background density
n = 1; g_s = -d p_s/dx, g = g_e + g_i plays the role of alpha.

    Q_s = chi0_s * g_s * max(1 - g/g_c, 0)          turbulent branch (QL-like:
                                                   growth rate falls with g)
        + chi_RR * g_e^3 / g_ref^2   (electrons)    RR-type residual channel
        + chi_neo_s * g_s                           neoclassical residual
    d p_e/dt = -dQ_e/dx + S_e + P_alpha,e - P_rad + Q_ei
    d p_i/dt = -dQ_i/dx + S_i + P_alpha,i          - Q_ei
    Q_ei = nu_ei (p_i - p_e)

Run:  python t3d_like.py            (writes t3d_like.npz and prints the end states)

STATUS (2026-10-22): DRAFT.  The integrator stalls (BDF with a sparse Jacobian and banded LSODA both take
> 15 min for t = 1 at N = 101): the g_e^3 branch plus the smoothed face gradients leave a grid-scale mode
that the hyperviscosity does not damp.  Not yet a result; see research/NOTES.md.
"""
import sys
import pathlib

import numpy as np
from scipy.integrate import solve_ivp

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from flux_transport_model_two_species_clean import sigma_v_DT, _alpha_ion_fraction  # noqa: E402

N = 101
x = np.linspace(0.0, 1.0, N)
dx = x[1] - x[0]
xf = 0.5 * (x[1:] + x[:-1])

P = dict(
    chi0_e=10.0, chi0_i=5.0, g_c=4.0,          # turbulent branch, stabilised by g = g_e + g_i
    chi_RR=0.05, g_ref=1.0,                    # RR-type electron branch chi_RR g_e^3 / g_ref^2
    chi_neo_e=0.01, chi_neo_i=0.05,            # linear residual channels
    nu_ei=2.0,                                 # collisional exchange rate
    nu4=5e-3, n_smooth=3,                      # hyperviscosity (front ~2 cells on this grid) and face smoothing
    S_ext=4.0, f_ext_e=0.9,                    # external heating, mostly electrons (EC)
    C_alpha=0.0, T_ref_keV=3.0,                # alpha heating (model units: T = p * T_ref)
    C_brem=0.0, C_sync=0.0,                    # electron sinks ~ sqrt(T_e) and T_e^2.5
    p_ped=0.5,
)


def source_profile(x):
    s = np.exp(-((x - 0.0) / 0.45) ** 2)
    return s / np.trapezoid(s, x)


S_SHAPE = source_profile(x)


def _smooth(g, n):
    for _ in range(n):
        g = g.copy(); g[1:-1] = 0.25 * g[:-2] + 0.5 * g[1:-1] + 0.25 * g[2:]
    return g


def fluxes(pe, pi, prm):
    ge = _smooth(-np.diff(pe) / dx, prm["n_smooth"])
    gi = _smooth(-np.diff(pi) / dx, prm["n_smooth"])
    g = np.maximum(ge + gi, 0.0)
    stab = np.maximum(1.0 - g / prm["g_c"], 0.0)
    Qe = prm["chi0_e"] * np.maximum(ge, 0) * stab + prm["chi_RR"] * np.maximum(ge, 0) ** 3 / prm["g_ref"] ** 2 \
        + prm["chi_neo_e"] * ge
    Qi = prm["chi0_i"] * np.maximum(gi, 0) * stab + prm["chi_neo_i"] * gi
    return Qe, Qi, ge, gi


def sources(pe, pi, prm):
    T_e = pe * prm["T_ref_keV"]
    T_i = pi * prm["T_ref_keV"]
    Se = prm["S_ext"] * prm["f_ext_e"] * S_SHAPE
    Si = prm["S_ext"] * (1 - prm["f_ext_e"]) * S_SHAPE
    if prm["C_alpha"] > 0:
        Pfus = prm["C_alpha"] * sigma_v_DT(T_i) * 1e20 * 0.25   # n_D n_T = n^2/4, n = 1e20
        fi = _alpha_ion_fraction(T_e, 1.0, 1, 2.5, 1.0)
        Se = Se + (1 - fi) * Pfus
        Si = Si + fi * Pfus
    Prad = prm["C_brem"] * np.sqrt(np.maximum(T_e, 1e-3)) + prm["C_sync"] * np.maximum(T_e, 0) ** 2.5
    Qei = prm["nu_ei"] * (pi - pe)
    return Se - Prad + Qei, Si - Qei, Se + Si - Prad


def rhs(t, y, prm):
    pe, pi = y[:N].copy(), y[N:].copy()
    pe[-1] = pi[-1] = prm["p_ped"]
    Qe, Qi, ge, gi = fluxes(pe, pi, prm)
    # hyperviscous flux  -nu4 d^3 p / dx^3 on faces (flux form, conservative)
    def hv(p):
        d3 = np.zeros(N - 1)
        d3[1:-1] = (p[3:] - 3 * p[2:-1] + 3 * p[1:-2] - p[:-3]) / dx ** 3
        return -prm["nu4"] * d3
    Qe = Qe + hv(pe)
    Qi = Qi + hv(pi)
    Re, Ri, _ = sources(pe, pi, prm)

    def div(Q):
        d = np.zeros(N)
        d[1:-1] = (Q[1:] - Q[:-1]) / dx
        d[0] = Q[0] / dx          # Q = 0 on axis
        d[-1] = 0.0               # Dirichlet edge
        return d
    dpe = -div(Qe) + Re
    dpi = -div(Qi) + Ri
    dpe[-1] = dpi[-1] = 0.0
    return np.r_[dpe, dpi]


def _rhs_interleaved(t, y, prm):
    f = rhs(t, np.r_[y[0::2], y[1::2]], prm)
    out = np.empty_like(y); out[0::2] = f[:N]; out[1::2] = f[N:]
    return out


def run(prm, T_end=10.0, y0=None, n_out=201):
    if y0 is None:
        p0 = prm["p_ped"] + 1.5 * (1 - x ** 2)
        y0 = np.r_[p0, 0.8 * p0 + 0.2 * prm["p_ped"]]
    yi = np.empty_like(y0); yi[0::2] = y0[:N]; yi[1::2] = y0[N:]
    bw = 2 * (3 + prm["n_smooth"]) + 1          # stencil half-width in the interleaved ordering
    sol = solve_ivp(_rhs_interleaved, (0, T_end), yi, args=(prm,), method="LSODA",
                    t_eval=np.linspace(0, T_end, n_out), lband=bw, uband=bw, rtol=1e-6, atol=1e-8)
    if sol.status != 0:
        print("  integration stopped:", sol.message, "at t =", sol.t[-1] if len(sol.t) else None)
    pe, pi = sol.y[0::2].T, sol.y[1::2].T
    W = np.trapezoid(pe + pi, x, axis=1)
    Qedge, S, gmax = [], [], []
    for a, b in zip(pe, pi):
        a = a.copy(); b = b.copy(); a[-1] = b[-1] = prm["p_ped"]
        Qe, Qi, ge, gi = fluxes(a, b, prm)
        Qedge.append(Qe[-1] + Qi[-1])
        S.append(np.trapezoid(sources(a, b, prm)[2], x))
        gmax.append((ge + gi).max())
    return dict(t=sol.t, pe=pe, pi=pi, W=W, Qedge=np.array(Qedge), S=np.array(S), gmax=np.array(gmax),
                status=sol.status)


if __name__ == "__main__":
    out = {}
    cases = {
        "ext_only":      dict(P),
        "alpha_no_sink": dict(P, C_alpha=2.0e-5),
        "alpha_brem":    dict(P, C_alpha=2.0e-5, C_brem=0.4),
        "alpha_sync":    dict(P, C_alpha=2.0e-5, C_brem=0.4, C_sync=0.004),
    }
    for name, prm in cases.items():
        r = run(prm, T_end=10.0)
        out[name] = r
        a, b = r["pe"][-1], r["pi"][-1]
        ge = -np.gradient(a, dx); gi = -np.gradient(b, dx)
        print(f"{name:14s} status={r['status']} W {r['W'][0]:.2f}->{r['W'][-1]:.2f}  dW/dt(end)={np.gradient(r['W'], r['t'])[-1]:+.3f}  "
              f"S(end)={r['S'][-1]:.2f} Qedge(end)={r['Qedge'][-1]:.2f}  max g_e={ge.max():.1f} max g_i={gi.max():.1f}  "
              f"T_e(0)={a[0]*prm['T_ref_keV']:.1f} T_i(0)={b[0]*prm['T_ref_keV']:.1f} keV")
    np.savez(HERE / "t3d_like.npz", **{f"{k}__{kk}": np.asarray(vv) for k, v in out.items() for kk, vv in v.items()})
