"""Out-of-sample test of the mechanism: step the drive down by the same 1.89 but remove it ONLY near the axis
(Gaussian, width 0.06), so the source inside the pedestal is unchanged and eta0 = 0.  The investigator's mechanism
predicts the 'source change' share of the lag vanishes and the half-time drops toward the front-motion-only value;
the Stefan law predicts (nearly) the same half-time as the paper's step.  Reduced model (paper discretisation) vs
own sharp two-phase solver vs Stefan law."""
import sys, numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import erf
from scipy.sparse import diags
RES = "/rds/project/rds-aSo1XX0UOlw/ir-kenn3/TRANSPORT_EQUATIONS/transport-equations/research"
sys.path.insert(0, RES); sys.path.insert(0, RES + "/..")
import reduced as R
from maxwell import maxwell, Q as Qf

L, PPED = 1.0, 1.0
x = R.grid(); dx = x[1]-x[0]; xfc = 0.5*(x[1:]+x[:-1]); x0 = dx/2
S0 = np.trapezoid(R.base_source(x), x)
Fb = lambda z: 5*0.25*np.sqrt(np.pi)/2*erf(np.asarray(z, float)/0.25)
W = 0.06
def make_src(Sint, drop=0.0):
    A = drop/(W*np.sqrt(np.pi)/2*erf(1/W))
    s = lambda z: Sint/S0*5*np.exp(-np.asarray(z, float)**2/0.0625) - A*np.exp(-np.asarray(z, float)**2/W**2)
    Fc = lambda z: Sint/S0*Fb(z) - A*W*np.sqrt(np.pi)/2*erf(np.asarray(z, float)/W)
    return s, Fc

def half(t, a):
    a = np.asarray(a); f = np.abs(a - a[0]) - 0.5*abs(a[-1]-a[0]); k = np.argmax(f > 0)
    return t[k-1] + (t[k]-t[k-1])*(-f[k-1])/(f[k]-f[k-1])

def cross(g, lev):
    j = np.where(g > lev)[0][0]
    return xfc[j-1] + (lev - g[j-1])/(g[j]-g[j-1])*dx

def reduced(P, s_to, p0, T=3.0):
    S_ext = s_to(x); rhs = R.make_rhs(x, P, S_ext); N = len(x); bw = 2 + P["n_smooth"]
    sp = diags([np.ones(N-k) for k in range(-bw, bw+1)], list(range(-bw, bw+1)), shape=(N, N))
    p0 = p0.copy(); p0[0] = p0[1]; p0[-1] = PPED
    t = np.unique(np.concatenate([np.linspace(0, 0.4, 2001), np.linspace(0.4, T, 300)]))
    s = solve_ivp(rhs, (0, T), p0, method="BDF", t_eval=t, jac_sparsity=sp, rtol=1e-8, atol=1e-10)
    Pt = s.y.T; Pt[:, 0] = Pt[:, 1]; Pt[:, -1] = PPED
    return s.t, Pt

def sharp(Pm, QM, ga, gb, src_from, src_to, Nc=120, Np=400, T=3.0):
    xi_c = (np.arange(Nc)+0.5)/Nc; xi_p = (np.arange(Np)+0.5)/Np
    def fac(src):
        _, Fc = src
        cs = lambda xl, xr: (Fc(xr) - Fc(xl))/(xr - xl)
        def rhs(t, y):
            u, v, xf = y[:Nc], y[Nc:Nc+Np], y[-1]; hc = (xf-x0)/Nc; hp = (L-xf)/Np
            ec = x0 + (xf-x0)*np.arange(Nc+1)/Nc
            Qc = np.empty(Nc+1); Qc[0] = 0; Qc[-1] = QM; Qc[1:-1] = Qf(np.maximum(-(u[1:]-u[:-1])/hc, 0), Pm)
            ptc = (Qc[:-1]-Qc[1:])/hc + cs(ec[:-1], ec[1:])
            ep = xf + (L-xf)*np.arange(Np+1)/Np
            Qp = np.empty(Np+1); Qp[0] = QM; Qp[1:-1] = Qf(-(v[1:]-v[:-1])/hp, Pm); Qp[-1] = Qf((v[-1]-PPED)/(hp/2), Pm)
            ptp = (Qp[:-1]-Qp[1:])/hp + cs(ep[:-1], ep[1:])
            gc = np.empty(Nc); gc[:-1] = -(u[1:]-u[:-1])/hc; gc[-1] = ga; gc = 0.5*(np.concatenate([[0], gc[:-1]]) + gc)
            gp = np.empty(Np); gp[1:] = -(v[1:]-v[:-1])/hp; gp[0] = gb; gp = 0.5*(gp + np.concatenate([gp[1:], [(v[-1]-PPED)/(hp/2)]]))
            cm = (u[-1]-ga*hc/2) - (v[0]+gb*hp/2)
            A = -gc[-1]*xi_c[-1] - ga/(2*Nc) + gp[0]*(1-xi_p[0]) + gb/(2*Np)
            V = (-(ptc[-1]-ptp[0]) - 50*cm)/A
            return np.concatenate([ptc - gc*xi_c*V, ptp - gp*(1-xi_p)*V, [V]])
        return rhs
    _, Ff = src_from
    xM = brentq(lambda z: Ff(z) - Ff(x0) - QM, x0+1e-3, L)
    gh = np.linspace(maxwell(Pm)["gmin"], 60, 200001); gl = np.linspace(0, 2, 40001)
    ghi = lambda m: np.interp(m, Qf(gh, Pm), gh); glo = lambda m: np.interp(m, Qf(gl, Pm), gl)
    zz = np.linspace(xM, L, 4001); gz = ghi(Ff(zz)-Ff(x0)); zp = xM + (L-xM)*xi_p
    v = PPED + np.array([np.trapezoid(gz[zz >= z], zz[zz >= z]) for z in zp]); pf = PPED + np.trapezoid(gz, zz)
    cc = np.linspace(x0, xM, 4001); gcz = glo(Ff(cc)-Ff(x0)); zc = x0 + (xM-x0)*xi_c
    u = pf + np.array([np.trapezoid(gcz[cc >= z], cc[cc >= z]) for z in zc])
    y0 = solve_ivp(fac(src_from), (0, 2), np.concatenate([u, v, [xM]]), method="BDF", rtol=1e-8, atol=1e-10).y[:, -1]
    t = np.unique(np.concatenate([np.linspace(0, 0.4, 2001), np.linspace(0.4, T, 300)]))
    s = solve_ivp(fac(src_to), (0, T), y0, method="BDF", t_eval=t, rtol=1e-8, atol=1e-10)
    return s.t, s.y[-1]

def stefan(QM, jump, src_to, xstart, t):
    _, Fc = src_to
    hit = lambda tt, y: min(y[0]-0.03, 0.97-y[0]); hit.terminal = True
    s = solve_ivp(lambda tt, y: [-(Fc(y[0]) - Fc(x0) - QM)/((y[0]-x0)*jump)], (0, t[-1]), [xstart], t_eval=t, rtol=1e-9, atol=1e-11, events=hit)
    return s.y[0]

for chi in (2.0, 20.0):
    Pm = dict(chi0=10.0, chi_RR=0.05, g_c=4.0, g_MHD=3.6, chi_MHD=chi)
    mx = maxwell(Pm); QM, ga, gb = mx["Q_M"], mx["g_a"], mx["g_b"]
    P = dict(R.PAPER, nu4_conservative=True, chi_MHD=chi, C_alpha=0.0, C_brem=0.0)
    S_lo, S_hi = QM + 1.71, QM + 3.60
    hi = make_src(S_hi); lo_paper = make_src(S_lo); lo_core = make_src(S_hi, drop=S_hi - S_lo)
    for nm, sv in (("hi", hi), ("lo_core", lo_core), ("lo_paper", lo_paper)):
        print(f"   chi {chi}: source {nm}: integral {sv[1](1.0)-sv[1](x0):.3f}, min S {sv[0](np.linspace(0,1,1001)).min():.2f}, "
              f"x_M {brentq(lambda z: sv[1](z)-sv[1](x0)-QM, x0+1e-3, 1):.4f}")
    # steady states
    st_hi = R.run(None, params=P, g_tgt=1.8, T_end=10.0, n_out=3, physics=False, source_scale=S_hi/S0)["p"][-1]
    t, Pt = reduced(P, lo_core[0], st_hi, T=10.0); st_lo_core = Pt[-1]
    for label, sf, stt, s0 in (("core-only step DOWN", hi, lo_core, st_hi), ("core-only step UP", lo_core, hi, st_lo_core)):
        t, Pt = reduced(P, stt[0], s0)
        G = -(Pt[:, 1:]-Pt[:, :-1])/dx
        mid = np.array([cross(g, 0.5*(ga+gb)) for g in G]); foot = np.array([cross(g, 2.0) for g in G])
        ts, xs = sharp(Pm, QM, ga, gb, sf, stt)
        law = stefan(QM, gb-ga, stt, mid[0], t)
        xMf = brentq(lambda z: sf[1](z)-sf[1](x0)-QM, x0+1e-3, 1)
        lawM = stefan(QM, gb-ga, stt, xMf, t)
        print(f"chi={chi:4.0f} {label}: x_mid {mid[0]:.4f}->{mid[-1]:.4f}; measured half mid {half(t, mid):.4f} foot {half(t, foot):.4f}; "
              f"sharp two-phase {half(ts, xs):.4f} (ratio {half(ts, xs)/half(t, mid):.2f}); Stefan law {half(t, law):.4f} (from x_M {half(t, lawM):.4f}); "
              f"measured/law {half(t, mid)/half(t, lawM):.2f}", flush=True)
