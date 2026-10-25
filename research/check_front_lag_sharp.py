"""Independent sharp-interface two-phase solver (own implementation, cell-centred FV in Landau coordinates, front
velocity from the continuity constraint of p across the front with flux Q_M on both sides).  No measured input.
Also lumped-core / frozen-pedestal / quasi-static variants and hand checks of Gamma, eta0, tau_p, tau_f."""
import sys, numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
from scipy.special import erf
RES = "/rds/project/rds-aSo1XX0UOlw/ir-kenn3/TRANSPORT_EQUATIONS/transport-equations/research"
sys.path.insert(0, RES); sys.path.insert(0, RES + "/..")
from maxwell import maxwell, Q as Qf

L, PPED = 1.0, 1.0
xg = np.linspace(0, 1, 200); dxg = xg[1]-xg[0]; x0 = dxg/2
S0 = np.trapezoid(5*np.exp(-xg**2/0.0625), xg)
F = lambda x: 5*0.25*np.sqrt(np.pi)/2*erf(np.asarray(x)/0.25)

def half(t, a):
    a = np.asarray(a); f = np.abs(a - a[0]) - 0.5*abs(a[-1]-a[0]); k = np.argmax(f > 0)
    return t[k-1] + (t[k]-t[k-1])*(-f[k-1])/(f[k]-f[k-1])

def solve(chi, Sfrom, Sto, Nc=120, Np=400, t_end=3.0, core_pde=True):
    Pm = dict(chi0=10.0, chi_RR=0.05, g_c=4.0, g_MHD=3.6, chi_MHD=chi)
    mx = maxwell(Pm); QM, ga, gb = mx["Q_M"], mx["g_a"], mx["g_b"]
    Qss = lambda x, Si: Si/S0*(F(x) - F(x0))
    cellsrc = lambda xl, xr, Si: Si/S0*(F(xr) - F(xl))/(xr - xl)
    xi_c = (np.arange(Nc)+0.5)/Nc; xi_p = (np.arange(Np)+0.5)/Np
    def rhs_factory(Si):
        def rhs(t, y):
            u, v, xf = y[:Nc], y[Nc:Nc+Np], y[-1]
            hc = (xf - x0)/Nc; hp = (L - xf)/Np
            # core
            ec = x0 + (xf - x0)*np.arange(Nc+1)/Nc
            Qc = np.empty(Nc+1); Qc[0] = 0.0; Qc[-1] = QM
            Qc[1:-1] = Qf(np.maximum(-(u[1:]-u[:-1])/hc, 0), Pm)
            ptc = (Qc[:-1] - Qc[1:])/hc + cellsrc(ec[:-1], ec[1:], Si)
            if not core_pde:   # lumped core: uniform lift
                ptc = np.full(Nc, (Qss(xf, Si) - QM)/(xf - x0))
            ep = xf + (L - xf)*np.arange(Np+1)/Np
            Qp = np.empty(Np+1); Qp[0] = QM
            Qp[1:-1] = Qf(-(v[1:]-v[:-1])/hp, Pm); Qp[-1] = Qf((v[-1]-PPED)/(hp/2), Pm)
            ptp = (Qp[:-1] - Qp[1:])/hp + cellsrc(ep[:-1], ep[1:], Si)
            gc = np.empty(Nc); gc[:-1] = -(u[1:]-u[:-1])/hc; gc[-1] = ga
            gc = 0.5*(np.concatenate([[0], gc[:-1]]) + gc)   # cell-centre gradient
            gp = np.empty(Np); gp[1:] = -(v[1:]-v[:-1])/hp; gp[0] = gb
            gp = 0.5*(gp + np.concatenate([gp[1:], [(v[-1]-PPED)/(hp/2)]]))
            # constraint c = (u_N - ga hc/2) - (v_1 + gb hp/2) = 0; dc/dt = 0 fixes V
            #   du_N/dt = ptc_N - gc_N xi_N V ; dv_1/dt = ptp_1 - gp_1 (1 - xi_1) V ; dhc/dt = V/Nc ; dhp/dt = -V/Np
            cmis = (u[-1] - ga*hc/2) - (v[0] + gb*hp/2)
            A = -gc[-1]*xi_c[-1] - ga/(2*Nc) + gp[0]*(1 - xi_p[0]) + gb/(2*Np)
            V = (-(ptc[-1] - ptp[0]) - 50.0*cmis)/A
            du = ptc - gc*xi_c*V; dv = ptp - gp*(1 - xi_p)*V
            return np.concatenate([du, dv, [V]])
        return rhs
    # initial: analytic steady state at Sfrom, then relax 2.0
    xM = lambda Si: brentq(lambda xx: Qss(xx, Si) - QM, x0+1e-4, L)
    gl = np.linspace(0, 2.0, 40001); gh = np.linspace(mx["gmin"], 60, 200001)
    glo = lambda m: np.interp(m, Qf(gl, Pm), gl); ghi = lambda m: np.interp(m, Qf(gh, Pm), gh)
    xf = xM(Sfrom)
    zp = xf + (L-xf)*xi_p; zz = np.linspace(xf, L, 4001); gpz = ghi(Qss(zz, Sfrom))
    Ip = np.array([np.trapezoid(gpz[zz >= z], zz[zz >= z]) for z in zp])
    v = PPED + Ip; pf = PPED + np.trapezoid(gpz, zz)
    zc = x0 + (xf-x0)*xi_c; cc = np.linspace(x0, xf, 4001); gcz = glo(Qss(cc, Sfrom))
    u = pf + np.array([np.trapezoid(gcz[cc >= z], cc[cc >= z]) for z in zc])
    y0 = np.concatenate([u, v, [xf]])
    y0 = solve_ivp(rhs_factory(Sfrom), (0, 2.0), y0, method="BDF", rtol=1e-8, atol=1e-10).y[:, -1]
    t = np.unique(np.concatenate([np.linspace(0, 0.4, 2001), np.linspace(0.4, t_end, 300)]))
    s = solve_ivp(rhs_factory(Sto), (0, t_end), y0, method="BDF", t_eval=t, rtol=1e-8, atol=1e-10)
    return s.t, s.y[-1], dict(QM=QM, ga=ga, gb=gb, xM_from=xM(Sfrom), xM_to=xM(Sto), x_relaxed=y0[-1], Pm=Pm, Qss=Qss, ghi=ghi)

for chi in (2.0, 20.0):
    mx = maxwell(dict(chi0=10.0, chi_RR=0.05, g_c=4.0, g_MHD=3.6, chi_MHD=chi))
    QM = mx["Q_M"]; S_lo, S_hi = QM+1.71, QM+3.60
    for a, b in ((S_hi, S_lo), (S_lo, S_hi)):
        res = {}
        for name, kw in (("PDE core", {}), ("lumped core", dict(core_pde=False))):
            t, xs, info = solve(chi, a, b, **kw); res[name] = half(t, xs)
        # hand checks
        ga, gb, D = info["ga"], info["gb"], 0.05+chi
        xT = info["xM_to"]; Sx = b/S0*5*np.exp(-xT**2/0.0625); l = L - xT
        Gam = Sx*l/D
        # eta0 = Hss(xf0, S_from) - Hqs(xf0, S_to) with linear cliff g+ = gb + (mu - QM)/D
        xF = info["xM_from"]; Qss = info["Qss"]
        Hss = quad(lambda z: gb + (Qss(z, a) - QM)/D, xF, L)[0]
        Hqs = quad(lambda z: gb + (Qss(z, b) - Qss(xF, b))/D, xF, L)[0]
        tauf = (xT - x0)*(gb-ga)/Sx
        print(f"chi={chi:4.0f} S {a:.2f}->{b:.2f}: sharp(own) half PDE core {res['PDE core']:.4f}, lumped core {res['lumped core']:.4f};"
              f" relaxed x {info['x_relaxed']:.4f} vs x_M {xF:.4f}, end {xs[-1]:.4f} vs {xT:.4f};"
              f" Gamma {Gam:.3f} (/[g] {Gam/(gb-ga):.3f}); eta0 {Hss-Hqs:+.4f} vs [g]|dxM| {(gb-ga)*abs(xT-xF):.4f};"
              f" tau_p l2/3D {l**2/3/D:.4f}, 4l2/pi2D {4*l**2/np.pi**2/D:.4f}, l2/pi2D {l**2/np.pi**2/D:.4f}; tau_f {tauf:.4f}", flush=True)
