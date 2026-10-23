"""Two-variable (W, N) model of the flux-driven T3D runs: phase portrait, nullclines, fold, hysteresis.

    dW/dt = eta_a P_aux + eta_al P_alpha(W,N) - eta_r P_rad(W,N;Z) - L(W,N)          (thermal layer)
    dN/dt = S_N - Gamma(W,N)                                                          (slow layer)

W = 3/2 int p dV (MJ), N = int n_e dV (1e20), Tbar = W / (3 N k) (keV).  Each source is weighted by the linear
response of the stored energy to power deposited at its radius (eta = 1 on axis, -> 0 at the edge); the part 1-eta
passes straight through the edge (Q_edge = L + sum (1-eta) P).  L is the barrier's residual (RR) loss.

Calibration: F on es_hkbm_F_frozen (Z_eff = 0, frozen, the ladder's template) only; H/I on I (EC only) only, with the
stiffness mu carried over from F.  Every other member is a prediction.  Notes: NOTES_phase.md.

Usage: python exp_phase.py         (one process; writes phase_cache.npz here and
                                     transport_equations_paper/data/t3d/phase_portrait.npz)
"""
import pathlib
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

HERE = pathlib.Path(__file__).resolve().parent
PAPER = HERE.parents[1] / "transport_equations_paper"
DATA = PAPER / "data" / "t3d"
sys.path.insert(0, str(PAPER / "scripts"))
import predict_endstate as PE  # noqa: E402  (constants and Bosch-Hale only)

K = 1e20 * 1e3 * 1.602176634e-19 * 1e-6      # MJ per (1e20 particles x keV)
sv = PE.bosch_hale_dt
OUT = []
SHELF = '--shelf' in sys.argv     # floor of L: the Dirichlet mean (default) or the shelf temperature (variant)


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------
LADDER = [("es_hkbm_F_frozen", 0.0), ("es_hkbm_F_frozen_brem", 3.4), ("es_hkbm_F_frozen_brem_z6", 6.0),
          ("es_hkbm_F_frozen_brem_z6.5", 6.5), ("es_hkbm_F_frozen_brem_z7", 7.0), ("es_hkbm_F_frozen_brem_z8", 8.0),
          ("es_hkbm_F_frozen_brem_z10", 10.0), ("es_hkbm_F_frozen_brem_z12", 12.0),
          ("es_hkbm_F_frozen_brem_z15", 15.0), ("es_hkbm_F_frozen_brem_z20", 20.0)]
REV = [("es_hkbm_F_rev_z20_to_z3.4", 3.4), ("es_hkbm_F_rev_z20_to_z5", 5.0), ("es_hkbm_F_rev_z20_to_z6", 6.0)]
T_REV = 1.46          # s, restart time of the reverse legs from the z20 member


def load_F(name):
    r = dict(np.load(DATA / "endstate" / f"{name}.npz"))
    N = r["n_e"] @ PE.DV
    W = r["Wtot_MJ"]
    return dict(t=r["t"], W=W, N=N, Tb=W / (3 * N * K), Pa=r["Palpha_int_MW"], Paux=r["Paux_MW"],
                Prad=r["Pradiation_model_MW"], Qe=r["Ptransport_MW"], S=r["Psource_net_MW"], r=r)


def load_HI(tag):
    f = {"I": "I_FS_low.npz", "H": "H_FS_low_QRAD.npz"}[tag]
    r = dict(np.load(DATA / "HI" / f, allow_pickle=True))
    N = r["n_e"] @ r["dV_m3"]
    W = r["Wtot_MJ"]
    return dict(t=r["t"], W=W, N=N, Tb=W / (3 * N * K), Pa=r["Palpha_int_MW"], Paux=r["Pec_MW"],
                Prad=r["Prad_MW"], Qe=r["Q_edge_MW"], r=r)


# ----------------------------------------------------------------------------------------------------------------
# deposition efficiencies: linear response of W to power deposited in cell c
# ----------------------------------------------------------------------------------------------------------------
def eta_profile(ne, Te, Ti, Qk, R, dV):
    """eta_c = Phi_c / Phi_0, Phi_c = sum_{k >= c} R_k M_k: R_k = d(dT_k)/dQ_k (keV/MW) the incremental resistance of
    tube k (between cells k and k+1), M_k = sum_{c <= k} 3 n_c dV_c k the heat capacity inside it (MJ/keV).  A power
    dP put in cell c raises every jump outside it by R_k dP, hence every temperature inside k by sum R dP, hence W by
    Phi_c dP in steady state.  The last cell is the Dirichlet edge (eta = 0)."""
    w = 3 * ne * dV * K
    M = np.cumsum(w)[:len(R)]
    phi = np.array([np.sum(R[c:] * M[c:]) for c in range(len(R))] + [0.0])
    return phi / phi[0], phi


# ----------------------------------------------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------------------------------------------
class Model:
    """Closures (all MW, MJ, 1e20, keV).  The profiles are the reference profile rescaled with the edge pinned:
    n_c -> n_c N/N_ref and T_c -> T_edge + s (T_c,ref - T_edge), with s = (Tbar - T_b)/(Tbar_ref - T_b) and
    T_b = (T_e,edge + T_i,edge)/2 the fixed pedestal (Dirichlet) temperature.  L is the barrier loss, driven by the
    excess theta = Tbar - T_b: L = C sgn(theta) |theta/theta_ref|^mu (N/N_ref)^nu (heat flows in from the boundary
    when the core is colder than the pedestal)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def s(self, W, N):
        return (W / (3 * N * K) - self.T_b) / (self.Tb_ref - self.T_b)

    def Palpha(self, W, N):
        s = np.atleast_1d(self.s(W, N))
        Ti = np.maximum(self.Ti_edge + np.multiply.outer(s, self.Ti_ref - self.Ti_edge), 0.3)
        return np.squeeze(self.ca * (np.atleast_1d(N) / self.N_ref) ** 2 * (sv(Ti) @ self.wa)).reshape(np.shape(W * N))

    def Prad(self, W, N, Z):
        if self.rad_fixed is not None:                 # imported QRAD: a fixed sink, independent of the state
            return self.rad_fixed * (Z > 0) + 0 * W
        s = np.atleast_1d(self.s(W, N))
        Te = np.maximum(self.Te_edge + np.multiply.outer(s, self.Te_ref - self.Te_edge), 0.3)
        return Z * (np.squeeze(5.35e-3 * (np.atleast_1d(N) / self.N_ref) ** 2 * (np.sqrt(Te) @ self.wr))
                    ).reshape(np.shape(W * N))

    def L(self, W, N, nu=None):
        nu = self.nu if nu is None else nu
        th = (W / (3 * N * K) - self.T_b) / (self.Tb_ref - self.T_b)
        return self.C * np.sign(th) * np.abs(th) ** self.mu * (N / self.N_ref) ** nu

    def fW(self, W, N, Z, Paux=None, nu=None):
        Paux = self.Paux if Paux is None else Paux
        return (self.eta_a * Paux + self.eta_al * self.Palpha(W, N) - self.eta_r * self.Prad(W, N, Z)
                - self.L(W, N, nu))

    def Qedge(self, W, N, Z, Paux=None, nu=None):
        Paux = self.Paux if Paux is None else Paux
        return (self.L(W, N, nu) + (1 - self.eta_a) * Paux + (1 - self.eta_al) * self.Palpha(W, N)
                - (1 - self.eta_r) * self.Prad(W, N, Z))

    def fN(self, W, N):
        return self.SN - self.G0 * (N / self.N_ref) ** self.p

    def roots(self, N, Z, Paux=None, nu=None, Wlo=40.0, Whi=6000.0, n=1500):
        """All W with fW = 0 at fixed N; returns list of (W, slope dfW/dW)."""
        Wg = np.geomspace(Wlo, Whi, n)
        f = self.fW(Wg, N, Z, Paux, nu)
        out = []
        for i in np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]:
            w = brentq(lambda x: self.fW(x, N, Z, Paux, nu), Wg[i], Wg[i + 1], xtol=1e-6)
            h = 1e-3 * w
            out.append((w, (self.fW(w + h, N, Z, Paux, nu) - self.fW(w - h, N, Z, Paux, nu)) / (2 * h)))
        return out

    def classify(self, N, Z, **kw):
        """hot (largest stable), unstable, cold (smallest stable) roots, NaN where absent."""
        rr = self.roots(N, Z, **kw)
        st = [w for w, s in rr if s < 0]
        un = [w for w, s in rr if s > 0]
        hot = st[-1] if st else np.nan
        cold = st[0] if len(st) > 1 else (st[0] if st and not np.isfinite(np.nan) else np.nan)
        if len(st) == 1:          # one stable root: call it hot if above the unstable/inflexion scale
            cold = np.nan
            if st[0] < self.W_split:
                cold, hot = st[0], np.nan
        return hot, (un[0] if un else np.nan), cold

    def integrate(self, W0, N0, t0, t1, Z, Paux=None, nu=None, freezeN=False):
        """ODE from (W0, N0); stops at T3D's temperature floor (Tbar = T_floor: the radiative collapse ends there)
        and holds W from then on."""
        def rhs(t, y):
            return [self.fW(y[0], y[1], Z, Paux, nu), 0.0 if freezeN else self.fN(y[0], y[1])]

        def floor(t, y):
            return y[0] / (3 * y[1] * K) - self.T_floor
        floor.terminal = True
        floor.direction = -1
        te = np.linspace(t0, t1, 400)
        s = solve_ivp(rhs, (t0, t1), [W0, N0], method="LSODA", rtol=1e-7, atol=1e-6, t_eval=te, events=floor)
        W, N = s.y[0], s.y[1]
        k = len(s.t)
        if k < len(te):
            W = np.r_[W, np.full(len(te) - k, W[-1])]
            N = np.r_[N, N[-1] + (te[k:] - te[k - 1]) * self.fN(W[-1], N[-1])]
        return te, W, N


def fit_pow(x, y):
    p = np.polyfit(np.log(x), np.log(y), 1)
    return p[0], np.exp(p[1])


# ----------------------------------------------------------------------------------------------------------------
# calibration F: es_hkbm_F_frozen only
# ----------------------------------------------------------------------------------------------------------------
def calibrate_F():
    R = load_F("es_hkbm_F_frozen")
    r, t = R["r"], R["t"]
    sel = t > 0.5
    iref = np.argmin(abs(t - 2.0))              # reference profile for the self-similar closures
    ne, Te, Ti = r["n_e"][iref], r["T_e"][iref], r["T_D"][iref]
    N_ref, Tb_ref = R["N"][iref], R["Tb"][iref]
    say(f"# F calibration on es_hkbm_F_frozen (Z_eff 0, frozen), t > 0.5 s; reference profile at t = {t[iref]:.2f} s:"
        f" W {R['W'][iref]:.0f} MJ, N {N_ref:.0f}e20, Tbar {Tb_ref:.2f} keV")
    Wchk = 1.5 * (r["n_e"][0] * (r["T_e"][0] + r["T_D"][0])) @ PE.DV * K
    say(f"  bookkeeping t=0: 3/2 int n(Te+Ti) dV = {Wchk:.1f} vs logged {R['W'][0]:.1f} MJ")
    # alpha: fuel fractions of the deck, 3.5 MeV per reaction
    wa = PE.F_D * PE.F_T * ne ** 2 * 1e40 * 3.5e3 * 1.602176634e-16 * 1e-6 * PE.DV
    Pa_formula = sv(Ti) @ wa
    ca = R["Pa"][iref] / Pa_formula
    say(f"  P_alpha: Bosch-Hale on the profile {Pa_formula:.1f} vs logged {R['Pa'][iref]:.1f} MW -> c_alpha = {ca:.4f}")
    cr = 5.35e-3 * (ne ** 2 * np.sqrt(Te)) @ PE.DV
    say(f"  brem per unit Z at the reference profile c_r = {cr:.1f} MW (40.6 at t = 0 in the paper)")
    # deposition efficiencies: incremental resistance of each tube along the reference path (t > 1 s)
    Q = r["Q_MW_e"] + r["Q_MW_D"]
    dT = 0.5 * ((r["T_e"][:, :-1] - r["T_e"][:, 1:]) + (r["T_D"][:, :-1] - r["T_D"][:, 1:]))
    s1 = t > 1.0
    Rinc = np.array([np.polyfit(Q[s1, k], dT[s1, k], 1)[0] for k in range(9)])
    Rinc = np.maximum(Rinc, 0.0)
    il = -1
    eta_c, phi = eta_profile(r["n_e"][il], r["T_e"][il], r["T_D"][il], Q[il], Rinc, PE.DV)
    Rsec = dT[il] / Q[il]
    eta_sec, _ = eta_profile(r["n_e"][il], r["T_e"][il], r["T_D"][il], Q[il], np.maximum(Rsec, 0), PE.DV)
    Paux_c = PE.SP_AUX_E_F * PE.P_REF_MWM3 * PE.DV
    brem_c = r["n_e"][il] ** 2 * np.sqrt(r["T_e"][il]) * PE.DV
    al_c = PE.F_D * PE.F_T * r["n_e"][il] ** 2 * sv(r["T_D"][il]) * PE.DV
    eff = lambda e, p: float(e @ p / p.sum())
    eta_a, eta_r, eta_al = eff(eta_c, Paux_c), eff(eta_c, brem_c), eff(eta_c, al_c)
    say(f"  incremental resistances R_k (keV/MW) {np.round(Rinc, 4)}")
    say(f"  eta_c (incremental) {np.round(eta_c, 2)};  1/L' = Phi_0 = {phi[0]:.2f} s")
    say(f"  eta_c (secant dT/Q)  {np.round(eta_sec, 2)}")
    say(f"  eta_aux {eta_a:.3f} (secant {eff(eta_sec, Paux_c):.3f}), eta_rad {eta_r:.3f} "
        f"(secant {eff(eta_sec, brem_c):.3f}), eta_alpha {eta_al:.3f} (secant {eff(eta_sec, al_c):.3f})")
    # density exponent of L from the fixed-temperature-profile argument
    i1 = np.argmin(abs(t - 1.0))
    dn = r["n_e"][il] - r["n_e"][i1]
    T2 = 0.5 * (r["T_e"][il] + r["T_D"][il])
    n0 = r["n_e"][il]
    e = 1e-3
    Tb = lambda n: (n * T2) @ PE.DV / (n @ PE.DV)
    xprime = (np.log(Tb(n0 + e * dn)) - np.log(Tb(n0))) / (np.log((n0 + e * dn) @ PE.DV) - np.log(n0 @ PE.DV))
    say(f"  d ln Tbar / d ln N at a fixed temperature profile, density added in the run's ramp pattern: x' = {xprime:.3f}")
    # the shelf: the outer region must carry the pass-through power (1 - eta_a) P_aux by RR; the reference's own
    # Q_edge - T_e(0.85) relation (t > 0.3 s) gives the electron temperature that does it.  A core with no net
    # drive cannot cool below it (ions equilibrated in a cold, collisional core).
    s3 = t > 0.3
    pT = np.polyfit(np.log(R["Qe"][s3]), np.log(r["T_e"][s3, 8]), 1)
    Qpass = (1 - eta_a) * 315.2
    T_b = float(np.exp(np.polyval(pT, np.log(Qpass)))) if SHELF else 0.5 * (PE.TE_EDGE + PE.TI_EDGE)
    xth = xprime * Tb_ref / (Tb_ref - T_b)
    say(f"  reference: T_e(0.85) = {np.exp(pT[1]):.2f} Q_edge^{pT[0]:.3f}; at the pass-through Q = {Qpass:.0f} MW "
        f"T_e(0.85) = {np.exp(np.polyval(pT, np.log(Qpass))):.2f} keV; Dirichlet mean {0.5 * (PE.TE_EDGE + PE.TI_EDGE):.2f}")
    say(f"  floor of the barrier loss T_b = {T_b:.3f} keV ({'shelf' if SHELF else 'Dirichlet'}); "
        f"d ln theta / d ln N at a fixed profile = {xth:.3f}")
    Lobs = (R["Qe"] - (1 - eta_al) * R["Pa"] - (1 - eta_a) * R["Paux"] + (1 - eta_r) * R["Prad"])[sel]
    x1, x2 = np.log((R["Tb"][sel] - T_b) / (Tb_ref - T_b)), np.log(R["N"][sel] / N_ref)

    def fitL(nu_rule):
        def res(p):
            lnC, mu = p
            return lnC + mu * x1 + nu_rule(mu) * x2 - np.log(Lobs)
        s = least_squares(res, [np.log(500), 3.0])
        return np.exp(s.x[0]), s.x[1], nu_rule(s.x[1]), np.std(np.exp(res(s.x) + np.log(Lobs)) - Lobs)
    C, mu, nu, rms = fitL(lambda m: -m * xth)
    C0, mu0, _, rms0 = fitL(lambda m: 0.0)
    C2, mu2, nu2, rms2 = fitL(lambda m: -2 * m * xth)
    say(f"  L = C sgn(theta)|theta/theta_ref|^mu (N/N_ref)^nu, theta = Tbar - T_b, nu = -mu dln theta/dlnN -> "
        f"C {C:.1f} MW, mu {mu:.2f}, nu {nu:.2f} (rms {rms:.1f} MW);  band: nu = 0 -> C {C0:.1f}, mu {mu0:.2f} "
        f"(rms {rms0:.1f}); nu doubled -> C {C2:.1f}, mu {mu2:.2f}, nu {nu2:.2f} (rms {rms2:.1f})")
    # particles
    P = dict(np.load(DATA / "endstate_pflux" / "pflux_endstate__es_hkbm_F_frozen.npz", allow_pickle=True))
    tp = P["t_s_abs"]
    Np = P["n_e"] @ PE.DV
    SN = float(P["Sn_aux_cumint_SI20_e"][0, 8])        # source inside face 0.9 (cells 0-8; cell 9 is Dirichlet)
    G = P["Gam_SI20_e"][:, 8]
    sp = tp > 0.5
    p, G0n = fit_pow(Np[sp] / N_ref, G[sp])
    say(f"  particles: S_N inside 0.9 = {SN:.1f}e20/s; Gamma(0.9) = {G0n:.1f} (N/N_ref)^{p:.2f} (hKBM on, hot branch)")
    m = Model(ca=ca, wa=wa, Ti_ref=Ti, Te_ref=Te, Ti_edge=PE.TI_EDGE, Te_edge=PE.TE_EDGE, T_b=T_b, wr=ne ** 2 * PE.DV,
              Tb_ref=Tb_ref, N_ref=N_ref, cr=cr, rad_fixed=None, C=C, mu=mu, nu=nu,
              eta_a=eta_a, eta_r=eta_r, eta_al=eta_al, Paux=315.2, SN=SN, G0=G0n, p=p, W_split=650.0, T_floor=0.5,
              band=dict(nu0=(C0, mu0, 0.0), nu2=(C2, mu2, nu2)), eta_c=eta_c, eta_sec=eta_sec, xprime=xprime)
    # check on the calibration run
    fw = m.fW(R["W"][sel], R["N"][sel], 0.0)
    dWdt = np.gradient(R["W"], t)[sel]
    say(f"  check on the calibration run: model dW/dt - run dW/dt: mean {np.mean(fw - dWdt):+.1f}, rms {np.std(fw - dWdt):.1f} MW;"
        f" model Q_edge - run: rms {np.std(m.Qedge(R['W'][sel], R['N'][sel], 0) - R['Qe'][sel]):.1f} MW;"
        f" P_alpha model/run {np.mean(m.Palpha(R['W'][sel], R['N'][sel]) / R['Pa'][sel]):.3f}")
    return m


def band_model(m, which):
    mm = Model(**m.__dict__)
    mm.C, mm.mu, mm.nu = m.band[which]
    return mm


# ----------------------------------------------------------------------------------------------------------------
# calibration H/I: I only (mu from F)
# ----------------------------------------------------------------------------------------------------------------
def calibrate_HI(mF):
    R = load_HI("I")
    r, t = R["r"], R["t"]
    dV = r["dV_m3"]
    sel = t > 0.5
    iref = np.argmin(abs(t - 1.0))
    ne, Te, Ti = r["n_e"][iref], r["T_e"][iref], r["T_D"][iref]
    N_ref, Tb_ref = R["N"][iref], R["Tb"][iref]
    say(f"# H/I calibration on I (EC only), t > 0.5 s; reference at t = {t[iref]:.2f} s: W {R['W'][iref]:.0f}, "
        f"N {N_ref:.0f}, Tbar {Tb_ref:.2f}")
    Wchk = 1.5 * (r["n_e"][0] * (r["T_e"][0] + r["T_D"][0])) @ dV * K
    say(f"  bookkeeping t=0: {Wchk:.1f} vs logged {R['W'][0]:.1f} MJ")
    wa = ne ** 2 * dV          # fuel fractions unknown: normalised to the logged P_alpha
    ca = R["Pa"][iref] / (sv(Ti) @ wa)
    Q = r["Q_MW_e"] + r["Q_MW_D"]
    dT = 0.5 * ((r["T_e"][:, :-1] - r["T_e"][:, 1:]) + (r["T_D"][:, :-1] - r["T_D"][:, 1:]))
    il = -1
    Rsec = np.maximum(dT[il] / Q[il], 0)
    eta_c, _ = eta_profile(r["n_e"][il], r["T_e"][il], r["T_D"][il], Q[il], Rsec, dV)
    Pec_c = np.diff(np.r_[0, r["Pec_cum_MW"][0]])
    H = load_HI("H")
    Prad_c = np.diff(np.r_[0, H["r"]["Prad_cum_MW"][0]])
    al_c = ne ** 2 * sv(Ti) * dV
    eff = lambda e, p: float(e @ p / p.sum())
    eta_a, eta_r, eta_al = eff(eta_c, Pec_c), eff(eta_c, Prad_c), eff(eta_c, al_c)
    say(f"  eta_c (secant; the I run is too steady for the incremental fit) {np.round(eta_c, 2)}")
    say(f"  EC per cell {np.round(Pec_c, 1)}, QRAD per cell {np.round(Prad_c, 1)} (last cell Dirichlet)")
    say(f"  eta_EC {eta_a:.3f}, eta_QRAD {eta_r:.3f} (fixed sink, {Prad_c.sum():.0f} MW of which "
        f"{Prad_c[:-1].sum():.0f} in the evolved cells), eta_alpha {eta_al:.3f}")
    mu = mF.mu
    Te_edge, Ti_edge = float(r["T_e"][0, -1]), float(r["T_D"][0, -1])
    T_b = 0.5 * (Te_edge + Ti_edge)
    th = (R["Tb"][sel] - T_b) / (Tb_ref - T_b)
    Lobs = (R["Qe"] - (1 - eta_al) * R["Pa"] - (1 - eta_a) * R["Paux"])[sel]
    C = np.exp(np.mean(np.log(Lobs) - mu * np.log(th)))
    # particles: N barely moves in I; take its measured drift as the slow source
    p = np.polyfit(t[sel], R["N"][sel], 1)
    say(f"  L: mu = {mu:.2f} (from F), T_b = {T_b:.2f} keV (I's Dirichlet values), C = {C:.1f} MW at Tbar_ref "
        f"(rms of L over I {np.std(Lobs - C * th ** mu):.1f}); I's dN/dt = {p[0]:.1f}e20/s (held: S_N - Gamma = const)")
    m = Model(ca=ca, wa=wa, Ti_ref=Ti, Te_ref=Te, Ti_edge=Ti_edge, Te_edge=Te_edge, T_b=T_b, wr=ne ** 2 * dV,
              Tb_ref=Tb_ref, N_ref=N_ref, cr=0.0, rad_fixed=float(Prad_c.sum()), C=C, mu=mu,
              nu=0.0, eta_a=eta_a, eta_r=eta_r, eta_al=eta_al, Paux=float(R["Paux"][0]), SN=p[0], G0=0.0, p=0.0,
              W_split=300.0, T_floor=0.5, eta_c=eta_c)
    fw = m.fW(R["W"][sel], R["N"][sel], 0.0)
    say(f"  check on I: model dW/dt - run: mean {np.mean(fw - np.gradient(R['W'], t)[sel]):+.1f} MW;"
        f" P_alpha model/run at t=0 {m.Palpha(R['W'][0], R['N'][0]) / R['Pa'][0]:.3f}")
    a = np.log(sv(Ti * 1.01) @ wa) - np.log(sv(Ti) @ wa)
    say(f"  d ln P_alpha / d ln Tbar at I's state = {a / np.log(1.01):.2f} vs mu = {mu:.2f}")
    return m


# ----------------------------------------------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------------------------------------------
def fold_Z(m, N, Zlo=0.0, Zhi=25.0, **kw):
    """Largest Z with a hot root (the fold of the burning branch) and smallest Z with a cold root, at fixed N."""
    Zs = np.linspace(Zlo, Zhi, 251)
    hot = np.array([m.classify(N, z, **kw)[0] for z in Zs])
    cold = np.array([m.classify(N, z, **kw)[2] for z in Zs])
    zf = Zs[np.isfinite(hot)].max() if np.isfinite(hot).any() else np.nan
    zc = Zs[np.isfinite(cold)].min() if np.isfinite(cold).any() else np.nan
    return zf, zc


def fold_in_N(m, Z, Nlo=1600.0, Nhi=3500.0, which="hot", **kw):
    """N at which the hot (or cold) stable root of the W equation merges with the unstable one at fixed Z, the
    merging W, and the normal-form coefficients fW ~ a (N - N_f) sgn - b (W - W_f)^2 there."""
    idx = 0 if which == "hot" else 2
    Ng = np.linspace(Nlo, Nhi, 96)
    have = np.array([np.isfinite(m.classify(N, Z, **kw)[idx]) for N in Ng])
    ch = np.where(have[:-1] != have[1:])[0]
    if not len(ch):
        return None
    lo, hi = Ng[ch[0]], Ng[ch[0] + 1]
    side = have[ch[0]]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if np.isfinite(m.classify(mid, Z, **kw)[idx]) == side:
            lo = mid
        else:
            hi = mid
    Nf = 0.5 * (lo + hi)
    rr = m.roots(lo if side else hi, Z, **kw)
    st = [w for w, sl in rr if sl < 0]
    un = [w for w, sl in rr if sl > 0]
    if not un:
        return None
    Wg = np.linspace(0.6 * min(st + un), 1.4 * max(st + un), 3000)
    f = m.fW(Wg, Nf, Z, **kw)
    j = np.argmin(np.abs(np.gradient(f, Wg)) + 1e3 * (np.abs(f) > 30))     # double root: f = f' = 0
    Wf = Wg[j]
    h = 2.0
    b = 0.5 * abs(m.fW(Wf + h, Nf, Z, **kw) - 2 * m.fW(Wf, Nf, Z, **kw) + m.fW(Wf - h, Nf, Z, **kw)) / h ** 2
    a = abs(m.fW(Wf, Nf + 1, Z, **kw) - m.fW(Wf, Nf - 1, Z, **kw)) / 2
    r = m.fN(Wf, Nf)
    td = 2.3381 * (a * r * b) ** (-1 / 3)
    return dict(Nf=Nf, Wf=Wf, a=a, b=b, r=r, td=td, appears=not side)


def slow_manifold(m, R, Z, paux=315.2):
    """Leading order W0 = W*(N(t)) and first order W0 + (dW*/dN)(dN/dt)/gamma_th along a run's own N(t)."""
    out = []
    for t, N, W in zip(R["t"], R["N"], R["W"]):
        hot = m.classify(N, Z, Paux=paux)[0]
        if not np.isfinite(hot):
            out.append((t, W, np.nan, np.nan, np.nan))
            continue
        hot2 = m.classify(N + 2, Z, Paux=paux)[0]
        dWdN = (hot2 - hot) / 2
        g = (m.fW(hot + 1, N, Z, Paux=paux) - m.fW(hot - 1, N, Z, Paux=paux)) / 2
        out.append((t, W, hot, hot + dWdN * m.fN(hot, N) / g, g))
    return np.array(out)


def main():
    mF = calibrate_F()
    mHI = calibrate_HI(mF)
    runs = {nm: load_F(nm) for nm, _ in LADDER + REV + [("es_hkbm_F_frozen_paux0", 0), ("hkbm_F_k1_v2", 0)]}
    N0 = runs["es_hkbm_F_frozen"]["N"][0]
    W0 = runs["es_hkbm_F_frozen"]["W"][0]

    # --- time scales (multiple scales)
    say("# time scales (F, model at the run states)")
    for nm, z in [("es_hkbm_F_frozen", 0), ("es_hkbm_F_frozen_brem_z6", 6), ("es_hkbm_F_rev_z20_to_z6", 6)]:
        R = runs[nm]
        i = -1
        h = 1.0
        g = (mF.fW(R["W"][i] + h, R["N"][i], z) - mF.fW(R["W"][i] - h, R["N"][i], z)) / (2 * h)
        tauN = R["N"][i] / mF.fN(R["W"][i], R["N"][i])
        say(f"  {nm}: at t {R['t'][i]:.2f}: gamma_th(fixed N) = {g:+.3f}/s (tau_th {1 / abs(g):.2f} s); "
            f"tau_N = N/(dN/dt) = {tauN:.1f} s; eps = {1 / abs(g) / tauN:.3f}")
    # paper's along-path slopes vs fixed-N slopes
    R = runs["es_hkbm_F_frozen"]
    sel = R["t"] > 0.5
    Wm, Nm = R["W"][sel], R["N"][sel]
    dS = np.polyfit(Wm, mF.eta_al * mF.Palpha(Wm, Nm) + mF.eta_a * 315.2, 1)[0]
    dL = np.polyfit(Wm, mF.L(Wm, Nm), 1)[0]
    say(f"  along the run path: d(eta P)/dW = {dS:.2f}, dL/dW = {dL:.2f} MW/MJ -> path rate {dS - dL:+.2f}/s")
    Wq = 1200.0
    Nq = np.interp(Wq, R["W"], R["N"])
    h = 1.0
    dSN = mF.eta_al * (mF.Palpha(Wq + h, Nq) - mF.Palpha(Wq - h, Nq)) / (2 * h)
    dLN = (mF.L(Wq + h, Nq) - mF.L(Wq - h, Nq)) / (2 * h)
    m_log = dLN * Wq / mF.L(Wq, Nq)
    n_log = (np.log(mF.Palpha(Wq * 1.001, Nq)) - np.log(mF.Palpha(Wq / 1.001, Nq))) / (2 * np.log(1.001))
    say(f"  at fixed N (W = {Wq:.0f}, N = {Nq:.0f}): d(eta_al P_alpha)/dW = {dSN:.2f}, dL/dW = {dLN:.2f} -> "
        f"gamma_th = {dSN - dLN:+.2f}/s; m = dlnL/dlnW = {m_log:.2f}, n = dlnP_alpha/dlnW = {n_log:.2f}")

    # --- fold vs N (S-curve)
    say("# fold of the burning branch and onset of the cold root, vs N (Z_eff at which they occur)")
    Nlist = [N0, 1767.0, 1850.0, 1950.0, 2100.0, 2300.0]
    folds = {}
    for N in Nlist:
        zf, zc = fold_Z(mF, N)
        zb0 = fold_Z(band_model(mF, "nu0"), N)
        zb1 = fold_Z(band_model(mF, "nu2"), N)
        folds[N] = (zf, zc, zb0, zb1)
        say(f"  N = {N:.0f}: hot root up to Z = {zf:.1f}, cold root from Z = {zc:.1f}  "
            f"[nu=0: {zb0[0]:.1f}/{zb0[1]:.1f}; nu doubled: {zb1[0]:.1f}/{zb1[1]:.1f}]")
    Zs = np.linspace(0, 20, 201)
    Scurve = {}
    for N in [N0, 1850.0, 2100.0]:
        Scurve[N] = np.array([mF.classify(N, z) for z in Zs])

    # --- predictions: integrate every member from its own state at t_s (barrier formed)
    say("# predictions (no refit): ODE from the member's own (W, N) at t_s; W at the member's last time")
    say("  member | Z | t_s | run W(t_s) | run t_end, W_end | model W(t_end) | model branch | run trend | "
        "model from the input (t=0)")
    traj = {}
    preds = []
    for nm, z in LADDER + REV + [("es_hkbm_F_frozen_paux0", 0)]:
        R = runs[nm]
        paux = 0.0 if "paux0" in nm else 315.2
        ts = T_REV + 0.02 if "rev" in nm else 0.5
        i = np.argmin(abs(R["t"] - ts))
        te = R["t"][-1]
        tt, WW, NN = mF.integrate(R["W"][i], R["N"][i], R["t"][i], max(te, R["t"][i] + 0.05), z, Paux=paux)
        tl, WL, NL = mF.integrate(R["W"][i], R["N"][i], R["t"][i], 20.0, z, Paux=paux)
        if "rev" in nm:
            t0m = WW0 = None
            Wi = np.nan
        else:
            tt0, WW0, NN0 = mF.integrate(W0, N0, 0.0, max(te, 0.1), z, Paux=paux)
            Wi = WW0[-1]
        trend = np.polyfit(R["t"][-20:], R["W"][-20:], 1)[0]
        hot, un, cold = mF.classify(NN[-1], z, Paux=paux)
        branch = "hot" if (np.isfinite(hot) and abs(WW[-1] - hot) < abs(WW[-1] - (cold if np.isfinite(cold) else -1e9))) \
            else "cold"
        if WW[-1] > (un if np.isfinite(un) else 0) and np.isfinite(hot):
            branch = "-> hot"
        elif np.isfinite(cold):
            branch = "-> cold"
        else:
            branch = "-> collapse (no root above)"
        # time of re-ignition / collapse in the long run
        jump = ""
        dW = np.diff(WL)
        if WL[-1] > 1.5 * WL[0] and WL[0] < 700:
            k = np.argmax(WL > 0.5 * (WL[0] + WL[-1]))
            jump = f"; re-ignites at {tl[k]:.1f} s"
        say(f"  {nm:28s} {z:4.1f} {R['t'][i]:.2f} {R['W'][i]:6.0f} | {te:.2f} {R['W'][-1]:6.0f} | {WW[-1]:6.0f} {branch} "
            f"| {trend:+5.0f} MJ/s | {Wi:6.0f}{jump}; model W(20 s) {WL[-1]:.0f}")
        preds.append((nm, z, R["t"][i], R["W"][i], te, R["W"][-1], WW[-1], trend, Wi, WL[-1]))
        traj[nm] = dict(t_run=R["t"], W_run=R["W"], N_run=R["N"], t_mod=tl, W_mod=WL, N_mod=NL, Z=z)

    # --- multiple scales: slow manifold along the runs, first-order lag
    say("# slow manifold: run W vs W*(N(t)) (leading order) and W* + (dW*/dN)(dN/dt)/gamma_th (first order)")
    sm = {}
    for nm, z, paux in [("es_hkbm_F_frozen", 0, 315.2), ("es_hkbm_F_frozen_brem", 3.4, 315.2),
                        ("es_hkbm_F_frozen_brem_z6", 6, 315.2), ("es_hkbm_F_frozen_paux0", 0, 0.0)]:
        R = runs[nm]
        k = np.r_[np.searchsorted(R["t"], [1.0, 2.0, 3.0]), len(R["t"]) - 1]
        Rk = {kk: R[kk][k] for kk in ["t", "N", "W"]}
        a = slow_manifold(mF, Rk, z, paux)
        sm[nm] = a
        say(f"  {nm}: " + "; ".join(f"t {x[0]:.1f}: run {x[1]:.0f}, W* {x[2]:.0f}, W*+lag {x[3]:.0f}, "
                                     f"tau_th {-1 / x[4]:.1f} s" for x in a))
    # --- folds crossed by the density ramp and the slow-passage delay
    say("# folds in N at fixed Z, normal form fW = a (N - N_f) - b (W - W_f)^2, slow passage t_d = 2.338 (a r b)^-1/3")
    sn = {}
    for z, which, mm, lab in [(7.8, "hot", mF, "base"), (8.0, "hot", mF, "base"), (6.5, "hot", band_model(mF, "nu2"), "nu doubled"),
                              (7.0, "hot", band_model(mF, "nu2"), "nu doubled"), (3.4, "cold", mF, "base"),
                              (4.0, "cold", mF, "base"), (5.0, "cold", mF, "base")]:
        d = fold_in_N(mm, z, which=which)
        if d is None:
            say(f"  Z {z} ({lab}) {which} root: no fold for N in 1600-3500")
            continue
        tauN = d["Nf"] / d["r"]
        g_ref = abs((mm.fW(d["Wf"] * 1.3 + 1, d["Nf"], z) - mm.fW(d["Wf"] * 1.3 - 1, d["Nf"], z)) / 2)
        eps = 1 / max(g_ref, 1e-3) / tauN
        say(f"  Z {z} ({lab}) {which} root {'appears' if d['appears'] else 'is lost'} at N_f = {d['Nf']:.0f}, W_f = {d['Wf']:.0f} MJ; "
            f"a = {d['a']:.3f} MW/1e20, b = {d['b']:.2e} MW/MJ^2, dN/dt = {d['r']:.1f}/s -> t_d = {d['td']:.2f} s "
            f"(tau_N = {tauN:.0f} s, eps ~ {eps:.3f}, eps^(2/3) tau_N = {eps ** (2 / 3) * tauN:.1f} s)")
        sn[f"{z}_{which}_{lab.replace(' ', '')}"] = d
    # z10: is there a cold root at all, and with the cold-state shape?
    R6 = runs["es_hkbm_F_rev_z20_to_z6"]["r"]
    mc = Model(**mF.__dict__)
    i6 = -1
    mc.Ti_ref, mc.Te_ref = R6["T_D"][i6], R6["T_e"][i6]
    mc.wa = PE.F_D * PE.F_T * R6["n_e"][i6] ** 2 * 1e40 * 3.5e3 * 1.602176634e-16 * 1e-6 * PE.DV
    mc.wr = R6["n_e"][i6] ** 2 * PE.DV
    mc.N_ref = runs["es_hkbm_F_rev_z20_to_z6"]["N"][i6]
    mc.Tb_ref_shape = runs["es_hkbm_F_rev_z20_to_z6"]["Tb"][i6]
    # keep L as calibrated (theta and N_ref of the hot reference)
    Lh = mF.L
    mc.L = lambda W, N, nu=None: Lh(W, N, nu)
    mc.s = lambda W, N: (W / (3 * N * K) - mF.T_b) / (mc.Tb_ref_shape - mF.T_b)
    for z in [5.0, 6.0, 10.0]:
        for lab, mm in [("hot-shape", mF), ("cold-shape", mc)]:
            rr = mm.roots(1920.0, z)
            say(f"  Z {z} at N 1920, {lab} closures: roots " + ", ".join(f"{w:.0f}{'s' if sl < 0 else 'u'}" for w, sl in rr))
    Rz = runs["es_hkbm_F_frozen_brem_z10"]
    tt, WW, NN = mc.integrate(Rz["W"][-1], Rz["N"][-1], Rz["t"][-1], 30.0, 10.0)
    say(f"  z10 from its last state with the cold-shape closures: W(10 s) {np.interp(10, tt, WW):.0f}, W(30 s) {WW[-1]:.0f}")

    # --- H and I
    say("# H and I (one knob: the imported radiation)")
    HI = {}
    for tag in ["I", "H"]:
        R = load_HI(tag)
        z = 1.0 if tag == "H" else 0.0
        tt, WW, NN = mHI.integrate(R["W"][0], R["N"][0], 0.0, R["t"][-1], z)
        rr = mHI.roots(R["N"][0], z)
        say(f"  {tag}: roots at N {R['N'][0]:.0f}: " + ", ".join(f"W {w:.0f} ({'stable' if s < 0 else 'unstable'})"
                                                              for w, s in rr)
            + f"; model from t=0 W({R['t'][-1]:.2f}) = {WW[-1]:.0f} vs run {R['W'][-1]:.0f}; "
              f"model fW at start {mHI.fW(R['W'][0], R['N'][0], z):+.0f} MW")
        HI[tag] = dict(t_run=R["t"], W_run=R["W"], N_run=R["N"], t_mod=tt, W_mod=WW, N_mod=NN, Z=z, roots=rr)
    # W slaved to the W-nullcline along the runs' own N(t) (leading order of the slow layer)
    for tag in ["I", "H"]:
        R = load_HI(tag)
        z = 1.0 if tag == "H" else 0.0
        k = np.searchsorted(R["t"], [0.1, 0.2, 0.3, 0.5, 1.0, R["t"][-1] - 1e-6])
        say(f"  {tag}: run W vs hot root W*(N_run(t)): " + "; ".join(
            f"t {R['t'][i]:.2f}: N {R['N'][i]:.0f}, W {R['W'][i]:.0f}, W* {mHI.classify(R['N'][i], z)[0]:.0f}" for i in k))
    # how much radiation would H tolerate (fold in the QRAD multiplier)
    fs = np.linspace(0, 1.5, 151)
    hot = []
    for f in fs:
        mm = Model(**mHI.__dict__)
        mm.rad_fixed = mHI.rad_fixed * f
        hot.append(mm.classify(load_HI("I")["N"][0], 1.0)[0])
    hot = np.array(hot)
    say(f"  H/I: the hot root exists up to a QRAD multiplier of {fs[np.isfinite(hot)].max():.2f} "
        f"(H = 1.00, I = 0)")

    # --- phase-portrait arrays
    pp = {}
    Wg = np.linspace(50, 2600, 260)
    Ng = np.linspace(1550, 2600, 106)
    for case, m, z, paux in [("F_Z0", mF, 0.0, 315.2), ("F_Z6", mF, 6.0, 315.2), ("F_Z10", mF, 10.0, 315.2),
                             ("F_Z20", mF, 20.0, 315.2), ("F_paux0", mF, 0.0, 0.0)]:
        WW_, NN_ = np.meshgrid(Wg, Ng, indexing="ij")
        pp[f"{case}__fW"] = m.fW(WW_, NN_, z, Paux=paux)
        pp[f"{case}__fN"] = m.fN(WW_, NN_)
        br = np.array([m.classify(N, z, Paux=paux) for N in Ng])
        pp[f"{case}__N_null"] = Ng
        pp[f"{case}__W_hot"], pp[f"{case}__W_unstable"], pp[f"{case}__W_cold"] = br.T
        pp[f"{case}__Z"] = z
        pp[f"{case}__Paux"] = paux
    pp["F__W_grid"], pp["F__N_grid"] = Wg, Ng
    WgH = np.linspace(50, 1200, 231)
    NgH = np.linspace(1200, 1900, 141)
    for case, z in [("I", 0.0), ("H", 1.0)]:
        WW_, NN_ = np.meshgrid(WgH, NgH, indexing="ij")
        pp[f"{case}__fW"] = mHI.fW(WW_, NN_, z)
        br = np.array([mHI.classify(N, z) for N in NgH])
        pp[f"{case}__N_null"] = NgH
        pp[f"{case}__W_hot"], pp[f"{case}__W_unstable"], pp[f"{case}__W_cold"] = br.T
    pp["HI__W_grid"], pp["HI__N_grid"] = WgH, NgH
    for nm, d in list(traj.items()) + [(k, v) for k, v in HI.items()]:
        for k in ["t_run", "W_run", "N_run", "t_mod", "W_mod", "N_mod"]:
            pp[f"traj_{nm}__{k}"] = d[k]
        pp[f"traj_{nm}__Z"] = d["Z"]
    for nm in ["hkbm_F_k1_v2"]:
        R = runs[nm]
        pp[f"traj_{nm}__t_run"], pp[f"traj_{nm}__W_run"], pp[f"traj_{nm}__N_run"] = R["t"], R["W"], R["N"]
    pp["Scurve__Z"] = Zs
    for N, a in Scurve.items():
        pp[f"Scurve__N{N:.0f}"] = a       # columns hot, unstable, cold
    pp["Scurve__N_list"] = np.array(list(Scurve))
    pp["fold__N"] = np.array(Nlist)
    pp["fold__Z_hot_end"] = np.array([folds[N][0] for N in Nlist])
    pp["fold__Z_cold_start"] = np.array([folds[N][1] for N in Nlist])
    pp["fold__Z_hot_end_nu0"] = np.array([folds[N][2][0] for N in Nlist])
    pp["fold__Z_hot_end_nu2"] = np.array([folds[N][3][0] for N in Nlist])
    pp["ladder_rows"] = np.array([p[1:] for p in preds], dtype=float)
    pp["ladder_names"] = np.array([p[0] for p in preds])
    pp["ladder_cols"] = np.array(["Z", "t_s", "W_run_ts", "t_end", "W_run_end", "W_model_end", "trend_run_MJs",
                                  "W_model_from_t0", "W_model_20s"])
    for tag, m in [("F", mF), ("HI", mHI)]:
        for k in ["ca", "Tb_ref", "N_ref", "cr", "C", "mu", "nu", "eta_a", "eta_r", "eta_al", "Paux", "SN", "G0", "p"]:
            pp[f"calib_{tag}__{k}"] = getattr(m, k)
        pp[f"calib_{tag}__Ti_ref"] = m.Ti_ref
        pp[f"calib_{tag}__wa"] = m.wa
        pp[f"calib_{tag}__eta_c"] = m.eta_c
    pp["calib_HI__rad_fixed"] = mHI.rad_fixed
    for z, d in sn.items():
        for k, v in d.items():
            pp[f"fold_{z}__{k}"] = v
    for nm, a in sm.items():
        pp[f"slowman_{nm}"] = a
    if SHELF:                       # the variant is kept beside the base model, never in the paper's data file
        np.savez(HERE / "phase_cache_shelf.npz", **pp)
        (HERE / "phase_log_shelf.txt").write_text("\n".join(OUT) + "\n")
        return
    np.savez(HERE / "phase_cache.npz", **pp)
    np.savez(DATA / "phase_portrait.npz", **pp)
    (HERE / "phase_log.txt").write_text("\n".join(OUT) + "\n")
    print("wrote", HERE / "phase_cache.npz", DATA / "phase_portrait.npz")


if __name__ == "__main__":
    main()
