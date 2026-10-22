"""Scans and runs for the GK-like flux law (gk_law.py).  Every product is cached as an .npz in this
directory with documented keys (see NOTES_gk_law.md); quick-look PNGs go to gk_law_figs/.

    python3 gk_law_runs.py fit      # bump fit         -> gk_law_bump_fit.npz
    python3 gk_law_runs.py table    # Maxwell table    -> gk_law_maxwell_table.npz
    python3 gk_law_runs.py runs     # cases (a)-(e)    -> gk_law_runs.npz
    python3 gk_law_runs.py alpha    # alpha heating    -> gk_law_alpha.npz
    python3 gk_law_runs.py figs     # quick-look PNGs from the caches
    python3 gk_law_runs.py all

Units: g in units of the bump-peak gradient g_pk, Q and the drive in units of the integrated external
source S (S = 1 unless stated), x in units of L, p = P_PED + int g dx with P_PED = 1.
"""
import json
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import brentq

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gk_law as G  # noqa: E402

FIGS = HERE / "gk_law_figs"
SHAPE_KIND = "plgauss"          # primary bump: power-law rise, log-Gaussian fall (see fit)
G_CAP = 10.0                     # 'runaway' ceiling on the barrier gradient, in units of g_pk (a choice)
NU4_FAC = 1.0                    # multiplier on gk_law.default_nu4 (5e-5 max|Q'|, the paper's rule); see do_nu4test
T_REF_KEV = 3.0                  # p = 1 (the pinned edge) <-> 3 keV in the alpha closure


def shapes():
    return {k: G.fit_bump(k) for k in ("plgauss", "lognormal", "general")}


def step_law(H, shape, ce_over_H=None, ci_over_H=None):
    st = G.step_ratios(shape)
    return G.Law(H, (ce_over_H or st["ce_over_H"]) * H, (ci_over_H or st["ci_over_H"]) * H, shape)


# ----------------------------------------------------------------------------- 1. fit
def do_fit():
    d = G.load_gk(); sh = shapes(); st = G.step_ratios(sh[SHAPE_KIND])
    g = np.logspace(-2, 0.7, 600)
    out = dict(bp=d["bp"], Q_gB=d["Q"], Qe_gB=d["Qe"], Qi_gB=d["Qi"], flag=d["flag"], saturated=d["saturated"],
               live=d["live"], Q_gB_MW=d["Q_gB_MW"], rr_gB=d["rr"], neo_gB=d["neo"], bp_nominal=G.BP_NOMINAL,
               g_grid=g, step_ratios=json.dumps(st))
    for k, s in sh.items():
        out[f"{k}_params"] = json.dumps({q: s[q] for q in ("g_pk", "Q_pk_gB", "sL", "sR", "mL", "mR", "rms_ln")})
        out[f"{k}_B"] = G.bump(g, s)
        out[f"{k}_resid"] = s["resid"]
        print(G.shape_summary(s))
        print("   residuals (ln data - ln fit):", " ".join(f"{-r:+.2f}" for r in s["resid"]))
    print("STEP ratios:", {k: f"{v:.3e}" for k, v in st.items()})
    np.savez(HERE / "gk_law_bump_fit.npz", **out)
    return sh, st


# ----------------------------------------------------------------------------- 2. Maxwell table
def do_table():
    sh = shapes(); st = G.step_ratios(sh[SHAPE_KIND])
    # (i) the one-parameter curve Q_M/H = f(c_e/H) for the three shapes and several c_i/H
    r = np.logspace(-8, 0, 161)
    curves = {}
    for k, s in sh.items():
        for ci_tag, ci in (("step", st["ci_over_H"]), ("zero", 0.0), ("1e-4", 1e-4), ("1e-2", 1e-2)):
            QM, ga, gb, est, Qmin, Alobe = [np.full(r.size, np.nan) for _ in range(6)]
            for i, ce in enumerate(r):
                law = G.Law(1.0, ce, ci, s); mx = law.maxwell()
                if mx is None: continue
                QM[i], ga[i], gb[i], est[i], Qmin[i], Alobe[i] = mx["Q_M"], mx["g_a"], mx["g_b"], mx["Q_M_est"], mx["Q_min"], mx["A_lobe"]
            curves[f"{k}_{ci_tag}"] = dict(Q_M=QM, g_a=ga, g_b=gb, Q_M_est=est, Q_min=Qmin, A_lobe=Alobe)
    s = sh[SHAPE_KIND]; A1 = G.Law(1, 1e-4, 0, s).maxwell()["A_full"]
    print(f"bump area int B dg = {A1:.4f} (g_pk units); leading-order Q_M/H = (4A/3)^(3/4) (c_e/H)^(1/4) = {(4*A1/3)**0.75:.4f} (c_e/H)^(1/4)")
    c = curves[f"{SHAPE_KIND}_step"]
    for ce in (1e-7, 1e-6, 1e-5, 1.069e-4, 1e-3, 1e-2, 1e-1):
        i = np.argmin(abs(np.log(r / ce)))
        print(f"  c_e/H={r[i]:.2e}: Q_M/H={c['Q_M'][i]:.4g} est={c['Q_M_est'][i]:.4g} ratio={c['Q_M'][i]/c['Q_M_est'][i]:.3f} g_a={c['g_a'][i]:.3f} g_b={c['g_b'][i]:.2f}")
    i = np.argmin(abs(np.log(r / st["ce_over_H"])))
    f_step = c["Q_M"][i]
    print(f"STEP line c_e/H={st['ce_over_H']:.3e}: Q_M/H = {f_step:.4f} -> Q_M = S at H/S = {1/f_step:.2f}; "
          + ", ".join(f"{k}: {curves[k + '_step']['Q_M'][i]:.4f}" for k in sh))

    # (ii) the 2-D table on (H/S, c_e/S) with c_i/S fixed small
    Hs = np.logspace(-1, 4, 51); ces = np.logspace(-9, 1, 81); ci = 1e-3
    x = G.R.grid(); S = G.source(x)
    keys = ("Q_M", "g_a", "g_b", "Q_min", "Q_max", "g_minus_S", "g_plus_S", "x_M", "W_L", "W_H", "regime")
    T = {k: np.full((Hs.size, ces.size), np.nan) for k in keys}
    for i, H in enumerate(Hs):
        for j, ce in enumerate(ces):
            law = G.Law(H, ce, ci, sh[SHAPE_KIND]); lm = law.landmarks(); mx = law.maxwell()
            T["Q_max"][i, j], T["Q_min"][i, j] = lm["Q_max"], lm["Q_min"]
            T["g_minus_S"][i, j] = law.g_minus(1.0)
            T["g_plus_S"][i, j] = law.g_plus(1.0) if not lm["monotone"] else law.g_minus(1.0)
            if mx is None:
                T["regime"][i, j] = 0                      # monotone: no falling branch, single branch
                continue
            T["Q_M"][i, j], T["g_a"][i, j], T["g_b"][i, j] = mx["Q_M"], mx["g_a"], mx["g_b"]
            T["x_M"][i, j] = G.x_M(x, S, mx["Q_M"])
            if lm["Q_max"] < 1.0:
                T["regime"][i, j] = 1                      # forced barrier: no L state
            elif mx["Q_M"] < 1.0:
                T["regime"][i, j] = 2                      # bistable
            else:
                T["regime"][i, j] = 3 if lm["Q_min"] < 1.0 else 4   # L only (4: even Q_min > S)
            aL = G.analytic_L(law, x, S); aH = G.analytic_H(law, x, S)
            T["W_L"][i, j] = aL["W"] if aL else np.nan
            T["W_H"][i, j] = aH["W"] if aH else np.nan
    # Maxwell boundary Q_M = S and the forced-transition boundary Q_max = S on the (H/S, c_e/S) plane
    H_b = np.logspace(0, 4, 81); ce_b = np.full(H_b.size, np.nan)
    for i, H in enumerate(H_b):
        def fQ(lce):
            mx = G.Law(H, 10 ** lce, ci, sh[SHAPE_KIND]).maxwell()
            return np.nan if mx is None else mx["Q_M"] - 1.0
        lce = np.linspace(-12, 1, 66); v = np.array([fQ(l) for l in lce])
        ok = np.isfinite(v)
        for k in range(len(lce) - 1):
            if ok[k] and ok[k + 1] and v[k] < 0 <= v[k + 1]:
                ce_b[i] = 10 ** brentq(fQ, lce[k], lce[k + 1]); break
    # the no-falling-branch threshold c_e/H above which the floor swallows the bump (H-independent when c_i << c_e)
    def mono(lr): return G.Law(1.0, 10 ** lr, 0.0, sh[SHAPE_KIND]).landmarks()["monotone"]
    lo_m, hi_m = -3.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo_m + hi_m)
        if mono(mid): hi_m = mid
        else: lo_m = mid
    ce_mono_over_H = 10 ** hi_m
    print(f"no falling branch for c_e/H > {ce_mono_over_H:.4f}")
    np.savez(HERE / "gk_law_maxwell_table.npz", ce_over_H=r, **{f"curve_{k}_{q}": v for k, d in curves.items() for q, v in d.items()},
             H_over_S=Hs, ce_over_S=ces, ci_over_S=ci, **{f"table_{k}": v for k, v in T.items()},
             maxwell_boundary_H=H_b, maxwell_boundary_ce=ce_b, step_ce_over_H=st["ce_over_H"], step_ci_over_H=st["ci_over_H"],
             g_cap=G_CAP, A_bump=A1, f_step=f_step, ce_mono_over_H=ce_mono_over_H, shape=SHAPE_KIND, shape_params=json.dumps({q: sh[SHAPE_KIND][q] for q in ("g_pk", "sL", "sR", "mL", "mR")}))
    print("regime counts:", {int(k): int(v) for k, v in zip(*np.unique(T["regime"], return_counts=True))})
    return curves, T


# ----------------------------------------------------------------------------- 3. runs
def case_list(sh, st):
    """(name, H, c_e, c_i, S_tot, ic, x0, T_end).  c_e, c_i default to the STEP line."""
    f = G.Law(1.0, st["ce_over_H"], st["ci_over_H"], sh).maxwell()["Q_M"]      # Q_M/H on the STEP line
    H_marg = 1.0 / f
    ce, ci = st["ce_over_H"], st["ci_over_H"]
    return [
        # (a) nonlinear bump, S < Q_M: collapse from a formed barrier; (d) subcritical start
        ("a_H1e3_super", 1e3, ce * 1e3, ci * 1e3, 1.0, "super", 0.3, 0.3),
        ("a_H1e3_sub",   1e3, ce * 1e3, ci * 1e3, 1.0, "sub",   0.3, 0.3),
        ("a_H30_super",  30., ce * 30., ci * 30., 1.0, "super", 0.3, 6.0),
        ("a_H30_sub",    30., ce * 30., ci * 30., 1.0, "sub",   0.3, 6.0),
        # (b) small bump, S > Q_M: barrier; subcritical start stays L (bistable)
        ("b_H3_super",   3.0, ce * 3.0, ci * 3.0, 1.0, "super", 0.5, 20.0),
        ("b_H3_sub",     3.0, ce * 3.0, ci * 3.0, 1.0, "sub",   0.5, 20.0),
        ("b_H3_ce3e-2_super", 3.0, 3e-2, ci * 3.0, 1.0, "super", 0.5, 20.0),   # off the STEP line: stiffer floor
        ("b_H3_ce3e-2_sub",   3.0, 3e-2, ci * 3.0, 1.0, "sub",   0.5, 20.0),
        # (c) marginal: H chosen so that Q_M = 1 on the STEP line, drive 0.9, 1.0, 1.1 (front time scale x_M (g_b-g_a)/S ~ 5)
        ("c_marg_S0.9_super", H_marg, ce * H_marg, ci * H_marg, 0.9, "super", 0.3, 40.0),
        ("c_marg_S1.0_super", H_marg, ce * H_marg, ci * H_marg, 1.0, "super", 0.3, 40.0),
        ("c_marg_S1.1_super", H_marg, ce * H_marg, ci * H_marg, 1.1, "super", 0.3, 40.0),
        ("c_marg_S1.1_sub",   H_marg, ce * H_marg, ci * H_marg, 1.1, "sub",   0.3, 40.0),
        # (e) QL-like bump below the drive: the L branch does not exist, forced transition from a subcritical start
        ("e_H0.3_sub",   0.3, ce * 0.3, ci * 0.3, 1.0, "sub",   0.3, 60.0),
    ]


def run_case(name, H, ce, ci, S_tot, ic, x0, T_end, sh, phys=None, p0=None, n_out=241):
    law = G.Law(H, ce, ci, sh); mx = law.maxwell(); lm = law.landmarks()
    x = G.R.grid(); S = G.source(x, S_tot)
    t0 = time.time()
    res = G.run(law, S_tot, ic=ic, x0=x0, nu4=G.default_nu4(law) * NU4_FAC, T_end=T_end, n_out=n_out, phys=phys, p0=p0)
    d = G.diagnostics(res, law)
    chk = G.steady_check(d)
    aL = G.analytic_L(law, x, S); aH = G.analytic_H(law, x, S)
    out = dict(H=H, c_e=ce, c_i=ci, S_tot=S_tot, ic=ic, x0=x0, T_end=T_end, nu4=res["nu4"], wall=time.time() - t0,
               Q_M=mx["Q_M"] if mx else np.nan, g_a=mx["g_a"] if mx else np.nan, g_b=mx["g_b"] if mx else np.nan,
               Q_min=lm["Q_min"], Q_max=lm["Q_max"], x_M=G.x_M(x, S, mx["Q_M"]) if mx else np.nan,
               W_L=aL["W"] if aL else np.nan, W_H=aH["W"] if aH else np.nan,
               g_L=aL["g"] if aL else np.full_like(x, np.nan), g_H=aH["g"] if aH else np.full_like(x, np.nan),
               p_L=aL["p"] if aL else np.full_like(x, np.nan), p_H=aH["p"] if aH else np.full_like(x, np.nan),
               x=x, t=d["t"], W=d["W"], Q_edge=d["Q_edge"], S_tot_t=d["S_tot"], P_alpha=d["P_alpha"], g_max=d["g_max"],
               g_edge=d["g_edge"], x_f=d["x_f"], p=res["p"], p_end=res["p"][-1], g_end=G.face_gradients(res["p"][-1], x[1] - x[0], 0)[0],
               dWdt_rel=chk["dWdt_rel"], flux_balance=chk["flux_balance"], nfev=res["nfev"])
    if mx and ic == "super":
        fl = G.front_law(law, x, S, x0, T_end, mx)
        out.update(law_t=fl["t"], law_x_f=fl["x_f"], law_t_wall=fl["t_wall"])
        W0 = d["W"][0]; WL = aL["W"] if aL else np.nan
        out["t_est_energy"] = (W0 - WL) / (mx["Q_M"] - S_tot) if mx["Q_M"] > S_tot else np.nan
    gone = np.where(np.isnan(d["x_f"]))[0]
    out["t_front_gone"] = d["t"][gone[0]] if len(gone) and ic == "super" else np.nan
    # collapse times: front reaches x = 0.95 (interpolated), and W within 10% of the way to W_L
    xf_ok = np.isfinite(d["x_f"])
    out["t_front_095"] = float(np.interp(0.95, d["x_f"][xf_ok], d["t"][xf_ok])) if xf_ok.any() and np.nanmax(d["x_f"]) > 0.95 and ic == "super" else np.nan
    if aL and ic == "super" and d["W"][0] > aL["W"]:
        thr = aL["W"] + 0.1 * (d["W"][0] - aL["W"])
        below = np.where(d["W"] < thr)[0]
        out["t_W_collapse"] = d["t"][below[0]] if len(below) else np.nan
    else:
        out["t_W_collapse"] = np.nan
    # end-state gradient against the analytic branches, away from the front (|x - x_f| > 0.1) and the axis (x > 0.1)
    g_end = out["g_end"]; xf_ = 0.5 * (x[1:] + x[:-1])
    ref = None
    if phys is None:
        if aH is not None and np.isfinite(d["x_f"][-1]):
            ref, xfront = np.interp(xf_, x, aH["g"]), aH["x_f"]
        elif aL is not None:
            ref, xfront = np.interp(xf_, x, aL["g"]), np.nan
    if ref is not None:
        sel = (xf_ > 0.1) & (xf_ < 0.98) & ((np.abs(xf_ - xfront) > 0.1) if np.isfinite(xfront) else True)
        out["g_dev_max"] = float(np.max(np.abs(g_end[sel] / ref[sel] - 1))) if sel.any() else np.nan
    else:
        out["g_dev_max"] = np.nan
    return out


def do_runs():
    sh = shapes()[SHAPE_KIND]; st = G.step_ratios(sh)
    results = {}
    for (name, H, ce, ci, S_tot, ic, x0, T_end) in case_list(sh, st):
        o = run_case(name, H, ce, ci, S_tot, ic, x0, T_end, sh)
        results[name] = o
        print(f"{name:22s} H={H:8.3g} c_e={ce:.3g} S={S_tot:.2f} ic={ic:5s} | Q_M={o['Q_M']:.4g} Qmin={o['Q_min']:.3g} x_M={o['x_M']:.3f} "
              f"| end: W={o['W'][-1]:.4f} (W_L={o['W_L']:.4f}, W_H={o['W_H']:.4f}) x_f={o['x_f'][-1]:.3f} g_max={o['g_max'][-1]:.3f} "
              f"Qe={o['Q_edge'][-1]:.4g} dW/dt/W={o['dWdt_rel']:+.1e} | t_front95={o['t_front_095']:.4f} t_W={o['t_W_collapse']:.4f} t_gone={o['t_front_gone']:.4f} "
              f"law={o.get('law_t_wall', np.nan):.4f} E-est={o.get('t_est_energy', np.nan):.4f} | g-dev={o['g_dev_max']:.3f} {o['wall']:.1f}s nfev={o['nfev']}", flush=True)
    flat = {}
    for name, o in results.items():
        for k, v in o.items():
            if k in ("x",): continue
            flat[f"{name}__{k}"] = v if not isinstance(v, str) else np.array(v)
    flat["x"] = results[next(iter(results))]["x"]
    flat["case_names"] = np.array(list(results))
    flat["shape"] = SHAPE_KIND; flat["T_REF_KEV"] = T_REF_KEV
    np.savez(HERE / "gk_law_runs.npz", **flat)
    return results


# ----------------------------------------------------------------------------- 4. alpha heating
def phys_params(C_alpha, C_brem=0.01):   # the paper's closure: Bosch-Hale alpha + bremsstrahlung (C_brem = 0.01)
    return dict(G.R.PAPER, C_alpha=C_alpha, C_brem=C_brem, T_ref_keV=T_REF_KEV, n_ref=1.0, n_ref_20=1.0)


def alpha_fixed_point(law, x, S_ext, phys, S_lo=None, S_hi=None):
    """Self-consistent drive on the H branch: S = S_ext + int P_alpha(p_H(S)) dx."""
    mx = law.maxwell()

    def F(S_tot):
        aH = G.analytic_H(law, x, S_ext * S_tot / np.trapezoid(S_ext, x))
        if aH is None: return np.nan
        return S_tot - (np.trapezoid(S_ext, x) + np.trapezoid(G.R.physics_source(aH["p"], phys), x))
    S0 = np.trapezoid(S_ext, x)
    lo = max(mx["Q_M"] * 1.001, S0) if S_lo is None else S_lo
    hi = 1e5 if S_hi is None else S_hi
    ss = np.logspace(np.log10(lo), np.log10(hi), 120)
    vals = np.array([F(s) for s in ss])
    roots = []
    for i in range(len(ss) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            roots.append(brentq(F, ss[i], ss[i + 1]))
    return roots, ss, vals


def alpha_fixed_point_sc(law, x, S_ext, phys, S0, n_iter=60, relax=0.5):
    """Self-consistent burning H-branch state: iterate p -> H-branch profile of the drive S_ext + P_alpha(p)
    (the drive profile of the burning state, not the Gaussian shape).  Starts from the H state of total drive
    S0.  Returns (S_tot, p, converged, history); diverges to the L state (S_tot -> S_ext) or upward if the
    quasi-static state is unstable under this map."""
    mx = law.maxwell()
    S_tot = np.trapezoid(S_ext, x)
    aH = G.analytic_H(law, x, S_ext * S0 / S_tot)
    if aH is None: return np.nan, None, False, []
    p = aH["p"]; hist = []
    for k in range(n_iter):
        drive = S_ext + G.R.physics_source(p, phys)
        drive = np.maximum(drive, 0.0)
        St = np.trapezoid(drive, x); hist.append(St)
        if St <= mx["Q_M"] * 1.001:
            return St, p, False, hist                      # the burning state has fallen below Q_M: L state
        aH = G.analytic_H(law, x, drive * St / np.trapezoid(drive, x))
        if aH is None: return St, p, False, hist
        p_new = aH["p"]
        if np.max(np.abs(p_new - p)) < 1e-6 * np.max(p): return St, p_new, True, hist
        p = relax * p_new + (1 - relax) * p
    return hist[-1], p, False, hist


def do_alpha():
    sh = shapes()[SHAPE_KIND]; st = G.step_ratios(sh)
    x = G.R.grid()
    results = {}
    # (i) access: nonlinear bump, supercritical start, does alpha heating hold the barrier?
    for C_alpha in (1e-5, 1e-4, 1.5e-4, 2e-4, 3e-4, 5e-4, 1e-3, 1e-2):
        H = 1e3
        law = G.Law(H, st["ce_over_H"] * H, st["ci_over_H"] * H, sh)
        roots, ss, vals = alpha_fixed_point(law, x, G.source(x, 1.0), phys_params(C_alpha))
        for start in ("super", "fp"):
            name = f"alpha_access_H1e3_Ca{C_alpha:.1e}" + ("" if start == "super" else "_fp")
            if start == "fp":
                if not roots: continue
                p0 = G.analytic_H(law, x, G.source(x, roots[0]))["p"]      # the burning fixed point (sustainment test)
            else:
                p0 = None
            o = run_case(name, H, st["ce_over_H"] * H, st["ci_over_H"] * H, 1.0, "super", 0.3, 0.3, sh, phys=phys_params(C_alpha), p0=p0)
            o["fixed_points"] = np.array(roots); o["fp_scan_S"] = ss; o["fp_scan_F"] = vals
            Ssc, psc, conv, hist = alpha_fixed_point_sc(law, x, G.source(x, 1.0), phys_params(C_alpha), roots[0] if roots else law.maxwell()["Q_M"] * 1.5)
            o["fixed_point_sc"] = Ssc; o["fixed_point_sc_converged"] = conv; o["fixed_point_sc_hist"] = np.array(hist)
            o["p_fixed_point_sc"] = psc if psc is not None else np.full_like(x, np.nan)
            results[name] = o
            print(f"{name:37s} P_alpha(0)={o['P_alpha'][0]:.3g} max P_alpha={o['P_alpha'].max():.3g} Q_M={o['Q_M']:.4g} | end W={o['W'][-1]:.4f} "
                  f"x_f={o['x_f'][-1]:.3f} S_tot={o['S_tot_t'][-1]:.3g} t_gone={o['t_front_gone']:.4f} dW/dt/W={o['dWdt_rel']:+.1e} | H-branch fixed points S_tot={np.round(roots, 1)} "
                  f"self-consistent {Ssc:.4g} ({'converged' if conv else 'diverged'}, {len(hist)} it) | {o['wall']:.1f}s", flush=True)
    # (ii) sustainment / saturation: scan C_alpha; start (A) from the barrier state of the external drive (H = 3,
    #      S_ext > Q_M: does the burn saturate on the H branch?) or from the self-consistent burning fixed point
    #      (H = 30, S_ext < Q_M: is the burning barrier state stable, i.e. the hysteresis lower branch?),
    #      and (B) from the supercritical front at x0 = 0.3 (access).
    for H, S_ext in ((3.0, 1.0), (30.0, 1.0)):
        law = G.Law(H, st["ce_over_H"] * H, st["ci_over_H"] * H, sh)
        S = G.source(x, S_ext)
        for C_alpha in ((3e-6, 1e-5, 2e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3) if H > 3 else (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)):
            phys = phys_params(C_alpha)
            roots, ss, vals = alpha_fixed_point(law, x, S, phys)
            for start in ("fp", "super"):
                name = f"alpha_sustain_H{H:g}_Ca{C_alpha:.0e}_{start}"
                if start == "fp":
                    if law.maxwell()["Q_M"] < S_ext:
                        p0 = G.analytic_H(law, x, S)["p"]                       # barrier of the external drive alone
                    elif roots:
                        p0 = G.analytic_H(law, x, S * roots[0] / S_ext)["p"]      # burning fixed point (lowest root)
                    else:
                        print(f"{name:36s} no H-branch fixed point: skipped", flush=True); continue
                else:
                    p0 = None
                o = run_case(name, H, st["ce_over_H"] * H, st["ci_over_H"] * H, S_ext, "super", 0.3, 20.0, sh, phys=phys, p0=p0)
                o["fixed_points"] = np.array(roots); o["fp_scan_S"] = ss; o["fp_scan_F"] = vals
                Ssc, psc, conv, hist = alpha_fixed_point_sc(law, x, S, phys, roots[0] if roots else law.maxwell()["Q_M"] * 1.5)
                o["fixed_point_sc"] = Ssc; o["fixed_point_sc_converged"] = conv; o["fixed_point_sc_hist"] = np.array(hist)
                o["p_fixed_point_sc"] = psc if psc is not None else np.full_like(x, np.nan)
                results[name] = o
                T_core = T_REF_KEV * o["p_end"].max()
                print(f"{name:36s} Q_M={o['Q_M']:.3g} P_a(0)={o['P_alpha'][0]:.3g} | end: W={o['W'][-1]:.3f} S_tot={o['S_tot_t'][-1]:.3f} Qe={o['Q_edge'][-1]:.3f} "
                      f"P_a={o['P_alpha'][-1]:.3f} x_f={o['x_f'][-1]:.3f} g_edge={o['g_edge'][-1]:.2f} T_core={T_core:.1f}keV dW/dt/W={o['dWdt_rel']:+.1e} "
                      f"| fixed points: Gaussian-shape S_tot={np.round(roots, 3)}, self-consistent {Ssc:.4g} ({'conv' if conv else 'div'}) | {o['wall']:.1f}s", flush=True)
    # (iii) L-mode burn: nonlinear bump, subcritical start, alpha on: the burn is set by the pinned edge
    for C_alpha in (1e-3, 1e-2):
        name = f"alpha_Lmode_H1e3_Ca{C_alpha:.0e}"
        H = 1e3
        o = run_case(name, H, st["ce_over_H"] * H, st["ci_over_H"] * H, 1.0, "sub", 0.3, 1.0, sh, phys=phys_params(C_alpha))
        results[name] = o
        print(f"{name:30s} end W={o['W'][-1]:.4f} S_tot={o['S_tot_t'][-1]:.3g} P_a={o['P_alpha'][-1]:.3g} g_max={o['g_max'][-1]:.3f} Q_M={o['Q_M']:.3g}", flush=True)
    flat = {}
    for name, o in results.items():
        for k, v in o.items():
            if k in ("x",): continue
            flat[f"{name}__{k}"] = v if not isinstance(v, str) else np.array(v)
    flat["x"] = x; flat["case_names"] = np.array(list(results)); flat["T_REF_KEV"] = T_REF_KEV
    np.savez(HERE / "gk_law_alpha.npz", **flat)
    return results


# ----------------------------------------------------------------------------- 4b. regularisation test
def do_nu4test():
    """End states and front motion against nu4 (x0.1 ... x10 of the default) and the gradient smoothing."""
    sh = shapes()[SHAPE_KIND]; st = G.step_ratios(sh)
    x = G.R.grid(); dx = x[1] - x[0]
    out = {}
    H_marg = 1.0 / G.Law(1.0, st["ce_over_H"], st["ci_over_H"], sh).maxwell()["Q_M"]
    cases = [("collapse_H1e3", 1e3, 1.0, 0.3), ("collapse_H30", 30., 1.0, 6.0), ("barrier_H3", 3., 1.0, 20.0),
             ("marginal_S0.95", H_marg, 0.95, 40.0), ("marginal_S1.05", H_marg, 1.05, 40.0)]
    for name, H, S_tot, T_end in cases:
        law = G.Law(H, st["ce_over_H"] * H, st["ci_over_H"] * H, sh); mx = law.maxwell(); nu0 = G.default_nu4(law)
        S = G.source(x, S_tot); aL = G.analytic_L(law, x, S); aH = G.analytic_H(law, x, S)
        fl = G.front_law(law, x, S, 0.3, T_end, mx)
        for fac, ns in ((0.1, 3), (0.3, 3), (1.0, 3), (3.0, 3), (10.0, 3), (1.0, 0)):
            t0 = time.time()
            try:
                res = G.run(law, S_tot, ic="super", x0=0.3, nu4=nu0 * fac, n_smooth=ns, T_end=T_end, n_out=161)
            except Exception as e:
                print(f"{name} fac={fac} ns={ns}: FAILED {e}", flush=True); continue
            d = G.diagnostics(res, law)
            g, _ = G.face_gradients(res["p"][3], dx, 0)
            lo, hi = mx["g_a"] + 0.1 * (mx["g_b"] - mx["g_a"]), mx["g_a"] + 0.9 * (mx["g_b"] - mx["g_a"])
            width = (0.5 * (x[1:] + x[:-1])[np.argmax(g > hi)] - 0.5 * (x[1:] + x[:-1])[np.argmax(g > lo)]) / dx
            xf_ok = np.isfinite(d["x_f"])
            t95 = float(np.interp(0.95, d["x_f"][xf_ok], d["t"][xf_ok])) if xf_ok.any() and np.nanmax(d["x_f"]) > 0.95 else np.nan
            law_xf_end = fl["sol"].sol(min(T_end, fl["t_wall"]) if np.isfinite(fl["t_wall"]) else T_end)[0]
            key = f"{name}__fac{fac:g}_ns{ns}"
            out[key] = dict(H=H, S_tot=S_tot, nu4=nu0 * fac, n_smooth=ns, t=d["t"], W=d["W"], x_f=d["x_f"], g_edge=d["g_edge"],
                            W_L=aL["W"] if aL else np.nan, W_H=aH["W"] if aH else np.nan, x_M=aH["x_f"] if aH else np.nan,
                            law_t_wall=fl["t_wall"], law_x_f_end=law_xf_end, t_front_095=t95, front_width_cells=width)
            print(f"{name:15s} fac={fac:4.1f} ns={ns}: W_end={d['W'][-1]:.4f} (W_L={out[key]['W_L']:.4f} W_H={out[key]['W_H']:.4f}) x_f_end={d['x_f'][-1]:.3f} "
                  f"(law {law_xf_end:.3f}, x_M {out[key]['x_M']:.3f}) t95={t95:.3f} (law {fl['t_wall']:.3f}) g_edge={d['g_edge'][-1]:.2f} width={width:.0f} cells {time.time()-t0:.0f}s", flush=True)
    flat = {f"{k}__{q}": v for k, o in out.items() for q, v in o.items()}
    flat["keys"] = np.array(list(out)); flat["x"] = x
    np.savez(HERE / "gk_law_nu4.npz", **flat)


# ----------------------------------------------------------------------------- 5. quick-look figures
def do_figs():
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/claude-21648/mplR")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    # bump fit
    f = dict(np.load(HERE / "gk_law_bump_fit.npz", allow_pickle=True))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    sat = f["saturated"]; live = f["live"]; g_pk = json.loads(str(f["plgauss_params"]))["g_pk"]; Qpk = json.loads(str(f["plgauss_params"]))["Q_pk_gB"]
    for a, scale in zip(ax, ("linear", "log")):
        a.plot(f["bp"][live & sat] / g_pk, f["Q_gB"][live & sat] / Qpk, "ko", label="stella (saturated)")
        a.plot(f["bp"][live & ~sat] / g_pk, f["Q_gB"][live & ~sat] / Qpk, "ko", mfc="none", label="stella (not converged)")
        for k, ls in (("plgauss", "-"), ("lognormal", "--"), ("general", ":")):
            a.plot(f["g_grid"], f[f"{k}_B"], ls, label=f"{k} (rms ln {json.loads(str(f[k + '_params']))['rms_ln']:.2f})")
        a.set_yscale(scale); a.set_xlabel("g / g_pk"); a.set_ylabel("Q / Q_pk")
        if scale == "log": a.set_ylim(1e-4, 2); a.set_xscale("log"); a.set_xlim(0.1, 3)
        else: a.set_xlim(0, 2.5)
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGS / "bump_fit.png", dpi=130); plt.close(fig)

    # Maxwell curve, scaling, and regime map
    T = dict(np.load(HERE / "gk_law_maxwell_table.npz", allow_pickle=True))
    r = T["ce_over_H"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for k, ls in (("plgauss", "-"), ("lognormal", "--"), ("general", ":")):
        ax[0].loglog(r, T[f"curve_{k}_step_Q_M"], ls, label=f"{k}")
    ax[0].loglog(r, T["curve_plgauss_step_Q_M_est"], "r-.", lw=0.8, label="(4A/3)^(3/4) c^(1/4)")
    ax[0].axvline(T["step_ce_over_H"], color="grey", ls=":"); ax[0].set_xlabel("c_e / H"); ax[0].set_ylabel("Q_M / H"); ax[0].legend(fontsize=7)
    ax[1].semilogx(r, T["curve_plgauss_step_Q_M"] / T["curve_plgauss_step_Q_M_est"], label="Q_M / estimate")
    ax[1].semilogx(r, T["curve_plgauss_step_g_b"] / 10, label="g_b / 10"); ax[1].semilogx(r, T["curve_plgauss_step_g_a"], label="g_a")
    ax[1].set_xlabel("c_e / H"); ax[1].legend(fontsize=7); ax[1].set_ylim(0, 2)
    Hs, ces = T["H_over_S"], T["ce_over_S"]
    reg = T["table_regime"]; gp = T["table_g_plus_S"]
    cmap = matplotlib.colors.ListedColormap(["#bbbbbb", "#d62728", "#ff7f0e", "#1f77b4", "#08306b"])
    ax[2].pcolormesh(Hs, ces, reg.T, cmap=cmap, vmin=-0.5, vmax=4.5, shading="nearest")
    ax[2].contour(Hs, ces, gp.T, levels=[T["g_cap"]], colors="w", linestyles="--")
    ax[2].loglog(T["maxwell_boundary_H"], T["maxwell_boundary_ce"], "k-", label="Q_M = S")
    ax[2].loglog(Hs, T["step_ce_over_H"] * Hs, "k:", label="STEP line c_e = 1.07e-4 H")
    ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("H / S"); ax[2].set_ylabel("c_e / S")
    ax[2].set_title("0 monotone | 1 forced barrier | 2 bistable | 3 L only | 4 L only, Q_min>S; white: g_+(S)=g_cap", fontsize=7)
    ax[2].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGS / "maxwell_and_map.png", dpi=130); plt.close(fig)

    # runs
    for fname, tag in (("gk_law_runs.npz", "runs"), ("gk_law_alpha.npz", "alpha")):
        p = HERE / fname
        if not p.exists(): continue
        Rn = dict(np.load(p, allow_pickle=True)); x = Rn["x"]; xf = 0.5 * (x[1:] + x[:-1])
        for name in Rn["case_names"]:
            k = lambda q: Rn[f"{name}__{q}"]
            fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
            t = k("t")
            ax[0].plot(t, k("W"), "k", label="W"); ax[0].set_xlabel("t"); ax[0].set_ylabel("W")
            ax[0].axhline(k("W_L"), color="b", ls=":", label="W_L"); ax[0].axhline(k("W_H"), color="r", ls=":", label="W_H")
            a2 = ax[0].twinx(); a2.semilogy(t, k("Q_edge"), "b--", label="Q_edge"); a2.semilogy(t, k("S_tot_t"), "r-.", label="S_tot")
            a2.axhline(k("Q_M"), color="g", ls=":", label="Q_M"); a2.set_ylabel("Q_edge, S_tot")
            ax[0].legend(fontsize=7, loc="upper left"); a2.legend(fontsize=7, loc="upper right")
            ax[1].plot(t, k("x_f"), "k", label="x_f numerical")
            if f"{name}__law_t" in Rn: ax[1].plot(k("law_t"), k("law_x_f"), "r--", label="front law")
            ax[1].axhline(k("x_M"), color="g", ls=":", label="x_M"); ax[1].set_xlabel("t"); ax[1].set_ylabel("x_f"); ax[1].legend(fontsize=7)
            if k("ic") == "super" and np.isfinite(k("t_front_gone")): ax[1].set_xlim(0, min(t[-1], 3 * k("t_front_gone")))
            ax[2].semilogy(xf, np.maximum(k("g_end"), 1e-4), "k", label="g end")
            ax[2].semilogy(x, np.maximum(k("g_L"), 1e-4), "b:", label="g_- (L)"); ax[2].semilogy(x, np.maximum(k("g_H"), 1e-4), "r:", label="H branch")
            P = k("p");
            for i in np.linspace(0, len(P) - 1, 6).astype(int):
                g, _ = G.face_gradients(P[i], x[1] - x[0], 0); ax[2].semilogy(xf, np.maximum(g, 1e-4), color=plt.cm.viridis(i / (len(P) - 1)), lw=0.6)
            ax[2].set_xlabel("x"); ax[2].set_ylabel("g"); ax[2].legend(fontsize=7)
            fig.suptitle(f"{name}: H={k('H'):.3g} c_e={k('c_e'):.3g} S={k('S_tot'):.2f} Q_M={k('Q_M'):.3g}", fontsize=9)
            fig.tight_layout(); fig.savefig(FIGS / f"{tag}_{name}.png", dpi=110); plt.close(fig)
    # alpha summary: end-state S_tot and W against C_alpha with the H-branch fixed points
    pa = HERE / "gk_law_alpha.npz"
    if pa.exists():
        A = dict(np.load(pa, allow_pickle=True))
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        for H, col in ((3.0, "C0"), (30.0, "C1")):
            for start, mk in (("fp", "o"), ("super", "s")):
                Ca, St, Wn, fp = [], [], [], []
                for name in A["case_names"]:
                    if name.startswith(f"alpha_sustain_H{H:g}_") and name.endswith(f"_{start}"):
                        Ca.append(float(name.split("_Ca")[1].split("_")[0])); St.append(A[f"{name}__S_tot_t"][-1]); Wn.append(A[f"{name}__W"][-1])
                        r = A[f"{name}__fixed_points"]; fp.append(r[0] if len(r) else np.nan)
                o = np.argsort(Ca); Ca, St, Wn, fp = np.array(Ca)[o], np.array(St)[o], np.array(Wn)[o], np.array(fp)[o]
                ax[0].loglog(Ca, St, mk, color=col, mfc="none" if start == "super" else col, label=f"H={H:g} {start} start")
                if start == "fp": ax[0].loglog(Ca, fp, "-", color=col, lw=0.8, label=f"H={H:g} H-branch fixed point")
                ax[1].loglog(Ca, Wn, mk, color=col, mfc="none" if start == "super" else col)
        ax[0].set_xlabel("C_alpha"); ax[0].set_ylabel("steady S_tot / S_ext"); ax[0].legend(fontsize=7)
        ax[1].set_xlabel("C_alpha"); ax[1].set_ylabel("steady W")
        fig.tight_layout(); fig.savefig(FIGS / "alpha_summary.png", dpi=130); plt.close(fig)
    # regime map with the run points
    pr = HERE / "gk_law_runs.npz"
    if pr.exists():
        Rn = dict(np.load(pr, allow_pickle=True))
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.pcolormesh(Hs, ces, reg.T, cmap=cmap, vmin=-0.5, vmax=4.5, shading="nearest")
        ax.contour(Hs, ces, np.nan_to_num(gp.T, nan=1e9), levels=[T["g_cap"]], colors="w", linestyles="--")
        ax.loglog(T["maxwell_boundary_H"], T["maxwell_boundary_ce"], "k-", label="Q_M = S")
        ax.loglog(Hs, T["step_ce_over_H"] * Hs, "k:", label="STEP line")
        for name in Rn["case_names"]:
            H, ce, S = float(Rn[f"{name}__H"]), float(Rn[f"{name}__c_e"]), float(Rn[f"{name}__S_tot"])
            ax.plot(H / S, ce / S, "k*" if "super" in name else "kx", ms=8)
            ax.annotate(name.split("_")[0], (H / S, ce / S), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("H / S"); ax.set_ylabel("c_e / S"); ax.legend(fontsize=7, loc="lower right")
        ax.set_title("grey monotone | red forced barrier | orange bistable | blue L only (dark: Q_min>S); white dashed g_+(S)=g_cap", fontsize=6)
        fig.tight_layout(); fig.savefig(FIGS / "regime_map_runs.png", dpi=140); plt.close(fig)
    print("figures in", FIGS)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("fit", "all"): do_fit()
    if what in ("table", "all"): do_table()
    if what in ("runs", "all"): do_runs()
    if what in ("alpha", "all"): do_alpha()
    if what in ("nu4test", "all"): do_nu4test()
    if what in ("figs", "all"): do_figs()
