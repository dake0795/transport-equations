"""
Two-species (electron + ion) 1D radial transport model.

Evolves three fields: p_e, p_i, n   (quasi-neutrality: n_e = n_i = n)

Heat flux driven by TEMPERATURE gradient: kappa_T = -d(ln T)/dx
Particle flux driven by DENSITY gradient:  kappa_n = -d(ln n)/dx

Electron-ion coupling via collisional energy exchange:
    Q_ei = n * (T_i - T_e) / tau_ei

Equations:
    dp_e/dt = -div(Q_e) + S_pe + Q_ei
    dp_i/dt = -div(Q_i) + S_pi - Q_ei
    dn/dt   = -div(Gamma) + S_n
"""

import numpy as np


# ==========================================================
# FLUX MODEL UTILITIES
# ==========================================================

def smooth_step(x, x0, delta):
    return 0.5 * (1.0 + np.tanh((x - x0) / delta))


def make_windows(x, boundaries, deltas):
    if len(boundaries) == 0:
        return [np.ones_like(x)]
    if len(boundaries) != len(deltas):
        raise ValueError("boundaries and deltas must have same length")
    S = [smooth_step(x, b, d) for b, d in zip(boundaries, deltas)]
    windows = [1.0 - S[0]]
    for i in range(len(S) - 1):
        windows.append(S[i] * (1.0 - S[i + 1]))
    windows.append(S[-1])
    return windows


def _smooth_face(g, n_passes):
    """Apply n_passes of (1/4, 1/2, 1/4) binomial filter to face array."""
    for _ in range(n_passes):
        g_s = g.copy()
        g_s[1:-1] = 0.25*g[:-2] + 0.5*g[1:-1] + 0.25*g[2:]
        g = g_s
    return g


def heat_flux_function(g, x, params):
    """
    Heat flux as function of log-temperature gradient.
    NL / core / linear models blended by spatial windows, plus background chi_RR.
    """
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    g = np.maximum(g, 0.0)

    chi0     = params.get("chi0",     1.0)
    chi_core = params.get("chi_core", 1.0)
    chi_RR   = params.get("chi_RR",   0.01)
    g_c      = params.get("g_c",      2.0)
    g_stiff  = params.get("g_stiff",  1.5)
    n_stiff  = params.get("n_stiff",  2)

    boundaries  = params.get("boundaries", [])
    deltas      = params.get("deltas", [0.01] * len(boundaries))
    flux_models = params.get("flux_models", ["nl"])

    ratio  = np.clip(g / g_stiff, 0.0, 20.0)
    Q_core = chi_core * g * (1.0 + ratio**n_stiff)
    Q_nl   = np.maximum(chi0 * g * (1.0 - g / g_c), 0.0)
    Q_lin  = chi_core * g

    Q_map = {"core": Q_core, "nl": Q_nl, "linear": Q_lin}
    W = make_windows(x, boundaries, deltas)

    if len(W) != len(flux_models):
        raise ValueError(f"flux_models length must equal number of regions ({len(W)})")

    Q_transport = sum(Wi * Q_map[m] for Wi, m in zip(W, flux_models))
    return Q_transport + chi_RR * g


def particle_flux_function(g_n, x, params):
    """
    Particle flux as function of log-density gradient.
    Same model structure as heat flux but with _n parameter suffixes.
    """
    g_n = np.nan_to_num(g_n, nan=0.0, posinf=0.0, neginf=0.0)
    g_n = np.maximum(g_n, 0.0)

    chi0_n     = params.get("chi0_n",     0.1)
    chi_core_n = params.get("chi_core_n", 0.1)
    chi_RR_n   = params.get("chi_RR_n",   0.01)
    g_c_n      = params.get("g_c_n",      2.0)
    g_stiff_n  = params.get("g_stiff_n",  1.5)
    n_stiff_n  = params.get("n_stiff_n",  2)

    boundaries    = params.get("boundaries",   [])
    deltas        = params.get("deltas",       [0.01] * len(boundaries))
    flux_models_n = params.get("flux_models_n", params.get("flux_models", ["nl"]))

    ratio  = np.clip(g_n / g_stiff_n, 0.0, 20.0)
    G_core = chi_core_n * g_n * (1.0 + ratio**n_stiff_n)
    G_nl   = np.maximum(chi0_n * g_n * (1.0 - g_n / g_c_n), 0.0)
    G_lin  = chi_core_n * g_n

    G_map = {"core": G_core, "nl": G_nl, "linear": G_lin}
    W = make_windows(x, boundaries, deltas)

    if len(W) != len(flux_models_n):
        raise ValueError(f"flux_models_n length must equal number of regions ({len(W)})")

    G_transport = sum(Wi * G_map[m] for Wi, m in zip(W, flux_models_n))
    return G_transport + chi_RR_n * g_n


# ==========================================================
# COLLISION COUPLING
# ==========================================================

def collision_exchange(n, p_e, p_i, params):
    """
    Collisional energy exchange: Q_ei = n * (T_i - T_e) / tau_ei.
    Electrons gain +Q_ei, ions lose -Q_ei.
    Set tau_ei = 0 to disable.
    """
    tau_ei = params.get("tau_ei", 0.0)
    if tau_ei <= 0.0:
        return np.zeros_like(p_e)
    T_e = p_e / np.maximum(n, 1e-10)
    T_i = p_i / np.maximum(n, 1e-10)
    return n * (T_i - T_e) / tau_ei


# ==========================================================
# SPECIES PARAMETER EXTRACTION
# ==========================================================

def _get_species_params(params, species):
    """
    Extract species-specific params.  For species 'e': looks for 'chi0_e'
    first, then falls back to 'chi0'.
    """
    suffix = f"_{species}"
    keys = [
        "chi0", "chi_core", "chi_RR", "g_c", "g_stiff", "n_stiff",
        "chi0_n", "chi_core_n", "chi_RR_n", "g_c_n", "g_stiff_n", "n_stiff_n",
        "V_n", "flux_models", "flux_models_n", "boundaries", "deltas", "nu4",
        "n_smooth",
    ]
    sp = {}
    for key in keys:
        spec_key = key + suffix
        if spec_key in params:
            sp[key] = params[spec_key]
        elif key in params:
            sp[key] = params[key]
    return sp


# ==========================================================
# PHYSICS SOURCES
# ==========================================================

def sigma_v_DT(T_keV):
    """Bosch-Hale D-T fusion reactivity [m^3/s].  Valid 0.2-100 keV."""
    T     = np.maximum(np.asarray(T_keV, dtype=float), 0.2)
    B_G   = 34.3827
    mr_c2 = 1.124656e6
    C1    = 1.17302e-9
    C2, C3 = 1.51361e-2,  7.51886e-2
    C4, C5 = 4.60643e-3,  1.35000e-2
    C6, C7 = -1.06750e-4, 1.36600e-5
    theta = T / (1.0 - T*(C2 + T*(C4 + T*C6)) / (1.0 + T*(C3 + T*(C5 + T*C7))))
    xi    = (B_G**2 / (4.0 * theta))**(1.0 / 3.0)
    sigv  = C1 * theta * np.sqrt(xi / (mr_c2 * T**3)) * np.exp(-3.0 * xi)
    return sigv * 1.0e-6


def _alpha_ion_fraction(T_e_keV, n_i, Z_i, A_i, n_e):
    """Stix critical energy: fraction of 3.5 MeV alpha power going to ions."""
    T_e  = np.maximum(T_e_keV, 0.1)
    z2m  = n_i * Z_i**2 / A_i / np.maximum(n_e, 1e-10)
    Ec   = 4.0 * 14.8 * T_e * np.maximum(z2m, 1e-10)**(2.0 / 3.0)
    x    = 3500.0 / Ec
    y    = np.sqrt(x)
    frac = (  np.log((1.0 + y**3) / (1.0 + y)**3) / (3.0 * x)
            + 2.0 * np.arctan((2.0*y - 1.0) / np.sqrt(3.0)) / (np.sqrt(3.0) * x)
            - 2.0 * np.arctan(-1.0          / np.sqrt(3.0)) / (np.sqrt(3.0) * x))
    return np.clip(frac, 0.0, 1.0)


def alpha_heating(n, p_e, p_i, params):
    """
    D-T alpha heating split between electrons and ions (Stix formula).
    Returns (P_alpha_e, P_alpha_i) in model source units.
    """
    C_alpha = params.get("C_alpha", 0.0)
    if C_alpha <= 0.0:
        return np.zeros_like(n), np.zeros_like(n)

    T_ref  = params.get("T_ref_keV", 10.0)
    n_ref  = params.get("n_ref_20",   1.0)
    Z_i    = params.get("Z_i",        1)
    A_i    = params.get("A_i",        2.5)
    f_D    = params.get("f_deuterium", 0.5)
    f_T    = params.get("f_tritium",   0.5)

    T_e_keV = p_e / np.maximum(n, 1e-10) * T_ref
    T_i_keV = p_i / np.maximum(n, 1e-10) * T_ref
    n_phys  = n * n_ref * 1.0e20

    n_D  = f_D * n_phys
    n_T  = f_T * n_phys
    sigv = sigma_v_DT(T_i_keV)
    P_fus = 5.6e-13 * sigv * n_D * n_T

    f_ion     = _alpha_ion_fraction(T_e_keV, n_phys, Z_i, A_i, n_phys)
    P_alpha_e = C_alpha * (1.0 - f_ion) * P_fus
    P_alpha_i = C_alpha * f_ion         * P_fus
    return P_alpha_e, P_alpha_i


def bremsstrahlung(n, p_e, params):
    """Bremsstrahlung loss from electrons: P_brem = C_brem * n^2 * Z_eff * sqrt(T_e_keV)."""
    C_brem = params.get("C_brem", 0.0)
    if C_brem <= 0.0:
        return np.zeros_like(n)
    Z_eff = params.get("Z_eff", 1.0)
    T_ref = params.get("T_ref_keV", 10.0)
    T_e   = p_e / np.maximum(n, 1e-10) * T_ref
    return C_brem * n**2 * Z_eff * np.sqrt(np.maximum(T_e, 1e-10))


# ==========================================================
# BOUNDARY CONDITIONS
# ==========================================================

def _get_bcs(params, p_e_ped, p_i_ped, n_ped):
    bc = params.get("bc", {})
    core_pe = bc.get("core_pe", {"type": "neumann",   "value": 0.0})
    edge_pe = bc.get("edge_pe", {"type": "dirichlet", "value": p_e_ped})
    core_pi = bc.get("core_pi", {"type": "neumann",   "value": 0.0})
    edge_pi = bc.get("edge_pi", {"type": "dirichlet", "value": p_i_ped})
    core_n  = bc.get("core_n",  {"type": "neumann",   "value": 0.0})
    edge_n  = bc.get("edge_n",  {"type": "dirichlet", "value": n_ped})
    return core_pe, edge_pe, core_pi, edge_pi, core_n, edge_n


def _apply_bc(prof, core, edge, dx):
    if core["type"] == "dirichlet":
        prof[0] = core["value"]
    elif core["type"] == "neumann":
        prof[0] = prof[1] - core["value"] * dx
    if edge["type"] == "dirichlet":
        prof[-1] = edge["value"]
    elif edge["type"] == "neumann":
        prof[-1] = prof[-2] + edge["value"] * dx


def _flux_div(flux, core_bc, edge_bc, dx, N):
    """Finite-volume divergence of face-centred flux array."""
    div = np.zeros(N)
    div[1:-1] = (flux[1:] - flux[:-1]) / dx
    if core_bc["type"] == "flux":
        div[0] = (flux[0] - core_bc["value"]) / dx
    else:
        div[0] = flux[0] / dx
    if edge_bc["type"] == "flux":
        div[-1] = (edge_bc["value"] - flux[-1]) / dx
    else:
        div[-1] = -flux[-1] / dx
    return div


def _log_kap(prof, dx):
    """Log gradient at cell faces: kappa = -(ln f[j+1] - ln f[j]) / dx."""
    return -(np.log(np.maximum(prof[1:], 1e-10))
             - np.log(np.maximum(prof[:-1], 1e-10))) / dx


# ==========================================================
# DIAGNOSTIC FLUX COMPUTATION
# ==========================================================

def compute_fluxes(p_e, p_i, n, dx, x, p_e_ped, p_i_ped, n_ped, params):
    """
    Return face-centred fluxes and collision exchange for diagnostics.
    """
    bcs = _get_bcs(params, p_e_ped, p_i_ped, n_ped)
    core_pe, edge_pe, core_pi, edge_pi, core_n, edge_n = bcs

    pe_bc = p_e.copy(); pi_bc = p_i.copy(); n_bc = n.copy()
    _apply_bc(pe_bc, core_pe, edge_pe, dx)
    _apply_bc(pi_bc, core_pi, edge_pi, dx)
    _apply_bc(n_bc,  core_n,  edge_n,  dx)

    x_face   = 0.5 * (x[1:] + x[:-1])
    params_e = _get_species_params(params, "e")
    params_i = _get_species_params(params, "i")

    T_e_bc = pe_bc / np.maximum(n_bc, 1e-10)
    T_i_bc = pi_bc / np.maximum(n_bc, 1e-10)

    Q_e_face   = heat_flux_function(_log_kap(T_e_bc, dx), x_face, params_e)
    Q_i_face   = heat_flux_function(_log_kap(T_i_bc, dx), x_face, params_i)
    Gamma_face = particle_flux_function(_log_kap(n_bc, dx), x_face, params_e)

    Q_ei = collision_exchange(n_bc, pe_bc, pi_bc, params)

    return Q_e_face, Q_i_face, Gamma_face, Q_ei


# ==========================================================
# RHS
# ==========================================================

def compute_rhs(p_e, p_i, n, dx, x,
                p_e_ped, p_i_ped, n_ped,
                source_pe_fn, source_pi_fn, source_n_fn,
                params):
    """
    RHS for the three-field system (p_e, p_i, n).
    """
    bcs = _get_bcs(params, p_e_ped, p_i_ped, n_ped)
    core_pe, edge_pe, core_pi, edge_pi, core_n, edge_n = bcs

    pe_bc = p_e.copy(); pi_bc = p_i.copy(); n_bc = n.copy()
    _apply_bc(pe_bc, core_pe, edge_pe, dx)
    _apply_bc(pi_bc, core_pi, edge_pi, dx)
    _apply_bc(n_bc,  core_n,  edge_n,  dx)

    x_face   = 0.5 * (x[1:] + x[:-1])
    N        = len(pe_bc)
    params_e = _get_species_params(params, "e")
    params_i = _get_species_params(params, "i")
    n_smooth = params.get("n_smooth", 0)

    # --- Electron heat flux (log T_e gradient) ---
    T_e_bc  = pe_bc / np.maximum(n_bc, 1e-10)
    kap_Te  = _smooth_face(_log_kap(T_e_bc, dx), n_smooth)
    Q_e     = heat_flux_function(kap_Te, x_face, params_e)
    div_Qe  = _flux_div(Q_e, core_pe, edge_pe, dx, N)

    # --- Ion heat flux (log T_i gradient) ---
    T_i_bc  = pi_bc / np.maximum(n_bc, 1e-10)
    kap_Ti  = _smooth_face(_log_kap(T_i_bc, dx), n_smooth)
    Q_i     = heat_flux_function(kap_Ti, x_face, params_i)
    div_Qi  = _flux_div(Q_i, core_pi, edge_pi, dx, N)

    # --- Particle flux (log n gradient) ---
    V_n     = params_e.get("V_n", 0.0)
    kap_n   = _smooth_face(_log_kap(n_bc, dx), n_smooth)
    Gamma   = particle_flux_function(kap_n, x_face, params_e) + V_n * n_bc[1:]
    div_G   = _flux_div(Gamma, core_n, edge_n, dx, N)

    # --- External sources ---
    S_pe = source_pe_fn(x, pe_bc, pi_bc, n_bc)
    S_pi = source_pi_fn(x, pe_bc, pi_bc, n_bc)
    S_n  = source_n_fn(x, pe_bc, pi_bc, n_bc)

    # --- Physics sources ---
    P_brem = bremsstrahlung(n_bc, pe_bc, params)
    P_alpha_e, P_alpha_i = alpha_heating(n_bc, pe_bc, pi_bc, params)
    S_pe = S_pe - P_brem + P_alpha_e
    S_pi = S_pi + P_alpha_i

    # --- Collisional exchange ---
    Q_ei = collision_exchange(n_bc, pe_bc, pi_bc, params)

    # --- Hyperviscosity (pressure fields only) ---
    nu4   = params.get("nu4", 0.0)
    d4pe  = np.zeros(N)
    d4pi  = np.zeros(N)
    if nu4 > 0 and N > 4:
        d4pe[2:-2] = (pe_bc[:-4] - 4*pe_bc[1:-3] + 6*pe_bc[2:-2] - 4*pe_bc[3:-1] + pe_bc[4:]) / dx**4
        d4pi[2:-2] = (pi_bc[:-4] - 4*pi_bc[1:-3] + 6*pi_bc[2:-2] - 4*pi_bc[3:-1] + pi_bc[4:]) / dx**4

    rhs_pe = -div_Qe - nu4*d4pe + S_pe + Q_ei
    rhs_pi = -div_Qi - nu4*d4pi + S_pi - Q_ei
    rhs_n  = -div_G                     + S_n

    return rhs_pe, rhs_pi, rhs_n


# ==========================================================
# RK4 TIME STEP
# ==========================================================

def rk4_step(p_e, p_i, n, dt, dx, x,
             p_e_ped, p_i_ped, n_ped,
             source_pe_fn, source_pi_fn, source_n_fn, params):

    def rhs(pe, pi_, nn):
        return compute_rhs(pe, pi_, nn, dx, x,
                           p_e_ped, p_i_ped, n_ped,
                           source_pe_fn, source_pi_fn, source_n_fn, params)

    k1 = rhs(p_e, p_i, n)
    k2 = rhs(p_e + 0.5*dt*k1[0], p_i + 0.5*dt*k1[1], n + 0.5*dt*k1[2])
    k3 = rhs(p_e + 0.5*dt*k2[0], p_i + 0.5*dt*k2[1], n + 0.5*dt*k2[2])
    k4 = rhs(p_e + dt*k3[0],     p_i + dt*k3[1],     n + dt*k3[2])

    return (p_e + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6,
            p_i + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6,
            n   + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6)


# ==========================================================
# SOLVER LOOP
# ==========================================================

def solving_loop(
        p_e_init, p_i_init, n_init,
        dt, dx, T, L,
        p_e_ped, p_i_ped, n_ped,
        source_pe_fn, source_pi_fn, source_n_fn,
        num_snapshots, params):

    N         = len(p_e_init)
    x         = np.linspace(0, L, N)
    num_steps = int(T / dt)
    save_idx  = np.linspace(0, num_steps, num_snapshots + 1, dtype=int)

    p_e = p_e_init.copy(); p_i = p_i_init.copy(); n = n_init.copy()
    saved_pe, saved_pi, saved_n = [], [], []
    save_counter = 0

    for i in range(num_steps + 1):
        if i == save_idx[save_counter]:
            saved_pe.append(p_e.copy())
            saved_pi.append(p_i.copy())
            saved_n.append(n.copy())
            save_counter = min(save_counter + 1, len(save_idx) - 1)

        p_e, p_i, n = rk4_step(
            p_e, p_i, n, dt, dx, x,
            p_e_ped, p_i_ped, n_ped,
            source_pe_fn, source_pi_fn, source_n_fn, params)

        # Enforce BCs
        p_e[0] = p_e[1];  p_e[-1] = p_e_ped
        p_i[0] = p_i[1];  p_i[-1] = p_i_ped
        n[0]   = n[1];    n[-1]   = n_ped

        if i % 20000 == 0:
            T_e = p_e / np.maximum(n, 1e-10)
            T_i = p_i / np.maximum(n, 1e-10)
            kap_Te = np.max(-np.gradient(np.log(np.maximum(T_e, 1e-10)), dx))
            kap_Ti = np.max(-np.gradient(np.log(np.maximum(T_i, 1e-10)), dx))
            print(f"Step {i:6d} | max kap_Te = {kap_Te:.4f} | max kap_Ti = {kap_Ti:.4f}")

    return np.array(saved_pe), np.array(saved_pi), np.array(saved_n)
