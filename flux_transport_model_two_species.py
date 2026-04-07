"""
Two-species (electron + ion) 1D radial transport model.

Evolves: p_e, n_e, p_i, n_i

Heat flux driven by TEMPERATURE gradient: g_T = -dT/dx  (T = p/n)
Particle flux driven by DENSITY gradient:  g_n = -dn/dx

Electron-ion coupling via collisional energy exchange:
    Q_ei = n_e * (T_i - T_e) / tau_ei

Equations:
    dp_e/dt = -∇·Q_e + S_pe + Q_ei
    dn_e/dt = -∇·Γ_e + S_ne
    dp_i/dt = -∇·Q_i + S_pi - Q_ei
    dn_i/dt = -∇·Γ_i + S_ni
"""

import numpy as np

TIME_SCHEME = "RK4"   # "Euler" or "RK4"


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


def heat_flux_function(g, x, params):
    """
    Heat flux as function of a gradient g (can be dT/dx, dp/dx, etc).

    NL model:   Q_nl  = chi0 * g * max(1 - g/g_c, 0)
    Core model: Q_core = chi_core * g * (1 + (g/g_stiff)^n_stiff)
    Linear:     Q_lin  = chi_core * g
    Plus background: chi_RR * g everywhere.
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
        raise ValueError(f"Number of flux_models must equal number of regions ({len(W)} expected)")

    Q_transport = sum(Wi * Q_map[m] for Wi, m in zip(W, flux_models))
    return Q_transport + chi_RR * g


def particle_flux_function(g_n, x, params):
    """
    Particle flux as function of density gradient g_n = -dn/dx.
    Same NL model structure as heat flux but with _n parameter suffixes.
    """
    g_n = np.nan_to_num(g_n, nan=0.0, posinf=0.0, neginf=0.0)
    g_n = np.maximum(g_n, 0.0)

    chi0_n     = params.get("chi0_n",     0.1)
    chi_core_n = params.get("chi_core_n", 0.1)
    chi_RR_n   = params.get("chi_RR_n",   0.01)
    g_c_n      = params.get("g_c_n",      2.0)
    g_stiff_n  = params.get("g_stiff_n",  1.5)
    n_stiff_n  = params.get("n_stiff_n",  2)

    boundaries   = params.get("boundaries",   [])
    deltas       = params.get("deltas",       [0.01] * len(boundaries))
    flux_models_n = params.get("flux_models_n", params.get("flux_models", ["nl"]))

    ratio     = np.clip(g_n / g_stiff_n, 0.0, 20.0)
    G_core    = chi_core_n * g_n * (1.0 + ratio**n_stiff_n)
    G_nl      = np.maximum(chi0_n * g_n * (1.0 - g_n / g_c_n), 0.0)
    G_lin     = chi_core_n * g_n

    G_map = {"core": G_core, "nl": G_nl, "linear": G_lin}
    W = make_windows(x, boundaries, deltas)

    if len(W) != len(flux_models_n):
        raise ValueError(f"Number of flux_models_n must equal number of regions ({len(W)} expected)")

    G_transport = sum(Wi * G_map[m] for Wi, m in zip(W, flux_models_n))
    return G_transport + chi_RR_n * g_n


# ==========================================================
# COLLISION COUPLING
# ==========================================================

def collision_exchange(n_e, p_e, n_i, p_i, params):
    """
    Collisional energy exchange between electrons and ions.

        Q_ei = n_e * (T_i - T_e) / tau_ei

    Electrons gain: dp_e/dt += +Q_ei
    Ions lose:      dp_i/dt += -Q_ei

    tau_ei: dimensionless equilibration time (larger = weaker coupling).
    Set tau_ei = 0 or omit to disable coupling.
    """
    tau_ei = params.get("tau_ei", 0.0)
    if tau_ei <= 0.0:
        return np.zeros_like(p_e)
    T_e = p_e / np.maximum(n_e, 1e-10)
    T_i = p_i / np.maximum(n_i, 1e-10)
    return n_e * (T_i - T_e) / tau_ei


# ==========================================================
# SPECIES PARAMETER EXTRACTION
# ==========================================================

def _get_species_params(params, species):
    """
    Extract species-specific params from the global params dict.

    For species 'e': looks for 'chi0_e' first, then falls back to 'chi0'.
    For species 'i': looks for 'chi0_i' first, then falls back to 'chi0'.
    """
    suffix = f"_{species}"
    keys = [
        "chi0", "chi_core", "chi_RR", "g_c", "g_stiff", "n_stiff",
        "chi0_n", "chi_core_n", "chi_RR_n", "g_c_n", "g_stiff_n", "n_stiff_n",
        "V_n", "flux_models", "flux_models_n", "boundaries", "deltas", "nu4",
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
# BOUNDARY CONDITIONS
# ==========================================================

def _get_bcs_two_species(params, p_e_ped, n_e_ped, p_i_ped, n_i_ped):
    bc = params.get("bc", {})
    core_pe = bc.get("core_pe", {"type": "neumann",    "value": 0.0})
    edge_pe = bc.get("edge_pe", {"type": "dirichlet",  "value": p_e_ped})
    core_ne = bc.get("core_ne", {"type": "neumann",    "value": 0.0})
    edge_ne = bc.get("edge_ne", {"type": "dirichlet",  "value": n_e_ped})
    core_pi = bc.get("core_pi", {"type": "neumann",    "value": 0.0})
    edge_pi = bc.get("edge_pi", {"type": "dirichlet",  "value": p_i_ped})
    core_ni = bc.get("core_ni", {"type": "neumann",    "value": 0.0})
    edge_ni = bc.get("edge_ni", {"type": "dirichlet",  "value": n_i_ped})
    return core_pe, edge_pe, core_ne, edge_ne, core_pi, edge_pi, core_ni, edge_ni


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
    """
    Log gradient at cell faces: κ = -d(ln f)/dx
    Computed as -(ln f[j+1] - ln f[j]) / dx  (exact finite-difference of log).
    This is equivalent to -(1/f) df/dx in the continuum limit.
    """
    return -(np.log(np.maximum(prof[1:], 1e-10))
             - np.log(np.maximum(prof[:-1], 1e-10))) / dx


# ==========================================================
# PHYSICS MODELS  (bremsstrahlung + alpha heating)
# ==========================================================

def sigma_v_DT(T_keV):
    """
    Bosch-Hale D-T fusion reactivity in m³/s.
    Bosch & Hale, Nucl. Fusion 32 (1992), Table 4.  Valid 0.2–100 keV.
    """
    T     = np.maximum(np.asarray(T_keV, dtype=float), 0.2)
    B_G   = 34.3827        # keV^{1/2}
    mr_c2 = 1.124656e6     # keV  (reduced mass × c²)
    C1    = 1.17302e-9     # cm³/s
    C2, C3 = 1.51361e-2,  7.51886e-2
    C4, C5 = 4.60643e-3,  1.35000e-2
    C6, C7 = -1.06750e-4, 1.36600e-5
    theta = T / (1.0 - T*(C2 + T*(C4 + T*C6)) / (1.0 + T*(C3 + T*(C5 + T*C7))))
    xi    = (B_G**2 / (4.0 * theta))**(1.0 / 3.0)
    sigv  = C1 * theta * np.sqrt(xi / (mr_c2 * T**3)) * np.exp(-3.0 * xi)
    return sigv * 1.0e-6   # cm³/s → m³/s


def _alpha_ion_fraction(T_e_keV, n_i, Z_i, A_i, n_e):
    """
    Stix critical energy formula: fraction of 3.5 MeV alpha power going to ions.
        E_crit = 4 × 14.8 × T_e × (n_i Z_i²/A_i / n_e)^(2/3)   [keV]
    """
    T_e  = np.maximum(T_e_keV, 0.1)
    z2m  = n_i * Z_i**2 / A_i / np.maximum(n_e, 1e-10)
    Ec   = 4.0 * 14.8 * T_e * np.maximum(z2m, 1e-10)**(2.0 / 3.0)
    x    = 3500.0 / Ec          # E_alpha = 3500 keV
    y    = np.sqrt(x)
    frac = (  np.log((1.0 + y**3) / (1.0 + y)**3) / (3.0 * x)
            + 2.0 * np.arctan((2.0*y - 1.0) / np.sqrt(3.0)) / (np.sqrt(3.0) * x)
            - 2.0 * np.arctan(-1.0          / np.sqrt(3.0)) / (np.sqrt(3.0) * x))
    return np.clip(frac, 0.0, 1.0)


def alpha_heating(n_e, p_e, n_i, p_i, params):
    """
    D-T alpha heating split between electrons and ions (Stix formula).

    Physical basis:
        P_α = 5.6e-13 J × ⟨σv⟩(T_i) × n_D × n_T   [W/m³]
    Ion/electron split via E_crit (T3D Physics.py lines 434-447).

    Returns (P_alpha_e, P_alpha_i) in model source units.
    C_alpha  encodes unit conversion + time-scale normalisation.
    T_ref_keV: model T=1 corresponds to T_ref_keV keV (default 10).
    n_ref_20:  model n=1 corresponds to n_ref_20 × 10²⁰ m⁻³ (default 1).
    """
    C_alpha = params.get("C_alpha", 0.0)
    if C_alpha <= 0.0:
        return np.zeros_like(n_e), np.zeros_like(n_i)

    T_ref  = params.get("T_ref_keV", 10.0)
    n_ref  = params.get("n_ref_20",   1.0)
    Z_i    = params.get("Z_i",        1)
    A_i    = params.get("A_i",        2.5)   # D-T average amu
    f_D    = params.get("f_deuterium", 0.5)
    f_T    = params.get("f_tritium",   0.5)

    T_e_keV = p_e / np.maximum(n_e, 1e-10) * T_ref
    T_i_keV = p_i / np.maximum(n_i, 1e-10) * T_ref
    n_e_phys = n_e * n_ref * 1.0e20   # m⁻³
    n_i_phys = n_i * n_ref * 1.0e20

    n_D  = f_D * n_i_phys
    n_T  = f_T * n_i_phys
    sigv = sigma_v_DT(T_i_keV)                        # m³/s
    P_fus_Wm3 = 5.6e-13 * sigv * n_D * n_T            # W/m³

    f_ion      = _alpha_ion_fraction(T_e_keV, n_i_phys, Z_i, A_i, n_e_phys)
    P_alpha_e  = C_alpha * (1.0 - f_ion) * P_fus_Wm3
    P_alpha_i  = C_alpha * f_ion         * P_fus_Wm3
    return P_alpha_e, P_alpha_i


def bremsstrahlung(n_e, p_e, params):
    """
    Bremsstrahlung radiation loss from electrons only (same as T3D).
        P_brem = 5.35e-3 × n_e² × Z_eff × √T_e   [MW/m³]
    with n_e in 10²⁰ m⁻³, T_e in keV.
    Returns power loss in model source units (same C_brem normalisation).
    """
    C_brem = params.get("C_brem", 0.0)
    if C_brem <= 0.0:
        return np.zeros_like(n_e)
    Z_eff  = params.get("Z_eff", 1.0)
    T_ref  = params.get("T_ref_keV", 10.0)
    T_e    = p_e / np.maximum(n_e, 1e-10) * T_ref   # in keV
    return C_brem * n_e**2 * Z_eff * np.sqrt(np.maximum(T_e, 1e-10))


# ==========================================================
# DIAGNOSTIC FLUX COMPUTATION
# ==========================================================

def compute_fluxes_two_species(p_e, n_e, p_i, n_i, dx, x,
                                p_e_ped, n_e_ped, p_i_ped, n_i_ped, params):
    """
    Return face-centred fluxes and collision exchange for diagnostics.
    Does not modify input arrays.
    """
    bcs = _get_bcs_two_species(params, p_e_ped, n_e_ped, p_i_ped, n_i_ped)
    core_pe, edge_pe, core_ne, edge_ne, core_pi, edge_pi, core_ni, edge_ni = bcs

    pe_bc = p_e.copy(); ne_bc = n_e.copy()
    pi_bc = p_i.copy(); ni_bc = n_i.copy()
    _apply_bc(pe_bc, core_pe, edge_pe, dx)
    _apply_bc(ne_bc, core_ne, edge_ne, dx)
    _apply_bc(pi_bc, core_pi, edge_pi, dx)
    _apply_bc(ni_bc, core_ni, edge_ni, dx)

    x_face   = 0.5 * (x[1:] + x[:-1])
    params_e = _get_species_params(params, "e")
    params_i = _get_species_params(params, "i")

    T_e_bc = pe_bc / np.maximum(ne_bc, 1e-10)
    T_i_bc = pi_bc / np.maximum(ni_bc, 1e-10)

    Q_e_face     = heat_flux_function(_log_kap(T_e_bc, dx), x_face, params_e)
    Gamma_e_face = particle_flux_function(_log_kap(ne_bc, dx), x_face, params_e)
    Q_i_face     = heat_flux_function(_log_kap(T_i_bc, dx), x_face, params_i)
    Gamma_i_face = particle_flux_function(_log_kap(ni_bc, dx), x_face, params_i)

    Q_ei = collision_exchange(ne_bc, pe_bc, ni_bc, pi_bc, params)

    return Q_e_face, Gamma_e_face, Q_i_face, Gamma_i_face, Q_ei


# ==========================================================
# RHS FOR TWO-SPECIES SYSTEM
# ==========================================================

def compute_rhs_two_species(
        p_e, n_e, p_i, n_i,
        dx, x,
        p_e_ped, n_e_ped, p_i_ped, n_i_ped,
        source_pe_fn, source_ne_fn, source_pi_fn, source_ni_fn,
        params):
    """
    Compute RHS for two-species coupled transport.

    Heat flux of each species driven by its own temperature gradient g_T = -dT/dx.
    Particle flux driven by density gradient g_n = -dn/dx.
    """
    bcs = _get_bcs_two_species(params, p_e_ped, n_e_ped, p_i_ped, n_i_ped)
    core_pe, edge_pe, core_ne, edge_ne, core_pi, edge_pi, core_ni, edge_ni = bcs

    pe_bc = p_e.copy(); ne_bc = n_e.copy()
    pi_bc = p_i.copy(); ni_bc = n_i.copy()
    _apply_bc(pe_bc, core_pe, edge_pe, dx)
    _apply_bc(ne_bc, core_ne, edge_ne, dx)
    _apply_bc(pi_bc, core_pi, edge_pi, dx)
    _apply_bc(ni_bc, core_ni, edge_ni, dx)

    x_face   = 0.5 * (x[1:] + x[:-1])
    N        = len(pe_bc)
    params_e = _get_species_params(params, "e")
    params_i = _get_species_params(params, "i")

    # --- Electron heat flux (log T_e gradient: κ_Te = -d ln T_e / dx) ---
    T_e_bc  = pe_bc / np.maximum(ne_bc, 1e-10)
    Q_e     = heat_flux_function(_log_kap(T_e_bc, dx), x_face, params_e)
    div_Qe  = _flux_div(Q_e, core_pe, edge_pe, dx, N)

    # --- Electron particle flux (log n_e gradient: κ_ne = -d ln n_e / dx) ---
    V_ne    = params_e.get("V_n", 0.0)
    Gamma_e = particle_flux_function(_log_kap(ne_bc, dx), x_face, params_e) + V_ne * ne_bc[1:]
    div_Ge  = _flux_div(Gamma_e, core_ne, edge_ne, dx, N)

    # --- Ion heat flux (log T_i gradient: κ_Ti = -d ln T_i / dx) ---
    T_i_bc  = pi_bc / np.maximum(ni_bc, 1e-10)
    Q_i     = heat_flux_function(_log_kap(T_i_bc, dx), x_face, params_i)
    div_Qi  = _flux_div(Q_i, core_pi, edge_pi, dx, N)

    # --- Ion particle flux (log n_i gradient: κ_ni = -d ln n_i / dx) ---
    V_ni    = params_i.get("V_n", 0.0)
    Gamma_i = particle_flux_function(_log_kap(ni_bc, dx), x_face, params_i) + V_ni * ni_bc[1:]
    div_Gi  = _flux_div(Gamma_i, core_ni, edge_ni, dx, N)

    # --- Sources ---
    S_pe = source_pe_fn(x, pe_bc, ne_bc, pi_bc, ni_bc)
    S_ne = source_ne_fn(x, pe_bc, ne_bc, pi_bc, ni_bc)
    S_pi = source_pi_fn(x, pe_bc, ne_bc, pi_bc, ni_bc)
    S_ni = source_ni_fn(x, pe_bc, ne_bc, pi_bc, ni_bc)

    # --- Power balance enforcement ---
    L           = x[-1]
    heating_mode = params.get("heating_mode", "global")
    edge_sigma  = params.get("edge_sigma", 0.05)

    def enforce_pb(S, flux_edge, pb_ratio):
        if pb_ratio is None or pb_ratio == 0:
            return S
        total_S = np.trapezoid(S, x)
        if heating_mode == "localized":
            g_test = np.exp(-((x - L)**2) / (2 * edge_sigma**2))
            g_norm = np.trapezoid(g_test, x)
            deficit = pb_ratio * flux_edge - total_S
            return S + (deficit / g_norm if g_norm > 0 else 0.0) * g_test
        else:
            if total_S > 0 and flux_edge > 0:
                return S * (pb_ratio * flux_edge / total_S)
        return S

    pb_pe = params.get("power_balance_pe", params.get("power_balance",   1.0))
    pb_ne = params.get("power_balance_ne", params.get("power_balance_n", 1.0))
    pb_pi = params.get("power_balance_pi", 1.0)
    pb_ni = params.get("power_balance_ni", 1.0)

    S_pe = enforce_pb(S_pe, Q_e[-1],     pb_pe)
    S_ne = enforce_pb(S_ne, Gamma_e[-1], pb_ne)
    S_pi = enforce_pb(S_pi, Q_i[-1],     pb_pi)
    S_ni = enforce_pb(S_ni, Gamma_i[-1], pb_ni)

    # --- Physics sources (added on top of aux after power balance) ---
    # Bremsstrahlung: radiation loss from electrons (T3D Physics.py L169)
    P_brem = bremsstrahlung(ne_bc, pe_bc, params)
    S_pe   = S_pe - P_brem

    # Alpha heating: D-T fusion power split via Stix formula (T3D Physics.py L416-432)
    P_alpha_e, P_alpha_i = alpha_heating(ne_bc, pe_bc, ni_bc, pi_bc, params)
    S_pe = S_pe + P_alpha_e
    S_pi = S_pi + P_alpha_i

    # --- Collisional energy exchange ---
    Q_ei = collision_exchange(ne_bc, pe_bc, ni_bc, pi_bc, params)

    # --- Hyperviscosity ---
    nu4  = params.get("nu4", 0.0)
    d4pe = np.zeros(N)
    d4pi = np.zeros(N)
    if nu4 > 0 and N > 4:
        d4pe[2:-2] = (pe_bc[:-4] - 4*pe_bc[1:-3] + 6*pe_bc[2:-2] - 4*pe_bc[3:-1] + pe_bc[4:]) / dx**4
        d4pi[2:-2] = (pi_bc[:-4] - 4*pi_bc[1:-3] + 6*pi_bc[2:-2] - 4*pi_bc[3:-1] + pi_bc[4:]) / dx**4

    rhs_pe = -div_Qe - nu4*d4pe + S_pe + Q_ei
    rhs_ne = -div_Ge                   + S_ne
    rhs_pi = -div_Qi - nu4*d4pi + S_pi - Q_ei
    rhs_ni = -div_Gi                   + S_ni

    return rhs_pe, rhs_ne, rhs_pi, rhs_ni


# ==========================================================
# TIME STEPPING
# ==========================================================

def _euler_step(p_e, n_e, p_i, n_i, dt, dx, x,
                p_e_ped, n_e_ped, p_i_ped, n_i_ped,
                source_pe_fn, source_ne_fn, source_pi_fn, source_ni_fn, params):
    r = compute_rhs_two_species(
        p_e, n_e, p_i, n_i, dx, x,
        p_e_ped, n_e_ped, p_i_ped, n_i_ped,
        source_pe_fn, source_ne_fn, source_pi_fn, source_ni_fn, params)
    return (p_e + dt*r[0], n_e + dt*r[1],
            p_i + dt*r[2], n_i + dt*r[3])


def _rk4_step(p_e, n_e, p_i, n_i, dt, dx, x,
              p_e_ped, n_e_ped, p_i_ped, n_i_ped,
              source_pe_fn, source_ne_fn, source_pi_fn, source_ni_fn, params):

    def rhs(pe, ne, pi_, ni):
        return compute_rhs_two_species(
            pe, ne, pi_, ni, dx, x,
            p_e_ped, n_e_ped, p_i_ped, n_i_ped,
            source_pe_fn, source_ne_fn, source_pi_fn, source_ni_fn, params)

    k1 = rhs(p_e, n_e, p_i, n_i)
    k2 = rhs(p_e + 0.5*dt*k1[0], n_e + 0.5*dt*k1[1],
             p_i + 0.5*dt*k1[2], n_i + 0.5*dt*k1[3])
    k3 = rhs(p_e + 0.5*dt*k2[0], n_e + 0.5*dt*k2[1],
             p_i + 0.5*dt*k2[2], n_i + 0.5*dt*k2[3])
    k4 = rhs(p_e + dt*k3[0], n_e + dt*k3[1],
             p_i + dt*k3[2], n_i + dt*k3[3])

    return (p_e + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6,
            n_e + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6,
            p_i + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6,
            n_i + dt*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6)


def step_two_species(p_e, n_e, p_i, n_i, dt, dx, x,
                     p_e_ped, n_e_ped, p_i_ped, n_i_ped,
                     source_pe_fn, source_ne_fn, source_pi_fn, source_ni_fn, params):
    if TIME_SCHEME == "Euler":
        pe_new, ne_new, pi_new, ni_new = _euler_step(
            p_e, n_e, p_i, n_i, dt, dx, x,
            p_e_ped, n_e_ped, p_i_ped, n_i_ped,
            source_pe_fn, source_ne_fn, source_pi_fn, source_ni_fn, params)
    elif TIME_SCHEME == "RK4":
        pe_new, ne_new, pi_new, ni_new = _rk4_step(
            p_e, n_e, p_i, n_i, dt, dx, x,
            p_e_ped, n_e_ped, p_i_ped, n_i_ped,
            source_pe_fn, source_ne_fn, source_pi_fn, source_ni_fn, params)
    else:
        raise ValueError(f"Unknown TIME_SCHEME '{TIME_SCHEME}'")

    # Re-enforce boundary conditions
    bcs = _get_bcs_two_species(params, p_e_ped, n_e_ped, p_i_ped, n_i_ped)
    core_pe, edge_pe, core_ne, edge_ne, core_pi, edge_pi, core_ni, edge_ni = bcs

    for prof, core, edge, ped in [
        (pe_new, core_pe, edge_pe, p_e_ped),
        (ne_new, core_ne, edge_ne, n_e_ped),
        (pi_new, core_pi, edge_pi, p_i_ped),
        (ni_new, core_ni, edge_ni, n_i_ped),
    ]:
        if core["type"] == "neumann" and core["value"] == 0.0:
            prof[0] = prof[1]
        if edge["type"] == "dirichlet":
            prof[-1] = ped

    return pe_new, ne_new, pi_new, ni_new


# ==========================================================
# SOLVER LOOP
# ==========================================================

def solving_loop_two_species(
        p_e_init, n_e_init, p_i_init, n_i_init,
        dt, dx, Trun, L,
        p_e_ped, n_e_ped, p_i_ped, n_i_ped,
        source_pe_fn, source_ne_fn, source_pi_fn, source_ni_fn,
        num_snapshots, params):
    """
    Evolve the two-species system from t=0 to t=Trun.

    Returns (saved_pe, saved_ne, saved_pi, saved_ni, pi_diag).
    pi_diag is a dict with PI controller diagnostics (empty if mode != "PI").
    """
    N         = len(p_e_init)
    x         = np.linspace(0, L, N)
    num_steps = int(Trun / dt)
    save_idx  = np.linspace(0, num_steps, num_snapshots + 1, dtype=int)

    p_e = p_e_init.copy(); n_e = n_e_init.copy()
    p_i = p_i_init.copy(); n_i = n_i_init.copy()

    saved_pe, saved_ne, saved_pi, saved_ni = [], [], [], []
    save_counter = 0

    # --- PI controller setup ---
    pb_mode = params.get("power_balance_mode", "instantaneous")
    use_pi  = (pb_mode == "PI")

    # Channel definitions: (name, initial_field, pb_key, Kp_key, Ki_key)
    ch_defs = [
        ("pe", p_e_init, "power_balance_pe", "Kp_pe", "Ki_pe"),
        ("ne", n_e_init, "power_balance_ne", "Kp_ne", "Ki_ne"),
        ("pi", p_i_init, "power_balance_pi", "Kp_pi", "Ki_pi"),
        ("ni", n_i_init, "power_balance_ni", "Kp_ni", "Ki_ni"),
    ]
    pi_scale   = {}   # current source multiplier per channel
    pi_int_err = {}   # integral of error per channel
    pi_target  = {}   # target stored quantity (= initial integral)
    pi_active  = {}   # whether PI is active for this channel

    for name, f_init, key_pb, _, _ in ch_defs:
        pb = params.get(key_pb, None)
        pi_active[name] = (use_pi and pb is not None)
        if pi_active[name]:
            pi_target[name]  = np.trapezoid(f_init, x)
            pi_scale[name]   = 1.0
            pi_int_err[name] = 0.0

    # Pre-balance: initialise pi_scale to the instantaneous power-balance
    # ratio so the controller starts near equilibrium rather than at 1.0
    if use_pi:
        Q_e0, Ge0, Q_i0, Gi0, _ = compute_fluxes_two_species(
            p_e, n_e, p_i, n_i, dx, x,
            p_e_ped, n_e_ped, p_i_ped, n_i_ped, params)
        init_src  = {
            "pe": source_pe_fn(x, p_e, n_e, p_i, n_i),
            "ne": source_ne_fn(x, p_e, n_e, p_i, n_i),
            "pi": source_pi_fn(x, p_e, n_e, p_i, n_i),
            "ni": source_ni_fn(x, p_e, n_e, p_i, n_i),
        }
        init_flux = {"pe": Q_e0[-1], "ne": Ge0[-1], "pi": Q_i0[-1], "ni": Gi0[-1]}
        pb_ratios = {name: params.get(key_pb, 1.0)
                     for name, _, key_pb, _, _ in ch_defs}
        for name in ["pe", "ne", "pi", "ni"]:
            if not pi_active.get(name):
                continue
            total_S0 = np.trapezoid(init_src[name], x)
            flux0    = init_flux[name]
            if total_S0 > 0 and flux0 > 0:
                pi_scale[name] = pb_ratios[name] * flux0 / total_S0

    # Diagnostics storage (recorded at save points)
    pi_diag = {f"scale_{n}": [] for n in ["pe","ne","pi","ni"]}
    pi_diag.update({f"W_{n}": [] for n in ["pe","ne","pi","ni"]})
    if not use_pi:
        pi_diag = {}

    # When using PI, disable instantaneous power balance in the RHS
    if use_pi:
        params = dict(params)  # shallow copy to avoid mutating original
        for key_pb in ["power_balance_pe","power_balance_ne",
                       "power_balance_pi","power_balance_ni"]:
            params[key_pb] = None  # disable instantaneous enforcement

    for i in range(num_steps + 1):
        if i == save_idx[save_counter]:
            saved_pe.append(p_e.copy()); saved_ne.append(n_e.copy())
            saved_pi.append(p_i.copy()); saved_ni.append(n_i.copy())
            # Record PI diagnostics at save points
            if use_pi:
                fields = {"pe": p_e, "ne": n_e, "pi": p_i, "ni": n_i}
                for name in ["pe","ne","pi","ni"]:
                    pi_diag[f"scale_{name}"].append(pi_scale.get(name, 1.0))
                    pi_diag[f"W_{name}"].append(np.trapezoid(fields[name], x))
            save_counter = min(save_counter + 1, len(save_idx) - 1)

        # --- PI controller: update source multipliers ---
        if use_pi:
            fields = {"pe": p_e, "ne": n_e, "pi": p_i, "ni": n_i}
            for name, _, _, key_Kp, key_Ki in ch_defs:
                if not pi_active[name]:
                    continue
                W_now = np.trapezoid(fields[name], x)
                error = pi_target[name] - W_now
                Kp = params.get(key_Kp, 10.0)
                Ki = params.get(key_Ki, 50.0)
                pi_int_err[name] += error * dt
                pi_scale[name] = 1.0 + Kp * error + Ki * pi_int_err[name]
                pi_scale[name] = max(pi_scale[name], 0.0)  # clamp non-negative

        # --- Build scaled source functions ---
        if use_pi:
            s_pe = pi_scale.get("pe", 1.0)
            s_ne = pi_scale.get("ne", 1.0)
            s_pi = pi_scale.get("pi", 1.0)
            s_ni = pi_scale.get("ni", 1.0)
            def src_pe_s(x, pe, ne, pi_, ni, _s=s_pe): return _s * source_pe_fn(x, pe, ne, pi_, ni)
            def src_ne_s(x, pe, ne, pi_, ni, _s=s_ne): return _s * source_ne_fn(x, pe, ne, pi_, ni)
            def src_pi_s(x, pe, ne, pi_, ni, _s=s_pi): return _s * source_pi_fn(x, pe, ne, pi_, ni)
            def src_ni_s(x, pe, ne, pi_, ni, _s=s_ni): return _s * source_ni_fn(x, pe, ne, pi_, ni)
            src_fns = (src_pe_s, src_ne_s, src_pi_s, src_ni_s)
        else:
            src_fns = (source_pe_fn, source_ne_fn, source_pi_fn, source_ni_fn)

        p_e, n_e, p_i, n_i = step_two_species(
            p_e, n_e, p_i, n_i, dt, dx, x,
            p_e_ped, n_e_ped, p_i_ped, n_i_ped,
            *src_fns, params)

        if i % 20000 == 0:
            T_e = p_e / np.maximum(n_e, 1e-10)
            T_i = p_i / np.maximum(n_i, 1e-10)
            kap_Te = np.max(-np.gradient(np.log(np.maximum(T_e, 1e-10)), dx))
            kap_Ti = np.max(-np.gradient(np.log(np.maximum(T_i, 1e-10)), dx))
            si = ""
            if use_pi and pi_active.get("pe"):
                si = f" | S_pe x{pi_scale['pe']:.3f}"
            print(f"Step {i:6d} | max kap_Te = {kap_Te:.4f} | max kap_Ti = {kap_Ti:.4f}{si}")

    return (np.array(saved_pe), np.array(saved_ne),
            np.array(saved_pi), np.array(saved_ni), pi_diag)
