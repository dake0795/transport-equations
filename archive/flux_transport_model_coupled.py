import numpy as np


# ==========================================================
# USER OPTIONS
# ==========================================================

TIME_SCHEME = "Euler"        # "Euler", "RK4", "CN"
SPACE_ORDER = 2              # 2 or 4
PICARD_MAXITER = 100
PICARD_TOL = 1e-6


# ==========================================================
# FLUX MODELS (density and pressure)
# ==========================================================

def smooth_step(x, x0, delta):
    return 0.5 * (1.0 + np.tanh((x - x0) / delta))


def make_windows(x, boundaries, deltas):
    """
    boundaries : sorted list [x1, x2, ..., xN]
    deltas     : same length as boundaries

    Returns N+1 smooth window functions that sum to 1.
    """
    if len(boundaries) == 0:
        return [np.ones_like(x)]

    if len(boundaries) != len(deltas):
        raise ValueError("boundaries and deltas must have same length")

    S = [smooth_step(x, b, d) for b, d in zip(boundaries, deltas)]
    windows = []
    windows.append(1.0 - S[0])
    for i in range(len(S) - 1):
        windows.append(S[i] * (1.0 - S[i + 1]))
    windows.append(S[-1])
    return windows


def particle_flux_function(g_n, x, params):
    """
    Particle flux: Γ = -D_n dn/dx with same NL model structure as heat flux

    Uses same flux models as pressure but applied to density:
    - NL: Γ_nl = χ₀_n g_n (1 - g_n/g_c_n)
    - Core: Γ_core = χ_n,core g_n (1 + (g_n/g_stiff_n)^n_stiff_n)
    - Linear: Γ_linear = χ_n,core g_n
    - Plus background diffusion χ_RR_n everywhere
    """
    g_n = np.nan_to_num(g_n, nan=0.0, posinf=0.0, neginf=0.0)
    g_n = np.maximum(g_n, 0.0)

    # Density transport parameters
    chi0_n = params.get("chi0_n", 0.1)
    chi_core_n = params.get("chi_core_n", 0.1)
    chi_RR_n = params.get("chi_RR_n", 0.01)

    g_c_n = params.get("g_c_n", 2.0)
    g_stiff_n = params.get("g_stiff_n", 1.5)
    n_stiff_n = params.get("n_stiff_n", 2)

    boundaries = params.get("boundaries", [])
    deltas = params.get("deltas", [0.01] * len(boundaries))
    flux_models_n = params.get("flux_models_n", params.get("flux_models", ["nl"]))

    # --- Core stiff model ---
    ratio = np.clip(g_n / g_stiff_n, 0.0, 20.0)
    Gamma_core = chi_core_n * g_n * (1.0 + ratio**n_stiff_n)

    # --- Nonlinear model ---
    Gamma_nl = chi0_n * g_n * np.maximum(1.0 - g_n / g_c_n, 0)
    Gamma_nl = np.maximum(Gamma_nl, 0.0)

    # --- Linear fallback model ---
    Gamma_linear = chi_core_n * g_n

    Gamma_map = {
        "core": Gamma_core,
        "nl": Gamma_nl,
        "linear": Gamma_linear,
    }

    W = make_windows(x, boundaries, deltas)

    if len(W) != len(flux_models_n):
        raise ValueError(
            "Number of flux_models_n must equal number of regions "
            f"({len(W)} regions expected)"
        )

    Gamma_transport = np.zeros_like(g_n)
    for Wi, model_name in zip(W, flux_models_n):
        if model_name not in Gamma_map:
            raise ValueError(f"Unknown flux model '{model_name}'")
        Gamma_transport += Wi * Gamma_map[model_name]

    # Plus background diffusion
    Gamma_total = Gamma_transport + chi_RR_n * g_n

    return Gamma_total


def heat_flux_function(g_p, x, params):
    """
    Heat flux: Q = -χ * dp/dx (similar to pressure solver)
    """
    chi0     = params["chi0"]
    chi_core = params["chi_core"]
    chi_RR   = params["chi_RR"]

    g_p = np.nan_to_num(g_p, nan=0.0, posinf=0.0, neginf=0.0)
    g_p = np.maximum(g_p, 0.0)

    g_c      = params["g_c"]
    g_stiff  = params["g_stiff"]
    n_stiff  = params["n_stiff"]

    boundaries = params.get("boundaries", [])
    deltas     = params.get("deltas", [0.01] * len(boundaries))
    flux_models = params.get("flux_models", ["core"])

    # --- Core stiff model ---
    ratio = np.clip(g_p / g_stiff, 0.0, 20.0)
    Q_core = chi_core * g_p * (1.0 + ratio**n_stiff)

    # --- Nonlinear model ---
    Q_nl = chi0 * g_p * np.maximum(1.0 - g_p / g_c, 0)
    Q_nl = np.maximum(Q_nl, 0.0)

    # --- Linear fallback model ---
    Q_linear = chi_core * g_p

    Q_map = {
        "core": Q_core,
        "nl": Q_nl,
        "linear": Q_linear,
    }

    W = make_windows(x, boundaries, deltas)

    if len(W) != len(flux_models):
        raise ValueError(
            "Number of flux_models must equal number of regions "
            f"({len(W)} regions expected)"
        )

    Q_transport = np.zeros_like(g_p)
    for Wi, model_name in zip(W, flux_models):
        if model_name not in Q_map:
            raise ValueError(f"Unknown flux model '{model_name}'")
        Q_transport += Wi * Q_map[model_name]

    Q_total = Q_transport + chi_RR * g_p

    return Q_total


# ==========================================================
# BOUNDARY CONDITIONS
# ==========================================================

def _get_bcs(params, p_ped, n_ped):
    bc = params.get("bc", {})
    core_p = bc.get("core_p", {"type": "neumann", "value": 0.0})
    edge_p = bc.get("edge_p", {"type": "dirichlet", "value": p_ped})
    core_n = bc.get("core_n", {"type": "neumann", "value": 0.0})
    edge_n = bc.get("edge_n", {"type": "dirichlet", "value": n_ped})
    return core_p, edge_p, core_n, edge_n


def _apply_bc_to_profile(p_bc, n_bc, dx, core_p, edge_p, core_n, edge_n):
    """Apply boundary conditions to pressure and density profiles."""
    # Pressure BCs
    ctype_p = core_p["type"]
    if ctype_p == "dirichlet":
        p_bc[0] = core_p["value"]
    elif ctype_p == "neumann":
        p_bc[0] = p_bc[1] - core_p["value"] * dx

    etype_p = edge_p["type"]
    if etype_p == "dirichlet":
        p_bc[-1] = edge_p["value"]
    elif etype_p == "neumann":
        p_bc[-1] = p_bc[-2] + edge_p["value"] * dx

    # Density BCs
    ctype_n = core_n["type"]
    if ctype_n == "dirichlet":
        n_bc[0] = core_n["value"]
    elif ctype_n == "neumann":
        n_bc[0] = n_bc[1] - core_n["value"] * dx

    etype_n = edge_n["type"]
    if etype_n == "dirichlet":
        n_bc[-1] = edge_n["value"]
    elif etype_n == "neumann":
        n_bc[-1] = n_bc[-2] + edge_n["value"] * dx


# ==========================================================
# RHS FOR COUPLED SYSTEM
# ==========================================================

def compute_rhs_coupled(p, n, dx, x, p_ped, n_ped, source_p_function, source_n_function, params):
    """
    Compute RHS for coupled density and pressure evolution with dual power balance.

    dp/dt = -∇·Q + S_p
    dn/dt = -∇·Γ + S_n
    """
    core_p, edge_p, core_n, edge_n = _get_bcs(params, p_ped, n_ped)

    p_bc = p.copy()
    n_bc = n.copy()
    _apply_bc_to_profile(p_bc, n_bc, dx, core_p, edge_p, core_n, edge_n)

    # ----- Pressure: compute heat flux -----
    g_p = -(p_bc[1:] - p_bc[:-1]) / dx
    x_face = 0.5 * (x[1:] + x[:-1])
    Q_face = heat_flux_function(g_p, x_face, params)

    # Divergence of heat flux
    dQdx = np.zeros_like(p_bc)
    dQdx[1:-1] = (Q_face[1:] - Q_face[:-1]) / dx

    if core_p["type"] == "flux":
        dQdx[0] = (Q_face[0] - core_p["value"]) / dx
    else:
        dQdx[0] = Q_face[0] / dx

    if edge_p["type"] == "flux":
        dQdx[-1] = (edge_p["value"] - Q_face[-1]) / dx
    else:
        dQdx[-1] = -Q_face[-1] / dx

    # ----- Density: compute particle flux -----
    g_n = -(n_bc[1:] - n_bc[:-1]) / dx
    Gamma_face = particle_flux_function(g_n, x_face, params)

    V_n = params.get("V_n", 0.0)
    Gamma_face = Gamma_face + V_n * n_bc[1:]

    # Divergence of particle flux
    dGammadx = np.zeros_like(n_bc)
    dGammadx[1:-1] = (Gamma_face[1:] - Gamma_face[:-1]) / dx

    if core_n["type"] == "flux":
        dGammadx[0] = (Gamma_face[0] - core_n["value"]) / dx
    else:
        dGammadx[0] = Gamma_face[0] / dx

    if edge_n["type"] == "flux":
        dGammadx[-1] = (edge_n["value"] - Gamma_face[-1]) / dx
    else:
        dGammadx[-1] = -Gamma_face[-1] / dx

    # ----- Sources -----
    S_p = source_p_function(x, p_bc, n_bc)
    S_n = source_n_function(x, p_bc, n_bc)

    # ----- Power balance enforcement (dual) -----
    heating_mode = params.get("heating_mode", "global")
    power_balance_mode = params.get("power_balance_mode", "separate")
    power_balance_p = params.get("power_balance", 1.0)
    power_balance_n = params.get("power_balance_n", 1.0)
    L = x[-1]
    edge_sigma = params.get("edge_sigma", 0.05)

    if heating_mode == "localized":
        # Edge-localized Gaussian heating (pressure)
        Q_edge = Q_face[-1]
        total_S_p = np.trapezoid(S_p, x)
        required_integral_p = power_balance_p * Q_edge
        deficit_p = required_integral_p - total_S_p

        gaussian_test = np.exp(-((x - L)**2) / (2 * edge_sigma**2))
        gaussian_norm = np.trapezoid(gaussian_test, x)

        if gaussian_norm > 0:
            edge_amp_p = deficit_p / gaussian_norm
        else:
            edge_amp_p = 0

        gaussian_edge_p = edge_amp_p * np.exp(-((x - L)**2) / (2 * edge_sigma**2))
        S_p = S_p + gaussian_edge_p

        # Edge-localized Gaussian heating (density)
        Gamma_edge = Gamma_face[-1]
        total_S_n = np.trapezoid(S_n, x)
        required_integral_n = power_balance_n * Gamma_edge
        deficit_n = required_integral_n - total_S_n

        if gaussian_norm > 0:
            edge_amp_n = deficit_n / gaussian_norm
        else:
            edge_amp_n = 0

        gaussian_edge_n = edge_amp_n * np.exp(-((x - L)**2) / (2 * edge_sigma**2))
        S_n = S_n + gaussian_edge_n

    elif heating_mode == "global":
        # Global power balance enforcement
        Q_edge = Q_face[-1]
        Gamma_edge = Gamma_face[-1]
        total_S_p = np.trapezoid(S_p, x)
        total_S_n = np.trapezoid(S_n, x)

        if power_balance_mode == "separate":
            # Scale each independently
            if total_S_p > 0 and Q_edge > 0:
                S_p = S_p * (power_balance_p * Q_edge / total_S_p)
            if total_S_n > 0 and Gamma_edge > 0:
                S_n = S_n * (power_balance_n * Gamma_edge / total_S_n)

        elif power_balance_mode == "coupled_to_p":
            # Scale density source to match pressure balance
            if total_S_p > 0 and Q_edge > 0:
                S_p = S_p * (power_balance_p * Q_edge / total_S_p)
            if total_S_n > 0 and Gamma_edge > 0:
                S_n = S_n * (power_balance_p * Q_edge / total_S_n)

        elif power_balance_mode == "coupled_to_n":
            # Scale pressure source to match density balance
            if total_S_n > 0 and Gamma_edge > 0:
                S_n = S_n * (power_balance_n * Gamma_edge / total_S_n)
            if total_S_p > 0 and Q_edge > 0:
                S_p = S_p * (power_balance_n * Gamma_edge / total_S_p)

    # Hyperviscosity (pressure only)
    nu4 = params.get("nu4", 0.0)
    d4p = np.zeros_like(p_bc)

    if nu4 > 0 and len(p_bc) > 4:
        d4p[2:-2] = (
              p_bc[0:-4]
            - 4*p_bc[1:-3]
            + 6*p_bc[2:-2]
            - 4*p_bc[3:-1]
            + p_bc[4:]
        ) / dx**4

    rhs_p = -dQdx - nu4*d4p + S_p
    rhs_n = -dGammadx + S_n

    return rhs_p, rhs_n


# ==========================================================
# TIME STEPPING
# ==========================================================

def euler_step_coupled(p, n, dt, dx, x, p_ped, n_ped, source_p_function, source_n_function, params):
    rhs_p, rhs_n = compute_rhs_coupled(p, n, dx, x, p_ped, n_ped, source_p_function, source_n_function, params)
    return p + dt*rhs_p, n + dt*rhs_n


def rk4_step_coupled(p, n, dt, dx, x, p_ped, n_ped, source_p_function, source_n_function, params):
    k1_p, k1_n = compute_rhs_coupled(p, n, dx, x, p_ped, n_ped, source_p_function, source_n_function, params)
    k2_p, k2_n = compute_rhs_coupled(p + 0.5*dt*k1_p, n + 0.5*dt*k1_n, dx, x, p_ped, n_ped, source_p_function, source_n_function, params)
    k3_p, k3_n = compute_rhs_coupled(p + 0.5*dt*k2_p, n + 0.5*dt*k2_n, dx, x, p_ped, n_ped, source_p_function, source_n_function, params)
    k4_p, k4_n = compute_rhs_coupled(p + dt*k3_p, n + dt*k3_n, dx, x, p_ped, n_ped, source_p_function, source_n_function, params)

    return p + dt*(k1_p + 2*k2_p + 2*k3_p + k4_p)/6, n + dt*(k1_n + 2*k2_n + 2*k3_n + k4_n)/6


# ==========================================================
# DISPATCH
# ==========================================================

def step_coupled(p, n, dt, dx, x, p_ped, n_ped, source_p_function, source_n_function, params):
    if TIME_SCHEME == "Euler":
        p_new, n_new = euler_step_coupled(p, n, dt, dx, x, p_ped, n_ped, source_p_function, source_n_function, params)
    elif TIME_SCHEME == "RK4":
        p_new, n_new = rk4_step_coupled(p, n, dt, dx, x, p_ped, n_ped, source_p_function, source_n_function, params)
    else:
        raise ValueError("Invalid TIME_SCHEME")

    # Re-enforce boundary conditions
    core_p, edge_p, core_n, edge_n = _get_bcs(params, p_ped, n_ped)

    if core_p["type"] == "neumann" and core_p["value"] == 0.0:
        p_new[0] = p_new[1]
    if edge_p["type"] == "dirichlet":
        p_new[-1] = p_ped

    if core_n["type"] == "neumann" and core_n["value"] == 0.0:
        n_new[0] = n_new[1]
    if edge_n["type"] == "dirichlet":
        n_new[-1] = n_ped

    return p_new, n_new


# ==========================================================
# SOLVER LOOP
# ==========================================================

def solving_loop_coupled(
    p_init,
    n_init,
    dt,
    dx,
    T,
    L,
    p_ped,
    n_ped,
    source_p_function,
    source_n_function,
    num_snapshots,
    params,
):
    N = len(p_init)
    x = np.linspace(0, L, N)

    num_steps = int(T/dt)
    save_indices = np.linspace(0, num_steps, num_snapshots+1, dtype=int)

    p = p_init.copy()
    n = n_init.copy()
    saved_p = []
    saved_n = []
    save_counter = 0

    for i in range(num_steps+1):

        if i == save_indices[save_counter]:
            saved_p.append(p.copy())
            saved_n.append(n.copy())
            save_counter = min(save_counter+1, len(save_indices)-1)

        p, n = step_coupled(p, n, dt, dx, x, p_ped, n_ped, source_p_function, source_n_function, params)

        # Diagnostic every 20000 steps
        if i % 20000 == 0:
            g_p = -(p[1:] - p[:-1]) / dx
            g_n = -(n[1:] - n[:-1]) / dx
            print(f"Step {i:6d} | max|g_p| = {np.max(np.abs(g_p)):.4f} | max|g_n| = {np.max(np.abs(g_n)):.4f}")

    return np.array(saved_p), np.array(saved_n)
