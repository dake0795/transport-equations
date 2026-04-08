import numpy as np
import scipy.linalg


# ==========================================================
# USER OPTIONS
# ==========================================================

TIME_SCHEME = "RK4"   # "Euler", "RK4", "CN"
SPACE_ORDER = 2        # 2 or 4


# ==========================================================
# FLUX MODEL
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
    windows = [1.0 - S[0]]
    for i in range(len(S) - 1):
        windows.append(S[i] * (1.0 - S[i + 1]))
    windows.append(S[-1])
    return windows


def flux_function(g, x, params):

    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    g = np.maximum(g, 0.0)

    chi0     = params["chi0"]
    chi_core = params["chi_core"]
    chi_RR   = params["chi_RR"]
    g_c      = params["g_c"]
    g_stiff  = params["g_stiff"]
    n_stiff  = params["n_stiff"]

    boundaries  = params.get("boundaries",  [])
    deltas      = params.get("deltas",      [0.01] * len(boundaries))
    flux_models = params.get("flux_models", ["core"])

    # Transport laws
    ratio  = np.clip(g / g_stiff, 0.0, 20.0)
    Q_core = chi_core * g * (1.0 + ratio**n_stiff)
    Q_nl   = np.maximum(chi0 * g * np.maximum(1.0 - g / g_c, 0), 0.0)
    Q_linear = chi_core * g

    Q_map = {"core": Q_core, "nl": Q_nl, "linear": Q_linear}

    W = make_windows(x, boundaries, deltas)
    if len(W) != len(flux_models):
        raise ValueError(
            f"Number of flux_models must equal number of regions ({len(W)} expected)"
        )

    Q_transport = np.zeros_like(g)
    for Wi, model_name in zip(W, flux_models):
        if model_name not in Q_map:
            raise ValueError(f"Unknown flux model '{model_name}'")
        Q_transport += Wi * Q_map[model_name]

    Q_total = Q_transport + chi_RR * g

    # Optional MHD stiff cliff: prevents unphysical blowup beyond g_MHD.
    # Set chi_MHD = 0 (default) to disable.
    g_MHD   = params.get("g_MHD",   None)
    chi_MHD = params.get("chi_MHD", 0.0)
    if g_MHD is not None and chi_MHD > 0.0:
        Q_total = Q_total + chi_MHD * np.maximum(g - g_MHD, 0.0)

    return Q_total


# ==========================================================
# BOUNDARY CONDITIONS
# ==========================================================

def _get_bcs(params, p_ped):
    bc   = params.get("bc", {})
    core = bc.get("core", {"type": "neumann",   "value": 0.0})
    edge = bc.get("edge", {"type": "dirichlet", "value": p_ped})
    return core, edge


def _apply_bc_to_profile(p_bc, dx, core_bc, edge_bc):
    if core_bc["type"] == "dirichlet":
        p_bc[0] = core_bc["value"]
    elif core_bc["type"] == "neumann":
        p_bc[0] = p_bc[1] - core_bc["value"] * dx

    if edge_bc["type"] == "dirichlet":
        p_bc[-1] = edge_bc["value"]
    elif edge_bc["type"] == "neumann":
        p_bc[-1] = p_bc[-2] + edge_bc["value"] * dx


# ==========================================================
# RHS
# ==========================================================

def compute_rhs(p, dx, x, p_ped, source_function, params):

    core_bc, edge_bc = _get_bcs(params, p_ped)
    p_bc = p.copy()
    _apply_bc_to_profile(p_bc, dx, core_bc, edge_bc)

    g_face = -(p_bc[1:] - p_bc[:-1]) / dx
    x_face = 0.5 * (x[1:] + x[:-1])
    Q_face = flux_function(g_face, x_face, params)

    dQdx = np.zeros_like(p_bc)
    dQdx[1:-1] = (Q_face[1:] - Q_face[:-1]) / dx

    if core_bc["type"] == "flux":
        dQdx[0] = (Q_face[0] - core_bc["value"]) / dx
    else:
        dQdx[0] = Q_face[0] / dx

    if edge_bc["type"] == "flux":
        dQdx[-1] = (edge_bc["value"] - Q_face[-1]) / dx
    else:
        dQdx[-1] = -Q_face[-1] / dx

    S = source_function(x, p_bc)

    # Power balance: rescale S so ∫S dx = power_balance × Q_edge
    power_balance = params.get("power_balance", None)
    heating_mode  = params.get("heating_mode", "global")

    if heating_mode == "global" and power_balance is not None:
        Q_edge   = Q_face[-1]
        total_S  = np.trapezoid(S, x)
        if total_S > 0:
            S = S * (power_balance * Q_edge / total_S)

    elif heating_mode == "localized" and power_balance is not None:
        Q_edge  = Q_face[-1]
        total_S = np.trapezoid(S, x)
        deficit = power_balance * Q_edge - total_S
        L = x[-1]
        edge_sigma  = params.get("edge_sigma", 0.05)
        gauss       = np.exp(-((x - L)**2) / (2 * edge_sigma**2))
        gauss_norm  = np.trapezoid(gauss, x)
        edge_amp    = deficit / gauss_norm if gauss_norm > 0 else 0.0
        S = S + edge_amp * gauss

    nu4 = params.get("nu4", 0.0)
    d4p = np.zeros_like(p_bc)
    if nu4 > 0 and len(p_bc) > 4:
        d4p[2:-2] = (
            p_bc[0:-4] - 4*p_bc[1:-3] + 6*p_bc[2:-2]
            - 4*p_bc[3:-1] + p_bc[4:]
        ) / dx**4

    return -dQdx - nu4*d4p + S


# ==========================================================
# TIME STEPPING
# ==========================================================

def euler_step(p, dt, dx, x, p_ped, source_function, params):
    return p + dt * compute_rhs(p, dx, x, p_ped, source_function, params)


def rk4_step(p, dt, dx, x, p_ped, source_function, params):
    k1 = compute_rhs(p,           dx, x, p_ped, source_function, params)
    k2 = compute_rhs(p+0.5*dt*k1, dx, x, p_ped, source_function, params)
    k3 = compute_rhs(p+0.5*dt*k2, dx, x, p_ped, source_function, params)
    k4 = compute_rhs(p+    dt*k3, dx, x, p_ped, source_function, params)
    return p + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def imex_step(p_old, dt, dx, x, p_ped, source_function, params):

    N      = len(p_old)
    chi_RR = params["chi_RR"]
    nu4    = params.get("nu4", 0.0)

    p_bc = p_old.copy()
    p_bc[0]  = p_bc[1]
    p_bc[-1] = p_ped

    g_face = -(p_bc[1:] - p_bc[:-1]) / dx
    x_face = 0.5 * (x[1:] + x[:-1])
    Q_full = flux_function(g_face, x_face, params)
    Q_nl   = Q_full - chi_RR * g_face

    dQ_nl = np.zeros(N)
    dQ_nl[1:-1] = (Q_nl[1:] - Q_nl[:-1]) / dx
    dQ_nl[0]    =  Q_nl[0]  / dx
    dQ_nl[-1]   = -Q_nl[-1] / dx

    S = source_function(x, p_bc)
    power_balance = params.get("power_balance", None)
    if power_balance is not None:
        Q_edge  = Q_full[-1]
        total_S = np.trapezoid(S, x)
        if total_S > 0:
            S = S * (power_balance * Q_edge / total_S)

    rhs_explicit = -dQ_nl + S

    ab = np.zeros((5, N))
    b  = p_old + dt * rhs_explicit

    alpha = dt * chi_RR / dx**2
    beta  = dt * nu4    / dx**4 if nu4 > 0.0 else 0.0

    inn = np.arange(1, N - 1)
    ab[3, inn - 1] -= alpha
    ab[2, inn    ] += 1.0 + 2.0 * alpha
    ab[1, inn + 1] -= alpha

    if nu4 > 0.0:
        inn4 = np.arange(2, N - 2)
        ab[4, inn4 - 2] +=       beta
        ab[3, inn4 - 1] -= 4.0 * beta
        ab[2, inn4     ] += 6.0 * beta
        ab[1, inn4 + 1] -= 4.0 * beta
        ab[0, inn4 + 2] +=       beta

    ab[2, 0] =  1.0;  ab[1, 1] = -1.0;  b[0]     = 0.0
    ab[2, N - 1] = 1.0;                  b[N - 1] = p_ped

    return scipy.linalg.solve_banded((2, 2), ab, b)


def step(p, dt, dx, x, p_ped, source_function, params):
    if TIME_SCHEME == "Euler":
        p_new = euler_step(p, dt, dx, x, p_ped, source_function, params)
    elif TIME_SCHEME == "RK4":
        p_new = rk4_step(p, dt, dx, x, p_ped, source_function, params)
    elif TIME_SCHEME == "CN":
        p_new = imex_step(p, dt, dx, x, p_ped, source_function, params)
    else:
        raise ValueError("Invalid TIME_SCHEME")

    p_new[0]  = p_new[1]   # Neumann at core
    p_new[-1] = p_ped       # Dirichlet at edge
    return p_new


# ==========================================================
# SOLVER LOOP
# ==========================================================

def solving_loop(p_init, dt, dx, T, L, p_ped, source_function, num_snapshots, params):

    N    = len(p_init)
    x    = np.linspace(0, L, N)
    num_steps   = int(T / dt)
    save_indices = np.linspace(0, num_steps, num_snapshots + 1, dtype=int)

    p            = p_init.copy()
    saved        = []
    save_counter = 0

    for i in range(num_steps + 1):
        if i == save_indices[save_counter]:
            saved.append(p.copy())
            save_counter = min(save_counter + 1, len(save_indices) - 1)

        p = step(p, dt, dx, x, p_ped, source_function, params)

        if i % 20000 == 0:
            g = -(p[1:] - p[:-1]) / dx
            print(f"Step {i:6d} | max|g| = {np.max(np.abs(g)):.4f}")

    return np.array(saved)
