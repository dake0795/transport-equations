import numpy as np


# ==========================================================
# USER OPTIONS
# ==========================================================

TIME_SCHEME = "Euler"        # "Euler", "RK4", "CN"
SPACE_ORDER = 2              # 2 or 4
PICARD_MAXITER = 100
PICARD_TOL = 1e-6


# ==========================================================
# FLUX MODEL
# ==========================================================

import numpy as np


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

    # Region 0 : x < x1
    windows.append(1.0 - S[0])

    # Middle regions
    for i in range(len(S) - 1):
        windows.append(S[i] * (1.0 - S[i + 1]))

    # Final region : x > xN
    windows.append(S[-1])

    return windows


def flux_function(g, x, params):

    # --------------------------------------------------
    # Enforce positivity and avoid catastrophic overflow
    # --------------------------------------------------
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    g = np.maximum(g, 0.0)

    # --------------------------------------------------
    # Read parameters
    # --------------------------------------------------
    chi0     = params["chi0"]
    chi_core = params["chi_core"]
    chi_RR   = params["chi_RR"]

    g_c      = params["g_c"]
    g_stiff  = params["g_stiff"]
    n_stiff  = params["n_stiff"]

    boundaries = params.get("boundaries", [])
    deltas     = params.get("deltas", [0.01] * len(boundaries))
    flux_models = params.get("flux_models", ["core"])

    # --------------------------------------------------
    # Define transport laws
    # --------------------------------------------------

    # --- Core stiff model ---
    ratio = np.clip(g / g_stiff, 0.0, 20.0)   # protects against overflow
    Q_core = chi_core * g * (1.0 + ratio**n_stiff)

    # --- Nonlinear model ---
    Q_nl = chi0 * g * np.maximum(1.0 - g / g_c, 0)
    Q_nl = np.maximum(Q_nl, 0.0)   # enforce positivity safely

    # --- Linear fallback model ---
    Q_linear = chi_core * g

    # Map string names to actual flux arrays
    Q_map = {
        "core": Q_core,
        "nl": Q_nl,
        "linear": Q_linear,
    }

    # --------------------------------------------------
    # Build smooth spatial windows
    # --------------------------------------------------
    W = make_windows(x, boundaries, deltas)

    if len(W) != len(flux_models):
        raise ValueError(
            "Number of flux_models must equal number of regions "
            f"({len(W)} regions expected)"
        )

    # --------------------------------------------------
    # Blend transport laws
    # --------------------------------------------------
    Q_transport = np.zeros_like(g)

    for Wi, model_name in zip(W, flux_models):
        if model_name not in Q_map:
            raise ValueError(f"Unknown flux model '{model_name}'")
        Q_transport += Wi * Q_map[model_name]

    # --------------------------------------------------
    # Add RR diffusion everywhere
    # --------------------------------------------------
    Q_total = Q_transport + chi_RR * g

    return Q_total

# ==========================================================
# FIRST DERIVATIVE (FIXED BOUNDARIES)
# ==========================================================

def first_derivative(p, dx):

    N = len(p)
    dp = np.zeros_like(p)

    if SPACE_ORDER == 2:

        # interior
        dp[1:-1] = (p[2:] - p[:-2]) / (2*dx)

    elif SPACE_ORDER == 4:

        dp[2:-2] = (-p[4:] + 8*p[3:-1] - 8*p[1:-3] + p[:-4]) / (12*dx)

        # fallback near boundaries
        dp[1] = (p[2] - p[0])/(2*dx)
        dp[-2] = (p[-1] - p[-3])/(2*dx)

    else:
        raise ValueError("SPACE_ORDER must be 2 or 4")

    # ---- Boundary derivatives ----

    # Forward difference at core
    dp[0] = (p[1] - p[0]) / dx

    # Backward difference at edge
    dp[-1] = (p[-1] - p[-2]) / dx

    return dp


# ==========================================================
# DIVERGENCE (FIXED BOUNDARIES)
# ==========================================================

def divergence(Q, dx):

    N = len(Q)
    dQ = np.zeros_like(Q)

    if SPACE_ORDER == 2:
        dQ[1:-1] = (Q[2:] - Q[:-2]) / (2*dx)

    elif SPACE_ORDER == 4:
        dQ[2:-2] = (-Q[4:] + 8*Q[3:-1] - 8*Q[1:-3] + Q[:-4]) / (12*dx)
        dQ[1] = (Q[2] - Q[0])/(2*dx)
        dQ[-2] = (Q[-1] - Q[-3])/(2*dx)

    # Boundary one-sided
    dQ[0] = (Q[1] - Q[0]) / dx
    dQ[-1] = (Q[-1] - Q[-2]) / dx

    return dQ


# ==========================================================
# BOUNDARY CONDITIONS
# ==========================================================
#
# Specify via params["bc"]:
#
#   params["bc"] = {
#       "core": {"type": "neumann",   "value": 0.0},   # default
#       "edge": {"type": "dirichlet", "value": p_ped},  # default
#   }
#
# Supported types:
#   "dirichlet" — fix p = value
#   "neumann"   — fix dp/dx = value  (value=0: zero-gradient / zero-flux)
#   "flux"      — prescribe total flux Q at the boundary
#                 core: Q_in  entering from x=0 (positive = into domain)
#                 edge: Q_out leaving  at  x=L (positive = out of domain)

def _get_bcs(params, p_ped):
    bc   = params.get("bc", {})
    core = bc.get("core", {"type": "neumann",   "value": 0.0})
    edge = bc.get("edge", {"type": "dirichlet", "value": p_ped})
    return core, edge


def _apply_bc_to_profile(p_bc, dx, core_bc, edge_bc):
    """Set boundary cell values in p_bc to reflect the chosen BC.
    Flux BCs leave the boundary cell free (it is evolved by the RHS)."""
    ctype = core_bc["type"]
    if ctype == "dirichlet":
        p_bc[0] = core_bc["value"]
    elif ctype == "neumann":
        # dp/dx|_0 = value  →  p[0] = p[1] - value*dx
        p_bc[0] = p_bc[1] - core_bc["value"] * dx

    etype = edge_bc["type"]
    if etype == "dirichlet":
        p_bc[-1] = edge_bc["value"]
    elif etype == "neumann":
        # dp/dx|_{N-1} = value  →  p[-1] = p[-2] + value*dx
        p_bc[-1] = p_bc[-2] + edge_bc["value"] * dx


# ==========================================================
# RHS
# ==========================================================

def compute_rhs(p, dx, x, p_ped, source_function, params):

    core_bc, edge_bc = _get_bcs(params, p_ped)

    p_bc = p.copy()
    _apply_bc_to_profile(p_bc, dx, core_bc, edge_bc)

    # ----- Compute face gradients -----
    g_face = -(p_bc[1:] - p_bc[:-1]) / dx

    # Face locations
    x_face = 0.5 * (x[1:] + x[:-1])

    # Flux at faces
    Q_face = flux_function(g_face, x_face, params)

    # Divergence — interior
    dQdx = np.zeros_like(p_bc)
    dQdx[1:-1] = (Q_face[1:] - Q_face[:-1]) / dx

    # Boundary divergence (one-sided finite-volume)
    # For Dirichlet/Neumann the ghost-cell sets Q_face, so the left/right
    # boundary flux is already encoded there; only the interior face is used.
    # For flux BCs the prescribed boundary flux replaces the missing face.
    if core_bc["type"] == "flux":
        dQdx[0] = (Q_face[0] - core_bc["value"]) / dx
    else:
        dQdx[0] = Q_face[0] / dx

    if edge_bc["type"] == "flux":
        dQdx[-1] = (edge_bc["value"] - Q_face[-1]) / dx
    else:
        dQdx[-1] = -Q_face[-1] / dx

    S = source_function(x, p_bc)

    # ---- Power balance enforcement ----
    # Two modes available:
    #   1. "global": rescale entire source field globally
    #   2. "localized": add Gaussian heating at edge to satisfy power balance

    heating_mode = params.get("heating_mode", "global")
    power_balance = params.get("power_balance", None)

    if heating_mode == "localized" and power_balance is not None:
        # Edge-localized Gaussian heating computed to satisfy power balance
        Q_edge  = Q_face[-1]
        total_S = np.trapezoid(S, x)

        # Compute required deficit
        required_integral = power_balance * Q_edge
        deficit = required_integral - total_S

        # Gaussian centered at edge
        L = x[-1]
        edge_sigma = params.get("edge_sigma", 0.05)

        # Compute actual integral of Gaussian over domain [0, L]
        # (not the full 2D integral, since Gaussian extends beyond x=L)
        gaussian_test = np.exp(-((x - L)**2) / (2 * edge_sigma**2))
        gaussian_norm = np.trapezoid(gaussian_test, x)

        # Amplitude needed to achieve deficit
        if gaussian_norm > 0:
            edge_amp = deficit / gaussian_norm
        else:
            edge_amp = 0

        gaussian_edge = edge_amp * np.exp(-((x - L)**2) / (2 * edge_sigma**2))
        S = S + gaussian_edge

    elif heating_mode == "global" and power_balance is not None:
        # Global rescaling of source
        # params["power_balance"] = r  →  ∫S dx = r · Q_edge
        #   r > 1 : source exceeds edge loss  (net heating)
        #   r < 1 : edge loss exceeds source  (net cooling)
        #   r = 1 : exact power balance
        Q_edge  = Q_face[-1]
        total_S = np.trapezoid(S, x)
        if total_S > 0 and Q_edge > 0:
            S = S * (power_balance * Q_edge / total_S)

    # Hyperviscosity (4th derivative)
    # --------------------------------------------------
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

    return -dQdx - nu4*d4p + S

# ==========================================================
# TIME STEPPING
# ==========================================================

def euler_step(p, dt, dx, x, p_ped, source_function, params):
    return p + dt*compute_rhs(p, dx, x, p_ped, source_function, params)


def rk4_step(p, dt, dx, x, p_ped, source_function, params):

    k1 = compute_rhs(p, dx, x, p_ped, source_function, params)
    k2 = compute_rhs(p + 0.5*dt*k1, dx, x, p_ped, source_function, params)
    k3 = compute_rhs(p + 0.5*dt*k2, dx, x, p_ped, source_function, params)
    k4 = compute_rhs(p + dt*k3, dx, x, p_ped, source_function, params)

    return p + dt*(k1 + 2*k2 + 2*k3 + k4)/6


# ==========================================================
# CN (Picard)
# ==========================================================

import numpy as np
import scipy.linalg


def imex_step(p_old, dt, dx, x, p_ped, source_function, params):

    N = len(p_old)

    chi_RR = params["chi_RR"]
    nu4 = params.get("nu4", 0.0)

    # --------------------------------------------------
    # 1. Explicit nonlinear transport term
    # --------------------------------------------------

    p_bc = p_old.copy()
    p_bc[0] = p_bc[1]
    p_bc[-1] = p_ped

    # Face gradients
    g_face = -(p_bc[1:] - p_bc[:-1]) / dx
    x_face = 0.5 * (x[1:] + x[:-1])

    # Full flux; strip the χ_RR part that is treated implicitly
    Q_full = flux_function(g_face, x_face, params)
    Q_nl   = Q_full - chi_RR * g_face

    # Divergence of nonlinear part
    dQ_nl = np.zeros(N)
    dQ_nl[1:-1] = (Q_nl[1:] - Q_nl[:-1]) / dx
    dQ_nl[0]    =  Q_nl[0]  / dx
    dQ_nl[-1]   = -Q_nl[-1] / dx

    S = source_function(x, p_bc)

    # Optional power-balance enforcement (same logic as compute_rhs)
    power_balance = params.get("power_balance", None)
    if power_balance is not None:
        Q_edge  = Q_full[-1]
        total_S = np.trapezoid(S, x)
        if total_S > 0 and Q_edge > 0:
            S = S * (power_balance * Q_edge / total_S)

    rhs_explicit = -dQ_nl + S

    # --------------------------------------------------
    # 2. Build banded matrix (row-based fill) and RHS
    #
    # scipy solve_banded convention: ab[u + i - j, j] = A[i, j], u = l = 2
    #
    # Filling row i maps to:
    #   A[i, i-2]  ->  ab[4, i-2]
    #   A[i, i-1]  ->  ab[3, i-1]
    #   A[i,  i ]  ->  ab[2,  i ]
    #   A[i, i+1]  ->  ab[1, i+1]
    #   A[i, i+2]  ->  ab[0, i+2]
    #
    # Modifying only rows (never whole columns) preserves the off-diagonal
    # coupling terms that interior rows have with the boundary nodes, so the
    # banded LU solve can correctly back-substitute the known BC values.
    # --------------------------------------------------

    ab = np.zeros((5, N))
    b  = p_old + dt * rhs_explicit

    alpha = dt * chi_RR / dx**2
    beta  = dt * nu4    / dx**4 if nu4 > 0.0 else 0.0

    # Interior rows 1..N-2: implicit χ_RR second derivative
    inn = np.arange(1, N - 1)
    ab[3, inn - 1] -= alpha
    ab[2, inn    ] += 1.0 + 2.0 * alpha
    ab[1, inn + 1] -= alpha

    # Interior rows 2..N-3: implicit ν₄ fourth derivative
    # (5-point stencil; limited to rows where all stencil points are in-domain)
    if nu4 > 0.0:
        inn4 = np.arange(2, N - 2)
        ab[4, inn4 - 2] +=       beta
        ab[3, inn4 - 1] -= 4.0 * beta
        ab[2, inn4     ] += 6.0 * beta
        ab[1, inn4 + 1] -= 4.0 * beta
        ab[0, inn4 + 2] +=       beta

    # --------------------------------------------------
    # 3. Boundary conditions — row-only modifications
    # --------------------------------------------------

    # Neumann at core (i = 0):  p_new[0] - p_new[1] = 0
    # Row 0 elements: A[0,0] at ab[2,0], A[0,1] at ab[1,1], A[0,2] at ab[0,2]
    # (ab[0,2] is already 0; setting ab[2,0] and ab[1,1] encodes the constraint
    #  without touching column 0, so the coupling A[1,0] in row 1 is preserved)
    ab[2, 0] =  1.0
    ab[1, 1] = -1.0
    b[0]     =  0.0

    # Dirichlet at edge (i = N-1):  p_new[N-1] = p_ped
    # Row N-1 elements: A[N-1, N-3] at ab[4, N-3], A[N-1, N-2] at ab[3, N-2],
    # A[N-1, N-1] at ab[2, N-1].  The first two are already 0 from initialisation
    # (they would only be set if i = N-1 appeared in the interior loops, which it
    # does not).  Setting only ab[2, N-1] leaves the coupling elements
    # A[N-2, N-1] = ab[1, N-1] and A[N-3, N-1] = ab[0, N-1] intact so the
    # banded solve correctly back-substitutes p_ped into those rows.
    ab[2, N - 1] = 1.0
    b[N - 1]     = p_ped

    # --------------------------------------------------
    # 4. Solve banded system
    # --------------------------------------------------

    return scipy.linalg.solve_banded((2, 2), ab, b)

# ==========================================================
# DISPATCH
# ==========================================================

def step(p, dt, dx, x, p_ped, source_function, params):

    if TIME_SCHEME == "Euler":
        p_new = euler_step(p, dt, dx, x, p_ped, source_function, params)

    elif TIME_SCHEME == "RK4":
        p_new = rk4_step(p, dt, dx, x, p_ped, source_function, params)

    elif TIME_SCHEME == "CN":
        #p_new = cn_step(p, dt, dx, x, p_ped, source_function, params)
        p_new = imex_step(p, dt, dx, x, p_ped, source_function, params)
    else:
        raise ValueError("Invalid TIME_SCHEME")

    # ---- Re-enforce boundary conditions ----
    p_new[0] = p_new[1]      # Neumann
    p_new[-1] = p_ped        # Dirichlet

    return p_new

# ==========================================================
# SOLVER LOOP
# ==========================================================

def solving_loop(
    p_init,
    dt,
    dx,
    T,
    L,
    p_ped,
    source_function,
    num_snapshots,
    params,
):

    N = len(p_init)
    x = np.linspace(0, L, N)

    num_steps = int(T/dt)
    save_indices = np.linspace(0, num_steps, num_snapshots+1, dtype=int)

    p = p_init.copy()
    saved = []
    save_counter = 0

    for i in range(num_steps+1):

        if i == save_indices[save_counter]:
            saved.append(p.copy())
            save_counter = min(save_counter+1, len(save_indices)-1)

        p = step(p, dt, dx, x, p_ped, source_function, params)

        # ---- Diagnostic every 2000 steps ----
        if i % 20000 == 0:
            g = -(p[1:] - p[:-1]) / dx
            print(f"Step {i:6d} | max|g| = {np.max(np.abs(g)):.4f}")

    return np.array(saved)
