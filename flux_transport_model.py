import numpy as np


# ==========================================================
# USER OPTIONS
# ==========================================================

TIME_SCHEME = "RK4"        # "Euler", "RK4", "CN"
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
# PHYSICS SOURCES (bremsstrahlung + alpha heating)
# ==========================================================

def sigma_v_DT(T_keV):
    """
    Bosch-Hale D-T fusion reactivity in m³/s.
    Bosch & Hale, Nucl. Fusion 32 (1992), Table 4.  Valid 0.2–100 keV.
    """
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
    return sigv * 1.0e-6   # cm³/s → m³/s


def alpha_heating(p, params):
    """
    Total D-T alpha heating for a single-pressure field.

        P_α = 5.6e-13 J × ⟨σv⟩(T_keV) × n_D × n_T   [W/m³]

    Temperature is inferred from the pressure via a fixed background density:
        T_keV = (p / n_ref) × T_ref_keV

    C_alpha encodes unit conversion and time-scale normalisation.
    Set C_alpha = 0 (default) to disable.
    """
    C_alpha = params.get("C_alpha", 0.0)
    if C_alpha <= 0.0:
        return np.zeros_like(p)

    T_ref   = params.get("T_ref_keV",  10.0)
    n_ref   = params.get("n_ref",       1.0)
    n_ref20 = params.get("n_ref_20",    1.0)
    f_D     = params.get("f_deuterium", 0.5)
    f_T     = params.get("f_tritium",   0.5)

    T_keV  = p / max(n_ref, 1e-10) * T_ref
    n_phys = n_ref * n_ref20 * 1.0e20   # m⁻³

    n_D  = f_D * n_phys
    n_T  = f_T * n_phys
    sigv = sigma_v_DT(T_keV)
    return C_alpha * 5.6e-13 * sigv * n_D * n_T


def bremsstrahlung(p, params):
    """
    Bremsstrahlung radiation loss (electron power sink).

        P_brem = 5.35e-3 × n² × Z_eff × √T_keV   [MW/m³]
    with n in 10²⁰ m⁻³, T in keV.

    Temperature is inferred via T_keV = (p / n_ref) × T_ref_keV.
    C_brem encodes unit conversion and time-scale normalisation.
    Set C_brem = 0 (default) to disable.
    """
    C_brem = params.get("C_brem", 0.0)
    if C_brem <= 0.0:
        return np.zeros_like(p)

    Z_eff  = params.get("Z_eff",     1.0)
    T_ref  = params.get("T_ref_keV", 10.0)
    n_ref  = params.get("n_ref",      1.0)

    T_keV = p / max(n_ref, 1e-10) * T_ref
    return C_brem * n_ref**2 * Z_eff * np.sqrt(np.maximum(T_keV, 1e-10))


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

def compute_rhs(p, dx, x, p_ped, source_function, params, physics_source_function=None):

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

    # External (controllable) source — subject to power balance scaling
    S_ext = source_function(x, p_bc)

    # Physics source (alpha heating, radiation, etc.) — never rescaled
    if physics_source_function is not None:
        S_phys = physics_source_function(x, p_bc)
    else:
        S_phys = np.zeros_like(p_bc)

    # ---- Power balance enforcement ----
    # Only S_ext is scaled; S_phys is always added unmodified.
    # Target:  ∫S_ext_scaled dx + ∫S_phys dx = power_balance × Q_edge
    #
    # "global":    rescale S_ext globally to hit target
    # "localized": add edge Gaussian to S_ext to close the deficit

    heating_mode = params.get("heating_mode", "global")
    power_balance = params.get("power_balance", None)

    if heating_mode == "localized" and power_balance is not None:
        Q_edge  = Q_face[-1]
        total_S = np.trapezoid(S_ext + S_phys, x)
        deficit = power_balance * Q_edge - total_S

        L = x[-1]
        edge_sigma = params.get("edge_sigma", 0.05)
        gaussian_test = np.exp(-((x - L)**2) / (2 * edge_sigma**2))
        gaussian_norm = np.trapezoid(gaussian_test, x)
        edge_amp = deficit / gaussian_norm if gaussian_norm > 0 else 0.0

        S_ext = S_ext + edge_amp * gaussian_test

    elif heating_mode == "global" and power_balance is not None:
        Q_edge       = Q_face[-1]
        total_S_ext  = np.trapezoid(S_ext,  x)
        total_S_phys = np.trapezoid(S_phys, x)
        target_S_ext = power_balance * Q_edge - total_S_phys
        if total_S_ext > 0 and target_S_ext > 0:
            S_ext = S_ext * (target_S_ext / total_S_ext)

    S = S_ext + S_phys

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

def euler_step(p, dt, dx, x, p_ped, source_function, params, physics_source_function=None):
    return p + dt*compute_rhs(p, dx, x, p_ped, source_function, params, physics_source_function)


def rk4_step(p, dt, dx, x, p_ped, source_function, params, physics_source_function=None):

    k1 = compute_rhs(p,            dx, x, p_ped, source_function, params, physics_source_function)
    k2 = compute_rhs(p+0.5*dt*k1,  dx, x, p_ped, source_function, params, physics_source_function)
    k3 = compute_rhs(p+0.5*dt*k2,  dx, x, p_ped, source_function, params, physics_source_function)
    k4 = compute_rhs(p+dt*k3,      dx, x, p_ped, source_function, params, physics_source_function)

    return p + dt*(k1 + 2*k2 + 2*k3 + k4)/6


# ==========================================================
# CN (Picard)
# ==========================================================

import numpy as np
import scipy.linalg


def imex_step(p_old, dt, dx, x, p_ped, source_function, params, physics_source_function=None):

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

    # External source (scaleable) and physics source (fixed)
    S_ext  = source_function(x, p_bc)
    S_phys = physics_source_function(x, p_bc) if physics_source_function is not None else np.zeros(N)

    # Power balance: scale only S_ext, same convention as compute_rhs
    power_balance = params.get("power_balance", None)
    if power_balance is not None:
        Q_edge       = Q_full[-1]
        total_S_ext  = np.trapezoid(S_ext,  x)
        total_S_phys = np.trapezoid(S_phys, x)
        target_S_ext = power_balance * Q_edge - total_S_phys
        if total_S_ext > 0 and target_S_ext > 0:
            S_ext = S_ext * (target_S_ext / total_S_ext)

    rhs_explicit = -dQ_nl + S_ext + S_phys

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

def step(p, dt, dx, x, p_ped, source_function, params, physics_source_function=None):

    if TIME_SCHEME == "Euler":
        p_new = euler_step(p, dt, dx, x, p_ped, source_function, params, physics_source_function)

    elif TIME_SCHEME == "RK4":
        p_new = rk4_step(p, dt, dx, x, p_ped, source_function, params, physics_source_function)

    elif TIME_SCHEME == "CN":
        p_new = imex_step(p, dt, dx, x, p_ped, source_function, params, physics_source_function)
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
    physics_source_function=None,
    source_controller=None,
):
    """
    Main time-integration loop.

    Parameters
    ----------
    source_controller : object or None
        If not None, must be callable as ``source_controller(x, p)`` returning
        the external source array, and must expose:
          - ``update(t, p, x, dt)``  — called each step before the RHS
          - ``_record_snapshot()``   — called each time a snapshot is saved
        When a controller is active it *replaces* ``source_function``; the
        ``power_balance`` key in ``params`` is ignored (set it to None).
    """

    N = len(p_init)
    x = np.linspace(0, L, N)

    num_steps = int(T/dt)
    save_indices = np.linspace(0, num_steps, num_snapshots+1, dtype=int)

    p = p_init.copy()
    saved = []
    save_counter = 0

    # Choose which callable provides the external source
    src_fn = source_controller if source_controller is not None else source_function

    for i in range(num_steps+1):

        t = i * dt

        if i == save_indices[save_counter]:
            saved.append(p.copy())
            if source_controller is not None:
                source_controller._record_snapshot()
            save_counter = min(save_counter+1, len(save_indices)-1)

        # Update controller *before* computing the RHS so the current step
        # uses the freshly updated amplitudes.
        if source_controller is not None:
            source_controller.update(t, p, x, dt)

        p = step(p, dt, dx, x, p_ped, src_fn, params, physics_source_function)

        # ---- Diagnostic every 20000 steps ----
        if i % 20000 == 0:
            g = -(p[1:] - p[:-1]) / dx
            print(f"Step {i:6d} | max|g| = {np.max(np.abs(g)):.4f}")

    return np.array(saved)
