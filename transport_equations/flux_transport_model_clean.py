import numpy as np
import scipy.linalg


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

    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    g = np.maximum(g, 0.0)

    chi0     = params["chi0"]
    chi_core = params["chi_core"]
    chi_RR   = params["chi_RR"]

    g_c      = params["g_c"]
    g_stiff  = params["g_stiff"]
    n_stiff  = params["n_stiff"]

    boundaries  = params.get("boundaries", [])
    deltas      = params.get("deltas", [0.01] * len(boundaries))
    flux_models = params.get("flux_models", ["core"])

    # --- Core stiff model ---
    ratio = np.clip(g / g_stiff, 0.0, 20.0)
    Q_core = chi_core * g * (1.0 + ratio**n_stiff)

    # --- Nonlinear model ---
    Q_nl = chi0 * g * np.maximum(1.0 - g / g_c, 0)
    Q_nl = np.maximum(Q_nl, 0.0)

    # --- Linear fallback model ---
    Q_linear = chi_core * g

    Q_map = {
        "core": Q_core,
        "nl": Q_nl,
        "linear": Q_linear,
    }

    W = make_windows(x, boundaries, deltas)

    if len(W) != len(flux_models):
        raise ValueError(
            f"Number of flux_models must equal number of regions ({len(W)} expected)"
        )

    Q_transport = np.zeros_like(g)
    for Wi, model_name in zip(W, flux_models):
        Q_transport += Wi * Q_map[model_name]

    Q_total = Q_transport + chi_RR * g

    # MHD stiff cliff (optional)
    g_MHD   = params.get("g_MHD",   None)
    chi_MHD = params.get("chi_MHD", 0.0)

    if g_MHD is not None and chi_MHD > 0.0:
        Q_total = Q_total + chi_MHD * np.maximum(g - g_MHD, 0.0)

    return Q_total


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
    return sigv * 1.0e-6   # cm^3/s -> m^3/s


def alpha_heating(p, params):
    """D-T alpha heating: P_a = C_alpha * 5.6e-13 * <sv> * n_D * n_T."""
    C_alpha = params.get("C_alpha", 0.0)
    if C_alpha <= 0.0:
        return np.zeros_like(p)

    T_ref   = params.get("T_ref_keV",  10.0)
    n_ref   = params.get("n_ref",       1.0)
    n_ref20 = params.get("n_ref_20",    1.0)
    f_D     = params.get("f_deuterium", 0.5)
    f_T     = params.get("f_tritium",   0.5)

    T_keV  = p / max(n_ref, 1e-10) * T_ref
    n_phys = n_ref * n_ref20 * 1.0e20

    n_D  = f_D * n_phys
    n_T  = f_T * n_phys
    sigv = sigma_v_DT(T_keV)
    return C_alpha * 5.6e-13 * sigv * n_D * n_T


def bremsstrahlung(p, params):
    """Bremsstrahlung loss: P_brem = C_brem * n^2 * Z_eff * sqrt(T_keV)."""
    C_brem = params.get("C_brem", 0.0)
    if C_brem <= 0.0:
        return np.zeros_like(p)

    Z_eff  = params.get("Z_eff",     1.0)
    T_ref  = params.get("T_ref_keV", 10.0)
    n_ref  = params.get("n_ref",      1.0)

    T_keV = p / max(n_ref, 1e-10) * T_ref
    return C_brem * n_ref**2 * Z_eff * np.sqrt(np.maximum(T_keV, 1e-10))


# ==========================================================
# BOUNDARY CONDITIONS
# ==========================================================

def _get_bcs(params, p_ped):
    bc   = params.get("bc", {})
    core = bc.get("core", {"type": "neumann",   "value": 0.0})
    edge = bc.get("edge", {"type": "dirichlet", "value": p_ped})
    return core, edge


def _apply_bc_to_profile(p_bc, dx, core_bc, edge_bc):
    """Set boundary cell values in p_bc to reflect the chosen BC."""
    ctype = core_bc["type"]
    if ctype == "dirichlet":
        p_bc[0] = core_bc["value"]
    elif ctype == "neumann":
        p_bc[0] = p_bc[1] - core_bc["value"] * dx

    etype = edge_bc["type"]
    if etype == "dirichlet":
        p_bc[-1] = edge_bc["value"]
    elif etype == "neumann":
        p_bc[-1] = p_bc[-2] + edge_bc["value"] * dx
    # robin: p[-1] is a free unknown — leave it as-is


# ==========================================================
# DIAGNOSTIC FLUX COMPUTATION
# ==========================================================

def compute_face_flux(p, dx, x, p_ped, params):
    """
    Compute face-centred *physical* Q for diagnostics.
    Uses the same face gradients and smoothing as compute_rhs,
    but omits the LLF numerical correction (which is a solver
    stabiliser, not a physical flux).
    Returns (Q_face, x_face).
    """
    core_bc, edge_bc = _get_bcs(params, p_ped)
    p_bc = p.copy()
    _apply_bc_to_profile(p_bc, dx, core_bc, edge_bc)

    g_face = -(p_bc[1:] - p_bc[:-1]) / dx
    x_face = 0.5 * (x[1:] + x[:-1])

    n_smooth = params.get("n_smooth", 0)
    for _ in range(n_smooth):
        g_s = g_face.copy()
        g_s[1:-1] = 0.25*g_face[:-2] + 0.5*g_face[1:-1] + 0.25*g_face[2:]
        g_face = g_s

    Q_face = flux_function(g_face, x_face, params)
    return Q_face, x_face


# ==========================================================
# RHS
# ==========================================================

def compute_rhs(p, dx, x, p_ped, source_function, params, physics_source_function=None):

    core_bc, edge_bc = _get_bcs(params, p_ped)

    p_bc = p.copy()
    _apply_bc_to_profile(p_bc, dx, core_bc, edge_bc)

    # Face gradients and flux
    g_face = -(p_bc[1:] - p_bc[:-1]) / dx
    x_face = 0.5 * (x[1:] + x[:-1])

    # Smooth face gradients before flux evaluation.
    # Each pass applies a (1/4, 1/2, 1/4) binomial filter to g_face,
    # suppressing grid-scale staircases that grow in the antidiffusion
    # region (falling branch of Q(g) where dQ/dg < 0).
    n_smooth = params.get("n_smooth", 0)
    for _ in range(n_smooth):
        g_s = g_face.copy()
        g_s[1:-1] = 0.25*g_face[:-2] + 0.5*g_face[1:-1] + 0.25*g_face[2:]
        g_face = g_s

    Q_face = flux_function(g_face, x_face, params)

    # Local Lax-Friedrichs stabilisation at each face.
    # Ensures the effective diffusivity (physical + numerical) never drops
    # below lf_dissipation.  Zero correction deep in the diffusive region
    # (where dQ/dg >> lf), active at the transition (dQ/dg ~ 0) and through
    # the antidiffusive region (dQ/dg < 0).
    lf = params.get("lf_dissipation", 0.0)
    if lf > 0:
        eps_fd = 1e-6
        dQdg = (flux_function(g_face + eps_fd, x_face, params)
              - flux_function(g_face - eps_fd, x_face, params)) / (2 * eps_fd)
        alpha = np.maximum(lf - dQdg, 0.0)
        Q_face = Q_face + 0.5 * alpha * g_face

    # Divergence
    dQdx = np.zeros_like(p_bc)
    dQdx[1:-1] = (Q_face[1:] - Q_face[:-1]) / dx

    if core_bc["type"] == "flux":
        dQdx[0] = (Q_face[0] - core_bc["value"]) / dx
    else:
        dQdx[0] = Q_face[0] / dx

    if edge_bc["type"] == "flux":
        dQdx[-1] = (edge_bc["value"] - Q_face[-1]) / dx
    elif edge_bc["type"] == "robin":
        # Q_edge = gamma * (p_edge - p_SOL): flux into SOL proportional to
        # how far edge pressure sits above the SOL sink value.
        Q_robin = edge_bc["gamma"] * (p_bc[-1] - edge_bc["p_SOL"])
        dQdx[-1] = (Q_robin - Q_face[-1]) / dx
    else:
        dQdx[-1] = -Q_face[-1] / dx

    # Source terms
    S_ext = source_function(x, p_bc)

    if physics_source_function is not None:
        S_phys = physics_source_function(x, p_bc)
    else:
        S_phys = np.zeros_like(p_bc)

    S = S_ext + S_phys

    # Hyperviscosity (4th derivative) — explicit only when not treated implicitly
    nu4 = params.get("nu4", 0.0)
    implicit_nu4 = params.get("implicit_nu4", False)
    d4p = np.zeros_like(p_bc)

    if nu4 > 0 and not implicit_nu4 and len(p_bc) > 4:
        d4p[2:-2] = (
              p_bc[0:-4]
            - 4*p_bc[1:-3]
            + 6*p_bc[2:-2]
            - 4*p_bc[3:-1]
            + p_bc[4:]
        ) / dx**4

    return -dQdx - nu4*d4p + S


# ==========================================================
# RK4 TIME STEP
# ==========================================================

def rk4_step(p, dt, dx, x, p_ped, source_function, params, physics_source_function=None):

    k1 = compute_rhs(p,            dx, x, p_ped, source_function, params, physics_source_function)
    k2 = compute_rhs(p+0.5*dt*k1,  dx, x, p_ped, source_function, params, physics_source_function)
    k3 = compute_rhs(p+0.5*dt*k2,  dx, x, p_ped, source_function, params, physics_source_function)
    k4 = compute_rhs(p+dt*k3,      dx, x, p_ped, source_function, params, physics_source_function)

    return p + dt*(k1 + 2*k2 + 2*k3 + k4)/6


# ==========================================================
# IMPLICIT HYPERVISCOSITY
# ==========================================================

def _implicit_nu4_step(p, dt, dx, nu4, p_ped):
    """
    Solve (I + dt * nu4 * D4) p_new = p for the 4th-derivative term.
    Pentadiagonal banded system, O(N) cost.
    Boundary rows are set to identity (BCs enforced after).
    """
    N = len(p)
    beta = dt * nu4 / dx**4

    # Banded storage: ab[u + i - j, j] = A[i, j], with u = l = 2
    ab = np.zeros((5, N))

    # Interior rows (2 .. N-3): pentadiagonal stencil
    inn = np.arange(2, N - 2)
    ab[0, inn + 2] = beta            # A[i, i+2]
    ab[1, inn + 1] = -4.0 * beta     # A[i, i+1]
    ab[2, inn]     = 1.0 + 6.0 * beta  # A[i, i]
    ab[3, inn - 1] = -4.0 * beta     # A[i, i-1]
    ab[4, inn - 2] = beta            # A[i, i-2]

    # Boundary rows: identity (rows 0, 1, N-2, N-1)
    for j in [0, 1, N - 2, N - 1]:
        ab[2, j] = 1.0

    return scipy.linalg.solve_banded((2, 2), ab, p)


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
):
    N = len(p_init)
    x = np.linspace(0, L, N)

    num_steps = int(T/dt)
    save_indices = np.linspace(0, num_steps, num_snapshots+1, dtype=int)

    nu4          = params.get("nu4", 0.0)
    implicit_nu4 = params.get("implicit_nu4", False) and nu4 > 0

    _, edge_bc = _get_bcs(params, p_ped)
    edge_is_dirichlet = (edge_bc["type"] == "dirichlet")

    p = p_init.copy()
    saved = []
    save_counter = 0

    for i in range(num_steps+1):

        if i == save_indices[save_counter]:
            saved.append(p.copy())
            save_counter = min(save_counter+1, len(save_indices)-1)

        # Explicit RK4 step (nu4 term skipped inside compute_rhs when implicit)
        p = rk4_step(p, dt, dx, x, p_ped, source_function, params, physics_source_function)

        # Implicit hyperviscosity substep
        if implicit_nu4:
            p = _implicit_nu4_step(p, dt, dx, nu4, p_ped)

        # Enforce BCs
        p[0] = p[1]       # Neumann core
        if edge_is_dirichlet:
            p[-1] = p_ped  # Dirichlet edge
        # robin / neumann: p[-1] evolves freely via RHS

        if i % 20000 == 0:
            g = -(p[1:] - p[:-1]) / dx
            int_p = np.trapezoid(p, x)
            print(f"Step {i:6d} | max|g| = {np.max(np.abs(g)):.4f} | int p dx = {int_p:.4f} | p_edge = {p[-1]:.4f}")

    return np.array(saved)
