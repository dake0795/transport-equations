#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from flux_transport_model import flux_function, solving_loop, alpha_heating, bremsstrahlung

# ==========================================
# Source Controller
# ==========================================

class SourceController:
    """
    Independent PI / open-loop controllers for a core and an edge heating channel.

    Core channel : Gaussian centred at x = 0,  targets stored energy ∫p dx
                   or core pressure p[1].
    Edge channel : Gaussian centred at x = L,  targets edge gradient g_edge.

    Each channel has three modes
    --------------------------------
    "off"      — channel contributes zero heating.
    "schedule" — open-loop: amplitude = schedule(t), a user-supplied callable.
    "PI"       — closed-loop: PI controller drives the observable toward ``target``.

    Set ``controller_mode = None`` in the driver section to bypass the controller
    entirely and run the model in its original form.

    PI update law (per channel)
    ----------------------------
        error = target - observable
        integral += error * dt           (anti-windup: frozen when output is clamped)
        A = max(A_ff + Kp * error + Ki * integral, 0)

    Config dict keys (``core_config`` / ``edge_config``)
    -----------------------------------------------------
    mode          : "off" | "schedule" | "PI"
    sigma         : Gaussian half-width
    A_init        : initial amplitude (used at t = 0 before first update)
    schedule      : callable A(t)  [schedule mode only]
    target        : scalar set-point [PI mode only]
    target_type   : "stored_energy" | "core_pressure"  (core);
                    "edge_gradient"                     (edge)
    A_ff          : feedforward amplitude added before PI correction
    Kp, Ki        : proportional and integral gains
    """

    def __init__(self, core_config, edge_config, L):
        self.L = L

        # ---- Core channel ----
        self.core_mode       = core_config.get("mode",        "off")
        self.core_sigma      = core_config.get("sigma",        0.25)
        self.core_A          = core_config.get("A_init",       0.0)
        self.core_schedule   = core_config.get("schedule",     None)
        self.core_target     = core_config.get("target",       None)
        self.core_target_type= core_config.get("target_type",  "stored_energy")
        self.core_A_ff       = core_config.get("A_ff",         0.0)
        self.core_Kp         = core_config.get("Kp",           1.0)
        self.core_Ki         = core_config.get("Ki",           5.0)
        self._core_integral  = 0.0

        # ---- Edge channel ----
        self.edge_mode       = edge_config.get("mode",        "off")
        self.edge_sigma      = edge_config.get("sigma",        0.05)
        self.edge_A          = edge_config.get("A_init",       0.0)
        self.edge_schedule   = edge_config.get("schedule",     None)
        self.edge_target     = edge_config.get("target",       None)
        self.edge_target_type= edge_config.get("target_type",  "edge_gradient")
        self.edge_A_ff       = edge_config.get("A_ff",         0.0)
        self.edge_Kp         = edge_config.get("Kp",           1.0)
        self.edge_Ki         = edge_config.get("Ki",           5.0)
        self._edge_integral  = 0.0

        # ---- Dense history (every step) ----
        self.t_hist        = []
        self.A_core_hist   = []
        self.A_edge_hist   = []
        self.obs_core_hist = []
        self.obs_edge_hist = []

        # ---- Snapshot-aligned history (at saved_p times) ----
        self.snap_A_core = []
        self.snap_A_edge = []

    # ----------------------------------------------------------
    # Shape functions
    # ----------------------------------------------------------

    def _core_shape(self, x):
        return np.exp(-(x**2) / (2.0 * self.core_sigma**2))

    def _edge_shape(self, x):
        return np.exp(-((x - self.L)**2) / (2.0 * self.edge_sigma**2))

    # ----------------------------------------------------------
    # Observables
    # ----------------------------------------------------------

    def _observe_core(self, p, x):
        if self.core_target_type == "stored_energy":
            return np.trapezoid(p, x)
        elif self.core_target_type == "core_pressure":
            return float(p[1])
        return np.trapezoid(p, x)

    def _observe_edge(self, p, x):
        dx = x[1] - x[0]
        return (p[-2] - p[-1]) / dx   # -(dp/dx) at edge ≈ g_edge

    # ----------------------------------------------------------
    # Snapshot recording (called by solving_loop at save times)
    # ----------------------------------------------------------

    def _record_snapshot(self):
        self.snap_A_core.append(self.core_A if self.core_mode != "off" else 0.0)
        self.snap_A_edge.append(self.edge_A if self.edge_mode != "off" else 0.0)

    # ----------------------------------------------------------
    # Reconstruct source at a given snapshot index
    # (used in diagnostics where we need the amplitude that was
    #  active when snapshot snap_i was saved)
    # ----------------------------------------------------------

    def source_at_snapshot(self, snap_i, x):
        S = np.zeros_like(x, dtype=float)
        if self.core_mode != "off" and snap_i < len(self.snap_A_core):
            S += self.snap_A_core[snap_i] * self._core_shape(x)
        if self.edge_mode != "off" and snap_i < len(self.snap_A_edge):
            S += self.snap_A_edge[snap_i] * self._edge_shape(x)
        return S

    # ----------------------------------------------------------
    # Step update (called by solving_loop every timestep)
    # ----------------------------------------------------------

    def update(self, t, p, x, dt):
        self.L = x[-1]   # keep in sync if domain ever changes

        # ---- Core ----
        if self.core_mode == "schedule":
            self.core_A = max(float(self.core_schedule(t)), 0.0)

        elif self.core_mode == "PI":
            obs   = self._observe_core(p, x)
            error = self.core_target - obs
            self._core_integral += error * dt
            A = self.core_A_ff + self.core_Kp * error + self.core_Ki * self._core_integral
            # Anti-windup: undo the integral accumulation if output is clamped
            if A < 0.0 and error < 0.0:
                self._core_integral -= error * dt
            self.core_A = max(A, 0.0)

        # ---- Edge ----
        if self.edge_mode == "schedule":
            self.edge_A = max(float(self.edge_schedule(t)), 0.0)

        elif self.edge_mode == "PI":
            obs   = self._observe_edge(p, x)
            error = self.edge_target - obs
            self._edge_integral += error * dt
            A = self.edge_A_ff + self.edge_Kp * error + self.edge_Ki * self._edge_integral
            if A < 0.0 and error < 0.0:
                self._edge_integral -= error * dt
            self.edge_A = max(A, 0.0)

        # ---- Record dense history ----
        self.t_hist.append(t)
        self.A_core_hist.append(self.core_A if self.core_mode != "off" else 0.0)
        self.A_edge_hist.append(self.edge_A if self.edge_mode != "off" else 0.0)
        self.obs_core_hist.append(self._observe_core(p, x))
        self.obs_edge_hist.append(self._observe_edge(p, x))

    # ----------------------------------------------------------
    # __call__ — makes the controller a drop-in source function
    # ----------------------------------------------------------

    def __call__(self, x, p):
        S = np.zeros_like(x, dtype=float)
        if self.core_mode != "off":
            S += self.core_A * self._core_shape(x)
        if self.edge_mode != "off":
            S += self.edge_A * self._edge_shape(x)
        return S


# ==========================================
# Golden-style plotting
# ==========================================
FIGSIZE = (6.33, 4.33)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.figsize": FIGSIZE,
    "lines.linewidth": 1.6,
    "axes.grid": False,
})

def style_plot(ax):
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()

def save_and_show(filename):
    """Save figure as PDF and display."""
    import os
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    filepath = os.path.join(plots_dir, f"{filename}.pdf")
    plt.savefig(filepath, format="pdf", bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.show()

# ==========================================
# Domain
# ==========================================
L = 1.0
dx = 0.005
dt = 1e-6
T = 0.5
num_snapshots = 8

x = np.linspace(0, L, int(L/dx))

# ==========================================
# Branch selection
# ==========================================
START_ON = "supercritical"
branch_label = START_ON

# ==========================================
# Power balance mode
# ==========================================
# "continuous"   : rescale source at every RHS call (original behaviour,
#                  4x per timestep with RK4 — artificially pins ∫p dx)
# "initial_only" : scale source once at t=0 to match initial edge flux,
#                  then fix amplitude — system evolves freely thereafter
# "free"         : no power balance enforcement at all
power_balance_mode = "initial_only"

# ====================================================
# Transport parameters (set here and passed to model)
# ====================================================

transport_params = {
    # ----- Transport strength -----
    "chi0": 10.0,        # NL transport coefficient
    "chi_core": 10.0,    # Core stiff transport coefficient
    "chi_RR": 0.05,     # Background diffusive transport (active everywhere)

    # ----- Nonlinear edge model -----
    "g_c": 4.0,         # Critical gradient where NL flux goes to zero

    # ----- Stiff core model -----
    "g_stiff": 3,     # Onset of stiff transport
    "n_stiff": 2,       # Stiffness exponent (reduce if unstable)

    # ----- Spatial structure -----
    "boundaries": [],   # No region boundaries → single region
    "deltas": [],       # No smoothing needed

    # ----- Flux model per region -----
    "flux_models": ["nl"], #, "nl", "core", "nl"],   # Entire domain uses stiff core model

    "nu4": 0,   # hyperviscosity strength (tune as needed)

    # ----- Power balance enforcement -----
    "heating_mode": "global",  # "global" or "localized"
    "power_balance": 0.6,      # < 1 → initial deficit: ∫S < Q_edge so ∫p dx falls first
    "edge_sigma": 0.05,        # width of edge-localized Gaussian heating (for localized mode)

    # ----- Physics sources -----
    # Bremsstrahlung radiation loss:  P_brem = C_brem × n_ref² × Z_eff × √T_keV
    #   C_brem = 0  disables; set ~0.03 for mild radiation at n~1, T~10 keV
    "C_brem":      0.03,
    "Z_eff":       1.0,
    # Alpha heating:  P_α = C_alpha × ⟨σv⟩(T_keV) × n_D × n_T
    #   C_alpha = 0  disables.
    #   The physical factor ⟨σv⟩ × n_D × n_T at n=1e20 m⁻³, T~20 keV is ~3e5 W/m³.
    #   C_alpha must bridge to model units where S_ext ~ O(1).
    #   C_alpha = 1e-7 → ∫P_alpha ~ 0.03  (small perturbation on S_ext ~ 1)
    #   C_alpha = 1e-6 → ∫P_alpha ~ 0.3   (mild, ~30% of S_ext)
    #   C_alpha = 1e-5 → ∫P_alpha ~ 3     (dominant, approaching ignition regime)
    "C_alpha":     1e-6,   # mild alpha heating — grows with T to eventually offset deficit
    "f_deuterium": 0.5,
    "f_tritium":   0.5,
    # Reference scales used to convert model p → physical T, n
    "T_ref_keV":   10.0,   # model T=1  ↔  T_ref_keV keV
    "n_ref":        1.0,   # background density in model units  (T = p / n_ref)
    "n_ref_20":     1.0,   # n_ref corresponds to n_ref_20 × 10²⁰ m⁻³
}

# ==========================================
# Initial profile
# ==========================================

g_c = transport_params["g_c"]

g_crit = g_c / 2.0

# --------------------------------------------------
# MHD stiff cliff
# g_MHD : gradient at which steep MHD-like transport switches on.
#         Set slightly below g_c so the cliff overlaps the falling NL
#         flux — total Q has a local minimum then rises steeply, preventing
#         unphysical blowup beyond the NL zero.
# chi_MHD : stiffness of the cliff (larger = harder gradient cap).
#           Set chi_MHD = 0.0 (or remove g_MHD) to disable entirely.
# --------------------------------------------------
transport_params["g_MHD"]   = 0.9 * g_c   # ~ 3.6 with default g_c = 4
transport_params["chi_MHD"] = 20        # disabled; enable (try 5–20) to cap supercritical blowup

p_core = 2
p_ped = 1


m = 2

def base_profile(x,m):
    return (1 - (x/L)**m)

p_shape = base_profile(x,m)
p_x_shape = np.gradient(p_shape, dx)
g_shape = -p_x_shape
max_g_shape = np.max(g_shape)

if START_ON == "subcritical":
    target_g = 0.6 * g_crit
elif START_ON == "supercritical":
    target_g = 1.8 * g_crit
else:
    raise ValueError("START_ON must be 'subcritical' or 'supercritical'")

scale = target_g / max_g_shape
p_init = p_ped + scale * (p_core - p_ped) * p_shape

g_init = -np.gradient(p_init, dx)
print("Initial max g =", np.max(g_init))
print("Critical g =", g_crit)

# ==========================================
# Source
# ==========================================
def base_source(x):
    S0 = 5.0
    sigma = 0.25 * L
    return S0 * np.exp(-(x**2)/(sigma**2))

def source(x, p):
    """External (Gaussian) heating only. This is the part scaled by power balance."""
    return base_source(x)

def physics_source(x, p):
    """
    Physics source terms: alpha heating minus bremsstrahlung.
    Determined by the plasma state; never rescaled by power balance enforcement.
    T_keV = (p / n_ref) × T_ref_keV with fixed background density n_ref.
    """
    return alpha_heating(p, transport_params) - bremsstrahlung(p, transport_params)

# ==========================================
# Compute initial gradient and flux using same discretization as compute_rhs
# ==========================================
from flux_transport_model import _get_bcs, _apply_bc_to_profile

# Apply boundary conditions first
core_bc, edge_bc = _get_bcs(transport_params, p_ped)
p_bc = p_init.copy()
_apply_bc_to_profile(p_bc, dx, core_bc, edge_bc)

# Face-centered gradients (same as in compute_rhs)
g_face = -(p_bc[1:] - p_bc[:-1]) / dx
x_face = 0.5 * (x[1:] + x[:-1])

# Flux at faces
Q_face_init = flux_function(g_face, x_face, transport_params)

# For display, interpolate to cell centers
g_init = -np.gradient(p_init, dx)
Q_init = flux_function(g_init, x, transport_params)

# ==========================================
# Initial power balance
# ==========================================
p_x_init = np.gradient(p_init, dx)
g_init = -p_x_init

S_ext_init   = source(x, p_init)          # base_source only
S_phys_init  = physics_source(x, p_init)  # P_alpha - P_brem
S_total_init = S_ext_init + S_phys_init

P_alpha_init = alpha_heating(p_init, transport_params)
P_brem_init  = bremsstrahlung(p_init, transport_params)

initial_edge_flux    = Q_face_init[-1]
initial_S_ext        = np.trapezoid(S_ext_init,   x)
initial_P_alpha      = np.trapezoid(P_alpha_init, x)
initial_P_brem       = np.trapezoid(P_brem_init,  x)
initial_S_total      = np.trapezoid(S_total_init, x)

print(f"Initial external heating  = {initial_S_ext:.4f}")
print(f"Initial alpha heating     = {initial_P_alpha:.4f}")
print(f"Initial bremsstrahlung    = {initial_P_brem:.4f}")
print(f"Initial net source        = {initial_S_total:.4f}")
print(f"Initial edge flux         = {initial_edge_flux:.4f}")
print(f"Initial balance           = {initial_edge_flux - initial_S_total:.4f}")

# ==========================================
# INITIAL DIAGNOSTIC PLOTS
# ==========================================

# ----------------------------------------------------------
# 1. Initial pressure profile with supercritical regions
# ----------------------------------------------------------

fig, ax = plt.subplots()

ax.plot(x, p_init, label="Initial pressure")

# Highlight where gradient exceeds critical value
supercrit_mask = g_init > g_crit

if np.any(supercrit_mask):
    ax.scatter(x[supercrit_mask],
               p_init[supercrit_mask],
               color="red",
               s=20,
               label=r"$g > g_\mathrm{crit}$")

ax.set_xlabel("$x$")
ax.set_ylabel("$p(x)$")
ax.set_title("$\mathrm{Initial\ pressure\ profile}$")
ax.legend()

style_plot(ax)
save_and_show("01_initial_pressure_profile")


# ----------------------------------------------------------
# 2. Initial gradient profile
# ----------------------------------------------------------

fig, ax = plt.subplots()

ax.plot(x, g_init, label=r"$g = -\partial_x p$")
ax.axhline(g_crit, linestyle="--", label=r"$g_\mathrm{crit}$")

ax.set_xlabel("$x$")
ax.set_ylabel("$g(x)$")
ax.set_title("$\mathrm{Initial \ gradient \ profile}$")
ax.legend()

style_plot(ax)
save_and_show("02_initial_gradient_profile")


# ----------------------------------------------------------
# 3. Initial flux profile with boundaries marked
# ----------------------------------------------------------

fig, ax = plt.subplots()

ax.plot(x, Q_init, label=r"$Q(x)$", color='k', linewidth=2)

H_ext   = np.cumsum(S_ext_init)   * dx
H_alpha = np.cumsum(P_alpha_init) * dx
H_brem  = np.cumsum(-P_brem_init) * dx
H_total = np.cumsum(S_total_init) * dx

ax.plot(x, H_ext,   linestyle="--", color='C0', label=r"$\int_0^x S_\mathrm{ext}$")
ax.plot(x, H_alpha, linestyle="--", color='C2', label=r"$\int_0^x P_\alpha$")
ax.plot(x, H_brem,  linestyle="--", color='C3', label=r"$-\int_0^x P_\mathrm{brem}$")
ax.plot(x, H_total, linestyle="-.", color='C1', label=r"$\int_0^x S_\mathrm{total}$")

boundaries = transport_params.get("boundaries", [])
for i, xb in enumerate(boundaries):
    ax.axvline(xb, linestyle="--", linewidth=1.5, alpha=0.7,
               label=r"$x = {:.2f}$".format(xb))

ax.set_xlabel("$x$")
ax.set_ylabel("$Q(x)$")
ax.set_title(r"$\mathrm{Initial\ flux\ profile}$")
ax.legend()

style_plot(ax)
save_and_show("03_initial_flux_profile")

# ----------------------------------------------------------
# 3b. Initial flux profile AFTER power balance enforcement
# ----------------------------------------------------------

# Recompute enforced external source, mirroring compute_rhs logic.
# Physics source (S_phys) is never rescaled.

heating_mode   = transport_params.get("heating_mode", "global")
power_balance  = transport_params.get("power_balance", None)
Q_edge_init    = Q_face_init[-1]

S_ext_enforced = S_ext_init.copy()

if heating_mode == "localized" and power_balance is not None:
    total_S_raw  = np.trapezoid(S_ext_init + S_phys_init, x)
    deficit      = power_balance * Q_edge_init - total_S_raw
    edge_sigma   = transport_params.get("edge_sigma", 0.05)
    gauss_shape  = np.exp(-((x - L)**2) / (2 * edge_sigma**2))
    gauss_norm   = np.trapezoid(gauss_shape, x)
    edge_amp     = deficit / gauss_norm if gauss_norm > 0 else 0.0
    S_ext_enforced = S_ext_init + edge_amp * gauss_shape

elif heating_mode == "global" and power_balance is not None:
    total_S_ext  = np.trapezoid(S_ext_init,  x)
    total_S_phys = np.trapezoid(S_phys_init, x)
    target_ext   = power_balance * Q_edge_init - total_S_phys
    if total_S_ext > 0 and target_ext > 0:
        S_ext_enforced = S_ext_init * (target_ext / total_S_ext)

S_total_enforced = S_ext_enforced + S_phys_init

H_total_enforced = np.cumsum(S_total_enforced) * dx
H_ext_enforced   = np.cumsum(S_ext_enforced)   * dx

fig, ax = plt.subplots()
ax.plot(x_face, Q_face_init,    label=r"$Q(x)$ at faces", color='k',
        marker='o', markersize=3, linewidth=2)
ax.plot(x, H_total_enforced,    label=r"$\int_0^x S_\mathrm{total,enforced}$",
        linestyle="-.", color='C1', linewidth=2)
ax.plot(x, H_ext_enforced,      label=r"$\int_0^x S_\mathrm{ext,enforced}$",
        linestyle="--", color='C0', linewidth=1.5)

boundaries = transport_params.get("boundaries", [])
for i, xb in enumerate(boundaries):
    ax.axvline(xb, linestyle="--", linewidth=1.5, alpha=0.7,
               label=r"$x = {:.2f}$".format(xb))

ax.set_xlabel("$x$")
ax.set_ylabel("$Q(x)$")
ax.set_title(r"$\mathrm{Initial\ flux\ profile\ (after\ power\ balance)}$")
ax.legend()
style_plot(ax)
save_and_show("03b_initial_flux_balanced")

# ----------------------------------------------------------
# 3c. Source components before and after power balance
# ----------------------------------------------------------

fig, ax = plt.subplots()

ax.plot(x, S_ext_init,       color='C0', linewidth=2,
        label=r"$S_\mathrm{ext}$ (raw)")
ax.plot(x, S_ext_enforced,   color='C0', linewidth=2, linestyle="--",
        label=r"$S_\mathrm{ext}$ (enforced)")
ax.plot(x, P_alpha_init,     color='C2', linewidth=2,
        label=r"$P_\alpha$ (alpha heating)")
ax.plot(x, -P_brem_init,     color='C3', linewidth=2,
        label=r"$-P_\mathrm{brem}$ (radiation loss)")
ax.plot(x, S_total_enforced, color='C1', linewidth=2, linestyle="-.",
        label=r"$S_\mathrm{total}$ (enforced)")
ax.axhline(0, color='k', linewidth=0.7, linestyle=':')

boundaries = transport_params.get("boundaries", [])
for i, xb in enumerate(boundaries):
    ax.axvline(xb, linestyle="--", linewidth=1.5, alpha=0.7)

ax.set_xlabel("$x$")
ax.set_ylabel("$S(x)$")
ax.set_title(r"$\mathrm{Source\ components\ (before\ and\ after\ power\ balance)}$")
ax.legend(fontsize=11)
style_plot(ax)
save_and_show("03c_source_comparison")

# ==========================================
# Solve
# ==========================================

# --- Power balance mode pre-processing ---
solve_params = dict(transport_params)   # copy so we don't mutate

if power_balance_mode == "continuous":
    # Enforcement runs inside every RHS call — source_fn is the external part only.
    # physics_source is passed separately and added unscaled inside compute_rhs.
    source_fn = source

elif power_balance_mode == "initial_only":
    # Scale factor computed once from initial state, then fixed.
    # Only the external source is scaled; physics terms evolve freely.
    pb_ratio         = transport_params.get("power_balance", 1.0)
    heating_mode_str = transport_params.get("heating_mode", "global")
    total_S0_ext     = np.trapezoid(source(x, p_init), x)      # ∫S_ext dx at t=0
    total_S0_phys    = np.trapezoid(physics_source(x, p_init), x)  # ∫S_phys dx at t=0
    Q_edge0          = Q_face_init[-1]

    if heating_mode_str == "global" and total_S0_ext > 0:
        target_ext = pb_ratio * Q_edge0 - total_S0_phys
        scale_0    = max(target_ext / total_S0_ext, 0.0)
        def source_fn(x, p, _s=scale_0): return _s * base_source(x)
        print(f"[initial_only] External source scale fixed at {scale_0:.4f} (t=0)")
    elif heating_mode_str == "localized":
        # Gaussian fills the gap between total initial source and target
        total_S0 = total_S0_ext + total_S0_phys
        deficit0 = pb_ratio * Q_edge0 - total_S0
        edge_sigma = transport_params.get("edge_sigma", 0.05)
        g_test   = np.exp(-((x - L)**2) / (2 * edge_sigma**2))
        g_norm   = np.trapezoid(g_test, x)
        edge_amp = deficit0 / g_norm if g_norm > 0 else 0.0
        def source_fn(x, p, _ea=edge_amp, _es=edge_sigma, _L=L):
            return base_source(x) + _ea * np.exp(-((x - _L)**2) / (2 * _es**2))
        print(f"[initial_only] Fixed edge Gaussian amplitude = {edge_amp:.4f} (t=0)")
    else:
        source_fn = source
        print("[initial_only] Could not scale source (zero source or flux)")

    # Disable continuous enforcement in RHS
    solve_params["power_balance"] = None

elif power_balance_mode == "free":
    source_fn = source
    solve_params["power_balance"] = None
    print("[free] No power balance enforcement")

else:
    raise ValueError(f"Unknown power_balance_mode '{power_balance_mode}'")

# ==========================================
# Source controller
# ==========================================
# Set controller_mode = None to run in the original form (no controller).
# "schedule" : open-loop ramps defined by core_schedule / edge_schedule below.
# "PI"       : closed-loop; core targets stored energy, edge targets g_edge.
#
# When a controller is active:
#   - it replaces source_fn (so base_source is no longer used directly)
#   - power_balance enforcement inside compute_rhs is disabled (set to None)
# ==========================================

controller_mode = None   # None | "schedule" | "PI"

if controller_mode is None:
    source_controller = None

elif controller_mode == "schedule":
    # Open-loop ramp examples — edit as needed.
    def core_schedule(t):
        """Ramp core heating from 0 to 5.0 over the first 0.1 time units."""
        return min(t / 0.1, 1.0) * 5.0

    def edge_schedule(t):
        """Switch on edge heating at t = 0.2, ramp up over 0.1 time units."""
        return max(t - 0.2, 0.0) / 0.1 * 3.0

    source_controller = SourceController(
        core_config={"mode": "schedule", "schedule": core_schedule, "sigma": 0.25},
        edge_config={"mode": "schedule", "schedule": edge_schedule, "sigma": 0.05},
        L=L,
    )

elif controller_mode == "PI":
    # Set-points derived from initial state.
    W0             = np.trapezoid(p_init, x)
    g_edge_target  = 0.9 * g_crit   # keep edge gradient just below g_crit

    source_controller = SourceController(
        core_config={
            "mode":        "PI",
            "target":      W0,
            "target_type": "stored_energy",
            "A_ff":        3.0,
            "Kp":          2.0,
            "Ki":          10.0,
            "sigma":       0.25,
        },
        edge_config={
            "mode":        "PI",
            "target":      g_edge_target,
            "target_type": "edge_gradient",
            "A_ff":        0.0,
            "Kp":          0.5,
            "Ki":          2.0,
            "sigma":       0.05,
        },
        L=L,
    )

else:
    raise ValueError(f"Unknown controller_mode '{controller_mode}'")

# When a controller is active it manages the source directly; disable the
# built-in power balance enforcement to avoid double-scaling.
if source_controller is not None:
    solve_params["power_balance"] = None
    print(f"[controller] mode='{controller_mode}' — power_balance enforcement disabled")

saved_p = solving_loop(
    p_init,
    dt,
    dx,
    T,
    L,
    p_ped,
    source_fn,
    num_snapshots,
    solve_params,
    physics_source_function=physics_source,
    source_controller=source_controller,
)

times = np.linspace(0, T, len(saved_p))

# ==========================================
# Diagnostics
# ==========================================
total_pressure_time  = []
edge_flux_time       = []
total_S_ext_time     = []
total_P_alpha_time   = []
total_P_brem_time    = []
total_heating_time   = []
max_gradient_time    = []

for snap_i, p in enumerate(saved_p):
    total_pressure_time.append(np.trapezoid(p, x))
    p_x = np.gradient(p, dx)
    Q = flux_function(p_x, x, transport_params)
    edge_flux_time.append(Q[-1])
    # External source: use controller snapshot amplitude when active
    if source_controller is not None:
        _S_ext = np.trapezoid(source_controller.source_at_snapshot(snap_i, x), x)
    else:
        _S_ext = np.trapezoid(source_fn(x, p), x)
    _P_alpha = np.trapezoid(alpha_heating(p, transport_params), x)
    _P_brem  = np.trapezoid(bremsstrahlung(p, transport_params), x)
    total_S_ext_time.append(_S_ext)
    total_P_alpha_time.append(_P_alpha)
    total_P_brem_time.append(_P_brem)
    total_heating_time.append(_S_ext + _P_alpha - _P_brem)
    max_gradient_time.append(np.max(-p_x))

total_pressure_time  = np.array(total_pressure_time)
edge_flux_time       = np.array(edge_flux_time)
total_S_ext_time     = np.array(total_S_ext_time)
total_P_alpha_time   = np.array(total_P_alpha_time)
total_P_brem_time    = np.array(total_P_brem_time)
total_heating_time   = np.array(total_heating_time)
max_gradient_time    = np.array(max_gradient_time)

# ==========================================
# Pressure evolution plot
# ==========================================
fig, ax = plt.subplots()
for i, p in enumerate(saved_p):
    ax.plot(x, p, label=rf"$t={times[i]:.3f}$")
ax.set_title(rf"$\mathrm{{Pressure\ Evolution\ ({branch_label})}}$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$p$")
boundaries = transport_params.get("boundaries", [])

for i, xb in enumerate(boundaries):
    ax.axvline(
        xb,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )
ax.legend(ncol=2)
style_plot(ax)
save_and_show("05_pressure_evolution")

# ==========================================
# Total pressure
# ==========================================
fig, ax = plt.subplots()
ax.plot(times, total_pressure_time)
ax.set_title(rf"$\mathrm{{Total\ Pressure\ \int p\,dx\ ({branch_label})}}$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\int p\,dx$")
style_plot(ax)
save_and_show("06_total_pressure")


# ==========================================
# Track gradient at selected points (MAX 8 curves)
# ==========================================

max_grad_curves = 8

# Evenly spaced locations in outer half (more relevant for pedestal physics)
track_points = np.linspace(0.4, 0.95, max_grad_curves)
track_indices = [np.argmin(np.abs(x - pt)) for pt in track_points]

grad_history = np.zeros((len(track_indices), len(saved_p)))

for t_idx, p in enumerate(saved_p):
    g = -np.gradient(p, dx)
    for i, idx in enumerate(track_indices):
        grad_history[i, t_idx] = g[idx]

fig, ax = plt.subplots()

for i, pt in enumerate(track_points):
    ax.plot(times,
            grad_history[i, :],
            label=rf"$x={pt:.2f}$")

ax.axhline(g_crit,
           linestyle='--',
           label=r"$g_\mathrm{crit}$")

ax.set_title(rf"$\mathrm{{Gradient\ Evolution\ (selected\ points,\ {branch_label})}}$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$g=-\partial_x p$")
ax.legend(ncol=2, fontsize=9)

style_plot(ax)
save_and_show("07_gradient_evolution")

# ==========================================
# Extra diagnostics (controlled by flag)
# ==========================================
extra_plots = True

if extra_plots:
    import matplotlib.cm as cm

    n_colors = 20
    colors = cm.tab20.colors

    # ==========================================================
    # 1. Gradient heatmap g(x,t)
    # ==========================================================
    g_all = np.array([-np.gradient(p, dx) for p in saved_p])

    fig, ax = plt.subplots(figsize=(7,4))
    im = ax.imshow(g_all.T,
                   extent=[0, T, 0, L],
                   aspect='auto',
                   origin='lower',
                   cmap='viridis')
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    
    boundaries = transport_params.get("boundaries", [])

    for i, xb in enumerate(boundaries):
        ax.axhline(
            xb,
            linestyle="--",
            linewidth=1.5,
            color="white",
            alpha=0.8,
        )


    ax.set_title(rf"$\mathrm{{Gradient\ g(x,t)\ ({branch_label})}}$")
    cbar = fig.colorbar(im)
    cbar.set_label(r"$g=-\partial_x p$")
    plt.tight_layout()
    save_and_show("08_gradient_heatmap")

    # ==========================================================
    # 2. Edge flux minus total heating
    # ==========================================================
    fig, ax = plt.subplots()
    ax.plot(times, edge_flux_time - total_heating_time,
            marker='o')
    ax.axhline(0, linestyle='--')
    ax.set_xlabel("t")
    ax.set_ylabel("Edge flux - total heating")
    ax.set_title(rf"$\mathrm{{Power\ Imbalance\ ({branch_label})}}$")
    style_plot(ax)
    save_and_show("09_power_imbalance")

    # ==========================================================
    # 3. Integrated source components vs time
    # ==========================================================
    fig, ax = plt.subplots()
    ax.plot(times, total_S_ext_time,                   color='C0', linewidth=2,
            label=r"$\int S_\mathrm{ext}\,dx$")
    ax.plot(times, total_P_alpha_time,                 color='C2', linewidth=2,
            label=r"$\int P_\alpha\,dx$")
    ax.plot(times, -np.array(total_P_brem_time),       color='C3', linewidth=2,
            label=r"$-\int P_\mathrm{brem}\,dx$")
    ax.plot(times, total_heating_time,                 color='C1', linewidth=2, linestyle="-.",
            label=r"$\int S_\mathrm{total}\,dx$")
    ax.plot(times, edge_flux_time,                     color='k',  linewidth=1.5, linestyle="--",
            label=r"$Q_\mathrm{edge}$")
    ax.set_xlabel("t")
    ax.set_ylabel("Integrated source / edge flux")
    ax.set_title(rf"$\mathrm{{Source\ components\ ({branch_label})}}$")
    ax.legend(fontsize=11)
    style_plot(ax)
    save_and_show("09b_source_components")

    # ==========================================================
    # 4. Local source vs flux (MAX 8 curves total)
    # ==========================================================
    max_curves = 8
    max_locations = max_curves // 2   # each location has Q and S

    track_points = np.linspace(0.2, 0.95, max_locations)
    track_indices = [np.argmin(np.abs(x - pt)) for pt in track_points]

    fig, ax = plt.subplots()

    for i, idx in enumerate(track_indices):
        color_idx = i % n_colors

        Q_local       = []
        S_total_local = []

        for p in saved_p:
            p_x = np.gradient(p, dx)
            Q = flux_function(p_x, x, transport_params)
            Q_local.append(Q[idx])
            S_net = source_fn(x, p)[idx] + physics_source(x, p)[idx]
            S_total_local.append(S_net)

        ax.plot(times, Q_local,
                label=f"Q @ x={x[idx]:.2f}",
                color=colors[color_idx])

        ax.plot(times, S_total_local,
                linestyle="--",
                label=r"$S_\mathrm{tot}$" + f" @ x={x[idx]:.2f}",
                color=colors[color_idx])

    ax.set_xlabel("t")
    ax.set_ylabel("Local flux / net source")
    ax.set_title(rf"$\mathrm{{Local\ Flux\ vs\ Net\ Source\ ({branch_label})}}$")
    ax.legend(ncol=2, fontsize=9)
    style_plot(ax)
    save_and_show("10_local_flux_vs_source")

    # ==========================================================
    # 4. Pressure heatmap p(x,t)
    # ==========================================================
    p_all = np.array(saved_p)

    fig, ax = plt.subplots(figsize=(7,4))
    im = ax.imshow(p_all.T,
                   extent=[0, T, 0, L],
                   aspect='auto',
                   origin='lower',
                   cmap='plasma')
    boundaries = transport_params.get("boundaries", [])

    for i, xb in enumerate(boundaries):
        ax.axhline(
            xb,
            linestyle="--",
            linewidth=1.5,
            color="white",
            alpha=0.8,
        )

    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(rf"$\mathrm{{Pressure\ p(x,t)\ ({branch_label})}}$")
    cbar = fig.colorbar(im)
    cbar.set_label("p")
    plt.tight_layout()
    save_and_show("11_pressure_heatmap")

    # ==========================================================
    # 5. Max gradient and location
    # ==========================================================
    max_g_location = []

    for p in saved_p:
        g = -np.gradient(p, dx)
        max_idx = np.argmax(g)
        max_g_location.append(x[max_idx])

    fig, ax = plt.subplots()
    ax.plot(times, max_gradient_time, label="Max gradient")
    ax.plot(times, max_g_location, label="Location of max g")
    ax.axhline(g_crit, linestyle='--', label="g_crit")
    ax.set_xlabel("t")
    ax.set_ylabel("Max gradient / location")
    ax.set_title(rf"$\mathrm{{Max\ gradient\ and\ location\ ({branch_label})}}$")
    ax.legend()
    style_plot(ax)
    save_and_show("12_max_gradient_and_location")

    # ==========================================================
    # 6. Edge gradient vs source
    # ==========================================================
    edge_gradient = [-np.gradient(p, dx)[-1] for p in saved_p]
    edge_S_total  = [source_fn(x, p)[-1] + physics_source(x, p)[-1] for p in saved_p]

    fig, ax = plt.subplots()
    ax.plot(times, edge_gradient, label=r"$g$ at edge")
    ax.plot(times, edge_S_total,  linestyle="--", label=r"$S_\mathrm{total}$ at edge")
    ax.axhline(g_crit, linestyle='--', label=r"$g_\mathrm{crit}$")
    ax.set_xlabel("t")
    ax.set_ylabel("Edge gradient / net source")
    ax.set_title(rf"$\mathrm{{Edge\ gradient\ vs\ net\ source\ ({branch_label})}}$")
    ax.legend()
    style_plot(ax)
    save_and_show("13_edge_gradient_vs_source")

    # ==========================================================
    # 7. Flux profile vs cumulative source (very informative)
    # ==========================================================
    for snap_idx in [-2, -1]:  # last 2 snapshots only
        p = saved_p[snap_idx]
        p_x = np.gradient(p, dx)
        Q = flux_function(-p_x, x, transport_params)

        F_ext   = np.array([np.trapezoid(source_fn(x[:j+1], p[:j+1]),     x[:j+1]) for j in range(len(x))])
        F_alpha = np.array([np.trapezoid(alpha_heating(p[:j+1], transport_params),  x[:j+1]) for j in range(len(x))])
        F_brem  = np.array([np.trapezoid(bremsstrahlung(p[:j+1], transport_params), x[:j+1]) for j in range(len(x))])
        F_total = F_ext + F_alpha - F_brem

        fig, ax = plt.subplots()
        ax.plot(x, Q,       color='k',  linewidth=2, label=r"$Q(x)$")
        ax.plot(x, F_total, color='C1', linewidth=2, linestyle="-.", label=r"$\int_0^x S_\mathrm{total}$")
        ax.plot(x, F_ext,   color='C0', linewidth=1.5, linestyle="--", label=r"$\int_0^x S_\mathrm{ext}$")
        ax.plot(x, F_alpha, color='C2', linewidth=1.5, linestyle="--", label=r"$\int_0^x P_\alpha$")
        ax.plot(x, -F_brem, color='C3', linewidth=1.5, linestyle="--", label=r"$-\int_0^x P_\mathrm{brem}$")
        ax.set_xlabel("x")
        ax.set_ylabel("Flux / cumulative source")
        ax.set_title(f"Flux balance at $t={times[snap_idx]:.3f}$")
        ax.legend(fontsize=11)
        style_plot(ax)
        save_and_show(f"14_flux_balance_t{snap_idx}")

    # ==========================================================
    # 8. Effective diffusivity profile (final snapshot)
    # ==========================================================

    p = saved_p[-1]
    p_x = np.gradient(p, dx)
    g = -p_x

    # Small perturbation for numerical derivative
    eps = 1e-6

    Q_plus  = flux_function(g + eps, x, transport_params)
    Q_minus = flux_function(g - eps, x, transport_params)

    chi_eff = (Q_plus - Q_minus) / (2 * eps)

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    fig, ax = plt.subplots()

    ax.plot(x, chi_eff, label=r"$\chi_{\mathrm{eff}}(x)$")

    # Plot boundaries if present
    boundaries = transport_params.get("boundaries", [])

    for i, xb in enumerate(boundaries):
        ax.axvline(
            xb,
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
            label=r"$x={:.2f}$".format(xb) if i == 0 else None
        )

    ax.set_xlabel("x")
    ax.set_ylabel(r"$\chi_{\mathrm{eff}}$")
    ax.set_title("Effective diffusivity profile (final state)")

    if boundaries:
        ax.legend()

    style_plot(ax)
    save_and_show("15_effective_diffusivity")

    # ==========================================================
    # 9. Controller diagnostics (only shown when controller active)
    # ==========================================================

    if source_controller is not None:
        t_ctrl = np.array(source_controller.t_hist)

        # ---- Plot 16: heating amplitudes vs time ----
        fig, ax = plt.subplots()

        if source_controller.core_mode != "off":
            ax.plot(t_ctrl, source_controller.A_core_hist,
                    color='C0', linewidth=1.5, label=r"$A_\mathrm{core}(t)$")
        if source_controller.edge_mode != "off":
            ax.plot(t_ctrl, source_controller.A_edge_hist,
                    color='C1', linewidth=1.5, label=r"$A_\mathrm{edge}(t)$")

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"Heating amplitude")
        ax.set_title(r"$\mathrm{Controller:\ heating\ amplitudes}$")
        ax.legend()
        style_plot(ax)
        save_and_show("16_controller_amplitudes")

        # ---- Plot 17: observables vs targets ----
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax_c, ax_e = axes

        # Core observable
        if source_controller.core_mode != "off":
            ax_c.plot(t_ctrl, source_controller.obs_core_hist,
                      color='C0', linewidth=1.5, label=r"observed")
            if source_controller.core_target is not None:
                ax_c.axhline(source_controller.core_target,
                             color='C0', linestyle='--', linewidth=1.2,
                             label=r"target")
            ax_c.set_xlabel(r"$t$")
            ax_c.set_ylabel(source_controller.core_target_type.replace("_", " "))
            ax_c.set_title(r"$\mathrm{Core\ channel}$")
            ax_c.legend()
            style_plot(ax_c)
        else:
            ax_c.set_visible(False)

        # Edge observable
        if source_controller.edge_mode != "off":
            ax_e.plot(t_ctrl, source_controller.obs_edge_hist,
                      color='C1', linewidth=1.5, label=r"observed")
            if source_controller.edge_target is not None:
                ax_e.axhline(source_controller.edge_target,
                             color='C1', linestyle='--', linewidth=1.2,
                             label=r"target")
            ax_e.axhline(g_crit, color='k', linestyle=':', linewidth=1.0,
                         label=r"$g_\mathrm{crit}$")
            ax_e.set_xlabel(r"$t$")
            ax_e.set_ylabel(source_controller.edge_target_type.replace("_", " "))
            ax_e.set_title(r"$\mathrm{Edge\ channel}$")
            ax_e.legend()
            style_plot(ax_e)
        else:
            ax_e.set_visible(False)

        plt.suptitle(r"$\mathrm{Controller:\ observables\ vs\ targets}$", fontsize=14)
        plt.tight_layout()
        save_and_show("17_controller_observables")

