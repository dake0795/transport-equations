#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from flux_transport_model import flux_function, solving_loop

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
# Feedback
# ==========================================
alpha = 0.1


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
    "power_balance": 1.0,      # ratio of integrated source to edge flux
    "edge_sigma": 0.05,        # width of edge-localized Gaussian heating (for localized mode)
}

# ==========================================
# Initial profile
# ==========================================

g_c = transport_params["g_c"]

g_crit = g_c / 2.0

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
    return base_source(x) * (1.0 + alpha * p)

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

initial_heating = np.trapezoid(source(x, p_init), x)
initial_edge_flux = Q_face_init[-1]

print("Initial total heating =", initial_heating)
print("Initial edge flux =", initial_edge_flux)
print("Initial balance (edge_flux - heating) =", initial_edge_flux - initial_heating)

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

ax.plot(x, Q_init, label="Initial Q(x)")

# Source
S = source(x, p_init)

# Cumulative integrated source
H = np.cumsum(S) * dx

ax.plot(x, H, linestyle="--", label=r"$\int_0^x \, S(x') \, \mathrm{d}x'$")
# Plot all region boundaries if they exist
boundaries = transport_params.get("boundaries", [])

for i, xb in enumerate(boundaries):
    ax.axvline(
        xb,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=r"$x = {:.2f}$".format(xb) 
    )

ax.set_xlabel("$x$")
ax.set_ylabel("$Q(x)$")
ax.set_title("$\mathrm{Initial \ flux \ profile}$")
ax.legend()

style_plot(ax)
save_and_show("03_initial_flux_profile")

# ----------------------------------------------------------
# 3b. Initial flux profile AFTER power balance enforcement
# ----------------------------------------------------------

# Recompute source after power balance is applied
# (mimicking what compute_rhs does)

S_init_raw = source(x, p_init)

# Apply power balance enforcement
heating_mode = transport_params.get("heating_mode", "global")
power_balance = transport_params.get("power_balance", None)

S_init_enforced = S_init_raw.copy()

if heating_mode == "localized" and power_balance is not None:
    Q_edge_init = Q_face_init[-1]
    total_S_raw = np.trapezoid(S_init_raw, x)

    required_integral = power_balance * Q_edge_init
    deficit = required_integral - total_S_raw

    L_val = L
    edge_sigma = transport_params.get("edge_sigma", 0.05)

    # Compute actual integral of Gaussian over domain [0, L]
    gaussian_test = np.exp(-((x - L_val)**2) / (2 * edge_sigma**2))
    gaussian_norm = np.trapezoid(gaussian_test, x)

    if gaussian_norm > 0:
        edge_amp = deficit / gaussian_norm
    else:
        edge_amp = 0

    gaussian_edge = edge_amp * np.exp(-((x - L_val)**2) / (2 * edge_sigma**2))
    S_init_enforced = S_init_enforced + gaussian_edge

elif heating_mode == "global" and power_balance is not None:
    Q_edge_init = Q_face_init[-1]
    total_S_raw = np.trapezoid(S_init_raw, x)
    if total_S_raw > 0 and Q_edge_init > 0:
        S_init_enforced = S_init_enforced * (power_balance * Q_edge_init / total_S_raw)

# Cumulative enforced source
H_enforced = np.cumsum(S_init_enforced) * dx

fig, ax = plt.subplots()

# Plot face fluxes at face locations for accuracy
ax.plot(x_face, Q_face_init, label="Initial Q(x) at faces", marker='o', markersize=3)
# Also show cumulative source at cell centers for reference
ax.plot(x, H_enforced, linestyle="--", label=r"$\int_0^x \, S_{\mathrm{enforced}}(x') \, \mathrm{d}x'$")

boundaries = transport_params.get("boundaries", [])

for i, xb in enumerate(boundaries):
    ax.axvline(
        xb,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=r"$x = {:.2f}$".format(xb)
    )

ax.set_xlabel("$x$")
ax.set_ylabel("$Q(x)$")
ax.set_title(r"$\mathrm{Initial\ flux\ profile\ (after\ power\ balance)}$")
ax.legend()

style_plot(ax)
save_and_show("03b_initial_flux_balanced")

# ----------------------------------------------------------
# 3c. Source before and after power balance enforcement
# ----------------------------------------------------------

fig, ax = plt.subplots()

ax.plot(x, S_init_raw, label="Original source", linewidth=2)
ax.plot(x, S_init_enforced, linestyle="--", label="Enforced source", linewidth=2)

boundaries = transport_params.get("boundaries", [])

for i, xb in enumerate(boundaries):
    ax.axvline(
        xb,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )

ax.set_xlabel("$x$")
ax.set_ylabel("$S(x)$")
ax.set_title(r"$\mathrm{Source\ profile\ (before\ and\ after\ power\ balance)}$")
ax.legend()

style_plot(ax)
save_and_show("03c_source_comparison")

# ==========================================
# Solve
# ==========================================
saved_p = solving_loop(
    p_init,
    dt,
    dx,
    T,
    L,
    p_ped,
    source,
    num_snapshots,
    transport_params
)

times = np.linspace(0, T, len(saved_p))

# ==========================================
# Diagnostics
# ==========================================
total_pressure_time = []
edge_flux_time = []
total_heating_time = []
max_gradient_time = []

for p in saved_p:
    total_pressure_time.append(np.trapezoid(p, x))
    p_x = np.gradient(p, dx)
    Q = flux_function(p_x,x, transport_params)
    edge_flux_time.append(Q[-1])
    total_heating_time.append(np.trapezoid(source(x, p), x))
    max_gradient_time.append(np.max(-p_x))

total_pressure_time = np.array(total_pressure_time)
edge_flux_time = np.array(edge_flux_time)
total_heating_time = np.array(total_heating_time)
max_gradient_time = np.array(max_gradient_time)

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
    # 3. Local source vs flux (MAX 8 curves total)
    # ==========================================================
    max_curves = 8
    max_locations = max_curves // 2   # each location has Q and S

    track_points = np.linspace(0.2, 0.95, max_locations)
    track_indices = [np.argmin(np.abs(x - pt)) for pt in track_points]

    fig, ax = plt.subplots()

    for i, idx in enumerate(track_indices):
        color_idx = i % n_colors

        Q_local = []
        S_local = []

        for p in saved_p:
            p_x = np.gradient(p, dx)
            Q = flux_function(p_x, x, transport_params)
            Q_local.append(Q[idx])
            S_local.append(source(x, p)[idx])

        ax.plot(times, Q_local,
                label=f"Q @ x={x[idx]:.2f}",
                color=colors[color_idx])

        ax.plot(times, S_local,
                linestyle="--",
                label=f"S @ x={x[idx]:.2f}",
                color=colors[color_idx])

    ax.set_xlabel("t")
    ax.set_ylabel("Local flux / source")
    ax.set_title(rf"$\mathrm{{Local\ Flux\ vs\ Source\ ({branch_label})}}$")
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
    edge_source = [source(x, p)[-1] for p in saved_p]

    fig, ax = plt.subplots()
    ax.plot(times, edge_gradient, label="g at edge")
    ax.plot(times, edge_source, linestyle="--", label="S at edge")
    ax.axhline(g_crit, linestyle='--', label="g_crit")
    ax.set_xlabel("t")
    ax.set_ylabel("Edge gradient / source")
    ax.set_title(rf"$\mathrm{{Edge\ gradient\ vs\ source\ ({branch_label})}}$")
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

        F = np.array([
            np.trapezoid(source(x[:j+1], p[:j+1]), x[:j+1])
            for j in range(len(x))
        ])

        fig, ax = plt.subplots()
        ax.plot(x, Q, label="Q(x)")
        ax.plot(x, F, linestyle="--", label=r"$\int_0^x S dx$")
        ax.set_xlabel("x")
        ax.set_ylabel("Flux / cumulative source")
        ax.set_title(f"Flux balance at t={times[snap_idx]:.3f}")
        ax.legend()
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

