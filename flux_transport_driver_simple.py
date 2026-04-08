#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from flux_transport_model_simple import flux_function, solving_loop, _get_bcs, _apply_bc_to_profile

# ==========================================
# Profile shape functions
# ==========================================

def tanh_profile(x, L, p_core, p_ped, x_sym=0.55, delta=0.25):
    """
    Smooth tokamak-like profile.
    x_sym : normalised transition midpoint (fraction of L).
    delta : transition half-width (fraction of L).
    Exactly p_core at x=0, p_ped at x=L.
    """
    xi = x / L
    s  = 0.5 * (1.0 - np.tanh((xi - x_sym) / delta))
    s0 = 0.5 * (1.0 - np.tanh((0.0        - x_sym) / delta))
    s1 = 0.5 * (1.0 - np.tanh((1.0        - x_sym) / delta))
    return p_ped + (p_core - p_ped) * (s - s1) / (s0 - s1)


def plateau_profile(x, L, p_core, p_ped, x_start=0.3, x_end=0.7, delta_ramp=0.05):
    """
    Profile with approximately constant absolute gradient g = -dp/dx over
    [x_start, x_end] (in normalised coordinates) and flat outside.
    Amplitude is set automatically so that p(0) = p_core and p(L) = p_ped.
    x_start, x_end : normalised positions (fraction of L).
    delta_ramp     : smoothness of ramp transitions (fraction of L).
    """
    xi = x / L
    window = 0.5 * (np.tanh((xi - x_start) / delta_ramp)
                  - np.tanh((xi - x_end)   / delta_ramp))
    # Integrate window to get the shape of p
    dxi   = xi[1] - xi[0]
    cum_w = np.cumsum(window) * dxi
    total = cum_w[-1]
    if total > 0:
        p = p_core - cum_w * (p_core - p_ped) / total
    else:
        p = np.full_like(x, p_core)
    return p


def power_law_profile(x, L, p_core, p_ped, m=2):
    """Power-law: p = p_ped + A*(1 - (x/L)^m)."""
    return p_ped + (p_core - p_ped) * (1.0 - (x / L)**m)


def build_profile(x, L, p_core, p_ped, profile_type, target_g, params):
    """
    Build an initial pressure profile of the chosen type, then scale its
    amplitude so that max(g) = target_g.  p_ped is preserved exactly.
    """
    pt = profile_type
    if pt == "power_law":
        p = power_law_profile(x, L, p_core, p_ped, m=params.get("m", 2))
    elif pt == "tanh":
        p = tanh_profile(x, L, p_core, p_ped,
                         x_sym=params.get("x_sym", 0.55),
                         delta=params.get("delta", 0.25))
    elif pt == "plateau":
        p = plateau_profile(x, L, p_core, p_ped,
                            x_start=params.get("x_start", 0.3),
                            x_end=params.get("x_end",   0.7),
                            delta_ramp=params.get("delta_ramp", 0.05))
    else:
        raise ValueError(f"Unknown profile_type '{pt}'")

    # Scale amplitude to hit target_g
    g_raw = -np.gradient(p, x[1] - x[0])
    g_max = np.max(g_raw)
    if g_max > 0:
        p = p_ped + (p - p_ped) * (target_g / g_max)
    return p

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
    import os
    plots_dir = "simple_plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    filepath = os.path.join(plots_dir, f"{filename}.pdf")
    plt.savefig(filepath, format="pdf", bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.show()

# ==========================================
# Domain
# ==========================================
L  = 1.0
dx = 0.005
dt = 1e-6
T  = 0.5
num_snapshots = 8

x = np.linspace(0, L, int(L / dx))

# ==========================================
# Branch selection
# ==========================================
START_ON    = "supercritical"   # "subcritical" or "supercritical"
branch_label = START_ON

# ==========================================
# Power balance mode
# ==========================================
# "continuous"   : rescale source at every RHS call
# "initial_only" : scale once at t=0 to match initial edge flux, then fixed
# "free"         : no enforcement — raw Gaussian amplitude throughout
power_balance_mode = "initial_only"

# ==========================================
# Transport parameters
# ==========================================
transport_params = {
    # Transport strength
    "chi0":     10.0,
    "chi_core": 10.0,
    "chi_RR":   0.05,

    # Nonlinear model
    "g_c":     4.0,

    # Stiff core model
    "g_stiff": 3.0,
    "n_stiff": 2,

    # Spatial structure
    "boundaries":  [],
    "deltas":      [],
    "flux_models": ["nl"],

    "nu4": 0.0,

    # Power balance enforcement
    # heating_mode = "global"    : rescale core Gaussian uniformly to hit target
    # heating_mode = "localized" : add an edge Gaussian to close the deficit
    #   (useful for L-H transition experiments — drives the edge gradient up)
    "heating_mode":  "global",
    "power_balance": 0.8,
    "edge_sigma":    0.05,   # width of edge Gaussian (localized mode only)

    # MHD stiff cliff (set chi_MHD = 0 to disable)
    # g_MHD set below after g_c is extracted
    "chi_MHD": 0.0,
}

# ==========================================
# Derived quantities
# ==========================================
g_c    = transport_params["g_c"]
g_crit = g_c / 2.0

transport_params["g_MHD"] = 0.9 * g_c   # cliff onset; only active if chi_MHD > 0

# ==========================================
# Initial profile
# ==========================================
p_core = 2.0
p_ped  = 1.0

# Profile shape: "power_law" (default), "tanh", or "plateau"
profile_type = "power_law"

# --- power_law parameters ---
#   p = p_ped + A * (1 - (x/L)^m);  higher m = flatter core, steeper edge
profile_params = {"m": 2}

# --- tanh parameters (used when profile_type = "tanh") ---
#   x_sym : normalised transition midpoint (fraction of L)
#   delta : transition half-width (fraction of L); smaller = steeper
# profile_params = {"x_sym": 0.55, "delta": 0.25}

# --- plateau parameters (used when profile_type = "plateau") ---
#   Constant g = -dp/dx in [x_start, x_end] (normalised), flat outside.
# profile_params = {"x_start": 0.3, "x_end": 0.7, "delta_ramp": 0.05}

if START_ON == "subcritical":
    target_g = 0.6 * g_crit
elif START_ON == "supercritical":
    target_g = 1.8 * g_crit
else:
    raise ValueError("START_ON must be 'subcritical' or 'supercritical'")

p_init = build_profile(x, L, p_core, p_ped, profile_type, target_g, profile_params)
g_init = -np.gradient(p_init, dx)
print(f"Profile type  = {profile_type}")
print(f"Initial max g = {np.max(g_init):.4f}")
print(f"Critical g    = {g_crit:.4f}")

# ==========================================
# Source — fixed Gaussian, constant in time
# ==========================================
S0    = 5.0
sigma = 0.25 * L

def base_source(x):
    return S0 * np.exp(-(x**2) / sigma**2)

def source(x, p):
    """Fixed-amplitude Gaussian. Scaled by power balance if mode = continuous."""
    return base_source(x)

# ==========================================
# Initial flux and power balance
# ==========================================
core_bc, edge_bc = _get_bcs(transport_params, p_ped)
p_bc = p_init.copy()
_apply_bc_to_profile(p_bc, dx, core_bc, edge_bc)

g_face       = -(p_bc[1:] - p_bc[:-1]) / dx
x_face       = 0.5 * (x[1:] + x[:-1])
Q_face_init  = flux_function(g_face, x_face, transport_params)
Q_init       = flux_function(-np.gradient(p_init, dx), x, transport_params)

initial_edge_flux = Q_face_init[-1]
initial_S         = np.trapezoid(source(x, p_init), x)

print(f"Initial source integral = {initial_S:.4f}")
print(f"Initial edge flux       = {initial_edge_flux:.4f}")
print(f"Initial balance         = {initial_edge_flux - initial_S:.4f}")

# ==========================================
# INITIAL DIAGNOSTIC PLOTS
# ==========================================

# 1. Initial pressure profile
fig, ax = plt.subplots()
ax.plot(x, p_init, label="Initial pressure")
supercrit_mask = g_init > g_crit
if np.any(supercrit_mask):
    ax.scatter(x[supercrit_mask], p_init[supercrit_mask],
               color="red", s=20, label=r"$g > g_\mathrm{crit}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$p(x)$")
ax.set_title(r"$\mathrm{Initial\ pressure\ profile}$")
ax.legend()
style_plot(ax)
save_and_show("01_initial_pressure")

# 2. Initial gradient
fig, ax = plt.subplots()
ax.plot(x, g_init, label=r"$g = -\partial_x p$")
ax.axhline(g_crit, linestyle="--", label=r"$g_\mathrm{crit}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$g(x)$")
ax.set_title(r"$\mathrm{Initial\ gradient\ profile}$")
ax.legend()
style_plot(ax)
save_and_show("02_initial_gradient")

# 3. Initial flux and cumulative source
fig, ax = plt.subplots()
ax.plot(x, Q_init, color='k', linewidth=2, label=r"$Q(x)$")
ax.plot(x, np.cumsum(source(x, p_init)) * dx,
        linestyle="--", color='C0', label=r"$\int_0^x S\,dx$")
ax.set_xlabel("$x$")
ax.set_ylabel("$Q(x)$")
ax.set_title(r"$\mathrm{Initial\ flux\ and\ cumulative\ source}$")
ax.legend()
style_plot(ax)
save_and_show("03_initial_flux")

# ==========================================
# Power balance pre-processing
# ==========================================
solve_params = dict(transport_params)

if power_balance_mode == "continuous":
    source_fn = source

elif power_balance_mode == "initial_only":
    pb_ratio    = transport_params.get("power_balance", 1.0)
    total_S0    = np.trapezoid(source(x, p_init), x)
    Q_edge0     = Q_face_init[-1]

    heating_mode_str = transport_params.get("heating_mode", "global")

    if heating_mode_str == "global" and total_S0 > 0:
        scale_0 = max(pb_ratio * Q_edge0 / total_S0, 0.0)
        def source_fn(x, p, _s=scale_0): return _s * base_source(x)
        print(f"[initial_only] Source scale fixed at {scale_0:.4f}")
    elif heating_mode_str == "localized":
        deficit0   = pb_ratio * Q_edge0 - total_S0
        edge_sigma = transport_params.get("edge_sigma", 0.05)
        g_test     = np.exp(-((x - L)**2) / (2 * edge_sigma**2))
        g_norm     = np.trapezoid(g_test, x)
        edge_amp   = deficit0 / g_norm if g_norm > 0 else 0.0
        def source_fn(x, p, _ea=edge_amp, _es=edge_sigma, _L=L):
            return base_source(x) + _ea * np.exp(-((x - _L)**2) / (2 * _es**2))
        print(f"[initial_only] Edge Gaussian amplitude fixed at {edge_amp:.4f}")
    else:
        source_fn = source

    solve_params["power_balance"] = None

elif power_balance_mode == "free":
    source_fn = source
    solve_params["power_balance"] = None
    print("[free] No power balance enforcement")

else:
    raise ValueError(f"Unknown power_balance_mode '{power_balance_mode}'")

# ==========================================
# Pre-solve plots: adjusted source
# ==========================================

S_raw      = source(x, p_init)          # before any enforcement
S_enforced = source_fn(x, p_init)       # after enforcement (scale or edge Gaussian)

H_raw      = np.cumsum(S_raw)      * dx
H_enforced = np.cumsum(S_enforced) * dx

# 03b. Q(x) vs cumulative enforced source (flux balance check)
fig, ax = plt.subplots()
ax.plot(x_face, Q_face_init, color='k', linewidth=2,
        marker='o', markersize=3, label=r"$Q(x)$ at faces")
ax.plot(x, H_enforced, color='C0', linewidth=2, linestyle="--",
        label=r"$\int_0^x S_\mathrm{enforced}\,dx$")
ax.set_xlabel("$x$")
ax.set_ylabel("$Q(x)$")
ax.set_title(r"$\mathrm{Initial\ flux\ vs\ enforced\ source}$")
ax.legend()
style_plot(ax)
save_and_show("03b_flux_enforced_source")

# 03c. Raw vs enforced source profile
fig, ax = plt.subplots()
ax.plot(x, S_raw,      color='C0', linewidth=2,
        label=r"$S$ (raw)")
ax.plot(x, S_enforced, color='C0', linewidth=2, linestyle="--",
        label=r"$S$ (enforced)")
ax.axhline(0, color='k', linewidth=0.7, linestyle=':')
ax.set_xlabel("$x$")
ax.set_ylabel("$S(x)$")
ax.set_title(r"$\mathrm{Source\ before\ and\ after\ power\ balance}$")
ax.legend()
style_plot(ax)
save_and_show("03c_source_comparison")

# ==========================================
# Solve
# ==========================================
saved_p = solving_loop(
    p_init, dt, dx, T, L, p_ped,
    source_fn, num_snapshots, solve_params,
)

times = np.linspace(0, T, len(saved_p))

# ==========================================
# Diagnostics
# ==========================================
total_pressure_time = []
edge_flux_time      = []
total_source_time   = []
max_gradient_time   = []

for p in saved_p:
    total_pressure_time.append(np.trapezoid(p, x))
    p_x = np.gradient(p, dx)
    Q   = flux_function(-p_x, x, transport_params)
    edge_flux_time.append(Q[-1])
    total_source_time.append(np.trapezoid(source_fn(x, p), x))
    max_gradient_time.append(np.max(-p_x))

total_pressure_time = np.array(total_pressure_time)
edge_flux_time      = np.array(edge_flux_time)
total_source_time   = np.array(total_source_time)
max_gradient_time   = np.array(max_gradient_time)

# ==========================================
# Evolution plots
# ==========================================

# 4. Pressure evolution
fig, ax = plt.subplots()
for i, p in enumerate(saved_p):
    ax.plot(x, p, label=rf"$t={times[i]:.3f}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$p$")
ax.set_title(rf"$\mathrm{{Pressure\ evolution\ ({branch_label})}}$")
ax.legend(ncol=2)
style_plot(ax)
save_and_show("04_pressure_evolution")

# 5. Total pressure vs time
fig, ax = plt.subplots()
ax.plot(times, total_pressure_time)
ax.set_xlabel("$t$")
ax.set_ylabel(r"$\int p\,dx$")
ax.set_title(rf"$\mathrm{{Total\ pressure\ ({branch_label})}}$")
style_plot(ax)
save_and_show("05_total_pressure")

# 6. Edge flux and total source vs time
fig, ax = plt.subplots()
ax.plot(times, edge_flux_time,    label=r"$Q_\mathrm{edge}$")
ax.plot(times, total_source_time, linestyle="--", label=r"$\int S\,dx$")
ax.set_xlabel("$t$")
ax.set_ylabel("Power")
ax.set_title(rf"$\mathrm{{Power\ balance\ ({branch_label})}}$")
ax.legend()
style_plot(ax)
save_and_show("06_power_balance")

# 7. Gradient evolution at selected points
track_points  = np.linspace(0.4, 0.95, 8)
track_indices = [np.argmin(np.abs(x - pt)) for pt in track_points]
grad_history  = np.zeros((len(track_indices), len(saved_p)))
for t_idx, p in enumerate(saved_p):
    g = -np.gradient(p, dx)
    for i, idx in enumerate(track_indices):
        grad_history[i, t_idx] = g[idx]

fig, ax = plt.subplots()
for i, pt in enumerate(track_points):
    ax.plot(times, grad_history[i, :], label=rf"$x={pt:.2f}$")
ax.axhline(g_crit, linestyle='--', label=r"$g_\mathrm{crit}$")
ax.set_xlabel("$t$")
ax.set_ylabel(r"$g = -\partial_x p$")
ax.set_title(rf"$\mathrm{{Gradient\ evolution\ ({branch_label})}}$")
ax.legend(ncol=2, fontsize=9)
style_plot(ax)
save_and_show("07_gradient_evolution")

# 8. Gradient heatmap
g_all = np.array([-np.gradient(p, dx) for p in saved_p])
fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(g_all.T, extent=[0, T, 0, L], aspect='auto',
               origin='lower', cmap='viridis')
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
ax.set_title(rf"$\mathrm{{Gradient\ g(x,t)\ ({branch_label})}}$")
fig.colorbar(im).set_label(r"$g = -\partial_x p$")
plt.tight_layout()
save_and_show("08_gradient_heatmap")

# 9. Pressure heatmap
p_all = np.array(saved_p)
fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(p_all.T, extent=[0, T, 0, L], aspect='auto',
               origin='lower', cmap='plasma')
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
ax.set_title(rf"$\mathrm{{Pressure\ p(x,t)\ ({branch_label})}}$")
fig.colorbar(im).set_label("$p$")
plt.tight_layout()
save_and_show("09_pressure_heatmap")

# 10. Flux balance at final snapshot
p    = saved_p[-1]
p_x  = np.gradient(p, dx)
Q    = flux_function(-p_x, x, transport_params)
F_S  = np.array([np.trapezoid(source_fn(x[:j+1], p[:j+1]), x[:j+1]) for j in range(len(x))])

fig, ax = plt.subplots()
ax.plot(x, Q,  color='k',  linewidth=2, label=r"$Q(x)$")
ax.plot(x, F_S, color='C0', linewidth=1.5, linestyle="--",
        label=r"$\int_0^x S\,dx$")
ax.set_xlabel("$x$")
ax.set_ylabel("Flux / cumulative source")
ax.set_title(rf"$\mathrm{{Flux\ balance\ at\ }}t={times[-1]:.3f}$")
ax.legend()
style_plot(ax)
save_and_show("10_flux_balance_final")

# 11. Effective diffusivity at final state
eps     = 1e-6
g_final = -p_x
Q_plus  = flux_function(g_final + eps, x, transport_params)
Q_minus = flux_function(g_final - eps, x, transport_params)
chi_eff = (Q_plus - Q_minus) / (2 * eps)

fig, ax = plt.subplots()
ax.plot(x, chi_eff, label=r"$\chi_\mathrm{eff}(x)$")
ax.set_xlabel("$x$")
ax.set_ylabel(r"$\chi_\mathrm{eff}$")
ax.set_title("Effective diffusivity (final state)")
style_plot(ax)
save_and_show("11_effective_diffusivity")
