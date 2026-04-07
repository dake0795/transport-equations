#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from flux_transport_model_coupled import heat_flux_function, particle_flux_function, solving_loop_coupled

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
# Transport parameters
# ==========================================

transport_params = {
    # --- Heat transport (pressure) ---
    "chi0": 10.0,
    "chi_core": 10.0,
    "chi_RR": 0.05,

    "g_c": 4.0,
    "g_stiff": 3,
    "n_stiff": 2,

    # --- Particle transport (density) - same NL model as pressure ---
    "chi0_n": 1.0,         # NL particle transport coefficient
    "chi_core_n": 1.0,     # Core stiff particle transport
    "chi_RR_n": 0.01,      # Background diffusion for density

    "g_c_n": 2.0,          # Critical density gradient
    "g_stiff_n": 1.5,      # Onset of stiff transport (density)
    "n_stiff_n": 2,        # Stiffness exponent (density)

    "V_n": 0.0,            # Convection velocity for density (pinch)

    # --- Spatial structure ---
    "boundaries": [],
    "deltas": [],
    "flux_models": ["nl"],
    "flux_models_n": ["nl"],  # Can differ from pressure flux model

    "nu4": 0,

    # --- Power balance ---
    "heating_mode": "global",
    "power_balance": 1.0,
    "edge_sigma": 0.05,
}

# ==========================================
# Initial profiles
# ==========================================

g_c = transport_params["g_c"]
g_crit = g_c / 2.0

p_core = 2.0
p_ped = 1.0

n_core = 1.0
n_ped = 0.5

m = 2

def base_profile(x, m):
    return (1 - (x/L)**m)

p_shape = base_profile(x, m)
n_shape = base_profile(x, m)

p_x_shape = np.gradient(p_shape, dx)
g_p_shape = -p_x_shape
max_g_p_shape = np.max(g_p_shape)

target_g = 1.8 * g_crit

scale_p = target_g / max_g_p_shape
p_init = p_ped + scale_p * (p_core - p_ped) * p_shape

# Density initialized proportional to pressure (in ideal gas limit)
# T ~ p/n, so n ~ p/T; if we use n proportional to p
scale_n = 2.0
n_init = n_ped + scale_n * (n_core - n_ped) * n_shape

g_p_init = -np.gradient(p_init, dx)
g_n_init = -np.gradient(n_init, dx)

print("Initial max grad_p =", np.max(g_p_init))
print("Initial max grad_n =", np.max(g_n_init))
print("Critical gradient =", g_crit)

# ==========================================
# Source functions
# ==========================================

alpha_p = 0.1
alpha_n = 0.05

def base_source_p(x):
    S0 = 5.0
    sigma = 0.25 * L
    return S0 * np.exp(-(x**2)/(sigma**2))

def base_source_n(x):
    S0 = 2.0
    sigma = 0.25 * L
    return S0 * np.exp(-(x**2)/(sigma**2))

def source_p(x, p, n):
    """Pressure source with feedback."""
    return base_source_p(x) * (1.0 + alpha_p * p)

def source_n(x, p, n):
    """Density source with feedback."""
    return base_source_n(x) * (1.0 + alpha_n * n)

# ==========================================
# Initial diagnostics
# ==========================================

print("\n" + "="*60)
print("INITIAL DIAGNOSTIC PLOTS")
print("="*60 + "\n")

# Pressure profile
fig, ax = plt.subplots()
ax.plot(x, p_init, label="Initial pressure")
ax.set_xlabel("$x$")
ax.set_ylabel("$p(x)$")
ax.set_title("$\mathrm{Initial\ pressure\ profile}$")
ax.legend()
style_plot(ax)
save_and_show("01_initial_pressure")

# Density profile
fig, ax = plt.subplots()
ax.plot(x, n_init, label="Initial density")
ax.set_xlabel("$x$")
ax.set_ylabel("$n(x)$")
ax.set_title("$\mathrm{Initial\ density\ profile}$")
ax.legend()
style_plot(ax)
save_and_show("02_initial_density")

# Gradients
fig, ax = plt.subplots()
ax.plot(x, g_p_init, label=r"$g_p = -\partial_x p$")
ax.plot(x, g_n_init, label=r"$g_n = -\partial_x n$")
ax.axhline(g_crit, linestyle="--", label=r"$g_\mathrm{crit}$")
ax.set_xlabel("$x$")
ax.set_ylabel("Gradient")
ax.set_title("$\mathrm{Initial\ gradients}$")
ax.legend()
style_plot(ax)
save_and_show("03_initial_gradients")

# Sources
fig, ax = plt.subplots()
S_p_init = source_p(x, p_init, n_init)
S_n_init = source_n(x, p_init, n_init)
ax.plot(x, S_p_init, label="$S_p$ (pressure source)")
ax.plot(x, S_n_init, label="$S_n$ (density source)")
ax.set_xlabel("$x$")
ax.set_ylabel("Source")
ax.set_title("$\mathrm{Initial\ sources}$")
ax.legend()
style_plot(ax)
save_and_show("04_initial_sources")

# ==========================================
# Solve coupled system
# ==========================================

print("\n" + "="*60)
print("SOLVING COUPLED SYSTEM")
print("="*60 + "\n")

saved_p, saved_n = solving_loop_coupled(
    p_init,
    n_init,
    dt,
    dx,
    T,
    L,
    p_ped,
    n_ped,
    source_p,
    source_n,
    num_snapshots,
    transport_params
)

times = np.linspace(0, T, len(saved_p))

print("\n" + "="*60)
print("EVOLUTION PLOTS")
print("="*60 + "\n")

# Pressure evolution
fig, ax = plt.subplots()
for i, p in enumerate(saved_p):
    ax.plot(x, p, label=rf"$t={times[i]:.3f}$")
ax.set_title(rf"$\mathrm{{Pressure\ Evolution}}$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$p$")
ax.legend(ncol=2)
style_plot(ax)
save_and_show("05_pressure_evolution")

# Density evolution
fig, ax = plt.subplots()
for i, n in enumerate(saved_n):
    ax.plot(x, n, label=rf"$t={times[i]:.3f}$")
ax.set_title(rf"$\mathrm{{Density\ Evolution}}$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$n$")
ax.legend(ncol=2)
style_plot(ax)
save_and_show("06_density_evolution")

# Total pressure over time
total_p_time = [np.trapezoid(p, x) for p in saved_p]
fig, ax = plt.subplots()
ax.plot(times, total_p_time)
ax.set_title(r"$\mathrm{Total\ Pressure}$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\int p\,dx$")
style_plot(ax)
save_and_show("07_total_pressure")

# Total density over time
total_n_time = [np.trapezoid(n, x) for n in saved_n]
fig, ax = plt.subplots()
ax.plot(times, total_n_time)
ax.set_title(r"$\mathrm{Total\ Density}$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\int n\,dx$")
style_plot(ax)
save_and_show("08_total_density")

# Temperature (p/n)
fig, ax = plt.subplots()
for i in [0, len(saved_p)//2, -1]:
    T_profile = saved_p[i] / (saved_n[i] + 1e-10)
    ax.plot(x, T_profile, label=rf"$t={times[i]:.3f}$")
ax.set_title(r"$\mathrm{Temperature\ Profile}$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$T = p/n$")
ax.legend()
style_plot(ax)
save_and_show("09_temperature_profile")

# ==========================================
# FINAL STATE PLOTS
# ==========================================

print("\n" + "="*60)
print("FINAL STATE (at t={:.3f})".format(times[-1]))
print("="*60 + "\n")

p_final = saved_p[-1]
n_final = saved_n[-1]
T_final = p_final / (n_final + 1e-10)

# Final pressure profile
fig, ax = plt.subplots()
ax.plot(x, p_final, linewidth=2.5, color='C0')
ax.fill_between(x, 0, p_final, alpha=0.3, color='C0')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$p(x)$")
ax.set_title(rf"$\mathrm{{Final\ Pressure\ Profile\ (t={times[-1]:.3f})}}$")
style_plot(ax)
save_and_show("10_final_pressure")

# Final density profile
fig, ax = plt.subplots()
ax.plot(x, n_final, linewidth=2.5, color='C1')
ax.fill_between(x, 0, n_final, alpha=0.3, color='C1')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$n(x)$")
ax.set_title(rf"$\mathrm{{Final\ Density\ Profile\ (t={times[-1]:.3f})}}$")
style_plot(ax)
save_and_show("11_final_density")

# Final temperature profile
fig, ax = plt.subplots()
ax.plot(x, T_final, linewidth=2.5, color='C2')
ax.fill_between(x, 0, T_final, alpha=0.3, color='C2')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$T(x) = p/n$")
ax.set_title(rf"$\mathrm{{Final\ Temperature\ Profile\ (t={times[-1]:.3f})}}$")
style_plot(ax)
save_and_show("12_final_temperature")

# All three on same plot for comparison
fig, ax = plt.subplots()
ax_n = ax.twinx()
ax_T = ax.twinx()
ax_T.spines['right'].set_position(('outward', 60))

p_line = ax.plot(x, p_final, linewidth=2, label=r"$p$", color='C0')
n_line = ax_n.plot(x, n_final, linewidth=2, label=r"$n$", color='C1')
T_line = ax_T.plot(x, T_final, linewidth=2, label=r"$T=p/n$", color='C2')

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$p(x)$", color='C0')
ax_n.set_ylabel(r"$n(x)$", color='C1')
ax_T.set_ylabel(r"$T(x)$", color='C2')

ax.tick_params(axis='y', labelcolor='C0')
ax_n.tick_params(axis='y', labelcolor='C1')
ax_T.tick_params(axis='y', labelcolor='C2')

ax.set_title(rf"$\mathrm{{Final\ State\ (t={times[-1]:.3f})}}$")

# Combined legend
lines = p_line + n_line + T_line
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='upper right')

style_plot(ax)
save_and_show("13_final_state_combined")

print("\nDone!")
