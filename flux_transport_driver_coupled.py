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
# Configuration
# ==========================================
L = 1.0
dx = 0.005
dt = 1e-6
T = 0.5
num_snapshots = 8

x = np.linspace(0, L, int(L/dx))

# ========================================
# INITIAL STATE CONTROL
# ========================================
# "separate": control p, n, T independently
# "coupled": link them (use p and T to set everything)
initial_state_mode = "separate"

# Pressure initial state: "supercritical" or "subcritical"
initial_state_p = "supercritical"

# Density initial state: "supercritical" or "subcritical"
initial_state_n = "supercritical"

# Temperature initial state: "supercritical" or "subcritical"
# (only used if mode is not "coupled")
initial_state_T = "supercritical"

# ========================================
# POWER BALANCE CONTROL
# ========================================
# "separate": enforce power balance on p and n independently
# "coupled_to_p": scale both sources to match pressure balance
# "coupled_to_n": scale both sources to match density balance
power_balance_mode = "separate"

power_balance_p = 1.0  # ∫S_p dx = ratio * Q_edge (pressure)
power_balance_n = 1.0  # ∫S_n dx = ratio * Γ_edge (density)

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

    # --- Particle transport (density) ---
    "chi0_n": 1.0,
    "chi_core_n": 1.0,
    "chi_RR_n": 0.01,
    "g_c_n": 2.0,
    "g_stiff_n": 1.5,
    "n_stiff_n": 2,
    "V_n": 0.0,

    # --- Spatial structure ---
    "boundaries": [],
    "deltas": [],
    "flux_models": ["nl"],
    "flux_models_n": ["nl"],
    "nu4": 0,

    # --- Power balance ---
    "heating_mode": "global",
    "power_balance": power_balance_p,
    "power_balance_n": power_balance_n,
    "power_balance_mode": power_balance_mode,
    "edge_sigma": 0.05,
}

# ==========================================
# Initial profiles
# ==========================================

g_c_p = transport_params["g_c"]
g_crit_p = g_c_p / 2.0

g_c_n = transport_params["g_c_n"]
g_crit_n = g_c_n / 2.0

p_core = 2.0
p_ped = 1.0
n_core = 1.0
n_ped = 0.5

m = 2

def base_profile(x, m):
    return (1 - (x/L)**m)

p_shape = base_profile(x, m)
n_shape = base_profile(x, m)

# --- Pressure gradient ---
p_x_shape = np.gradient(p_shape, dx)
g_p_shape = -p_x_shape
max_g_p_shape = np.max(g_p_shape)

if initial_state_p == "supercritical":
    target_g_p = 1.8 * g_crit_p
else:
    target_g_p = 0.6 * g_crit_p

scale_p = target_g_p / max_g_p_shape
p_init = p_ped + scale_p * (p_core - p_ped) * p_shape

# --- Density gradient ---
n_x_shape = np.gradient(n_shape, dx)
g_n_shape = -n_x_shape
max_g_n_shape = np.max(g_n_shape)

if initial_state_n == "supercritical":
    target_g_n = 1.8 * g_crit_n
else:
    target_g_n = 0.6 * g_crit_n

scale_n = target_g_n / max_g_n_shape
n_init = n_ped + scale_n * (n_core - n_ped) * n_shape

# --- Temperature (if using separate mode) ---
if initial_state_mode == "coupled":
    # In coupled mode, T = p/n, derive n from T if needed
    # For now, just use p and n as computed
    pass

g_p_init = -np.gradient(p_init, dx)
g_n_init = -np.gradient(n_init, dx)
T_init = p_init / (n_init + 1e-10)
g_T_init = -np.gradient(T_init, dx)

print("\n" + "="*60)
print("INITIAL CONDITIONS")
print("="*60)
print(f"Initial state mode: {initial_state_mode}")
print(f"Pressure state: {initial_state_p}, max g_p = {np.max(g_p_init):.4f} (crit: {g_crit_p:.4f})")
print(f"Density state:  {initial_state_n}, max g_n = {np.max(g_n_init):.4f} (crit: {g_crit_n:.4f})")
print(f"Temperature:    max g_T = {np.max(g_T_init):.4f}")
print("\n" + "="*60)
print("POWER BALANCE MODE")
print("="*60)
print(f"Power balance mode: {power_balance_mode}")
print(f"Balance ratio (p): {power_balance_p}")
print(f"Balance ratio (n): {power_balance_n}")

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

# ==========================================
# Compute diagnostics
# ==========================================

total_p_time = [np.trapezoid(p, x) for p in saved_p]
total_n_time = [np.trapezoid(n, x) for n in saved_n]

saved_T = [p / (n + 1e-10) for p, n in zip(saved_p, saved_n)]
total_T_time = [np.trapezoid(T_prof, x) for T_prof in saved_T]

# ==========================================
# INITIAL DIAGNOSTIC PLOTS (a, b, c)
# ==========================================

print("\n" + "="*60)
print("CREATING DIAGNOSTIC PLOTS")
print("="*60 + "\n")

# 01a. Initial pressure profile
fig, ax = plt.subplots()
ax.plot(x, p_init, linewidth=2.5, label="Pressure")
ax.fill_between(x, 0, p_init, alpha=0.3)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$p(x)$")
ax.set_title(r"$\mathrm{Initial\ Pressure\ Profile}$")
ax.legend()
style_plot(ax)
save_and_show("01a_initial_pressure")

# 01b. Initial density profile
fig, ax = plt.subplots()
ax.plot(x, n_init, linewidth=2.5, label="Density", color='C1')
ax.fill_between(x, 0, n_init, alpha=0.3, color='C1')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$n(x)$")
ax.set_title(r"$\mathrm{Initial\ Density\ Profile}$")
ax.legend()
style_plot(ax)
save_and_show("01b_initial_density")

# 01c. Initial temperature profile
fig, ax = plt.subplots()
ax.plot(x, T_init, linewidth=2.5, label="Temperature", color='C2')
ax.fill_between(x, 0, T_init, alpha=0.3, color='C2')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$T(x) = p/n$")
ax.set_title(r"$\mathrm{Initial\ Temperature\ Profile}$")
ax.legend()
style_plot(ax)
save_and_show("01c_initial_temperature")

# 02a. Initial pressure gradients
fig, ax = plt.subplots()
ax.plot(x, g_p_init, linewidth=2, label=r"$g_p = -\partial_x p$")
ax.axhline(g_crit_p, linestyle="--", linewidth=1.5, label=r"$g_\mathrm{crit,p}$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"Gradient")
ax.set_title(r"$\mathrm{Initial\ Pressure\ Gradient}$")
ax.legend()
style_plot(ax)
save_and_show("02a_initial_grad_pressure")

# 02b. Initial density gradients
fig, ax = plt.subplots()
ax.plot(x, g_n_init, linewidth=2, label=r"$g_n = -\partial_x n$", color='C1')
ax.axhline(g_crit_n, linestyle="--", linewidth=1.5, label=r"$g_\mathrm{crit,n}$", color='C1')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"Gradient")
ax.set_title(r"$\mathrm{Initial\ Density\ Gradient}$")
ax.legend()
style_plot(ax)
save_and_show("02b_initial_grad_density")

# 02c. Initial temperature gradients
fig, ax = plt.subplots()
ax.plot(x, g_T_init, linewidth=2, label=r"$g_T = -\partial_x T$", color='C2')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"Gradient")
ax.set_title(r"$\mathrm{Initial\ Temperature\ Gradient}$")
ax.legend()
style_plot(ax)
save_and_show("02c_initial_grad_temperature")

# 03a/b/c. Initial sources
S_p_init = source_p(x, p_init, n_init)
S_n_init = source_n(x, p_init, n_init)

fig, ax = plt.subplots()
ax.plot(x, S_p_init, linewidth=2.5, label=r"$S_p$")
ax.fill_between(x, 0, S_p_init, alpha=0.3)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$S_p(x)$")
ax.set_title(r"$\mathrm{Initial\ Pressure\ Source}$")
style_plot(ax)
save_and_show("03a_initial_source_pressure")

fig, ax = plt.subplots()
ax.plot(x, S_n_init, linewidth=2.5, label=r"$S_n$", color='C1')
ax.fill_between(x, 0, S_n_init, alpha=0.3, color='C1')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$S_n(x)$")
ax.set_title(r"$\mathrm{Initial\ Density\ Source}$")
style_plot(ax)
save_and_show("03b_initial_source_density")

# ==========================================
# EVOLUTION PLOTS (a, b, c)
# ==========================================

# 04a. Pressure evolution
fig, ax = plt.subplots()
for i, p in enumerate(saved_p[::len(saved_p)//4]):  # Show 5 snapshots
    idx = i * len(saved_p) // 4
    ax.plot(x, p, label=rf"$t={times[idx]:.3f}$")
ax.set_title(r"$\mathrm{Pressure\ Evolution}$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$p$")
ax.legend(ncol=2)
style_plot(ax)
save_and_show("04a_pressure_evolution")

# 04b. Density evolution
fig, ax = plt.subplots()
for i, n in enumerate(saved_n[::len(saved_n)//4]):
    idx = i * len(saved_n) // 4
    ax.plot(x, n, label=rf"$t={times[idx]:.3f}$", color=f'C{i}')
ax.set_title(r"$\mathrm{Density\ Evolution}$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$n$")
ax.legend(ncol=2)
style_plot(ax)
save_and_show("04b_density_evolution")

# 04c. Temperature evolution
fig, ax = plt.subplots()
for i, T_prof in enumerate(saved_T[::len(saved_T)//4]):
    idx = i * len(saved_T) // 4
    ax.plot(x, T_prof, label=rf"$t={times[idx]:.3f}$", color=f'C{i+5}')
ax.set_title(r"$\mathrm{Temperature\ Evolution}$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$T = p/n$")
ax.legend(ncol=2)
style_plot(ax)
save_and_show("04c_temperature_evolution")

# ==========================================
# INTEGRATED QUANTITIES (a, b, c)
# ==========================================

# 05a. Total pressure
fig, ax = plt.subplots()
ax.plot(times, total_p_time, linewidth=2, marker='o', markersize=4)
ax.set_title(r"$\mathrm{Total\ Pressure}$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\int p\,dx$")
style_plot(ax)
save_and_show("05a_total_pressure")

# 05b. Total density
fig, ax = plt.subplots()
ax.plot(times, total_n_time, linewidth=2, marker='o', markersize=4, color='C1')
ax.set_title(r"$\mathrm{Total\ Density}$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\int n\,dx$")
style_plot(ax)
save_and_show("05b_total_density")

# 05c. Total temperature
fig, ax = plt.subplots()
ax.plot(times, total_T_time, linewidth=2, marker='o', markersize=4, color='C2')
ax.set_title(r"$\mathrm{Total\ Temperature}$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\int T\,dx$")
style_plot(ax)
save_and_show("05c_total_temperature")

# ==========================================
# FINAL STATE PLOTS (a, b, c)
# ==========================================

print("\n" + "="*60)
print("FINAL STATE (at t={:.3f})".format(times[-1]))
print("="*60 + "\n")

p_final = saved_p[-1]
n_final = saved_n[-1]
T_final = saved_T[-1]

# 06a. Final pressure
fig, ax = plt.subplots()
ax.plot(x, p_final, linewidth=2.5, color='C0')
ax.fill_between(x, 0, p_final, alpha=0.3, color='C0')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$p(x)$")
ax.set_title(rf"$\mathrm{{Final\ Pressure}}$")
style_plot(ax)
save_and_show("06a_final_pressure")

# 06b. Final density
fig, ax = plt.subplots()
ax.plot(x, n_final, linewidth=2.5, color='C1')
ax.fill_between(x, 0, n_final, alpha=0.3, color='C1')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$n(x)$")
ax.set_title(rf"$\mathrm{{Final\ Density}}$")
style_plot(ax)
save_and_show("06b_final_density")

# 06c. Final temperature
fig, ax = plt.subplots()
ax.plot(x, T_final, linewidth=2.5, color='C2')
ax.fill_between(x, 0, T_final, alpha=0.3, color='C2')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$T(x)$")
ax.set_title(rf"$\mathrm{{Final\ Temperature}}$")
style_plot(ax)
save_and_show("06c_final_temperature")

# 07. Final state combined
fig, ax = plt.subplots()
ax_n = ax.twinx()
ax_T = ax.twinx()
ax_T.spines['right'].set_position(('outward', 60))

p_line = ax.plot(x, p_final, linewidth=2.5, label=r"$p$", color='C0')
n_line = ax_n.plot(x, n_final, linewidth=2.5, label=r"$n$", color='C1')
T_line = ax_T.plot(x, T_final, linewidth=2.5, label=r"$T=p/n$", color='C2')

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$p(x)$", color='C0', fontsize=14)
ax_n.set_ylabel(r"$n(x)$", color='C1', fontsize=14)
ax_T.set_ylabel(r"$T(x)$", color='C2', fontsize=14)

ax.tick_params(axis='y', labelcolor='C0')
ax_n.tick_params(axis='y', labelcolor='C1')
ax_T.tick_params(axis='y', labelcolor='C2')

ax.set_title(rf"$\mathrm{{Final\ State}}$")

lines = p_line + n_line + T_line
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='upper right', fontsize=12)

style_plot(ax)
save_and_show("07_final_state_combined")

print(f"\n{'='*60}")
print("SIMULATION COMPLETE")
print(f"{'='*60}")
print(f"Total simulation time: {T:.3f} time units")
print(f"Final integrated pressure: {total_p_time[-1]:.4f}")
print(f"Final integrated density:  {total_n_time[-1]:.4f}")
print(f"Final integrated temp:     {total_T_time[-1]:.4f}")
print(f"Plots saved to 'plots/' directory\n")
