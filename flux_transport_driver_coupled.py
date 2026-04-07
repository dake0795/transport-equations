#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import AutoMinorLocator

from flux_transport_model_coupled import (
    heat_flux_function, particle_flux_function,
    solving_loop_coupled, _get_bcs, _apply_bc_to_profile
)

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
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    filepath = os.path.join(plots_dir, f"{filename}.pdf")
    plt.savefig(filepath, format="pdf", bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.show()

colors_tab20 = cm.tab20.colors

def plot_profile_trio(x, ya, yb, yc,
                      labels, ylabels, titles, filenames,
                      crits=(None, None, None)):
    """Plot three profiles (a=pressure, b=density, c=temperature)."""
    for y, label, ylabel, title, filename, crit, col in zip(
        [ya, yb, yc], labels, ylabels, titles, filenames, crits, ['C0','C1','C2']
    ):
        fig, ax = plt.subplots()
        ax.plot(x, y, linewidth=2, label=label, color=col)
        if crit is not None:
            ax.axhline(crit, linestyle="--", linewidth=1.5,
                       label=r"$g_\mathrm{crit}$", color=col, alpha=0.7)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        style_plot(ax)
        save_and_show(filename)

# ==========================================
# Domain
# ==========================================
L = 1.0
dx = 0.005
dt = 1e-6
Tmax = 0.5
num_snapshots = 8

x = np.linspace(0, L, int(L/dx))

# ==========================================
# INITIAL STATE CONTROL
# ==========================================
# "separate": control p, n independently
# "coupled":  set n from p and T targets
initial_state_mode = "separate"

initial_state_p = "supercritical"   # "supercritical" or "subcritical"
initial_state_n = "supercritical"   # "supercritical" or "subcritical"

# ==========================================
# POWER BALANCE CONTROL
# ==========================================
# "separate":    enforce p and n independently
# "coupled_to_p": scale both to match pressure edge flux
# "coupled_to_n": scale both to match density edge flux
power_balance_mode = "separate"

power_balance_p = 1.0   # ∫S_p dx = ratio * Q_edge
power_balance_n = 1.0   # ∫S_n dx = ratio * Γ_edge

heating_mode = "global"  # "global" or "localized"

# ==========================================
# Transport parameters
# ==========================================
transport_params = {
    # --- Heat transport (pressure) ---
    "chi0":     10.0,
    "chi_core": 10.0,
    "chi_RR":   0.05,
    "g_c":      4.0,
    "g_stiff":  3.0,
    "n_stiff":  2,

    # --- Particle transport (density) ---
    "chi0_n":    1.0,
    "chi_core_n": 1.0,
    "chi_RR_n":  0.01,
    "g_c_n":     2.0,
    "g_stiff_n": 1.5,
    "n_stiff_n": 2,
    "V_n":       0.0,

    # --- Spatial structure ---
    "boundaries":   [],
    "deltas":       [],
    "flux_models":  ["nl"],
    "flux_models_n": ["nl"],
    "nu4": 0,

    # --- Power balance ---
    "heating_mode":       heating_mode,
    "power_balance":      power_balance_p,
    "power_balance_n":    power_balance_n,
    "power_balance_mode": power_balance_mode,
    "edge_sigma": 0.05,
}

# ==========================================
# Initial profiles
# ==========================================
g_c_p  = transport_params["g_c"]
g_crit_p = g_c_p / 2.0

g_c_n_val  = transport_params["g_c_n"]
g_crit_n = g_c_n_val / 2.0

p_core = 2.0;  p_ped = 1.0
n_core = 1.0;  n_ped = 0.5
m_p = 2   # shape exponent for pressure
m_n = 3   # shape exponent for density (different → T = p/n has a gradient)

def base_profile(x, m):
    return 1.0 - (x/L)**m

p_shape = base_profile(x, m_p)
n_shape = base_profile(x, m_n)

# Pressure
max_g_p_shape = np.max(-np.gradient((p_core - p_ped) * p_shape, dx))
target_g_p = 1.8 * g_crit_p if initial_state_p == "supercritical" else 0.6 * g_crit_p
scale_p = target_g_p / max_g_p_shape
p_init = p_ped + scale_p * (p_core - p_ped) * p_shape

# Density
max_g_n_shape = np.max(-np.gradient((n_core - n_ped) * n_shape, dx))
target_g_n = 1.8 * g_crit_n if initial_state_n == "supercritical" else 0.6 * g_crit_n
scale_n = target_g_n / max_g_n_shape
n_init = n_ped + scale_n * (n_core - n_ped) * n_shape

# Temperature
T_init = p_init / (n_init + 1e-10)

g_p_init = -np.gradient(p_init, dx)
g_n_init = -np.gradient(n_init, dx)
g_T_init = -np.gradient(T_init, dx)

print(f"\n{'='*60}\nINITIAL CONDITIONS\n{'='*60}")
print(f"Pressure: {initial_state_p}, max g_p = {np.max(g_p_init):.4f} (crit: {g_crit_p:.4f})")
print(f"Density:  {initial_state_n}, max g_n = {np.max(g_n_init):.4f} (crit: {g_crit_n:.4f})")
print(f"Temp:     max g_T = {np.max(g_T_init):.4f}")
print(f"\n{'='*60}\nPOWER BALANCE MODE: {power_balance_mode}\n{'='*60}")
print(f"  p ratio = {power_balance_p},  n ratio = {power_balance_n}")

# ==========================================
# Source functions
# ==========================================
alpha_p = 0.1
alpha_n = 0.05

def base_source_p(x):
    return 5.0 * np.exp(-(x**2) / (0.25*L)**2)

def base_source_n(x):
    return 2.0 * np.exp(-(x**2) / (0.25*L)**2)

def source_p(x, p, n):
    return base_source_p(x) * (1.0 + alpha_p * p)

def source_n(x, p, n):
    return base_source_n(x) * (1.0 + alpha_n * n)

# ==========================================
# Initial face fluxes for diagnostic plots
# ==========================================
core_p_bc, edge_p_bc, core_n_bc, edge_n_bc = _get_bcs(transport_params, p_ped, n_ped)
p_bc0 = p_init.copy(); n_bc0 = n_init.copy()
_apply_bc_to_profile(p_bc0, n_bc0, dx, core_p_bc, edge_p_bc, core_n_bc, edge_n_bc)

x_face = 0.5 * (x[1:] + x[:-1])
g_p_face0 = -(p_bc0[1:] - p_bc0[:-1]) / dx
g_n_face0 = -(n_bc0[1:] - n_bc0[:-1]) / dx

Q_face_init    = heat_flux_function(g_p_face0, x_face, transport_params)
Gamma_face_init = particle_flux_function(g_n_face0, x_face, transport_params)
T_bc0 = p_bc0 / (n_bc0 + 1e-10)
g_T_face0 = -(T_bc0[1:] - T_bc0[:-1]) / dx
# Temperature flux proxy: chi_eff * g_T
chi_eff_T = transport_params["chi0"] * np.ones_like(g_T_face0)
T_flux_init = chi_eff_T * np.maximum(g_T_face0, 0.0)

S_p_init = source_p(x, p_init, n_init)
S_n_init = source_n(x, p_init, n_init)
S_T_init = (S_p_init - T_init * S_n_init) / (n_init + 1e-10)


def _apply_power_balance(S, flux_face, x, params, mode_key="power_balance"):
    """Mimic the power balance enforcement from compute_rhs for diagnostics."""
    heating_mode_loc = params.get("heating_mode", "global")
    ratio = params.get(mode_key, 1.0)
    S_out = S.copy()

    if ratio is None:
        return S_out

    Q_edge = flux_face[-1]
    total_S = np.trapezoid(S_out, x)

    if heating_mode_loc == "localized":
        L_val = x[-1]
        sigma = params.get("edge_sigma", 0.05)
        g_test = np.exp(-((x - L_val)**2) / (2*sigma**2))
        g_norm = np.trapezoid(g_test, x)
        deficit = ratio * Q_edge - total_S
        amp = deficit / g_norm if g_norm > 0 else 0
        S_out = S_out + amp * g_test

    elif heating_mode_loc == "global":
        if total_S > 0 and Q_edge > 0:
            S_out = S_out * (ratio * Q_edge / total_S)

    return S_out


S_p_enforced = _apply_power_balance(S_p_init, Q_face_init, x, transport_params, "power_balance")
S_n_enforced = _apply_power_balance(S_n_init, Gamma_face_init, x, transport_params, "power_balance_n")
S_T_enforced = (S_p_enforced - T_init * S_n_enforced) / (n_init + 1e-10)

# ==========================================
# INITIAL DIAGNOSTIC PLOTS (before solve)
# ==========================================
print(f"\n{'='*60}\nCREATING INITIAL PLOTS\n{'='*60}\n")

# 01a/b/c  Initial profiles
plot_profile_trio(
    x,
    p_init, n_init, T_init,
    [r"$p$", r"$n$", r"$T=p/n$"],
    [r"$p(x)$", r"$n(x)$", r"$T(x)$"],
    [r"$\mathrm{Initial\ Pressure\ Profile}$",
     r"$\mathrm{Initial\ Density\ Profile}$",
     r"$\mathrm{Initial\ Temperature\ Profile}$"],
    ["01a_initial_pressure", "01b_initial_density", "01c_initial_temperature"]
)

# 02a/b/c  Initial gradients
plot_profile_trio(
    x,
    g_p_init, g_n_init, g_T_init,
    [r"$g_p$", r"$g_n$", r"$g_T$"],
    [r"$g_p = -\partial_x p$", r"$g_n = -\partial_x n$", r"$g_T = -\partial_x T$"],
    [r"$\mathrm{Initial\ Pressure\ Gradient}$",
     r"$\mathrm{Initial\ Density\ Gradient}$",
     r"$\mathrm{Initial\ Temperature\ Gradient}$"],
    ["02a_initial_grad_p", "02b_initial_grad_n", "02c_initial_grad_T"],
    crits=(g_crit_p, g_crit_n, None)
)

# 03a/b/c  Initial flux vs cumulative source
H_p = np.cumsum(S_p_init) * dx
H_n = np.cumsum(S_n_init) * dx
H_T = np.cumsum(S_T_init) * dx

for flux_face, H, ylabel, title, filename, col in [
    (Q_face_init,     H_p, r"$Q(x)$",       r"$\mathrm{Initial\ Heat\ Flux\ vs\ Source}$",        "03a_flux_vs_source_p", 'C0'),
    (Gamma_face_init, H_n, r"$\Gamma(x)$",  r"$\mathrm{Initial\ Particle\ Flux\ vs\ Source}$",    "03b_flux_vs_source_n", 'C1'),
    (T_flux_init,     H_T, r"$Q_T(x)$",     r"$\mathrm{Initial\ Temperature\ Flux\ vs\ Source}$", "03c_flux_vs_source_T", 'C2'),
]:
    fig, ax = plt.subplots()
    ax.plot(x_face, flux_face, label="Flux", color=col)
    ax.plot(x, H, linestyle="--", label=r"$\int_0^x S\,dx'$", color=col)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); style_plot(ax); save_and_show(filename)

# 03Xb  Flux vs enforced source
H_p_enf = np.cumsum(S_p_enforced) * dx
H_n_enf = np.cumsum(S_n_enforced) * dx
H_T_enf = np.cumsum(S_T_enforced) * dx

for flux_face, H_enf, ylabel, title, filename, col in [
    (Q_face_init,     H_p_enf, r"$Q(x)$",      r"$\mathrm{Heat\ Flux\ vs\ Enforced\ Source}$",     "03ab_flux_balanced_p", 'C0'),
    (Gamma_face_init, H_n_enf, r"$\Gamma(x)$", r"$\mathrm{Particle\ Flux\ vs\ Enforced\ Source}$", "03bb_flux_balanced_n", 'C1'),
    (T_flux_init,     H_T_enf, r"$Q_T(x)$",    r"$\mathrm{Temp\ Flux\ vs\ Enforced\ Source}$",     "03cb_flux_balanced_T", 'C2'),
]:
    fig, ax = plt.subplots()
    ax.plot(x_face, flux_face, label="Flux", color=col)
    ax.plot(x, H_enf, linestyle="--", label=r"$\int_0^x S_\mathrm{enforced}\,dx'$", color=col)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); style_plot(ax); save_and_show(filename)

# 03Xc  Source before and after power balance
for S_raw, S_enf, ylabel, title, filename, col in [
    (S_p_init, S_p_enforced, r"$S_p(x)$", r"$\mathrm{Pressure\ Source\ (before/after\ balance)}$", "03ac_source_comparison_p", 'C0'),
    (S_n_init, S_n_enforced, r"$S_n(x)$", r"$\mathrm{Density\ Source\ (before/after\ balance)}$",  "03bc_source_comparison_n", 'C1'),
    (S_T_init, S_T_enforced, r"$S_T(x)$", r"$\mathrm{Temp\ Source\ (before/after\ balance)}$",     "03cc_source_comparison_T", 'C2'),
]:
    fig, ax = plt.subplots()
    ax.plot(x, S_raw, linewidth=2, label="Raw", color=col)
    ax.plot(x, S_enf, linestyle="--", linewidth=2, label="Enforced", color=col)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); style_plot(ax); save_and_show(filename)

# ==========================================
# SOLVE
# ==========================================
print(f"\n{'='*60}\nSOLVING COUPLED SYSTEM\n{'='*60}\n")

saved_p, saved_n = solving_loop_coupled(
    p_init, n_init, dt, dx, Tmax, L,
    p_ped, n_ped, source_p, source_n,
    num_snapshots, transport_params
)

times = np.linspace(0, Tmax, len(saved_p))
saved_T_prof = [p / (n + 1e-10) for p, n in zip(saved_p, saved_n)]

# ==========================================
# Post-solve diagnostics
# ==========================================
total_p_time = [np.trapezoid(p, x) for p in saved_p]
total_n_time = [np.trapezoid(n, x) for n in saved_n]
total_T_time = [np.trapezoid(T, x) for T in saved_T_prof]

edge_Q_time     = []
edge_Gamma_time = []
total_Sp_time   = []
total_Sn_time   = []
max_gp_time     = []
max_gn_time     = []
max_gT_time     = []

for p, n in zip(saved_p, saved_n):
    T_snap = p / (n + 1e-10)
    p_bc = p.copy(); n_bc = n.copy()
    _apply_bc_to_profile(p_bc, n_bc, dx, core_p_bc, edge_p_bc, core_n_bc, edge_n_bc)

    gp = -(p_bc[1:] - p_bc[:-1]) / dx
    gn = -(n_bc[1:] - n_bc[:-1]) / dx
    Q     = heat_flux_function(gp, x_face, transport_params)
    Gamma = particle_flux_function(gn, x_face, transport_params)
    edge_Q_time.append(Q[-1])
    edge_Gamma_time.append(Gamma[-1])

    total_Sp_time.append(np.trapezoid(source_p(x, p, n), x))
    total_Sn_time.append(np.trapezoid(source_n(x, p, n), x))

    max_gp_time.append(np.max(-np.gradient(p, dx)))
    max_gn_time.append(np.max(-np.gradient(n, dx)))
    max_gT_time.append(np.max(-np.gradient(T_snap, dx)))

edge_Q_time     = np.array(edge_Q_time)
edge_Gamma_time = np.array(edge_Gamma_time)

# ==========================================
# POST-SOLVE DIAGNOSTIC PLOTS
# ==========================================
print(f"\n{'='*60}\nCREATING POST-SOLVE PLOTS\n{'='*60}\n")

# ==========================================
# 04a/b/c  Evolution profiles
# ==========================================
snap_stride = max(len(saved_p) // 4, 1)

for saved, ylabel, title, filename, col_offset in [
    (saved_p,      r"$p$",     r"$\mathrm{Pressure\ Evolution}$",    "04a_pressure_evolution",    0),
    (saved_n,      r"$n$",     r"$\mathrm{Density\ Evolution}$",     "04b_density_evolution",     0),
    (saved_T_prof, r"$T=p/n$", r"$\mathrm{Temperature\ Evolution}$", "04c_temperature_evolution", 5),
]:
    fig, ax = plt.subplots()
    for i, prof in enumerate(saved[::snap_stride]):
        idx = min(i * snap_stride, len(times)-1)
        ax.plot(x, prof, label=rf"$t={times[idx]:.3f}$", color=colors_tab20[i % 20])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=2)
    style_plot(ax)
    save_and_show(filename)

# ==========================================
# 05a/b/c  Total integrated quantities
# ==========================================
for total_t, ylabel, title, filename, col in [
    (total_p_time, r"$\int p\,dx$", r"$\mathrm{Total\ Pressure}$",    "05a_total_pressure",    'C0'),
    (total_n_time, r"$\int n\,dx$", r"$\mathrm{Total\ Density}$",     "05b_total_density",     'C1'),
    (total_T_time, r"$\int T\,dx$", r"$\mathrm{Total\ Temperature}$", "05c_total_temperature", 'C2'),
]:
    fig, ax = plt.subplots()
    ax.plot(times, total_t, linewidth=2, marker='o', markersize=4, color=col)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_plot(ax)
    save_and_show(filename)

# ==========================================
# 06a/b/c  Gradient evolution at tracked points
# ==========================================
track_points = np.linspace(0.4, 0.95, 8)
track_indices = [np.argmin(np.abs(x - pt)) for pt in track_points]

for saved, g_crit_val, ylabel, title, filename, col in [
    (saved_p,      g_crit_p, r"$g_p$",   r"$\mathrm{Pressure\ Gradient\ Evolution}$",    "06a_grad_evolution_p", 'C0'),
    (saved_n,      g_crit_n, r"$g_n$",   r"$\mathrm{Density\ Gradient\ Evolution}$",     "06b_grad_evolution_n", 'C1'),
    (saved_T_prof, None,     r"$g_T$",   r"$\mathrm{Temperature\ Gradient\ Evolution}$", "06c_grad_evolution_T", 'C2'),
]:
    grad_hist = np.array([
        [-np.gradient(prof, dx)[idx] for idx in track_indices]
        for prof in saved
    ])
    fig, ax = plt.subplots()
    for i, pt in enumerate(track_points):
        ax.plot(times, grad_hist[:, i], label=rf"$x={pt:.2f}$", color=colors_tab20[i % 20])
    if g_crit_val is not None:
        ax.axhline(g_crit_val, linestyle='--', linewidth=1.5, label=r"$g_\mathrm{crit}$", color='k')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=9)
    style_plot(ax)
    save_and_show(filename)

# ==========================================
# 07a/b/c  Gradient heatmaps
# ==========================================
for saved, label, title, filename in [
    (saved_p,      r"$g_p$", r"$\mathrm{Pressure\ Gradient\ }g_p(x,t)$",    "07a_heatmap_gp"),
    (saved_n,      r"$g_n$", r"$\mathrm{Density\ Gradient\ }g_n(x,t)$",     "07b_heatmap_gn"),
    (saved_T_prof, r"$g_T$", r"$\mathrm{Temperature\ Gradient\ }g_T(x,t)$", "07c_heatmap_gT"),
]:
    g_all = np.array([-np.gradient(prof, dx) for prof in saved])
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(g_all.T, extent=[0, Tmax, 0, L], aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$x$")
    ax.set_title(title)
    fig.colorbar(im).set_label(label)
    plt.tight_layout()
    save_and_show(filename)

# ==========================================
# 08a/b/c  Profile heatmaps
# ==========================================
for saved, label, title, filename, cmap in [
    (saved_p,      r"$p$",     r"$\mathrm{Pressure\ }p(x,t)$",         "08a_heatmap_p", 'plasma'),
    (saved_n,      r"$n$",     r"$\mathrm{Density\ }n(x,t)$",          "08b_heatmap_n", 'inferno'),
    (saved_T_prof, r"$T=p/n$", r"$\mathrm{Temperature\ }T(x,t)$",      "08c_heatmap_T", 'magma'),
]:
    arr = np.array(saved)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(arr.T, extent=[0, Tmax, 0, L], aspect='auto', origin='lower', cmap=cmap)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$x$")
    ax.set_title(title)
    fig.colorbar(im).set_label(label)
    plt.tight_layout()
    save_and_show(filename)

# ==========================================
# 09a/b  Power imbalance over time
# ==========================================
for total_S_t, edge_flux_t, ylabel, title, filename, col in [
    (total_Sp_time, edge_Q_time,     r"$Q_\mathrm{edge} - \int S_p\,dx$",     r"$\mathrm{Pressure\ Power\ Imbalance}$", "09a_power_imbalance_p", 'C0'),
    (total_Sn_time, edge_Gamma_time, r"$\Gamma_\mathrm{edge} - \int S_n\,dx$", r"$\mathrm{Density\ Power\ Imbalance}$",  "09b_power_imbalance_n", 'C1'),
]:
    fig, ax = plt.subplots()
    ax.plot(times, np.array(edge_flux_t) - np.array(total_S_t), marker='o', markersize=4, color=col)
    ax.axhline(0, linestyle='--', color='k', linewidth=1)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_plot(ax)
    save_and_show(filename)

# ==========================================
# 10a/b  Max gradient and location
# ==========================================
for saved, g_crit_val, ylabel, title, filename, col in [
    (saved_p,      g_crit_p, r"$\max(g_p)$",   r"$\mathrm{Max\ Pressure\ Gradient}$",    "10a_max_grad_p", 'C0'),
    (saved_n,      g_crit_n, r"$\max(g_n)$",   r"$\mathrm{Max\ Density\ Gradient}$",     "10b_max_grad_n", 'C1'),
    (saved_T_prof, None,     r"$\max(g_T)$",   r"$\mathrm{Max\ Temperature\ Gradient}$", "10c_max_grad_T", 'C2'),
]:
    max_g = [np.max(-np.gradient(prof, dx)) for prof in saved]
    loc_g = [x[np.argmax(-np.gradient(prof, dx))] for prof in saved]

    fig, ax = plt.subplots()
    ax.plot(times, max_g, label="Max gradient", color=col)
    ax.plot(times, loc_g, linestyle="--", label="Location", color=col)
    if g_crit_val is not None:
        ax.axhline(g_crit_val, linestyle=':', color='k', linewidth=1.5, label=r"$g_\mathrm{crit}$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    style_plot(ax)
    save_and_show(filename)

# ==========================================
# 11a/b/c  Edge gradient vs edge source
# ==========================================
for saved, source_fn, g_crit_val, ylabel, title, filename, col in [
    (saved_p,      lambda p, n: source_p(x, p, n), g_crit_p, r"$g_p\ \mathrm{at\ edge}$", r"$\mathrm{Pressure\ Edge\ Gradient\ vs\ Source}$",    "11a_edge_grad_p", 'C0'),
    (saved_n,      lambda p, n: source_n(x, p, n), g_crit_n, r"$g_n\ \mathrm{at\ edge}$", r"$\mathrm{Density\ Edge\ Gradient\ vs\ Source}$",     "11b_edge_grad_n", 'C1'),
    (saved_T_prof, None,                            None,     r"$g_T\ \mathrm{at\ edge}$", r"$\mathrm{Temperature\ Edge\ Gradient\ vs\ Source}$", "11c_edge_grad_T", 'C2'),
]:
    edge_g = [-np.gradient(prof, dx)[-1] for prof in saved]
    fig, ax = plt.subplots()
    ax.plot(times, edge_g, label="Edge gradient", color=col)
    if source_fn is not None:
        edge_s = [source_fn(p, n)[-1] for p, n in zip(saved_p, saved_n)]
        ax.plot(times, edge_s, linestyle="--", label="Edge source", color=col)
    if g_crit_val is not None:
        ax.axhline(g_crit_val, linestyle=':', color='k', linewidth=1.5, label=r"$g_\mathrm{crit}$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    style_plot(ax)
    save_and_show(filename)

# ==========================================
# 12a/b/c  Flux balance at final snapshots
# ==========================================
for saved, flux_fn, source_fn, flux_label, title_prefix, prefix, col in [
    (saved_p,      lambda p, n: heat_flux_function(-(p[1:]-p[:-1])/dx, x_face, transport_params),
                   lambda p, n: source_p(x, p, n), r"$Q(x)$",     r"$\mathrm{Heat}$",     "12a_flux_balance_p", 'C0'),
    (saved_n,      lambda p, n: particle_flux_function(-(n[1:]-n[:-1])/dx, x_face, transport_params),
                   lambda p, n: source_n(x, p, n), r"$\Gamma(x)$", r"$\mathrm{Particle}$", "12b_flux_balance_n", 'C1'),
]:
    for snap_idx in [-2, -1]:
        p_snap = saved_p[snap_idx]; n_snap = saved_n[snap_idx]
        flux = flux_fn(p_snap, n_snap)
        S_snap = source_fn(p_snap, n_snap)
        F = np.array([np.trapezoid(S_snap[:j+1], x[:j+1]) for j in range(len(x))])

        fig, ax = plt.subplots()
        ax.plot(x_face, flux, label=flux_label, color=col)
        ax.plot(x, F, linestyle="--", label=r"$\int_0^x S\,dx'$", color=col)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"Flux / cumulative source")
        ax.set_title(rf"$\mathrm{{{title_prefix}\ Flux\ Balance}}\ (t={times[snap_idx]:.3f})$")
        ax.legend()
        style_plot(ax)
        save_and_show(f"{prefix}_t{snap_idx}")

# ==========================================
# 13a/b  Effective diffusivity (final state)
# ==========================================
eps = 1e-6

g_p_final = -np.gradient(saved_p[-1], dx)
Q_plus  = heat_flux_function(g_p_final + eps, x, transport_params)
Q_minus = heat_flux_function(g_p_final - eps, x, transport_params)
chi_eff_p = (Q_plus - Q_minus) / (2 * eps)

fig, ax = plt.subplots()
ax.plot(x, chi_eff_p, linewidth=2, color='C0')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\chi_\mathrm{eff,p}$")
ax.set_title(r"$\mathrm{Effective\ Heat\ Diffusivity\ (final)}$")
style_plot(ax)
save_and_show("13a_chi_eff_p")

g_n_final = -np.gradient(saved_n[-1], dx)
D_plus  = particle_flux_function(g_n_final + eps, x, transport_params)
D_minus = particle_flux_function(g_n_final - eps, x, transport_params)
chi_eff_n = (D_plus - D_minus) / (2 * eps)

fig, ax = plt.subplots()
ax.plot(x, chi_eff_n, linewidth=2, color='C1')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$D_\mathrm{eff,n}$")
ax.set_title(r"$\mathrm{Effective\ Particle\ Diffusivity\ (final)}$")
style_plot(ax)
save_and_show("13b_chi_eff_n")

# ==========================================
# 14. Final state combined
# ==========================================
p_final = saved_p[-1]
n_final = saved_n[-1]
T_final = saved_T_prof[-1]

fig, ax = plt.subplots()
ax_n = ax.twinx()
ax_T = ax.twinx()
ax_T.spines['right'].set_position(('outward', 60))

p_line = ax.plot(x, p_final, linewidth=2.5, label=r"$p$",     color='C0')
n_line = ax_n.plot(x, n_final, linewidth=2.5, label=r"$n$",   color='C1')
T_line = ax_T.plot(x, T_final, linewidth=2.5, label=r"$T$",   color='C2')

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$p(x)$",       color='C0')
ax_n.set_ylabel(r"$n(x)$",     color='C1')
ax_T.set_ylabel(r"$T(x)=p/n$", color='C2')
ax.tick_params(axis='y', labelcolor='C0')
ax_n.tick_params(axis='y', labelcolor='C1')
ax_T.tick_params(axis='y', labelcolor='C2')
ax.set_title(rf"$\mathrm{{Final\ State}}\ (t={times[-1]:.3f})$")
lines = p_line + n_line + T_line
ax.legend(lines, [l.get_label() for l in lines], loc='upper right')
style_plot(ax)
save_and_show("14_final_state_combined")

print(f"\n{'='*60}")
print("SIMULATION COMPLETE")
print(f"{'='*60}")
print(f"Final integrated pressure: {total_p_time[-1]:.4f}")
print(f"Final integrated density:  {total_n_time[-1]:.4f}")
print(f"Final integrated temp:     {total_T_time[-1]:.4f}")
print(f"Plots saved to 'plots/' directory\n")
