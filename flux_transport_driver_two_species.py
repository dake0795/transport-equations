#!/usr/bin/env python3
"""
Two-species (electron + ion) transport driver.

Heat flux driven by log temperature gradient κ_T = -d(ln T)/dx.
Particle flux driven by log density gradient κ_n = -d(ln n)/dx.
Electron-ion coupling via Q_ei = n_e (T_i - T_e) / tau_ei.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import AutoMinorLocator

from flux_transport_model_two_species import (
    heat_flux_function, particle_flux_function,
    collision_exchange,
    compute_fluxes_two_species,
    solving_loop_two_species,
    _get_bcs_two_species, _apply_bc,
    _get_species_params, _log_kap,
)

# ==========================================
# Plotting style
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
    os.makedirs("plots", exist_ok=True)
    filepath = f"plots/{filename}.pdf"
    plt.savefig(filepath, format="pdf", bbox_inches="tight")
    print(f"Saved: {filepath}")
    plt.show()


colors_tab20 = cm.tab20.colors


def tanh_profile(x, L, f_core, f_ped, x_sym=0.55, delta=0.25):
    """
    Smooth tokamak-like profile: f_core at centre, f_ped at edge.
    x_sym : normalised midpoint of transition (fraction of L).
    delta : transition half-width (fraction of L).
    Normalised so that the profile equals exactly f_core at x=0
    and f_ped at x=L.
    """
    xi = x / L
    s  = 0.5 * (1.0 - np.tanh((xi - x_sym) / delta))
    s0 = 0.5 * (1.0 - np.tanh((0.0        - x_sym) / delta))
    s1 = 0.5 * (1.0 - np.tanh((1.0        - x_sym) / delta))
    return f_ped + (f_core - f_ped) * (s - s1) / (s0 - s1)


def log_kap_cell(prof, dx):
    """Log gradient at cell centres: κ = -d(ln f)/dx."""
    return -np.gradient(np.log(np.maximum(prof, 1e-10)), dx)


def plot_single(x, y, label, ylabel, title, filename, color='C0', crit=None):
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2, label=label, color=color)
    if crit is not None:
        ax.axhline(crit, linestyle="--", linewidth=1.5,
                   label=r"$\kappa_\mathrm{crit}$", color=color, alpha=0.7)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    style_plot(ax)
    save_and_show(filename)


def plot_comparison(x, ye, yi, label_e, label_i, ylabel, title, filename,
                    crit_e=None, crit_i=None):
    """Overlay electron and ion profiles on the same axes."""
    fig, ax = plt.subplots()
    ax.plot(x, ye, linewidth=2, label=label_e, color='C0')
    ax.plot(x, yi, linewidth=2, label=label_i, color='C3', linestyle='--')
    if crit_e is not None:
        ax.axhline(crit_e, linestyle=":", color='C0', linewidth=1.5,
                   label=r"$\kappa_{\mathrm{crit},e}$", alpha=0.8)
    if crit_i is not None:
        ax.axhline(crit_i, linestyle=":", color='C3', linewidth=1.5,
                   label=r"$\kappa_{\mathrm{crit},i}$", alpha=0.8)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    style_plot(ax)
    save_and_show(filename)


# ==========================================
# Domain
# ==========================================
L            = 1.0
dx           = 0.005
dt           = 1e-6
Tmax         = 0.2
num_snapshots = 8

x = np.linspace(0, L, int(L / dx))

# ==========================================
# INITIAL STATE CONTROL
# ==========================================
initial_state_e = "supercritical"   # T_e gradient: "supercritical" or "subcritical"
initial_state_i = "supercritical"   # T_i gradient: "supercritical" or "subcritical"
initial_state_n = "subcritical"   # n gradient:   "supercritical" or "subcritical"

# ==========================================
# POWER BALANCE CONTROL
# ==========================================
power_balance_pe = 1.0   # ∫S_pe dx = ratio * Q_e_edge
power_balance_pi = 1.0   # ∫S_pi dx = ratio * Q_i_edge
power_balance_ne = 1.0   # ∫S_ne dx = ratio * Γ_e_edge
power_balance_ni = 1.0   # ∫S_ni dx = ratio * Γ_i_edge
heating_mode     = "global"   # "global" or "localized"

# ==========================================
# Transport parameters
# ==========================================
transport_params = {
    # --- Electron heat transport (driven by g_Te = -dT_e/dx) ---
    "chi0_e":      10.0,
    "chi_core_e":  10.0,
    "chi_RR_e":    0.05,
    "g_c_e":        4.0,
    "g_stiff_e":    3.0,
    "n_stiff_e":    2,

    # --- Electron particle transport (driven by g_ne = -dn_e/dx) ---
    "chi0_n_e":    1.0,
    "chi_core_n_e": 1.0,
    "chi_RR_n_e":  0.01,
    "g_c_n_e":     2.0,
    "g_stiff_n_e": 1.5,
    "n_stiff_n_e": 2,
    "V_n_e":       0.0,

    # --- Ion heat transport (driven by g_Ti = -dT_i/dx) ---
    "chi0_i":      3.0,
    "chi_core_i":  3.0,
    "chi_RR_i":    0.05,
    "g_c_i":        3.0,
    "g_stiff_i":    2.5,
    "n_stiff_i":    2,

    # --- Ion particle transport (driven by g_ni = -dn_i/dx) ---
    "chi0_n_i":    1.0,
    "chi_core_n_i": 1.0,
    "chi_RR_n_i":  0.01,
    "g_c_n_i":     2.0,
    "g_stiff_n_i": 1.5,
    "n_stiff_n_i": 2,
    "V_n_i":       0.0,

    # --- Electron-ion coupling ---
    "tau_ei":      0.05,   # equilibration time (smaller = stronger coupling)

    # --- Spatial structure ---
    "boundaries":    [],
    "deltas":        [],
    "flux_models":   ["nl"],
    "flux_models_n": ["nl"],
    "nu4":           0.0,

    # --- Power balance ---
    "heating_mode":    heating_mode,
    "power_balance_pe": power_balance_pe,
    "power_balance_pi": power_balance_pi,
    "power_balance_ne": power_balance_ne,
    "power_balance_ni": power_balance_ni,
    "edge_sigma":       0.05,

    # --- Physics sources (T3D-like) ---
    # Bremsstrahlung: P_brem = C_brem * n_e^2 * Z_eff * sqrt(T_e_keV)
    #   C_brem = 0 disables; set ~0.03 for mild radiation at n~2, T~15 keV
    "C_brem":     0.03,
    "Z_eff":      1.0,
    # Alpha heating: P_alpha = C_alpha * <sigma_v>(T_i_keV) * n_D * n_T
    #   C_alpha = 0 disables; set ~1e-3 for mild alpha heating at reactor conditions
    "C_alpha":    1e-3,
    "T_ref_keV":  10.0,   # model T=1 corresponds to T_ref_keV keV
    "n_ref_20":    1.0,   # model n=1 corresponds to n_ref_20 × 10^20 m^-3
    "Z_i":         1,
    "A_i":         2.5,   # D-T average amu
    "f_deuterium": 0.5,
    "f_tritium":   0.5,
}

# ==========================================
# Initial profiles
# ==========================================
# Critical log-gradients  (κ = -d ln T / dx  or  -d ln n / dx)
g_c_e    = transport_params["g_c_e"]
g_c_i    = transport_params["g_c_i"]
g_crit_e = g_c_e / 2.0
g_crit_i = g_c_i / 2.0
g_c_n_e  = transport_params["g_c_n_e"]
g_crit_n = g_c_n_e / 2.0

# Pedestal / core values
# (dimensionless; T_ref_keV=10 → T_e_core=2.0 ≈ 20 keV, n_core=2.0 ≈ 2×10²⁰ m⁻³)
T_e_core = 2.0;  T_e_ped = 0.5
T_i_core = 1.5;  T_i_ped = 0.4
n_core   = 2.0;  n_ped   = 0.2

# Tanh-based profiles: flat core + smooth drop to pedestal, matching typical tokamak shape
#   x_sym  : normalised transition midpoint  (r/a)
#   delta  : supercritical → small (steep); subcritical → large (broad)
x_sym_T       = 0.55
delta_T_super = 0.20
delta_T_sub   = 0.70
x_sym_n       = 0.80
delta_n_super = 0.20
delta_n_sub   = 0.70

delta_Te = delta_T_super if initial_state_e == "supercritical" else delta_T_sub
delta_Ti = delta_T_super if initial_state_i == "supercritical" else delta_T_sub
delta_n  = delta_n_super if initial_state_n == "supercritical" else delta_n_sub

T_e_init = tanh_profile(x, L, T_e_core, T_e_ped, x_sym=x_sym_T, delta=delta_Te)
T_i_init = tanh_profile(x, L, T_i_core, T_i_ped, x_sym=x_sym_T, delta=delta_Ti)
n_init   = tanh_profile(x, L, n_core,   n_ped,   x_sym=x_sym_n, delta=delta_n)

# Shared boundary condition values
p_e_ped  = T_e_ped * n_ped
p_i_ped  = T_i_ped * n_ped
n_e_ped  = n_ped
n_i_ped  = n_ped

# Pressure profiles  (p = n × T, the evolved quantity)
p_e_init = T_e_init * n_init
p_i_init = T_i_init * n_init
n_e_init = n_init.copy()
n_i_init = n_init.copy()

# Log gradients of initial profiles (κ = -d ln f / dx)
kap_Te_init = log_kap_cell(T_e_init, dx)
kap_Ti_init = log_kap_cell(T_i_init, dx)
kap_n_init  = log_kap_cell(n_init,   dx)

print(f"\n{'='*60}\nINITIAL CONDITIONS\n{'='*60}")
print(f"Electrons: {initial_state_e}, max κ_Te = {np.max(kap_Te_init):.4f} (crit: {g_crit_e:.4f})")
print(f"Ions:      {initial_state_i}, max κ_Ti = {np.max(kap_Ti_init):.4f} (crit: {g_crit_i:.4f})")
print(f"Density:   max κ_n  = {np.max(kap_n_init):.4f} (crit: {g_crit_n:.4f})")
print(f"tau_ei = {transport_params['tau_ei']}")

# ==========================================
# Source functions
# ==========================================
# f(x, p_e, n_e, p_i, n_i) -> array
alpha_p = 0.05

def source_pe(x, p_e, n_e, p_i, n_i):
    """External heating into electrons only (e.g. ECRH)."""
    return 5.0 * np.exp(-(x**2) / (0.25*L)**2) * (1.0 + alpha_p * p_e)

def source_ne(x, p_e, n_e, p_i, n_i):
    """Electron particle source."""
    return 2.0 * np.exp(-(x**2) / (0.25*L)**2)

def source_pi(x, p_e, n_e, p_i, n_i):
    """Ion external heating (can be zero; ions get energy via coupling)."""
    return 0.0 * x   # zero: ions heated only by Q_ei

def source_ni(x, p_e, n_e, p_i, n_i):
    """Ion particle source (same as electrons for quasi-neutrality)."""
    return 2.0 * np.exp(-(x**2) / (0.25*L)**2)

# ==========================================
# Initial face fluxes for diagnostic plots
# ==========================================
bcs_init = _get_bcs_two_species(transport_params, p_e_ped, n_e_ped, p_i_ped, n_i_ped)
core_pe, edge_pe, core_ne, edge_ne, core_pi, edge_pi, core_ni, edge_ni = bcs_init

pe_bc0 = p_e_init.copy(); ne_bc0 = n_e_init.copy()
pi_bc0 = p_i_init.copy(); ni_bc0 = n_i_init.copy()
_apply_bc(pe_bc0, core_pe, edge_pe, dx)
_apply_bc(ne_bc0, core_ne, edge_ne, dx)
_apply_bc(pi_bc0, core_pi, edge_pi, dx)
_apply_bc(ni_bc0, core_ni, edge_ni, dx)

x_face = 0.5 * (x[1:] + x[:-1])
params_e_init = _get_species_params(transport_params, "e")
params_i_init = _get_species_params(transport_params, "i")

T_e_bc0 = pe_bc0 / np.maximum(ne_bc0, 1e-10)
T_i_bc0 = pi_bc0 / np.maximum(ni_bc0, 1e-10)

Q_e_init     = heat_flux_function(_log_kap(T_e_bc0, dx), x_face, params_e_init)
Q_i_init     = heat_flux_function(_log_kap(T_i_bc0, dx), x_face, params_i_init)
Gamma_e_init = particle_flux_function(_log_kap(ne_bc0, dx), x_face, params_e_init)
Gamma_i_init = particle_flux_function(_log_kap(ni_bc0, dx), x_face, params_i_init)

Q_ei_init  = collision_exchange(ne_bc0, pe_bc0, ni_bc0, pi_bc0, transport_params)

S_pe_init  = source_pe(x, pe_bc0, ne_bc0, pi_bc0, ni_bc0)
S_ne_init  = source_ne(x, pe_bc0, ne_bc0, pi_bc0, ni_bc0)
S_pi_init  = source_pi(x, pe_bc0, ne_bc0, pi_bc0, ni_bc0)
S_ni_init  = source_ni(x, pe_bc0, ne_bc0, pi_bc0, ni_bc0)

# ==========================================
# INITIAL DIAGNOSTIC PLOTS
# ==========================================
print(f"\n{'='*60}\nCREATING INITIAL PLOTS\n{'='*60}\n")

# 01a/b  Initial T_e and T_i profiles (overlay)
plot_comparison(x, T_e_init, T_i_init,
                r"$T_e$", r"$T_i$",
                r"$T(x)$", r"$\mathrm{Initial\ Temperature\ Profiles}$",
                "01a_initial_temperatures")

# 01b  Density profile
plot_single(x, n_init, r"$n$", r"$n(x)$",
            r"$\mathrm{Initial\ Density\ Profile}$",
            "01b_initial_density", color='C2')

# 01c  Pressure profiles
plot_comparison(x, p_e_init, p_i_init,
                r"$p_e = n T_e$", r"$p_i = n T_i$",
                r"$p(x)$", r"$\mathrm{Initial\ Pressure\ Profiles}$",
                "01c_initial_pressures")

# 02a  Temperature log-gradients (overlay)
plot_comparison(x, kap_Te_init, kap_Ti_init,
                r"$\kappa_{T_e}$", r"$\kappa_{T_i}$",
                r"$\kappa_T = -\partial_x \ln T$",
                r"$\mathrm{Initial\ Temperature\ Log\mbox{-}Gradients}$",
                "02a_initial_kap_T",
                crit_e=g_crit_e, crit_i=g_crit_i)

# 02b  Density log-gradient
plot_single(x, kap_n_init, r"$\kappa_n$", r"$\kappa_n = -\partial_x \ln n$",
            r"$\mathrm{Initial\ Density\ Log\mbox{-}Gradient}$",
            "02b_initial_kap_n", color='C2', crit=g_crit_n)

# 03a/b  Initial flux vs cumulative source (electrons and ions)
for flux_f, H_src, ylabel, title, filename, col in [
    (Q_e_init,     np.cumsum(S_pe_init) * dx, r"$Q_e(x)$",
     r"$\mathrm{Electron\ Heat\ Flux\ vs\ Source}$",    "03a_flux_Qe",     'C0'),
    (Q_i_init,     np.cumsum(S_pi_init) * dx, r"$Q_i(x)$",
     r"$\mathrm{Ion\ Heat\ Flux\ vs\ Source}$",         "03b_flux_Qi",     'C3'),
    (Gamma_e_init, np.cumsum(S_ne_init) * dx, r"$\Gamma_e(x)$",
     r"$\mathrm{Electron\ Particle\ Flux\ vs\ Source}$", "03c_flux_Gammae", 'C0'),
    (Gamma_i_init, np.cumsum(S_ni_init) * dx, r"$\Gamma_i(x)$",
     r"$\mathrm{Ion\ Particle\ Flux\ vs\ Source}$",      "03d_flux_Gammai", 'C3'),
]:
    fig, ax = plt.subplots()
    ax.plot(x_face, flux_f, label="Flux", color=col)
    ax.plot(x, H_src, linestyle="--", label=r"$\int_0^x S\,dx'$", color=col)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); style_plot(ax); save_and_show(filename)

# 03e  Initial Q_ei profile
plot_single(x, Q_ei_init, r"$Q_{ei}$",
            r"$Q_{ei}(x) = n_e\,(T_i - T_e)\,/\,\tau_{ei}$",
            r"$\mathrm{Initial\ Electron\mbox{-}Ion\ Exchange}$",
            "03e_Qei_initial", color='C4')

# ==========================================
# SOLVE
# ==========================================
print(f"\n{'='*60}\nSOLVING TWO-SPECIES SYSTEM\n{'='*60}\n")

saved_pe, saved_ne, saved_pi, saved_ni = solving_loop_two_species(
    p_e_init, n_e_init, p_i_init, n_i_init,
    dt, dx, Tmax, L,
    p_e_ped, n_e_ped, p_i_ped, n_i_ped,
    source_pe, source_ne, source_pi, source_ni,
    num_snapshots, transport_params
)

times = np.linspace(0, Tmax, len(saved_pe))
saved_Te = [p / np.maximum(n, 1e-10) for p, n in zip(saved_pe, saved_ne)]
saved_Ti = [p / np.maximum(n, 1e-10) for p, n in zip(saved_pi, saved_ni)]

# ==========================================
# Post-solve diagnostics
# ==========================================
total_pe_t = [np.trapezoid(p, x) for p in saved_pe]
total_pi_t = [np.trapezoid(p, x) for p in saved_pi]
total_ne_t = [np.trapezoid(n, x) for n in saved_ne]
total_ni_t = [np.trapezoid(n, x) for n in saved_ni]
total_Te_t = [np.trapezoid(T, x) for T in saved_Te]
total_Ti_t = [np.trapezoid(T, x) for T in saved_Ti]

edge_Qe_t = []; edge_Qi_t = []
edge_Ge_t = []; edge_Gi_t = []
max_gTe_t = []; max_gTi_t = []; max_gn_t = []
total_Spe_t = []; total_Spi_t = []
integral_Qei_t = []

for pe, ne, pi_, ni in zip(saved_pe, saved_ne, saved_pi, saved_ni):
    Qe, Ge, Qi, Gi, Qei = compute_fluxes_two_species(
        pe, ne, pi_, ni, dx, x,
        p_e_ped, n_e_ped, p_i_ped, n_i_ped, transport_params)
    edge_Qe_t.append(Qe[-1]); edge_Qi_t.append(Qi[-1])
    edge_Ge_t.append(Ge[-1]); edge_Gi_t.append(Gi[-1])

    Te_s = pe / np.maximum(ne, 1e-10)
    Ti_s = pi_ / np.maximum(ni, 1e-10)
    max_gTe_t.append(np.max(-np.gradient(np.log(np.maximum(Te_s, 1e-10)), dx)))
    max_gTi_t.append(np.max(-np.gradient(np.log(np.maximum(Ti_s, 1e-10)), dx)))
    max_gn_t.append( np.max(-np.gradient(np.log(np.maximum(ne, 1e-10)),   dx)))

    total_Spe_t.append(np.trapezoid(source_pe(x, pe, ne, pi_, ni), x))
    total_Spi_t.append(np.trapezoid(source_pi(x, pe, ne, pi_, ni), x))
    integral_Qei_t.append(np.trapezoid(np.abs(Qei), x))

edge_Qe_t = np.array(edge_Qe_t); edge_Qi_t = np.array(edge_Qi_t)
edge_Ge_t = np.array(edge_Ge_t); edge_Gi_t = np.array(edge_Gi_t)

# ==========================================
# POST-SOLVE PLOTS
# ==========================================
print(f"\n{'='*60}\nCREATING POST-SOLVE PLOTS\n{'='*60}\n")

snap_stride = max(len(saved_pe) // 4, 1)

# ==========================================
# 04  Temperature evolution (electrons and ions)
# ==========================================
for saved_T, ylabel, title, filename, col in [
    (saved_Te, r"$T_e$", r"$\mathrm{Electron\ Temperature\ Evolution}$", "04a_Te_evolution", 'C0'),
    (saved_Ti, r"$T_i$", r"$\mathrm{Ion\ Temperature\ Evolution}$",      "04b_Ti_evolution", 'C3'),
]:
    fig, ax = plt.subplots()
    for i, prof in enumerate(saved_T[::snap_stride]):
        idx = min(i * snap_stride, len(times)-1)
        ax.plot(x, prof, label=rf"$t={times[idx]:.3f}$", color=colors_tab20[i % 20])
    ax.set_xlabel(r"$x$"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(ncol=2); style_plot(ax); save_and_show(filename)

# 04c  T_e vs T_i at final snapshot (overlay)
plot_comparison(x, saved_Te[-1], saved_Ti[-1],
                r"$T_e$ (final)", r"$T_i$ (final)",
                r"$T(x)$", r"$\mathrm{Final\ Temperature\ Comparison\ }T_e\ \mathrm{vs}\ T_i$",
                "04c_Te_vs_Ti_final")

# ==========================================
# 05  Density evolution
# ==========================================
for saved_n, ylabel, title, filename, col in [
    (saved_ne, r"$n_e$", r"$\mathrm{Electron\ Density\ Evolution}$", "05a_ne_evolution", 'C0'),
    (saved_ni, r"$n_i$", r"$\mathrm{Ion\ Density\ Evolution}$",      "05b_ni_evolution", 'C3'),
]:
    fig, ax = plt.subplots()
    for i, prof in enumerate(saved_n[::snap_stride]):
        idx = min(i * snap_stride, len(times)-1)
        ax.plot(x, prof, label=rf"$t={times[idx]:.3f}$", color=colors_tab20[i % 20])
    ax.set_xlabel(r"$x$"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(ncol=2); style_plot(ax); save_and_show(filename)

# ==========================================
# 06  Total integrated quantities
# ==========================================
fig, ax = plt.subplots()
ax.plot(times, total_pe_t, color='C0', label=r"$\int p_e\,dx$")
ax.plot(times, total_pi_t, color='C3', linestyle='--', label=r"$\int p_i\,dx$")
ax.plot(times, [a+b for a,b in zip(total_pe_t, total_pi_t)], color='k',
        linestyle=':', linewidth=2, label=r"$\int (p_e+p_i)\,dx$")
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\int p\,dx$")
ax.set_title(r"$\mathrm{Total\ Pressure\ (Electron,\ Ion,\ Combined)}$")
ax.legend(); style_plot(ax); save_and_show("06a_total_pressure")

fig, ax = plt.subplots()
ax.plot(times, total_ne_t, color='C0', label=r"$\int n_e\,dx$")
ax.plot(times, total_ni_t, color='C3', linestyle='--', label=r"$\int n_i\,dx$")
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\int n\,dx$")
ax.set_title(r"$\mathrm{Total\ Density}$")
ax.legend(); style_plot(ax); save_and_show("06b_total_density")

fig, ax = plt.subplots()
ax.plot(times, total_Te_t, color='C0', label=r"$\int T_e\,dx$")
ax.plot(times, total_Ti_t, color='C3', linestyle='--', label=r"$\int T_i\,dx$")
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\int T\,dx$")
ax.set_title(r"$\mathrm{Total\ Temperature}$")
ax.legend(); style_plot(ax); save_and_show("06c_total_temperature")

# ==========================================
# 07  Profile and gradient heatmaps
# ==========================================
for saved_T, label, title, filename, cmap in [
    (saved_Te, r"$T_e$",     r"$T_e(x,t)$", "07a_heatmap_Te", 'plasma'),
    (saved_Ti, r"$T_i$",     r"$T_i(x,t)$", "07b_heatmap_Ti", 'inferno'),
    (saved_ne, r"$n_e = n_i$", r"$n_e(x,t)$", "07c_heatmap_ne", 'viridis'),
]:
    arr = np.array(saved_T)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(arr.T, extent=[0, Tmax, 0, L], aspect='auto', origin='lower', cmap=cmap)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$x$"); ax.set_title(title)
    fig.colorbar(im).set_label(label); plt.tight_layout(); save_and_show(filename)

# Gradient heatmaps
gTe_all = np.array([-np.gradient(np.log(np.maximum(T, 1e-10)), dx) for T in saved_Te])
gTi_all = np.array([-np.gradient(np.log(np.maximum(T, 1e-10)), dx) for T in saved_Ti])
for g_arr, label, title, filename, cmap in [
    (gTe_all, r"$\kappa_{T_e}$", r"$\kappa_{T_e}(x,t)$", "07d_heatmap_kapTe", 'RdYlBu_r'),
    (gTi_all, r"$\kappa_{T_i}$", r"$\kappa_{T_i}(x,t)$", "07e_heatmap_kapTi", 'RdYlBu_r'),
]:
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(g_arr.T, extent=[0, Tmax, 0, L], aspect='auto', origin='lower', cmap=cmap)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$x$"); ax.set_title(title)
    fig.colorbar(im).set_label(label); plt.tight_layout(); save_and_show(filename)

# ==========================================
# 08  Q_ei collision exchange
# ==========================================
Qei_all = np.array([
    collision_exchange(ne, pe, ni, pi_,transport_params)
    for pe, ne, pi_, ni in zip(saved_pe, saved_ne, saved_pi, saved_ni)
])

fig, ax = plt.subplots()
for i, profile in enumerate(Qei_all[::snap_stride]):
    idx = min(i * snap_stride, len(times)-1)
    ax.plot(x, profile, label=rf"$t={times[idx]:.3f}$", color=colors_tab20[i % 20])
ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$Q_{ei}(x)$")
ax.set_title(r"$\mathrm{Electron\mbox{-}Ion\ Exchange\ } Q_{ei} = n_e(T_i-T_e)/\tau_{ei}$")
ax.legend(ncol=2); style_plot(ax); save_and_show("08a_Qei_profile")

fig, ax = plt.subplots()
ax.plot(times, integral_Qei_t, linewidth=2, color='C4', marker='o', markersize=4)
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\int |Q_{ei}|\,dx$")
ax.set_title(r"$\mathrm{Integrated\ Coupling\ Strength}$")
style_plot(ax); save_and_show("08b_Qei_integral")

# Heatmap of Q_ei(x,t)
fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(Qei_all.T, extent=[0, Tmax, 0, L], aspect='auto', origin='lower',
               cmap='RdBu_r')
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$x$")
ax.set_title(r"$Q_{ei}(x,t)$\ (red = ions$\to$electrons)")
fig.colorbar(im).set_label(r"$Q_{ei}$"); plt.tight_layout(); save_and_show("08c_Qei_heatmap")

# ==========================================
# 09  Power imbalance
# ==========================================
for total_S_t, edge_flux_t, ylabel, title, filename, col in [
    (total_Spe_t, edge_Qe_t, r"$Q_{e,\mathrm{edge}} - \int S_{pe}\,dx$",
     r"$\mathrm{Electron\ Heat\ Power\ Imbalance}$", "09a_imbalance_pe", 'C0'),
    (total_Spi_t, edge_Qi_t, r"$Q_{i,\mathrm{edge}} - \int S_{pi}\,dx$",
     r"$\mathrm{Ion\ Heat\ Power\ Imbalance}$",      "09b_imbalance_pi", 'C3'),
]:
    fig, ax = plt.subplots()
    ax.plot(times, np.array(edge_flux_t) - np.array(total_S_t),
            marker='o', markersize=4, color=col)
    ax.axhline(0, linestyle='--', color='k', linewidth=1)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(ylabel); ax.set_title(title)
    style_plot(ax); save_and_show(filename)

# ==========================================
# 10  Max gradient evolution
# ==========================================
fig, ax = plt.subplots()
ax.plot(times, max_gTe_t, color='C0', label=r"$\max(\kappa_{T_e})$")
ax.plot(times, max_gTi_t, color='C3', linestyle='--', label=r"$\max(\kappa_{T_i})$")
ax.axhline(g_crit_e, linestyle=':', color='C0', linewidth=1.5,
           label=r"$\kappa_{\mathrm{crit},e}$", alpha=0.8)
ax.axhline(g_crit_i, linestyle=':', color='C3', linewidth=1.5,
           label=r"$\kappa_{\mathrm{crit},i}$", alpha=0.8)
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\max(\kappa_T)$")
ax.set_title(r"$\mathrm{Max\ Temperature\ Log\mbox{-}Gradient\ vs\ Critical}$")
ax.legend(); style_plot(ax); save_and_show("10a_max_gT")

fig, ax = plt.subplots()
ax.plot(times, max_gn_t, color='C2', label=r"$\max(\kappa_n)$")
ax.axhline(g_crit_n, linestyle=':', color='C2', linewidth=1.5,
           label=r"$\kappa_{\mathrm{crit},n}$", alpha=0.8)
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\max(\kappa_n)$")
ax.set_title(r"$\mathrm{Max\ Density\ Log\mbox{-}Gradient}$")
ax.legend(); style_plot(ax); save_and_show("10b_max_gn")

# ==========================================
# 11  Gradient evolution at tracked points
# ==========================================
track_points  = np.linspace(0.4, 0.95, 8)
track_indices = [np.argmin(np.abs(x - pt)) for pt in track_points]

for saved_T, g_crit_val, ylabel, title, filename, col in [
    (saved_Te, g_crit_e, r"$\kappa_{T_e}$",
     r"$\mathrm{Electron\ Log\mbox{-}Gradient\ Evolution}$", "11a_kap_evol_Te", 'C0'),
    (saved_Ti, g_crit_i, r"$\kappa_{T_i}$",
     r"$\mathrm{Ion\ Log\mbox{-}Gradient\ Evolution}$",      "11b_kap_evol_Ti", 'C3'),
]:
    grad_hist = np.array([
        [-np.gradient(np.log(np.maximum(T, 1e-10)), dx)[idx] for idx in track_indices]
        for T in saved_T
    ])
    fig, ax = plt.subplots()
    for i, pt in enumerate(track_points):
        ax.plot(times, grad_hist[:, i], label=rf"$x={pt:.2f}$",
                color=colors_tab20[i % 20])
    ax.axhline(g_crit_val, linestyle='--', color='k', linewidth=1.5,
               label=r"$\kappa_\mathrm{crit}$")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(ncol=2, fontsize=9); style_plot(ax); save_and_show(filename)

# ==========================================
# 12  Flux balance at final snapshot
# ==========================================
pe_final = saved_pe[-1]; ne_final = saved_ne[-1]
pi_final = saved_pi[-1]; ni_final = saved_ni[-1]
Te_final = saved_Te[-1]; Ti_final = saved_Ti[-1]

Qe_f, Ge_f, Qi_f, Gi_f, Qei_f = compute_fluxes_two_species(
    pe_final, ne_final, pi_final, ni_final,
    dx, x, p_e_ped, n_e_ped, p_i_ped, n_i_ped, transport_params)

Spe_f = source_pe(x, pe_final, ne_final, pi_final, ni_final)
Spi_f = source_pi(x, pe_final, ne_final, pi_final, ni_final)

for flux_f, cumS, ylabel, title, filename, col in [
    (Qe_f, np.cumsum(Spe_f)*dx, r"$Q_e(x)$",
     rf"$\mathrm{{Electron\ Flux\ Balance\ }}(t={times[-1]:.3f})$", "12a_fluxbal_e", 'C0'),
    (Qi_f, np.cumsum(Spi_f)*dx, r"$Q_i(x)$",
     rf"$\mathrm{{Ion\ Flux\ Balance\ }}(t={times[-1]:.3f})$",      "12b_fluxbal_i", 'C3'),
]:
    fig, ax = plt.subplots()
    ax.plot(x_face, flux_f, label="Flux", color=col)
    ax.plot(x, cumS, linestyle="--", label=r"$\int_0^x S\,dx'$", color=col)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); style_plot(ax); save_and_show(filename)

# ==========================================
# 13  Effective diffusivity (final state)
# ==========================================
eps = 1e-6
for T_f, flux_fn, sp_label, ylabel, title, filename, col in [
    (Te_final, heat_flux_function, params_e_init,
     r"$\chi_{\mathrm{eff},e}$",
     r"$\mathrm{Electron\ Effective\ Diffusivity}$", "13a_chi_eff_e", 'C0'),
    (Ti_final, heat_flux_function, params_i_init,
     r"$\chi_{\mathrm{eff},i}$",
     r"$\mathrm{Ion\ Effective\ Diffusivity}$",      "13b_chi_eff_i", 'C3'),
]:
    kap_f   = -np.gradient(np.log(np.maximum(T_f, 1e-10)), dx)
    Q_plus  = flux_fn(kap_f + eps, x, sp_label)
    Q_minus = flux_fn(kap_f - eps, x, sp_label)
    chi_eff = (Q_plus - Q_minus) / (2 * eps)
    fig, ax = plt.subplots()
    ax.plot(x, chi_eff, linewidth=2, color=col)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(ylabel); ax.set_title(title)
    style_plot(ax); save_and_show(filename)

# ==========================================
# 14  Final state combined
# ==========================================
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax3 = ax.twinx()
ax3.spines['right'].set_position(('outward', 60))

l1 = ax.plot(x,  Te_final, linewidth=2.5, label=r"$T_e$", color='C0')
l2 = ax.plot(x,  Ti_final, linewidth=2.5, label=r"$T_i$", color='C3', linestyle='--')
l3 = ax2.plot(x, ne_final, linewidth=2.5, label=r"$n_e$", color='C2')
l4 = ax3.plot(x, Qei_f,   linewidth=2.0, label=r"$Q_{ei}$", color='C4', linestyle=':')

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$T(x)$", color='k')
ax2.set_ylabel(r"$n(x)$", color='C2')
ax3.set_ylabel(r"$Q_{ei}(x)$", color='C4')
ax2.tick_params(axis='y', labelcolor='C2')
ax3.tick_params(axis='y', labelcolor='C4')
ax.set_title(rf"$\mathrm{{Final\ State}}\ (t={times[-1]:.3f})$")
lines = l1 + l2 + l3 + l4
ax.legend(lines, [l.get_label() for l in lines], loc='upper right')
style_plot(ax); save_and_show("14_final_state_combined")

print(f"\n{'='*60}")
print("SIMULATION COMPLETE")
print(f"{'='*60}")
print(f"Final ∫p_e dx  = {total_pe_t[-1]:.4f}")
print(f"Final ∫p_i dx  = {total_pi_t[-1]:.4f}")
print(f"Final ∫n_e dx  = {total_ne_t[-1]:.4f}")
print(f"Final ∫T_e dx  = {total_Te_t[-1]:.4f}")
print(f"Final ∫T_i dx  = {total_Ti_t[-1]:.4f}")
print(f"Final ∫|Q_ei|dx = {integral_Qei_t[-1]:.4f}")
print(f"Plots saved to 'plots/' directory\n")
