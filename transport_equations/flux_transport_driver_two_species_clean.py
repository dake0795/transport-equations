#!/usr/bin/env python3
"""Two-species flux transport driver (import-safe).

Generated from flux_transport_driver_two_species_clean.py.

- main(..., do_plots=True, do_videos=True)
- If do_plots=False: no matplotlib import, no figures created.
- If do_videos=False: no animations/MP4s generated.
"""

import os
import numpy as np

from .flux_transport_model_two_species_clean import (
    heat_flux_function, particle_flux_function,
    collision_exchange, alpha_heating, bremsstrahlung,
    compute_fluxes, solving_loop,
    _get_bcs, _apply_bc, _get_species_params, _log_kap,
)

"""
Two-species (electron + ion) transport driver.
Three evolved fields: p_e, p_i, n  (quasi-neutrality: n_e = n_i = n).
"""

def main(
    skip_solve: bool = False,
    data_file: str = "plots_2sp/run_data.npz",
    do_plots: bool = True,
    do_videos: bool = True,
):
    """Run the two-species simulation."""

    if do_plots or do_videos:
        import matplotlib
        if not do_plots:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.cm as cm
        from matplotlib.ticker import AutoMinorLocator

        FIGSIZE = (6.33, 4.33)
        plt.rcParams.update({
            "text.usetex": bool(do_plots),
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

        PLOTS_DIR = "plots_2sp"

        def style_plot(ax):
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            ax.yaxis.set_minor_locator(AutoMinorLocator(4))
            ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6)
            ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
            ax.set_axisbelow(True)
            plt.tight_layout()

        def save_and_show(filename):
            os.makedirs(PLOTS_DIR, exist_ok=True)
            filepath = os.path.join(PLOTS_DIR, f"{filename}.pdf")
            plt.savefig(filepath, format="pdf", bbox_inches="tight")
            print(f"Saved: {filepath}")
            plt.show()

        def plot_single(x, y, label, ylabel, title, filename, color='C0', crit=None):
            fig, ax = plt.subplots()
            ax.plot(x, y, linewidth=2, label=label, color=color)
            if crit is not None:
                ax.axhline(crit, linestyle="--", linewidth=1.5, color=color, alpha=0.7)
            ax.set_xlabel(r""); ax.set_ylabel(ylabel); ax.set_title(title)
            ax.legend(); style_plot(ax); save_and_show(filename)

        def plot_comparison(x, ye, yi, label_e, label_i, ylabel, title, filename, crit_e=None, crit_i=None):
            fig, ax = plt.subplots()
            ax.plot(x, ye, linewidth=2, label=label_e, color='C0')
            ax.plot(x, yi, linewidth=2, label=label_i, color='C3', linestyle='--')
            if crit_e is not None:
                ax.axhline(crit_e, linestyle=":", color='C0', linewidth=1.5, alpha=0.8)
            if crit_i is not None:
                ax.axhline(crit_i, linestyle=":", color='C3', linewidth=1.5, alpha=0.8)
            ax.set_xlabel(r""); ax.set_ylabel(ylabel); ax.set_title(title)
            ax.legend(); style_plot(ax); save_and_show(filename)
    else:
        plt = None
        animation = None
        cm = None

    # ==========================================
    # Plotting style
    # ==========================================

    PLOTS_DIR = "plots_2sp"


    def log_kap_cell(prof, dx):
        """Log gradient at cell centres."""
        return -np.gradient(np.log(np.maximum(prof, 1e-10)), dx)


    def power_law_profile(x, L, f_core, f_ped, m=2, kap_target=None):
        """Power-law profile with optional log-gradient target."""
        shape = 1.0 - (x / L)**m
        if kap_target is not None:
            trial = f_ped + (f_core - f_ped) * shape
            kap_max = np.max(-np.gradient(np.log(np.maximum(trial, 1e-10)), x[1]-x[0]))
            if kap_max > 0:
                return f_ped + (kap_target / kap_max) * (f_core - f_ped) * shape
        return f_ped + (f_core - f_ped) * shape


    # ==========================================
    # Domain
    # ==========================================
    L  = 1.0
    dx = 0.005
    dt = None       # None = auto (largest stable dt for RK4 given nu4, chi, dx)
    T  = 0.2
    num_video_frames  = 300
    num_plot_snapshots = 8

    x = np.linspace(0, L, int(L / dx))

    # ==========================================
    # Initial state control
    # ==========================================
    initial_state_e = "supercritical"
    initial_state_i = "supercritical"
    initial_state_n = "subcritical"

    # ==========================================
    # Power balance mode
    # ==========================================
    # "initial_only" : scale sources once at t=0 to match initial edge fluxes
    # None           : no power balance enforcement
    power_balance_mode = None

    # ==========================================
    # Transport parameters
    # ==========================================
    transport_params = {
        # --- Electron heat transport ---
        "chi0_e":      10.0,
        "chi_core_e":  10.0,
        "chi_RR_e":    0.05,
        "g_c_e":        4.0,
        "g_stiff_e":    3.0,
        "n_stiff_e":    2,

        # --- Electron particle transport ---
        "chi0_n_e":    1.0,
        "chi_core_n_e": 1.0,
        "chi_RR_n_e":  0.01,
        "g_c_n_e":     2.0,
        "g_stiff_n_e": 1.5,
        "n_stiff_n_e": 2,
        "V_n_e":       0.0,

        # --- Ion heat transport ---
        "chi0_i":      3.0,
        "chi_core_i":  3.0,
        "chi_RR_i":    0.05,
        "g_c_i":        3.0,
        "g_stiff_i":    2.5,
        "n_stiff_i":    2,

        # --- Spatial structure ---
        "boundaries":    [],
        "deltas":        [],
        "flux_models":   ["nl"],
        "flux_models_n": ["nl"],
        "nu4":           0.0,
        "n_smooth":      2,

        # --- Electron-ion coupling ---
        "tau_ei":      0.05,

        # --- Physics sources ---
        "C_brem":     0.03,
        "Z_eff":      1.0,
        "C_alpha":    1e-3,
        "T_ref_keV":  10.0,
        "n_ref_20":    1.0,
        "Z_i":         1,
        "A_i":         2.5,
        "f_deuterium": 0.5,
        "f_tritium":   0.5,
    }

    # ==========================================
    # Auto timestep
    # ==========================================
    if dt is None:
        _CFL_RK4 = 2.785
        _safety  = 0.8

        nu4     = transport_params.get("nu4", 0.0)
        chi_max = max(
            transport_params.get("chi0_e", 0.0),
            transport_params.get("chi_core_e", 0.0),
            transport_params.get("chi0_i", 0.0),
            transport_params.get("chi_core_i", 0.0),
        )

        dt_candidates = []
        if nu4 > 0:
            dt_candidates.append(_CFL_RK4 / (16 * nu4 / dx**4))
        if chi_max > 0:
            dt_candidates.append(_CFL_RK4 / (2 * chi_max / dx**2))

        dt = _safety * min(dt_candidates) if dt_candidates else 1e-6
        print(f"[auto-dt] dt = {dt:.2e}  (nu4={nu4:.0e}, chi_max={chi_max:.1f}, dx={dx})")

    # ==========================================
    # Critical gradients
    # ==========================================
    g_c_e    = transport_params["g_c_e"]
    g_c_i    = transport_params["g_c_i"]
    g_crit_e = g_c_e / 2.0
    g_crit_i = g_c_i / 2.0
    g_c_n    = transport_params["g_c_n_e"]
    g_crit_n = g_c_n / 2.0

    # ==========================================
    # Initial profiles
    # ==========================================
    T_e_core = 2.0;  T_e_ped = 0.5
    T_i_core = 1.5;  T_i_ped = 0.4
    n_core   = 2.0;  n_ped   = 0.2
    n_ped_sub = 1.2

    m_T = 2
    m_n = 3
    kap_ratio_super = 1.8
    kap_ratio_sub   = 0.6

    n_ped_eff = n_ped if initial_state_n == "supercritical" else n_ped_sub

    kap_target_Te = (kap_ratio_super if initial_state_e == "supercritical" else kap_ratio_sub) * g_crit_e
    kap_target_Ti = (kap_ratio_super if initial_state_i == "supercritical" else kap_ratio_sub) * g_crit_i
    kap_target_n  = (kap_ratio_super if initial_state_n == "supercritical" else kap_ratio_sub) * g_crit_n

    T_e_init = power_law_profile(x, L, T_e_core, T_e_ped, m=m_T, kap_target=kap_target_Te)
    T_i_init = power_law_profile(x, L, T_i_core, T_i_ped, m=m_T, kap_target=kap_target_Ti)
    n_init   = power_law_profile(x, L, n_core,   n_ped_eff, m=m_n, kap_target=kap_target_n)

    # Boundary values
    p_e_ped = T_e_ped * n_ped_eff
    p_i_ped = T_i_ped * n_ped_eff

    # Evolved fields: pressure = n * T
    p_e_init = T_e_init * n_init
    p_i_init = T_i_init * n_init

    kap_Te_init = log_kap_cell(T_e_init, dx)
    kap_Ti_init = log_kap_cell(T_i_init, dx)
    kap_n_init  = log_kap_cell(n_init,   dx)

    print(f"\n{'='*60}\nINITIAL CONDITIONS\n{'='*60}")
    print(f"Electrons: {initial_state_e}, max kap_Te = {np.max(kap_Te_init):.4f} (crit: {g_crit_e:.4f})")
    print(f"Ions:      {initial_state_i}, max kap_Ti = {np.max(kap_Ti_init):.4f} (crit: {g_crit_i:.4f})")
    print(f"Density:   {initial_state_n}, max kap_n  = {np.max(kap_n_init):.4f} (crit: {g_crit_n:.4f})")

    # ==========================================
    # Source functions
    # ==========================================
    # f(x, p_e, p_i, n) -> array

    def source_pe(x, p_e, p_i, n):
        """External heating into electrons (e.g. ECRH)."""
        return 5.0 * np.exp(-(x**2) / (0.25*L)**2)

    def source_pi(x, p_e, p_i, n):
        """Ion external heating (zero: ions heated only by Q_ei)."""
        return 0.0 * x

    def source_n(x, p_e, p_i, n):
        """Particle source."""
        return 2.0 * np.exp(-(x**2) / (0.25*L)**2)

    # ==========================================
    # Initial fluxes (for diagnostics)
    # ==========================================
    bcs = _get_bcs(transport_params, p_e_ped, p_i_ped, n_ped_eff)
    core_pe, edge_pe, core_pi, edge_pi, core_n, edge_n = bcs

    pe_bc0 = p_e_init.copy(); pi_bc0 = p_i_init.copy(); n_bc0 = n_init.copy()
    _apply_bc(pe_bc0, core_pe, edge_pe, dx)
    _apply_bc(pi_bc0, core_pi, edge_pi, dx)
    _apply_bc(n_bc0,  core_n,  edge_n,  dx)

    x_face   = 0.5 * (x[1:] + x[:-1])
    params_e = _get_species_params(transport_params, "e")
    params_i = _get_species_params(transport_params, "i")

    T_e_bc0 = pe_bc0 / np.maximum(n_bc0, 1e-10)
    T_i_bc0 = pi_bc0 / np.maximum(n_bc0, 1e-10)

    Q_e_init     = heat_flux_function(_log_kap(T_e_bc0, dx), x_face, params_e)
    Q_i_init     = heat_flux_function(_log_kap(T_i_bc0, dx), x_face, params_i)
    Gamma_init   = particle_flux_function(_log_kap(n_bc0, dx), x_face, params_e)
    Q_ei_init    = collision_exchange(n_bc0, pe_bc0, pi_bc0, transport_params)

    S_pe_init = source_pe(x, pe_bc0, pi_bc0, n_bc0)
    S_pi_init = source_pi(x, pe_bc0, pi_bc0, n_bc0)
    S_n_init  = source_n(x, pe_bc0, pi_bc0, n_bc0)

    print(f"\nInitial Q_e edge  = {Q_e_init[-1]:.4f}")
    print(f"Initial Q_i edge  = {Q_i_init[-1]:.4f}")
    print(f"Initial Gamma edge = {Gamma_init[-1]:.4f}")
    print(f"Initial int S_pe  = {np.trapezoid(S_pe_init, x):.4f}")
    print(f"Initial int S_n   = {np.trapezoid(S_n_init, x):.4f}")

    if do_plots and plt is not None:
        # ==========================================
        # INITIAL DIAGNOSTIC PLOTS
        # ==========================================

        # 01  Temperature profiles
        plot_comparison(x, T_e_init, T_i_init,
                        r"$T_e$", r"$T_i$",
                        r"$T(x)$", r"$\mathrm{Initial\ Temperature\ Profiles}$",
                        "01a_initial_temperatures")

        # 01b  Density
        plot_single(x, n_init, r"$n$", r"$n(x)$",
                    r"$\mathrm{Initial\ Density\ Profile}$",
                    "01b_initial_density", color='C2')

        # 01c  Pressure
        plot_comparison(x, p_e_init, p_i_init,
                        r"$p_e = nT_e$", r"$p_i = nT_i$",
                        r"$p(x)$", r"$\mathrm{Initial\ Pressure\ Profiles}$",
                        "01c_initial_pressures")

        # 02  Log-gradients
        plot_comparison(x, kap_Te_init, kap_Ti_init,
                        r"$\kappa_{T_e}$", r"$\kappa_{T_i}$",
                        r"$\kappa_T = -\partial_x \ln T$",
                        r"$\mathrm{Initial\ Temperature\ Log\mbox{-}Gradients}$",
                        "02a_initial_kap_T", crit_e=g_crit_e, crit_i=g_crit_i)

        plot_single(x, kap_n_init, r"$\kappa_n$", r"$\kappa_n = -\partial_x \ln n$",
                    r"$\mathrm{Initial\ Density\ Log\mbox{-}Gradient}$",
                    "02b_initial_kap_n", color='C2', crit=g_crit_n)

        # 03  Flux vs cumulative source
        for flux_f, H_src, ylabel, title, filename, col in [
            (Q_e_init,   np.cumsum(S_pe_init)*dx, r"$Q_e(x)$",
             r"$\mathrm{Electron\ Heat\ Flux\ vs\ Source}$",    "03a_flux_Qe",    'C0'),
            (Q_i_init,   np.cumsum(S_pi_init)*dx, r"$Q_i(x)$",
             r"$\mathrm{Ion\ Heat\ Flux\ vs\ Source}$",         "03b_flux_Qi",    'C3'),
            (Gamma_init, np.cumsum(S_n_init)*dx,  r"$\Gamma(x)$",
             r"$\mathrm{Particle\ Flux\ vs\ Source}$",          "03c_flux_Gamma", 'C2'),
        ]:
            fig, ax = plt.subplots()
            ax.plot(x_face, flux_f, label="Flux", color=col)
            ax.plot(x, H_src, linestyle="--", label=r"$\int_0^x S\,dx'$", color=col)
            ax.set_xlabel(r"$x$"); ax.set_ylabel(ylabel); ax.set_title(title)
            ax.legend(); style_plot(ax); save_and_show(filename)

        # 03d  Q_ei
        plot_single(x, Q_ei_init, r"$Q_{ei}$",
                    r"$Q_{ei}(x)$",
                    r"$\mathrm{Initial\ Electron\mbox{-}Ion\ Exchange}$",
                    "03d_Qei_initial", color='C4')

    # ==========================================
    # Solve (or load)
    # ==========================================

    # Build source functions (with optional initial_only power balance)
    source_pe_fn = source_pe
    source_pi_fn = source_pi
    source_n_fn  = source_n

    if power_balance_mode == "initial_only":
        int_Spe0 = np.trapezoid(S_pe_init, x)
        int_Spi0 = np.trapezoid(S_pi_init, x)
        int_Sn0  = np.trapezoid(S_n_init,  x)

        Qe_edge0    = Q_e_init[-1]
        Qi_edge0    = Q_i_init[-1]
        Gamma_edge0 = Gamma_init[-1]

        scale_pe = max(Qe_edge0 / int_Spe0, 0.0)    if int_Spe0 > 0 else 1.0
        scale_pi = max(Qi_edge0 / int_Spi0, 0.0)    if int_Spi0 > 0 else 1.0
        scale_n  = max(Gamma_edge0 / int_Sn0, 0.0)  if int_Sn0  > 0 else 1.0

        def source_pe_fn(x, pe, pi_, n, _s=scale_pe): return _s * source_pe(x, pe, pi_, n)
        def source_pi_fn(x, pe, pi_, n, _s=scale_pi): return _s * source_pi(x, pe, pi_, n)
        def source_n_fn(x, pe, pi_, n, _s=scale_n):   return _s * source_n(x, pe, pi_, n)
        print(f"\n[initial_only] scale_pe={scale_pe:.4f}, scale_pi={scale_pi:.4f}, scale_n={scale_n:.4f}")

    if skip_solve and os.path.exists(data_file):
        print(f"\n[skip_solve] Loading data from {data_file}")
        _data = np.load(data_file)
        saved_pe = _data["saved_pe"]
        saved_pi = _data["saved_pi"]
        saved_n  = _data["saved_n"]
        times    = _data["times"]
        print(f"  Loaded {len(saved_pe)} frames, T={times[-1]:.4f}")
    else:
        if skip_solve:
            print(f"\n[skip_solve] {data_file} not found -- running solver")

        print(f"\n{'='*60}\nSOLVING\n{'='*60}\n")
        saved_pe, saved_pi, saved_n = solving_loop(
            p_e_init, p_i_init, n_init,
            dt, dx, T, L,
            p_e_ped, p_i_ped, n_ped_eff,
            source_pe_fn, source_pi_fn, source_n_fn,
            num_video_frames, transport_params,
        )
        times = np.linspace(0, T, len(saved_pe))

    # Derived temperature arrays
    saved_Te = np.array([pe / np.maximum(n, 1e-10) for pe, n in zip(saved_pe, saved_n)])
    saved_Ti = np.array([pi / np.maximum(n, 1e-10) for pi, n in zip(saved_pi, saved_n)])

    # Snapshot indices for static overlay plots
    snap_indices = np.linspace(0, len(saved_pe) - 1, num_plot_snapshots, dtype=int)

    # ==========================================
    # Diagnostics
    # ==========================================
    total_pe_t = []; total_pi_t = []; total_n_t = []
    edge_Qe_t  = []; edge_Qi_t  = []; edge_Gamma_t = []
    total_Spe_t = []; total_Spi_t = []; total_Sn_t = []
    max_gTe_t  = []; max_gTi_t  = []; max_gn_t  = []
    int_Qei_t  = []

    for pe, pi_, nn in zip(saved_pe, saved_pi, saved_n):
        total_pe_t.append(np.trapezoid(pe, x))
        total_pi_t.append(np.trapezoid(pi_, x))
        total_n_t.append(np.trapezoid(nn, x))

        Qe, Qi, Gam, Qei = compute_fluxes(pe, pi_, nn, dx, x, p_e_ped, p_i_ped, n_ped_eff, transport_params)
        edge_Qe_t.append(Qe[-1]); edge_Qi_t.append(Qi[-1]); edge_Gamma_t.append(Gam[-1])
        int_Qei_t.append(np.trapezoid(np.abs(Qei), x))

        total_Spe_t.append(np.trapezoid(source_pe_fn(x, pe, pi_, nn), x))
        total_Spi_t.append(np.trapezoid(source_pi_fn(x, pe, pi_, nn), x))
        total_Sn_t.append(np.trapezoid(source_n_fn(x, pe, pi_, nn), x))

        Te = pe / np.maximum(nn, 1e-10)
        Ti = pi_ / np.maximum(nn, 1e-10)
        max_gTe_t.append(np.max(log_kap_cell(Te, dx)))
        max_gTi_t.append(np.max(log_kap_cell(Ti, dx)))
        max_gn_t.append(np.max(log_kap_cell(nn, dx)))

    total_pe_t = np.array(total_pe_t); total_pi_t = np.array(total_pi_t)
    total_n_t  = np.array(total_n_t)
    edge_Qe_t  = np.array(edge_Qe_t);  edge_Qi_t  = np.array(edge_Qi_t)
    edge_Gamma_t = np.array(edge_Gamma_t)
    total_Spe_t = np.array(total_Spe_t); total_Spi_t = np.array(total_Spi_t)
    total_Sn_t  = np.array(total_Sn_t)
    max_gTe_t = np.array(max_gTe_t); max_gTi_t = np.array(max_gTi_t)
    max_gn_t  = np.array(max_gn_t)
    int_Qei_t = np.array(int_Qei_t)

    # ==========================================
    # Save all data
    # ==========================================
    os.makedirs(PLOTS_DIR, exist_ok=True)
    np.savez(
        data_file,
        x=x, times=times,
        saved_pe=saved_pe, saved_pi=saved_pi, saved_n=saved_n,
        total_pe_t=total_pe_t, total_pi_t=total_pi_t, total_n_t=total_n_t,
        edge_Qe_t=edge_Qe_t, edge_Qi_t=edge_Qi_t, edge_Gamma_t=edge_Gamma_t,
        total_Spe_t=total_Spe_t, total_Spi_t=total_Spi_t, total_Sn_t=total_Sn_t,
        max_gTe_t=max_gTe_t, max_gTi_t=max_gTi_t, max_gn_t=max_gn_t,
        int_Qei_t=int_Qei_t,
        g_crit_e=g_crit_e, g_crit_i=g_crit_i, g_crit_n=g_crit_n,
    )
    print(f"Saved run data to {data_file}")

    if do_plots and plt is not None:
        # ==========================================
        # POST-SOLVE PLOTS
        # ==========================================

        # 04  Temperature evolution (static snapshots)
        for saved_T, ylabel, title, filename in [
            (saved_Te, r"$T_e$", r"$\mathrm{Electron\ Temperature\ Evolution}$", "04a_Te_evolution"),
            (saved_Ti, r"$T_i$", r"$\mathrm{Ion\ Temperature\ Evolution}$",      "04b_Ti_evolution"),
        ]:
            fig, ax = plt.subplots()
            for i in snap_indices:
                ax.plot(x, saved_T[i], label=rf"$t={times[i]:.3f}$")
            ax.set_xlabel(r"$x$"); ax.set_ylabel(ylabel); ax.set_title(title)
            ax.legend(ncol=2); style_plot(ax); save_and_show(filename)

        # 04c  Final T_e vs T_i
        plot_comparison(x, saved_Te[-1], saved_Ti[-1],
                        r"$T_e$ (final)", r"$T_i$ (final)",
                        r"$T(x)$", r"$\mathrm{Final\ Temperature\ Comparison}$",
                        "04c_Te_vs_Ti_final")

        # 05  Density evolution
        fig, ax = plt.subplots()
        for i in snap_indices:
            ax.plot(x, saved_n[i], label=rf"$t={times[i]:.3f}$")
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$n$")
        ax.set_title(r"$\mathrm{Density\ Evolution}$")
        ax.legend(ncol=2); style_plot(ax); save_and_show("05_n_evolution")

        # 06  Total integrated quantities
        fig, ax = plt.subplots()
        ax.plot(times, total_pe_t, color='C0', label=r"$\int p_e\,dx$")
        ax.plot(times, total_pi_t, color='C3', linestyle='--', label=r"$\int p_i\,dx$")
        ax.plot(times, total_pe_t + total_pi_t, color='k', linestyle=':', linewidth=2,
                label=r"$\int (p_e+p_i)\,dx$")
        ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\int p\,dx$")
        ax.set_title(r"$\mathrm{Total\ Pressure}$")
        ax.legend(); style_plot(ax); save_and_show("06a_total_pressure")

        fig, ax = plt.subplots()
        ax.plot(times, total_n_t, color='C2')
        ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\int n\,dx$")
        ax.set_title(r"$\mathrm{Total\ Density}$")
        style_plot(ax); save_and_show("06b_total_density")

        # 07  Heatmaps
        for arr, label, title, filename, cmap_name in [
            (saved_Te, r"$T_e$", r"$T_e(x,t)$", "07a_heatmap_Te", 'plasma'),
            (saved_Ti, r"$T_i$", r"$T_i(x,t)$", "07b_heatmap_Ti", 'inferno'),
            (saved_n,  r"$n$",   r"$n(x,t)$",   "07c_heatmap_n",  'viridis'),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(np.array(arr).T, extent=[0, T, 0, L],
                           aspect='auto', origin='lower', cmap=cmap_name)
            ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$x$"); ax.set_title(title)
            fig.colorbar(im).set_label(label); plt.tight_layout()
            save_and_show(filename)

        # Gradient heatmaps
        gTe_all = np.array([log_kap_cell(T, dx) for T in saved_Te])
        gTi_all = np.array([log_kap_cell(T, dx) for T in saved_Ti])
        for g_arr, label, title, filename in [
            (gTe_all, r"$\kappa_{T_e}$", r"$\kappa_{T_e}(x,t)$", "07d_heatmap_kapTe"),
            (gTi_all, r"$\kappa_{T_i}$", r"$\kappa_{T_i}(x,t)$", "07e_heatmap_kapTi"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(g_arr.T, extent=[0, T, 0, L], aspect='auto', origin='lower', cmap='RdYlBu_r')
            ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$x$"); ax.set_title(title)
            fig.colorbar(im).set_label(label); plt.tight_layout()
            save_and_show(filename)

        # 08  Q_ei
        Qei_all = np.array([collision_exchange(nn, pe, pi_, transport_params)
                             for pe, pi_, nn in zip(saved_pe, saved_pi, saved_n)])

        fig, ax = plt.subplots()
        for i in snap_indices:
            ax.plot(x, Qei_all[i], label=rf"$t={times[i]:.3f}$")
        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$Q_{ei}(x)$")
        ax.set_title(r"$\mathrm{Electron\mbox{-}Ion\ Exchange}$")
        ax.legend(ncol=2); style_plot(ax); save_and_show("08a_Qei_profile")

        fig, ax = plt.subplots()
        ax.plot(times, int_Qei_t, linewidth=2, color='C4', marker='o', markersize=3)
        ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\int |Q_{ei}|\,dx$")
        ax.set_title(r"$\mathrm{Integrated\ Coupling\ Strength}$")
        style_plot(ax); save_and_show("08b_Qei_integral")

        # 10  Max gradient evolution
        fig, ax = plt.subplots()
        ax.plot(times, max_gTe_t, color='C0', label=r"$\max(\kappa_{T_e})$")
        ax.plot(times, max_gTi_t, color='C3', linestyle='--', label=r"$\max(\kappa_{T_i})$")
        ax.axhline(g_crit_e, linestyle=':', color='C0', linewidth=1.5, alpha=0.8,
                   label=r"$\kappa_{\mathrm{crit},e}$")
        ax.axhline(g_crit_i, linestyle=':', color='C3', linewidth=1.5, alpha=0.8,
                   label=r"$\kappa_{\mathrm{crit},i}$")
        ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\max(\kappa_T)$")
        ax.set_title(r"$\mathrm{Max\ Temperature\ Log\mbox{-}Gradient}$")
        ax.legend(); style_plot(ax); save_and_show("10a_max_gT")

        fig, ax = plt.subplots()
        ax.plot(times, max_gn_t, color='C2')
        ax.axhline(g_crit_n, linestyle=':', color='C2', linewidth=1.5, alpha=0.8,
                   label=r"$\kappa_{\mathrm{crit},n}$")
        ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\max(\kappa_n)$")
        ax.set_title(r"$\mathrm{Max\ Density\ Log\mbox{-}Gradient}$")
        ax.legend(); style_plot(ax); save_and_show("10b_max_gn")

        # 11  Gradient evolution at tracked points
        track_points  = np.linspace(0.4, 0.95, 8)
        track_indices = [np.argmin(np.abs(x - pt)) for pt in track_points]

        for saved_T, g_crit_val, ylabel, title, filename in [
            (saved_Te, g_crit_e, r"$\kappa_{T_e}$",
             r"$\mathrm{Electron\ Log\mbox{-}Gradient\ Evolution}$", "11a_kap_evol_Te"),
            (saved_Ti, g_crit_i, r"$\kappa_{T_i}$",
             r"$\mathrm{Ion\ Log\mbox{-}Gradient\ Evolution}$",      "11b_kap_evol_Ti"),
        ]:
            grad_hist = np.array([
                [log_kap_cell(T, dx)[idx] for idx in track_indices]
                for T in saved_T
            ])
            fig, ax = plt.subplots()
            for i, pt in enumerate(track_points):
                ax.plot(times, grad_hist[:, i], label=rf"$x={pt:.2f}$")
            ax.axhline(g_crit_val, linestyle='--', color='k', linewidth=1.5,
                       label=r"$\kappa_\mathrm{crit}$")
            ax.set_xlabel(r"$t$"); ax.set_ylabel(ylabel); ax.set_title(title)
            ax.legend(ncol=2, fontsize=9); style_plot(ax); save_and_show(filename)

        # 14  Final state combined
        Te_final = saved_Te[-1]; Ti_final = saved_Ti[-1]
        n_final  = saved_n[-1]
        _, _, _, Qei_f = compute_fluxes(saved_pe[-1], saved_pi[-1], n_final,
                                        dx, x, p_e_ped, p_i_ped, n_ped_eff, transport_params)

        fig, ax = plt.subplots()
        ax2 = ax.twinx(); ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        l1 = ax.plot(x, Te_final, linewidth=2.5, label=r"$T_e$", color='C0')
        l2 = ax.plot(x, Ti_final, linewidth=2.5, label=r"$T_i$", color='C3', linestyle='--')
        l3 = ax2.plot(x, n_final, linewidth=2.5, label=r"$n$", color='C2')
        l4 = ax3.plot(x, Qei_f, linewidth=2.0, label=r"$Q_{ei}$", color='C4', linestyle=':')
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$T(x)$")
        ax2.set_ylabel(r"$n(x)$", color='C2'); ax2.tick_params(axis='y', labelcolor='C2')
        ax3.set_ylabel(r"$Q_{ei}(x)$", color='C4'); ax3.tick_params(axis='y', labelcolor='C4')
        ax.set_title(rf"$\mathrm{{Final\ State}}\ (t={times[-1]:.3f})$")
        lines = l1 + l2 + l3 + l4
        ax.legend(lines, [l.get_label() for l in lines], loc='upper right')
        style_plot(ax); save_and_show("14_final_state_combined")

        print(f"\n{'='*60}")
        print("SIMULATION COMPLETE")
        print(f"{'='*60}")
        print(f"Final int p_e dx = {total_pe_t[-1]:.4f}")
        print(f"Final int p_i dx = {total_pi_t[-1]:.4f}")
        print(f"Final int n dx   = {total_n_t[-1]:.4f}")

    if  do_videos and plt is not None:
        # ==========================================
        # VIDEO 1: Temperature profile evolution (T_e and T_i)
        # ==========================================

        Te_min = float(min(np.min(saved_Te), np.min(saved_Ti)))
        Te_max = float(max(np.max(saved_Te), np.max(saved_Ti)))
        T_pad  = 0.05 * (Te_max - Te_min)

        fig_v, ax_v = plt.subplots(figsize=FIGSIZE)
        (line_Te,) = ax_v.plot([], [], lw=1.8, color="C0", label=r"$T_e(x,t)$")
        (line_Ti,) = ax_v.plot([], [], lw=1.8, color="C3", linestyle="--", label=r"$T_i(x,t)$")

        ax_v.set_xlim(0, L)
        ax_v.set_ylim(Te_min - T_pad, Te_max + T_pad)
        ax_v.set_xlabel(r"$x$"); ax_v.set_ylabel(r"$T$")
        ax_v.legend(loc="upper right", fontsize=11)
        ax_v.minorticks_on()
        ax_v.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax_v.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax_v.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        ax_v.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
        ax_v.set_axisbelow(True)
        fig_v.tight_layout(rect=[0, 0, 1, 0.93])

        def init_T():
            line_Te.set_data([], []); line_Ti.set_data([], [])
            ax_v.set_title(r"$\mathrm{Temperature\ Evolution},\ t=0.000$", fontsize=11)
            return line_Te, line_Ti

        def update_T(frame):
            line_Te.set_data(x, saved_Te[frame])
            line_Ti.set_data(x, saved_Ti[frame])
            ax_v.set_title(rf"$\mathrm{{Temperature\ Evolution}},\ t={times[frame]:.3f}$", fontsize=11)
            return line_Te, line_Ti

        anim_T = animation.FuncAnimation(fig_v, update_T, frames=len(saved_Te),
                                          init_func=init_T, blit=True, interval=30)
        video_path = os.path.join(PLOTS_DIR, "temperature_evolution.mp4")
        writer = animation.FFMpegWriter(fps=30, bitrate=2000)
        anim_T.save(video_path, writer=writer, dpi=200)
        print(f"Saved: {video_path}")
        plt.close(fig_v)

        # ==========================================
        # VIDEO 2: Pressure profile evolution (p_e and p_i)
        # ==========================================

        pe_min = float(min(np.min(saved_pe), np.min(saved_pi)))
        pe_max = float(max(np.max(saved_pe), np.max(saved_pi)))
        p_pad  = 0.05 * (pe_max - pe_min)

        fig_vp, ax_vp = plt.subplots(figsize=FIGSIZE)
        (line_pe,) = ax_vp.plot([], [], lw=1.8, color="C0", label=r"$p_e(x,t)$")
        (line_pi,) = ax_vp.plot([], [], lw=1.8, color="C3", linestyle="--", label=r"$p_i(x,t)$")

        ax_vp.set_xlim(0, L)
        ax_vp.set_ylim(pe_min - p_pad, pe_max + p_pad)
        ax_vp.set_xlabel(r"$x$"); ax_vp.set_ylabel(r"$p$")
        ax_vp.legend(loc="upper right", fontsize=11)
        ax_vp.minorticks_on()
        ax_vp.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax_vp.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax_vp.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        ax_vp.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
        ax_vp.set_axisbelow(True)
        fig_vp.tight_layout(rect=[0, 0, 1, 0.93])

        def init_p():
            line_pe.set_data([], []); line_pi.set_data([], [])
            ax_vp.set_title(r"$\mathrm{Pressure\ Evolution},\ t=0.000$", fontsize=11)
            return line_pe, line_pi

        def update_p(frame):
            line_pe.set_data(x, saved_pe[frame])
            line_pi.set_data(x, saved_pi[frame])
            ax_vp.set_title(rf"$\mathrm{{Pressure\ Evolution}},\ t={times[frame]:.3f}$", fontsize=11)
            return line_pe, line_pi

        anim_p = animation.FuncAnimation(fig_vp, update_p, frames=len(saved_pe),
                                          init_func=init_p, blit=True, interval=30)
        video_path = os.path.join(PLOTS_DIR, "pressure_evolution.mp4")
        writer = animation.FFMpegWriter(fps=30, bitrate=2000)
        anim_p.save(video_path, writer=writer, dpi=200)
        print(f"Saved: {video_path}")
        plt.close(fig_vp)

        # ==========================================
        # VIDEO 3: Density profile evolution
        # ==========================================

        n_min = float(np.min(saved_n)); n_max = float(np.max(saved_n))
        n_pad = 0.05 * (n_max - n_min)

        fig_v2, ax_v2 = plt.subplots(figsize=FIGSIZE)
        (line_n,) = ax_v2.plot([], [], lw=1.8, color="C2", label=r"$n(x,t)$")

        ax_v2.set_xlim(0, L)
        ax_v2.set_ylim(n_min - n_pad, n_max + n_pad)
        ax_v2.set_xlabel(r"$x$"); ax_v2.set_ylabel(r"$n$")
        ax_v2.legend(loc="upper right", fontsize=11)
        ax_v2.minorticks_on()
        ax_v2.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax_v2.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax_v2.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        ax_v2.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
        ax_v2.set_axisbelow(True)
        fig_v2.tight_layout(rect=[0, 0, 1, 0.93])

        def init_n():
            line_n.set_data([], [])
            ax_v2.set_title(r"$\mathrm{Density\ Evolution},\ t=0.000$", fontsize=11)
            return (line_n,)

        def update_n(frame):
            line_n.set_data(x, saved_n[frame])
            ax_v2.set_title(rf"$\mathrm{{Density\ Evolution}},\ t={times[frame]:.3f}$", fontsize=11)
            return (line_n,)

        anim_n = animation.FuncAnimation(fig_v2, update_n, frames=len(saved_n),
                                          init_func=init_n, blit=True, interval=30)
        video_path = os.path.join(PLOTS_DIR, "density_evolution.mp4")
        writer = animation.FFMpegWriter(fps=30, bitrate=2000)
        anim_n.save(video_path, writer=writer, dpi=200)
        print(f"Saved: {video_path}")
        plt.close(fig_v2)

        # ==========================================
        # VIDEO 4: Edge fluxes & total sources vs time
        # ==========================================

        all_vals = np.concatenate([edge_Qe_t, edge_Qi_t, edge_Gamma_t,
                                   total_Spe_t, total_Spi_t, total_Sn_t])
        y_min = float(np.min(all_vals)); y_max = float(np.max(all_vals))
        y_pad = 0.08 * (y_max - y_min)

        fig_s, ax_s = plt.subplots(figsize=FIGSIZE)
        (ln_Qe,)   = ax_s.plot([], [], color='C0', lw=1.8, label=r"$Q_{e,\mathrm{edge}}$")
        (ln_Spe,)  = ax_s.plot([], [], color='C0', lw=1.2, linestyle=':', label=r"$\int S_{p_e}\,dx$")
        (ln_Qi,)   = ax_s.plot([], [], color='C3', lw=1.8, linestyle='--', label=r"$Q_{i,\mathrm{edge}}$")
        (ln_Spi,)  = ax_s.plot([], [], color='C3', lw=1.2, linestyle=':', label=r"$\int S_{p_i}\,dx$")
        (ln_Gam,)  = ax_s.plot([], [], color='C2', lw=1.8, linestyle='-.', label=r"$\Gamma_\mathrm{edge}$")
        (ln_Sn,)   = ax_s.plot([], [], color='C2', lw=1.2, linestyle=':', label=r"$\int S_n\,dx$")

        ax_s.set_xlim(0, T)
        ax_s.set_ylim(y_min - y_pad, y_max + y_pad)
        ax_s.set_xlabel(r"$t$"); ax_s.set_ylabel(r"Edge flux / total source")
        ax_s.legend(fontsize=9, loc="best", ncol=2)
        ax_s.minorticks_on()
        ax_s.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax_s.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax_s.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        ax_s.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
        ax_s.set_axisbelow(True)
        fig_s.tight_layout(rect=[0, 0, 1, 0.93])

        _all_lines = (ln_Qe, ln_Spe, ln_Qi, ln_Spi, ln_Gam, ln_Sn)

        def init_flux():
            for ln in _all_lines:
                ln.set_data([], [])
            ax_s.set_title(r"$\mathrm{Edge\ Fluxes\ \&\ Sources},\ t=0.000$", fontsize=11)
            return _all_lines

        def update_flux(frame):
            n_pts = frame + 1
            t_sl = times[:n_pts]
            ln_Qe.set_data(t_sl,  edge_Qe_t[:n_pts])
            ln_Spe.set_data(t_sl, total_Spe_t[:n_pts])
            ln_Qi.set_data(t_sl,  edge_Qi_t[:n_pts])
            ln_Spi.set_data(t_sl, total_Spi_t[:n_pts])
            ln_Gam.set_data(t_sl, edge_Gamma_t[:n_pts])
            ln_Sn.set_data(t_sl,  total_Sn_t[:n_pts])
            ax_s.set_title(rf"$\mathrm{{Edge\ Fluxes\ \&\ Sources}},\ t={times[frame]:.3f}$", fontsize=11)
            return _all_lines

        anim_flux = animation.FuncAnimation(fig_s, update_flux, frames=len(saved_pe),
                                             init_func=init_flux, blit=True, interval=30)
        video_path = os.path.join(PLOTS_DIR, "edge_fluxes_and_sources.mp4")
        writer = animation.FFMpegWriter(fps=30, bitrate=2000)
        anim_flux.save(video_path, writer=writer, dpi=200)
        print(f"Saved: {video_path}")
        plt.close(fig_s)


if __name__ == "__main__":
    main()

