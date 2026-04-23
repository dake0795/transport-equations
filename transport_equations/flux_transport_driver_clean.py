#!/usr/bin/env python3
"""Single-species flux transport driver (import-safe).

Generated from flux_transport_driver_clean.py.

- main(..., do_plots=True, do_videos=True)
- If do_plots=False: no matplotlib import, no figures created.
- If do_videos=False: no animations/MP4s generated.
"""

import os
import numpy as np

from .flux_transport_model_clean import (
    flux_function, solving_loop, alpha_heating, bremsstrahlung, compute_face_flux,
    _get_bcs, _apply_bc_to_profile,
)


def main(
    skip_solve: bool = False,
    data_file: str = "plots/run_data.npz",
    do_plots: bool = True,
    do_videos: bool = True,
):
    """Run the single-species simulation."""

    # Optional matplotlib import
    if do_plots or do_videos:
        import matplotlib
        if not do_plots:
            matplotlib.use("Agg")

        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.ticker import AutoMinorLocator

        FIGSIZE = (6.33, 4.33)
        # Avoid requiring LaTeX unless making static plots
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

        def style_plot(ax):
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            ax.yaxis.set_minor_locator(AutoMinorLocator(4))
            ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6)
            ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
            ax.set_axisbelow(True)
            plt.tight_layout()

        def save_and_show(filename, plots_dir="plots"):
            os.makedirs(plots_dir, exist_ok=True)
            filepath = os.path.join(plots_dir, f"{filename}.pdf")
            plt.savefig(filepath, format="pdf", bbox_inches="tight")
            print(f"Saved: {filepath}")
            plt.show()
    else:
        plt = None
        animation = None
        FIGSIZE = None


    # ==========================================
    # Domain
    # ==========================================
    L = 1.0
    dx = 0.005
    dt = None       # None = auto (largest stable dt for RK4 given nu4, chi0, dx)
    T = 0.2
    num_video_frames  = 300   # total frames saved from solver (videos + time-series)
    num_plot_snapshots = 8    # subset shown on static overlay plots

    x = np.linspace(0, L, int(L/dx) + 1)
    dx = x[1] - x[0]

    # ==========================================
    # Branch selection
    # ==========================================
    START_ON = "supercritical"
    branch_label = START_ON

    # ==========================================
    # Power balance mode
    # ==========================================
    # "initial_only" : scale source once at t=0 to match initial edge flux,
    #                  then fix amplitude — system evolves freely thereafter
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
        "flux_models": ["nl"],

        "nu4": 5e-4,          # hyperviscosity coefficient
        "implicit_nu4": True, # treat hyperviscosity implicitly (removes nu4 CFL limit)
        "n_smooth": 3,        # binomial smoothing passes on face gradients before flux eval
        "lf_dissipation": 0,  # local Lax-Friedrichs coefficient (0 to disable)

        # ----- Power balance enforcement -----
        "heating_mode": "global",  # "global" or "localized"
        "power_balance": 0.65,      # < 1 → initial deficit: ∫S < Q_edge so ∫p dx falls first
        "edge_sigma": 0.05,        # width of edge-localized Gaussian heating (for localized mode)

        # ----- Edge boundary condition -----
        # Default: Dirichlet (fixed p_ped). Uncomment for floating edge (Robin):
        #   "bc": {"edge": {"type": "robin", "gamma": 5.0, "p_SOL": 0.5}},
        # Q_edge = gamma * (p_edge - p_SOL). Edge pressure floats freely.
        # gamma = SOL coupling strength (larger = edge pinned closer to p_SOL).

        # ----- Physics sources -----
        "C_brem":      0.01,
        "Z_eff":       1.0,
        "C_alpha":     4e-6,
        "f_deuterium": 0.5,
        "f_tritium":   0.5,
        "T_ref_keV":   10.0,
        "n_ref":        1.0,
        "n_ref_20":     1.0,

        # ----- Adaptive timestepping -----
        "adaptive": False,      # Enable adaptive timestepping (default: False)
        "method": "RK4",        # Integration method: "RK4", "Euler"
        "atol": 1e-6,           # Absolute tolerance for adaptive stepping
        "rtol": 1e-3,           # Relative tolerance for adaptive stepping
        "dt_min": 1e-12,        # Minimum timestep
        "dt_max": 1e-2,         # Maximum timestep
        "safety_factor": 0.9,   # Safety factor for timestep adjustment
    }
    

    # ==========================================
    # Auto timestep
    # ==========================================
    # RK4 stability on the negative real axis: |lambda * dt| < 2.785
    # Two constraints:
    #   Hyperviscosity (4th derivative): lambda = 16 * nu4 / dx^4
    #   Diffusion      (2nd derivative): lambda = 2 * chi_max / dx^2
    # Take the tighter limit with a safety margin.

    adaptive = transport_params.get("adaptive", False)

    if dt is None:
        _CFL_RK4 = 2.785
        _safety  = 0.8

        nu4      = transport_params.get("nu4", 0.0)
        chi_max  = max(transport_params.get("chi0", 0.0),
                       transport_params.get("chi_core", 0.0),
                       transport_params.get("chi_MHD", 0.0))

        implicit_nu4 = transport_params.get("implicit_nu4", False)

        dt_candidates = []
        if nu4 > 0 and not implicit_nu4:
            dt_candidates.append(_CFL_RK4 / (16 * nu4 / dx**4))
        if chi_max > 0:
            dt_candidates.append(_CFL_RK4 / (2 * chi_max / dx**2))

        dt = _safety * min(dt_candidates)
        if adaptive:
            print(f"[auto-dt] Initial dt guess = {dt:.2e} (adaptive enabled, nu4={nu4:.0e}, chi_max={chi_max:.1f}, dx={dx})")
        else:
            print(f"[auto-dt] dt = {dt:.2e}  (nu4={nu4:.0e}, chi_max={chi_max:.1f}, dx={dx})")

    # ==========================================
    # Initial profile
    # ==========================================

    g_c = transport_params["g_c"]
    g_crit = g_c / 2.0

    transport_params["g_MHD"]   = 0.9 * g_c
    transport_params["chi_MHD"] = 2

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
        target_g = 0.8 * g_crit
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
        """
        return alpha_heating(p, transport_params) - bremsstrahlung(p, transport_params)

    # ==========================================
    # Compute initial gradient and flux using same discretization as compute_rhs
    # ==========================================

    core_bc, edge_bc = _get_bcs(transport_params, p_ped)
    p_bc = p_init.copy()
    _apply_bc_to_profile(p_bc, dx, core_bc, edge_bc)

    g_face = -(p_bc[1:] - p_bc[:-1]) / dx
    x_face = 0.5 * (x[1:] + x[:-1])

    Q_face_init = flux_function(g_face, x_face, transport_params)

    g_init = -np.gradient(p_init, dx)
    Q_init = flux_function(g_init, x, transport_params)

    # ==========================================
    # Initial power balance
    # ==========================================
    p_x_init = np.gradient(p_init, dx)
    g_init = -p_x_init

    S_ext_init   = source(x, p_init)
    S_phys_init  = physics_source(x, p_init)
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

    if do_plots and plt is not None:
        # ==========================================
        # INITIAL DIAGNOSTIC PLOTS
        # ==========================================

        # ----------------------------------------------------------
        # 1. Initial pressure profile with supercritical regions
        # ----------------------------------------------------------

        fig, ax = plt.subplots()

        ax.plot(x, p_init, label="Initial pressure")

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
    # Solve (or load)
    # ==========================================

    solve_params = dict(transport_params)

    # Scale factor computed once from initial state, then fixed.
    pb_ratio         = transport_params.get("power_balance", 1.0)
    heating_mode_str = transport_params.get("heating_mode", "global")
    total_S0_ext     = np.trapezoid(source(x, p_init), x)
    total_S0_phys    = np.trapezoid(physics_source(x, p_init), x)
    Q_edge0          = Q_face_init[-1]

    if heating_mode_str == "global" and total_S0_ext > 0:
        target_ext = pb_ratio * Q_edge0 - total_S0_phys
        scale_0    = max(target_ext / total_S0_ext, 0.0)
        def source_fn(x, p, _s=scale_0): return _s * base_source(x)
        print(f"[initial_only] External source scale fixed at {scale_0:.4f} (t=0)")
    elif heating_mode_str == "localized":
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

    if skip_solve and os.path.exists(data_file):
        print(f"[skip_solve] Loading data from {data_file}")
        _data = np.load(data_file)
        saved_p = _data["saved_p"]
        times   = _data["times"]
        print(f"  Loaded {len(saved_p)} frames, T={times[-1]:.4f}")
    else:
        if skip_solve:
            print(f"[skip_solve] {data_file} not found — running solver anyway")

        saved_p = solving_loop(
            p_init,
            dt,
            dx,
            T,
            L,
            p_ped,
            source_fn,
            num_video_frames,
            solve_params,
            physics_source_function=physics_source,
        )

        times = np.linspace(0, T, len(saved_p))

    # Indices for the static overlay plots (num_plot_snapshots evenly spaced)
    snap_indices = np.linspace(0, len(saved_p) - 1, num_plot_snapshots, dtype=int)

    # ==========================================
    # Diagnostics (computed from saved_p)
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
        Q_f, _ = compute_face_flux(p, dx, x, p_ped, transport_params)
        edge_flux_time.append(Q_f[-1])
        _S_ext = np.trapezoid(source_fn(x, p), x)
        _P_alpha = np.trapezoid(alpha_heating(p, transport_params), x)
        _P_brem  = np.trapezoid(bremsstrahlung(p, transport_params), x)
        total_S_ext_time.append(_S_ext)
        total_P_alpha_time.append(_P_alpha)
        total_P_brem_time.append(_P_brem)
        total_heating_time.append(_S_ext + _P_alpha - _P_brem)
        max_gradient_time.append(np.max(-np.gradient(p, dx)))

    total_pressure_time  = np.array(total_pressure_time)
    edge_flux_time       = np.array(edge_flux_time)
    total_S_ext_time     = np.array(total_S_ext_time)
    total_P_alpha_time   = np.array(total_P_alpha_time)
    total_P_brem_time    = np.array(total_P_brem_time)
    total_heating_time   = np.array(total_heating_time)
    max_gradient_time    = np.array(max_gradient_time)

    # ==========================================
    # Save all data
    # ==========================================
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    np.savez(
        data_file,
        # Grid and time
        x=x,
        times=times,
        # Simulation output
        saved_p=saved_p,
        # Diagnostics
        total_pressure_time=total_pressure_time,
        edge_flux_time=edge_flux_time,
        total_S_ext_time=total_S_ext_time,
        total_P_alpha_time=total_P_alpha_time,
        total_P_brem_time=total_P_brem_time,
        total_heating_time=total_heating_time,
        max_gradient_time=max_gradient_time,
        # Scalars
        g_crit=g_crit,
        g_c=g_c,
        L=L,
        T=T,
    )
    print(f"Saved run data to {data_file}")

    if do_plots and plt is not None:
        # ==========================================
        # Pressure evolution plot (static, subset of snapshots)
        # ==========================================
        fig, ax = plt.subplots()
        for i in snap_indices:
            ax.plot(x, saved_p[i], label=rf"$t={times[i]:.3f}$")
        ax.set_title(rf"$\mathrm{{Pressure\ Evolution\ ({branch_label})}}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$p$")
        boundaries = transport_params.get("boundaries", [])

        for i, xb in enumerate(boundaries):
            ax.axvline(xb, linestyle="--", linewidth=1.5, alpha=0.7)
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
        # Extra diagnostics
        # ==========================================

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
            ax.axhline(xb, linestyle="--", linewidth=1.5, color="white", alpha=0.8)

        ax.set_title(rf"$\mathrm{{Gradient\ g(x,t)\ ({branch_label})}}$")
        cbar = fig.colorbar(im)
        cbar.set_label(r"$g=-\partial_x p$")
        plt.tight_layout()
        save_and_show("08_gradient_heatmap")

        # ==========================================================
        # 2. Edge flux minus total heating
        # ==========================================================
        fig, ax = plt.subplots()
        ax.plot(times, edge_flux_time - total_heating_time, marker='o')
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
        max_locations = max_curves // 2

        track_points = np.linspace(0.2, 0.95, max_locations)
        track_indices = [np.argmin(np.abs(x - pt)) for pt in track_points]

        fig, ax = plt.subplots()

        for i, idx in enumerate(track_indices):
            color_idx = i % n_colors

            Q_local       = []
            S_total_local = []

            face_idx = min(idx, len(x) - 2)   # nearest face index
            for p in saved_p:
                Q_f, _ = compute_face_flux(p, dx, x, p_ped, transport_params)
                Q_local.append(Q_f[face_idx])
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
        # 5. Pressure heatmap p(x,t)
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
            ax.axhline(xb, linestyle="--", linewidth=1.5, color="white", alpha=0.8)

        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title(rf"$\mathrm{{Pressure\ p(x,t)\ ({branch_label})}}$")
        cbar = fig.colorbar(im)
        cbar.set_label("p")
        plt.tight_layout()
        save_and_show("11_pressure_heatmap")

        # ==========================================================
        # 6. Max gradient and location
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
        # 7. Edge gradient vs source
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
        # 8. Flux profile vs cumulative source (selected snapshots)
        # ==========================================================
        for si in snap_indices:
            p = saved_p[si]
            Q_f, x_f = compute_face_flux(p, dx, x, p_ped, transport_params)

            F_ext   = np.array([np.trapezoid(source_fn(x[:j+1], p[:j+1]),     x[:j+1]) for j in range(len(x))])
            F_alpha = np.array([np.trapezoid(alpha_heating(p[:j+1], transport_params),  x[:j+1]) for j in range(len(x))])
            F_brem  = np.array([np.trapezoid(bremsstrahlung(p[:j+1], transport_params), x[:j+1]) for j in range(len(x))])
            F_total = F_ext + F_alpha - F_brem

            fig, ax = plt.subplots()
            ax.plot(x_f, Q_f,   color='k',  linewidth=2, label=r"$Q(x)$")
            ax.plot(x, F_total, color='C1', linewidth=2, linestyle="-.", label=r"$\int_0^x S_\mathrm{total}$")
            ax.plot(x, F_ext,   color='C0', linewidth=1.5, linestyle="--", label=r"$\int_0^x S_\mathrm{ext}$")
            ax.plot(x, F_alpha, color='C2', linewidth=1.5, linestyle="--", label=r"$\int_0^x P_\alpha$")
            ax.plot(x, -F_brem, color='C3', linewidth=1.5, linestyle="--", label=r"$-\int_0^x P_\mathrm{brem}$")
            ax.set_xlabel("x")
            ax.set_ylabel("Flux / cumulative source")
            ax.set_title(f"Flux balance at $t={times[si]:.3f}$")
            ax.legend(fontsize=11)
            style_plot(ax)
            save_and_show(f"14_flux_balance_snap{si:03d}")

        # ==========================================================
        # 9. Effective diffusivity evolution (selected snapshots)
        # ==========================================================

        eps = 1e-6
        chi_snap_indices = np.linspace(0, len(saved_p) - 1, min(6, len(saved_p)), dtype=int)

        fig, ax = plt.subplots()

        for si in chi_snap_indices:
            p = saved_p[si]
            g = -np.gradient(p, dx)
            Q_plus  = flux_function(g + eps, x, transport_params)
            Q_minus = flux_function(g - eps, x, transport_params)
            chi_eff = (Q_plus - Q_minus) / (2 * eps)
            ax.plot(x, chi_eff, label=rf"$t={times[si]:.3f}$")

        boundaries = transport_params.get("boundaries", [])
        for i, xb in enumerate(boundaries):
            ax.axvline(xb, linestyle="--", linewidth=1.2, alpha=0.6)

        ax.set_xlabel("x")
        ax.set_ylabel(r"$\chi_{\mathrm{eff}}$")
        ax.set_title(rf"$\mathrm{{Effective\ diffusivity\ evolution\ ({branch_label})}}$")
        ax.legend(ncol=2, fontsize=10)
        style_plot(ax)
        save_and_show("15_effective_diffusivity")

        # ==========================================================
        # 10. Q(g) phase diagram — snapshots on the theoretical curve
        # ==========================================================

        g_theory = np.linspace(0, 1.5 * g_c, 500)
        Q_theory = flux_function(g_theory, np.full_like(g_theory, 0.5), transport_params)

        fig, ax = plt.subplots()

        # Integrated total source as horizontal lines (behind everything)
        cmap = cm.coolwarm
        for idx, si in enumerate(chi_snap_indices):
            color = cmap(idx / max(len(chi_snap_indices) - 1, 1))
            ax.axhline(total_heating_time[si], color=color, linewidth=2.0,
                       linestyle=':', alpha=0.8, zorder=1)

        ax.plot(g_theory, Q_theory, color='k', linewidth=2, label=r"$Q(g)$ theory", zorder=2)
        ax.axvline(g_crit, linestyle=':', color='gray', linewidth=1, label=r"$g^*$")
        ax.axvline(g_c,    linestyle='--', color='gray', linewidth=1, label=r"$g_c$")

        for idx, si in enumerate(chi_snap_indices):
            p = saved_p[si]
            g_local = -np.gradient(p, dx)
            Q_local = flux_function(np.maximum(g_local, 0), x, transport_params)
            color = cmap(idx / max(len(chi_snap_indices) - 1, 1))
            ax.scatter(g_local[1:-1], Q_local[1:-1], s=16, color=color,
                       label=rf"$t={times[si]:.3f}$", zorder=3)

        ax.set_xlabel(r"$g = -\partial_x p$")
        ax.set_ylabel(r"$Q(g)$")
        ax.set_title(rf"$\mathrm{{Phase\ diagram\ Q(g)\ ({branch_label})}}$")
        ax.legend(ncol=2, fontsize=9)
        style_plot(ax)
        save_and_show("16_Qg_phase_diagram")

    if do_videos and plt is not None:
        # ==========================================
        # VIDEO 1: Pressure profile evolution
        # ==========================================

        p_min = float(np.min(saved_p))
        p_max = float(np.max(saved_p))
        p_pad = 0.05 * (p_max - p_min)

        fig_v, ax_v = plt.subplots(figsize=FIGSIZE)
        (line_p,) = ax_v.plot([], [], lw=1.8, color="C0", label=r"$p(x,t)$")
        scat_p = ax_v.scatter([], [], s=20, color="red", zorder=3,
                               label=r"$g > g_\mathrm{crit}$")

        ax_v.set_xlim(0, L)
        ax_v.set_ylim(p_min - p_pad, p_max + p_pad)
        ax_v.set_xlabel(r"$x$")
        ax_v.set_ylabel(r"$p$")
        ax_v.legend(loc="upper right", fontsize=11)
        ax_v.minorticks_on()
        ax_v.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax_v.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax_v.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        ax_v.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
        ax_v.set_axisbelow(True)
        fig_v.tight_layout(rect=[0, 0, 1, 0.93])

        def init_profile():
            line_p.set_data([], [])
            scat_p.set_offsets(np.empty((0, 2)))
            ax_v.set_title(
                rf"$\mathrm{{Pressure\ Evolution\ ({branch_label})}},\ t=0.000$",
                fontsize=11)
            return line_p, scat_p

        def update_profile(frame):
            p = saved_p[frame]
            line_p.set_data(x, p)
            g = -np.gradient(p, dx)
            mask = g > g_crit
            if np.any(mask):
                scat_p.set_offsets(np.column_stack([x[mask], p[mask]]))
            else:
                scat_p.set_offsets(np.empty((0, 2)))
            ax_v.set_title(
                rf"$\mathrm{{Pressure\ Evolution\ ({branch_label})}},\ t={times[frame]:.3f}$",
                fontsize=11)
            return line_p, scat_p

        anim_profile = animation.FuncAnimation(
            fig_v, update_profile,
            frames=len(saved_p),
            init_func=init_profile,
            blit=True,
            interval=30,
        )

        video_path = os.path.join(plots_dir, "pressure_evolution.mp4")
        writer = animation.FFMpegWriter(fps=30, bitrate=2000)
        anim_profile.save(video_path, writer=writer, dpi=200)
        print(f"Saved: {video_path}")
        plt.close(fig_v)

        # ==========================================
        # VIDEO 2: Integrated source components & edge flux vs time
        # ==========================================

        all_vals = np.concatenate([
            total_S_ext_time,
            total_P_alpha_time,
            -total_P_brem_time,
            total_heating_time,
            edge_flux_time,
        ])
        y_min = float(np.min(all_vals))
        y_max = float(np.max(all_vals))
        y_pad = 0.08 * (y_max - y_min)

        fig_s, ax_s = plt.subplots(figsize=FIGSIZE)

        (ln_ext,)   = ax_s.plot([], [], color='C0', lw=1.8,
                                 label=r"$\int S_\mathrm{ext}\,dx$")
        (ln_alpha,) = ax_s.plot([], [], color='C2', lw=1.8,
                                 label=r"$\int P_\alpha\,dx$")
        (ln_brem,)  = ax_s.plot([], [], color='C3', lw=1.8,
                                 label=r"$-\int P_\mathrm{brem}\,dx$")
        (ln_total,) = ax_s.plot([], [], color='C1', lw=1.8, linestyle="-.",
                                 label=r"$\int S_\mathrm{total}\,dx$")
        (ln_edge,)  = ax_s.plot([], [], color='k',  lw=1.5, linestyle="--",
                                 label=r"$Q_\mathrm{edge}$")

        ax_s.set_xlim(0, T)
        ax_s.set_ylim(y_min - y_pad, y_max + y_pad)
        ax_s.set_xlabel(r"$t$")
        ax_s.set_ylabel(r"Integrated source / edge flux")
        ax_s.legend(fontsize=11, loc="best")
        ax_s.minorticks_on()
        ax_s.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax_s.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax_s.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.6)
        ax_s.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.5)
        ax_s.set_axisbelow(True)
        fig_s.tight_layout(rect=[0, 0, 1, 0.93])

        def init_source():
            for ln in (ln_ext, ln_alpha, ln_brem, ln_total, ln_edge):
                ln.set_data([], [])
            ax_s.set_title(
                rf"$\mathrm{{Source\ components\ ({branch_label})}},\ t=0.000$",
                fontsize=11)
            return ln_ext, ln_alpha, ln_brem, ln_total, ln_edge

        def update_source(frame):
            n = frame + 1
            t_slice = times[:n]
            ln_ext.set_data(t_slice, total_S_ext_time[:n])
            ln_alpha.set_data(t_slice, total_P_alpha_time[:n])
            ln_brem.set_data(t_slice, -total_P_brem_time[:n])
            ln_total.set_data(t_slice, total_heating_time[:n])
            ln_edge.set_data(t_slice, edge_flux_time[:n])
            ax_s.set_title(
                rf"$\mathrm{{Source\ components\ ({branch_label})}},\ t={times[frame]:.3f}$",
                fontsize=11)
            return ln_ext, ln_alpha, ln_brem, ln_total, ln_edge

        anim_source = animation.FuncAnimation(
            fig_s, update_source,
            frames=len(saved_p),
            init_func=init_source,
            blit=True,
            interval=30,
        )

        video_path = os.path.join(plots_dir, "source_components.mp4")
        writer = animation.FFMpegWriter(fps=30, bitrate=2000)
        anim_source.save(video_path, writer=writer, dpi=200)
        print(f"Saved: {video_path}")
        plt.close(fig_s)



if __name__ == "__main__":
    main()

