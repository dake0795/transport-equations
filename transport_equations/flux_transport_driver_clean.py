#!/usr/bin/env python3
"""Single-species flux transport driver (import-safe).

Generated from flux_transport_driver_clean.py.

- main(..., do_plots=True, do_videos=True, config_file=None)
- If do_plots=False: no matplotlib import, no figures created.
- If do_videos=False: no animations/MP4s generated.
- If config_file provided: overrides default variables.
"""

import os
import json
import collections.abc
import numpy as np

from .flux_transport_model_clean import (
    flux_function, solving_loop, alpha_heating, bremsstrahlung, compute_face_flux,
    _get_bcs, _apply_bc_to_profile,
)

def deep_update(d, u):
    """Recursively updates nested dictionaries."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def main(
    skip_solve: bool = False,
    data_file: str = "plots/run_data.npz",
    do_plots: bool = True,
    do_videos: bool = True,
    config_file: str = None,
):
    """Run the single-species simulation."""

    # ==========================================
    # Parse Config File
    # ==========================================
    # Auto-detect 'inputs.json' if nothing is passed
    if config_file is None and os.path.exists("inputs.json"):
        config_file = "inputs.json"
    user_config = {}
    if config_file:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                user_config = json.load(f)
            print(f"Loaded configuration from {config_file}")
        else:
            print(f"Warning: Config file '{config_file}' not found. Using defaults.")

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
    L = user_config.get("L", 1.0)
    dx = user_config.get("dx", 0.005)
    dt = user_config.get("dt", None)
    T = user_config.get("T", 0.2)
    num_video_frames = user_config.get("num_video_frames", 300)
    num_plot_snapshots = user_config.get("num_plot_snapshots", 8)

    x = np.linspace(0, L, int(L/dx) + 1)
    dx = x[1] - x[0]

    # ==========================================
    # Branch selection
    # ==========================================
    START_ON = user_config.get("START_ON", "supercritical")
    branch_label = START_ON

    # ==========================================
    # Power balance mode
    # ==========================================
    power_balance_mode = user_config.get("power_balance_mode", "initial_only")

    # ====================================================
    # Transport parameters
    # ====================================================
    transport_params = {
        "chi0": 10.0,
        "chi_core": 10.0,
        "chi_RR": 0.05,
        "g_c": 4.0,
        "g_stiff": 3,
        "n_stiff": 2,
        "boundaries": [],
        "deltas": [],
        "flux_models": ["nl"],
        "nu4": 5e-4,
        "implicit_nu4": True,
        "n_smooth": 3,
        "lf_dissipation": 0,
        "heating_mode": "global",
        "power_balance": 0.65,
        "edge_sigma": 0.05,
        "C_brem": 0.01,
        "Z_eff": 1.0,
        "C_alpha": 4e-6,
        "f_deuterium": 0.5,
        "f_tritium": 0.5,
        "T_ref_keV": 10.0,
        "n_ref": 1.0,
        "n_ref_20": 1.0,
        "adaptive": False,
        "method": "RK4",
        "atol": 1e-6,
        "rtol": 1e-3,
        "dt_min": 1e-12,
        "dt_max": 1e-2,
        "safety_factor": 0.9,
    }

    if "transport_params" in user_config:
        transport_params = deep_update(transport_params, user_config["transport_params"])

    # ==========================================
    # Auto timestep
    # ==========================================
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
        return base_source(x)

    def physics_source(x, p):
        return alpha_heating(p, transport_params) - bremsstrahlung(p, transport_params)

    # ==========================================
    # Compute initial gradient and flux
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
        pass # All of your plotting code will still work identically here. 
             # For brevity in this snippet, the execution directly proceeds.
             # Note: You can paste your existing 01-16 plotting blocks here if you wish, 
             # they require ZERO changes since the variables remain identical.

    # ==========================================
    # Solve (or load)
    # ==========================================
    solve_params = dict(transport_params)

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

    solve_params["power_balance"] = None

    if skip_solve and os.path.exists(data_file):
        print(f"[skip_solve] Loading data from {data_file}")
        _data = np.load(data_file)
        saved_p = _data["saved_p"]
        times   = _data["times"]
    else:
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

    snap_indices = np.linspace(0, len(saved_p) - 1, num_plot_snapshots, dtype=int)

    # ==========================================
    # Diagnostics & Saving
    # ==========================================
    total_pressure_time, edge_flux_time = [], []
    total_S_ext_time, total_P_alpha_time, total_P_brem_time = [], [], []
    total_heating_time, max_gradient_time = [], []

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

    plots_dir = os.path.dirname(data_file) or "plots"
    os.makedirs(plots_dir, exist_ok=True)

    np.savez(
        data_file,
        x=x, times=times, saved_p=saved_p,
        total_pressure_time=total_pressure_time,
        edge_flux_time=edge_flux_time,
        total_S_ext_time=total_S_ext_time,
        total_P_alpha_time=total_P_alpha_time,
        total_P_brem_time=total_P_brem_time,
        total_heating_time=total_heating_time,
        max_gradient_time=max_gradient_time,
        g_crit=g_crit, g_c=g_c, L=L, T=T,
    )
    print(f"Saved run data to {data_file}")

    # Plotting/Video logic from your original script can safely remain here unedited.
    # Note: Omitted in this snippet to focus on the exact driver logic you needed,
    # but you can paste the rest of your `if do_plots...` blocks exactly as they were.

if __name__ == "__main__":
    main()
