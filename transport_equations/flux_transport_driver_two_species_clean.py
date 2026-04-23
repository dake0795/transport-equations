#!/usr/bin/env python3
"""Two-species flux transport driver (import-safe).

Generated from flux_transport_driver_two_species_clean.py.

- main(..., do_plots=True, do_videos=True, config_file=None)
"""

import os
import json
import collections.abc
import numpy as np

from .flux_transport_model_two_species_clean import (
    heat_flux_function, particle_flux_function,
    collision_exchange, alpha_heating, bremsstrahlung,
    compute_fluxes, solving_loop,
    _get_bcs, _apply_bc, _get_species_params, _log_kap,
)

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def main(
    skip_solve: bool = False,
    data_file: str = "plots_2sp/run_data.npz",
    do_plots: bool = True,
    do_videos: bool = True,
    config_file: str = None,
):
    """Run the two-species simulation."""

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

    if do_plots or do_videos:
        import matplotlib
        if not do_plots:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.cm as cm
        from matplotlib.ticker import AutoMinorLocator
        # ... (Your standard plotting defs remain identical here)
    else:
        plt = None
        animation = None
        cm = None

    def log_kap_cell(prof, dx):
        return -np.gradient(np.log(np.maximum(prof, 1e-10)), dx)

    def power_law_profile(x, L, f_core, f_ped, m=2, kap_target=None):
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
    L = user_config.get("L", 1.0)
    dx = user_config.get("dx", 0.005)
    dt = user_config.get("dt", None)
    T = user_config.get("T", 0.2)
    num_video_frames = user_config.get("num_video_frames", 300)
    num_plot_snapshots = user_config.get("num_plot_snapshots", 8)

    x = np.linspace(0, L, int(L / dx))

    # ==========================================
    # Initial state control
    # ==========================================
    initial_state_e = user_config.get("initial_state_e", "supercritical")
    initial_state_i = user_config.get("initial_state_i", "supercritical")
    initial_state_n = user_config.get("initial_state_n", "subcritical")
    power_balance_mode = user_config.get("power_balance_mode", None)

    # ==========================================
    # Transport parameters
    # ==========================================
    transport_params = {
        "chi0_e": 10.0,
        "chi_core_e": 10.0,
        "chi_RR_e": 0.05,
        "g_c_e": 4.0,
        "g_stiff_e": 3.0,
        "n_stiff_e": 2,
        "chi0_n_e": 1.0,
        "chi_core_n_e": 1.0,
        "chi_RR_n_e": 0.01,
        "g_c_n_e": 2.0,
        "g_stiff_n_e": 1.5,
        "n_stiff_n_e": 2,
        "V_n_e": 0.0,
        "chi0_i": 3.0,
        "chi_core_i": 3.0,
        "chi_RR_i": 0.05,
        "g_c_i": 3.0,
        "g_stiff_i": 2.5,
        "n_stiff_i": 2,
        "boundaries": [],
        "deltas": [],
        "flux_models": ["nl"],
        "flux_models_n": ["nl"],
        "nu4": 0.0,
        "n_smooth": 2,
        "tau_ei": 0.05,
        "C_brem": 0.03,
        "Z_eff": 1.0,
        "C_alpha": 1e-3,
        "T_ref_keV": 10.0,
        "n_ref_20": 1.0,
        "Z_i": 1,
        "A_i": 2.5,
        "f_deuterium": 0.5,
        "f_tritium": 0.5,
    }

    if "transport_params" in user_config:
        transport_params = deep_update(transport_params, user_config["transport_params"])

    # ==========================================
    # Auto timestep
    # ==========================================
    if dt is None:
        _CFL_RK4 = 2.785
        _safety  = 0.8

        nu4 = transport_params.get("nu4", 0.0)
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
    # Critical gradients & Setup
    # ==========================================
    g_c_e    = transport_params["g_c_e"]
    g_c_i    = transport_params["g_c_i"]
    g_crit_e = g_c_e / 2.0
    g_crit_i = g_c_i / 2.0
    g_c_n    = transport_params["g_c_n_e"]
    g_crit_n = g_c_n / 2.0

    T_e_core = 2.0;  T_e_ped = 0.5
    T_i_core = 1.5;  T_i_ped = 0.4
    n_core   = 2.0;  n_ped   = 0.2
    n_ped_sub = 1.2

    m_T = 2; m_n = 3
    kap_ratio_super = 1.8; kap_ratio_sub = 0.6

    n_ped_eff = n_ped if initial_state_n == "supercritical" else n_ped_sub
    kap_target_Te = (kap_ratio_super if initial_state_e == "supercritical" else kap_ratio_sub) * g_crit_e
    kap_target_Ti = (kap_ratio_super if initial_state_i == "supercritical" else kap_ratio_sub) * g_crit_i
    kap_target_n  = (kap_ratio_super if initial_state_n == "supercritical" else kap_ratio_sub) * g_crit_n

    T_e_init = power_law_profile(x, L, T_e_core, T_e_ped, m=m_T, kap_target=kap_target_Te)
    T_i_init = power_law_profile(x, L, T_i_core, T_i_ped, m=m_T, kap_target=kap_target_Ti)
    n_init   = power_law_profile(x, L, n_core,   n_ped_eff, m=m_n, kap_target=kap_target_n)

    p_e_ped = T_e_ped * n_ped_eff
    p_i_ped = T_i_ped * n_ped_eff
    p_e_init = T_e_init * n_init
    p_i_init = T_i_init * n_init

    # ==========================================
    # Source functions & Solve
    # ==========================================
    def source_pe(x, p_e, p_i, n): return 5.0 * np.exp(-(x**2) / (0.25*L)**2)
    def source_pi(x, p_e, p_i, n): return 0.0 * x
    def source_n(x, p_e, p_i, n):  return 2.0 * np.exp(-(x**2) / (0.25*L)**2)

    source_pe_fn = source_pe
    source_pi_fn = source_pi
    source_n_fn  = source_n

    if power_balance_mode == "initial_only":
        # Note: In a full copy, your exact initialization logic for Q_e_init goes here.
        pass

    if skip_solve and os.path.exists(data_file):
        _data = np.load(data_file)
        saved_pe, saved_pi, saved_n = _data["saved_pe"], _data["saved_pi"], _data["saved_n"]
        times = _data["times"]
    else:
        saved_pe, saved_pi, saved_n = solving_loop(
            p_e_init, p_i_init, n_init,
            dt, dx, T, L,
            p_e_ped, p_i_ped, n_ped_eff,
            source_pe_fn, source_pi_fn, source_n_fn,
            num_video_frames, transport_params,
        )
        times = np.linspace(0, T, len(saved_pe))

    os.makedirs(os.path.dirname(data_file) or "plots_2sp", exist_ok=True)
    np.savez(data_file, x=x, times=times, saved_pe=saved_pe, saved_pi=saved_pi, saved_n=saved_n)
    
    # ... (Plotting code from original file directly follows here)

if __name__ == "__main__":
    main()
