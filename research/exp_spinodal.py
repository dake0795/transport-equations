"""Source-free spinodal decomposition on a periodic domain in the g-form:
    g_t = d_xx [ Q(g) - nu4 g_xx ].
Start uniform on the falling branch plus noise.  Prediction: plateaus at the
Maxwell values g_a, g_b with lever-rule fractions, then coarsening."""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scipy.integrate import solve_ivp
import flux_transport_model_clean as m
from maxwell import maxwell
mx = maxwell(); P = dict(chi0=10., chi_core=10., chi_RR=0.05, g_c=4., g_stiff=3, n_stiff=2,
                         boundaries=[], deltas=[], flux_models=["nl"], g_MHD=3.6, chi_MHD=2.)
L, N, nu4 = 2.0, 400, 5e-4
x = np.arange(N) * L / N; dx = L / N
def rhs(t, g):
    mu = m.flux_function(g, x, P) - nu4 * (np.roll(g, -1) - 2*g + np.roll(g, 1)) / dx**2
    return (np.roll(mu, -1) - 2*mu + np.roll(mu, 1)) / dx**2
rng = np.random.default_rng(1)
for g0 in [1.5, 3.0, 5.0]:
    gi = g0 + 0.01 * rng.standard_normal(N)
    ts = [0.01, 0.05, 0.2, 1.0, 3.0]
    sol = solve_ivp(rhs, (0, ts[-1]), gi, method="BDF", t_eval=ts, rtol=1e-7, atol=1e-9)
    print(f"\ng0={g0} (mean conserved: {sol.y[:,-1].mean():.4f}); lever-rule high fraction = {(g0-mx['g_a'])/(mx['g_b']-mx['g_a']):.3f}; predicted g_a={mx['g_a']:.3f} g_b={mx['g_b']:.3f}")
    for k, t in enumerate(sol.t):
        g = sol.y[:, k]; mid = 0.5*(mx['g_a']+mx['g_b'])
        hi = g > mid
        n_steps = np.sum(hi != np.roll(hi, 1)) // 2
        lo_val = np.median(g[~hi]) if (~hi).any() else np.nan
        hi_val = np.median(g[hi]) if hi.any() else np.nan
        print(f"  t={t:5.2f}: steps={n_steps:3d}  low plateau={lo_val:.3f}  high plateau={hi_val:.3f}  high fraction={hi.mean():.3f}  g in [{g.min():.2f},{g.max():.2f}]", flush=True)
