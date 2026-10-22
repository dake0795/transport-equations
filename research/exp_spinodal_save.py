import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scipy.integrate import solve_ivp
import flux_transport_model_clean as m
P = dict(chi0=10., chi_core=10., chi_RR=0.05, g_c=4., g_stiff=3, n_stiff=2, boundaries=[], deltas=[], flux_models=["nl"], g_MHD=3.6, chi_MHD=2.)
L, N, nu4 = 2.0, 400, 5e-4
x = np.arange(N) * L / N; dx = L / N
def rhs(t, g):
    mu = m.flux_function(g, x, P) - nu4 * (np.roll(g, -1) - 2*g + np.roll(g, 1)) / dx**2
    return (np.roll(mu, -1) - 2*mu + np.roll(mu, 1)) / dx**2
g0 = 3.0; gi = g0 + 0.01 * np.random.default_rng(1).standard_normal(N)
ts = [0.02, 0.05, 0.2, 1.0, 3.0, 10.0]
sol = solve_ivp(rhs, (0, ts[-1]), gi, method="BDF", t_eval=ts, rtol=1e-7, atol=1e-9)
np.savez("spinodal.npz", x=x, t=np.array(ts), g=sol.y.T, g0=g0)
print("saved; steps per snapshot:", [int(np.sum((sol.y[:,k]>3.5) != np.roll(sol.y[:,k]>3.5,1))//2) for k in range(len(ts))])
