"""Figures summarising the saturation study (research/, not the paper)."""
import sys, pathlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import reduced as R
from maxwell import maxwell, Q, landmarks
from scipy.optimize import brentq

OUT = pathlib.Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
mx = maxwell(); lm = landmarks()
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5})

# ---------------------------------------------------------------- 1. Maxwell
fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
g = np.linspace(0, 8, 2000); q = Q(g)
ax.plot(g, q, "k", lw=1.4, label="$Q(g)$")
ax.axhline(mx["Q_M"], color="C3", lw=1, ls="--", label=f"$Q_M={mx['Q_M']:.2f}$")
ax.axhline(lm["Qmin"], color="C0", lw=0.8, ls=":", label=f"$Q_{{\\min}}=Q(g_c)={lm['Qmin']:.1f}$")
ax.axhline(lm["Qmax"], color="C2", lw=0.8, ls=":", label=f"$Q_{{\\max}}={lm['Qmax']:.1f}$")
sel = (g >= mx["g_a"]) & (g <= mx["g_b"])
ax.fill_between(g[sel], mx["Q_M"], q[sel], where=q[sel] > mx["Q_M"], color="C3", alpha=0.25)
ax.fill_between(g[sel], mx["Q_M"], q[sel], where=q[sel] < mx["Q_M"], color="C3", alpha=0.25)
for gg, lab in [(mx["g_a"], "$g_a$"), (mx["g_b"], "$g_b$"), (lm["gstar"], "$g_*$"), (4.0, "$g_c$"), (3.6, "$g_{\\rm MHD}$")]:
    ax.axvline(gg, color="grey", lw=0.5, ls=":")
    ax.text(gg, 15.3, lab, ha="center", fontsize=8)
ax.set_xlabel("$g=-\\partial_x p$"); ax.set_ylabel("$Q$"); ax.set_ylim(0, 16); ax.set_xlim(0, 8)
ax.legend(fontsize=7, loc="upper left")
ax.set_title("Equal-area (Maxwell) construction on $Q(g)$", fontsize=9)
fig.savefig(OUT / "maxwell_construction.pdf"); fig.savefig(OUT / "maxwell_construction.png", dpi=150)

# --------------------------------------------------------- 2. bifurcation
b = np.load(pathlib.Path(__file__).resolve().parent / "bifurcation.npy")
S, WL, WH, Wsub, _, Wsup, _ = b.T
fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
ax.plot(S, WL, "C0-", lw=1.2, label="analytic L branch ($g_-$)")
ax.plot(S, WH, "C3-", lw=1.2, label="analytic H branch (Maxwell front)")
ax.plot(S, Wsub, "C0o", ms=5, mfc="none", label="numerical, subcritical start")
ax.plot(S, Wsup, "C3s", ms=4, mfc="none", label="numerical, supercritical start")
ax.axvline(mx["Q_M"], color="C3", ls="--", lw=0.8); ax.axvline(lm["Qmax"], color="C2", ls=":", lw=0.8)
ax.text(mx["Q_M"], 5.6, "$Q_M$", color="C3", ha="right", fontsize=8); ax.text(lm["Qmax"], 5.6, "$Q_{\\max}$", color="C2", ha="left", fontsize=8)
ax.set_xlabel("integrated drive $\\mathcal{S}$"); ax.set_ylabel("$W=\\int p\\,dx$ (steady)")
ax.legend(fontsize=7, loc="upper left"); ax.set_title("Bistability and hysteresis", fontsize=9)
fig.savefig(OUT / "bifurcation.pdf"); fig.savefig(OUT / "bifurcation.png", dpi=150)

# ------------------------------------------------- 3. profiles + front
dat = np.load(pathlib.Path(__file__).resolve().parent / "scan_pb_nophys_ns3.npz")
x = dat["x"]; xf = dat["xf"]; dx = x[1] - x[0]
fig, axs = plt.subplots(1, 3, figsize=(9.0, 2.8), constrained_layout=True)
for pb, c in [(1.7, "C0"), (2.0, "C1"), (2.5, "C2")]:
    gE = dat[f"pb{pb}__g_end"]; xM = float(dat[f"pb{pb}__xM"])
    axs[0].plot(x, dat[f"pb{pb}__p_end"], color=c, label=f"pb={pb}")
    axs[1].plot(xf, gE, color=c); axs[1].axvline(xM, color=c, ls="--", lw=0.8)
    axs[2].plot(dat[f"pb{pb}__t"], dat[f"pb{pb}__xfoot"], color=c); axs[2].axhline(xM, color=c, ls="--", lw=0.8)
axs[1].axhline(mx["g_a"], color="grey", ls=":", lw=0.8); axs[1].axhline(mx["g_b"], color="grey", ls=":", lw=0.8)
axs[0].set_xlabel("$x$"); axs[0].set_ylabel("$p$ (steady)"); axs[0].legend(fontsize=7)
axs[1].set_xlabel("$x$"); axs[1].set_ylabel("$g$ (steady)"); axs[1].set_title("dashed: predicted $x_M$; dotted: $g_a$, $g_b$", fontsize=8)
axs[2].set_xlabel("$t$"); axs[2].set_ylabel("barrier foot $x_f(t)$"); axs[2].set_xlim(0, 4)
fig.savefig(OUT / "barrier_front.pdf"); fig.savefig(OUT / "barrier_front.png", dpi=150)

# --------------------------------------------------------- 4. spinodal
sp = pathlib.Path(__file__).resolve().parent / "spinodal.npz"
if sp.exists():
    d = np.load(sp)
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    for k, t in enumerate(d["t"]):
        ax.plot(d["x"], d["g"][k], lw=0.9, label=f"$t={t:g}$", color=plt.cm.viridis(k / (len(d["t"]) - 1)))
    ax.axhline(mx["g_a"], color="grey", ls=":", lw=0.8); ax.axhline(mx["g_b"], color="grey", ls=":", lw=0.8)
    ax.set_xlabel("$x$ (periodic)"); ax.set_ylabel("$g$"); ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"Source-free: uniform $g_0={float(d['g0']):g}$ on the falling branch", fontsize=9)
    fig.savefig(OUT / "spinodal.pdf"); fig.savefig(OUT / "spinodal.png", dpi=150)
print("figures written to", OUT)
