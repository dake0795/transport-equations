# The thermal fold, the hysteresis window and the edge-flux response (reduced model with a radiation sink)

Script `exp_fold.py` (stages `scan newton sharp nu4 edge highS step sharp_edge extra highS_core figs tables`), caches
`fold_*.npz`, logs `fold_*.log`, all tables in `fold_tables.txt`, quick-look plots in `fold_figs/`.  Local python only.

**Set-up.** The paper's production reduced model of the radiation scan (Section 5, `scripts/make_brem_scan.py`):
`reduced.PAPER` with conservative hyperviscosity $\nu_4=5\times10^{-4}$, $C_\alpha=2\times10^{-5}$, $T_{\rm ref}=3$ keV,
$\int S_{\rm ext}\,\dd x=4.94$ (core Gaussian), $p(L)=1$, 200 points.  Maxwell construction: $\QM=5.851$, $g_a=0.706$,
$g_b=6.367$, $\Qmax=10.1$, $\Qmin=1.0$, $\chi_{\rm ped}\equiv\chiRR+\chiMHD=2.05$.

Conventions. $\mathcal S$ below is the *injected* drive (interior-cell sum), equal to $\Qedge$ to $10^{-4}$ in every
steady state; the paper's trapezoid is larger by the axis half-cell ($+0.06$ to $+0.10$).  Where it matters both are
quoted.  "BDF" = time-integrated steady states of the discretised model; "sharp" = the sharp-interface (Maxwell-front)
model with self-consistent sources (`sharp_profile`), which is the theory of Section 4 with $P_\alpha-P_{\rm brem}$ inside
$Q_{\rm ss}$.  $Q_{\rm ss}(x)=\int_0^x S\,\dd x'$ with $S=S_{\rm ext}+P_\alpha-P_{\rm brem}$, so $\mathcal S=Q_{\rm ss}(L)$.

---------------------------------------------------------------------------------------------------------------------

## 1. Thermal fold or transport threshold?

### 1.1 Numbers

Burning branch of the discretised model near its end ($f=0$, `fold_scan.npz`; $\mathcal S$ injected):

| $C_{\rm brem}$ | $\Wtot$ | $\mathcal S=\Qedge$ | $P_\alpha$ | $P_{\rm brem}$ (in / out of front) | $x_f$ | $\max_x Q_{\rm ss}$ @ $x$ | $\gamma_{\rm th}$ (Jacobian leading eigenvalue) |
|---|---|---|---|---|---|---|---|
| 1.20 | 4.212 | 6.653 | 5.909 | 4.144 (1.36 / 2.78) | 0.269 | 7.287 @ 0.62 | −2.84 |
| 1.25 | 4.115 | 6.279 | 5.660 | 4.270 (1.48 / 2.80) | 0.284 | 6.979 @ 0.60 | −2.86 |
| 1.30 | 3.950 | 5.762 | 5.231 | 4.357 (1.67 / 2.69) | 0.319 | 6.541 @ 0.58 | −2.84 |
| 1.3062 | 3.909 | 5.653 | 5.122 | 4.357 (1.72 / 2.64) | 0.329 | 6.445 @ 0.58 | −2.85 |
| **1.3094** (last burn) | **3.888** | **5.598** (trapezoid 5.657) | 5.066 | 4.357 (1.74 / 2.62) | 0.334 | **6.396** @ 0.57 | −2.85 |
| 1.3125 (lost) | 1.228 | 2.510 | 0.125 | 2.504 | wall | 3.80 | −17.7 |

The burning state survives with $\mathcal S<\QM$ (by 0.25; by 0.19 on the paper's trapezoid) because 60% of the sink
is *outside* the front, in the pedestal, where it lowers $\Qedge=\mathcal S$ without touching the drive the front sees.
What the front sees is $Q_{\rm ss}(x_f)=\QM$, and $\max_x Q_{\rm ss}=6.40$ is still $0.55$ above $\QM$ when the burn is
lost.  So the transport threshold has **not** been reached: the burn ends at a fold.

The sharp-interface branch (`fold_sharp.npz`) shows the fold directly, including the unstable middle branch that the
discretised Newton continuation could not reach:

| $x_f$ | $C_{\rm brem}$ | $\Wtot$ | $\mathcal S$ | $P_\alpha$ | $P_{\rm brem}$ | $S(x_f)$ | $\gamma_{\rm th}$ (Stefan family) | $\dd\mathcal S/\dd\Wtot$ (Stefan family, fixed $C$) | $\dd\Qedge/\dd\Wtot$ (same) |
|---|---|---|---|---|---|---|---|---|---|
| 0.160 | 0.037 | 5.876 | 14.67 | 9.88 | 0.15 | 31.7 | −2.24 | 2.11 | 4.35 |
| 0.250 | 1.111 | 4.418 | 7.47 | 6.46 | 3.94 | 14.8 | −1.16 | 2.11 | 3.27 |
| 0.300 | 1.287 | 4.087 | 6.16 | 5.61 | 4.40 | 9.60 | −0.44 | 2.13 | 2.57 |
| 0.325 | 1.319 | 3.971 | 5.80 | 5.31 | 4.45 | 7.70 | −0.05 | 2.16 | 2.21 |
| **0.350** | **1.327** | 3.875 | 5.57 | 5.05 | 4.43 | 6.18 | +0.34 | 2.19 | 1.85 |
| 0.400 | 1.287 | 3.717 | 5.35 | 4.62 | 4.21 | 4.01 | +1.00 | 2.24 | 1.24 |
| 0.600 | 0.787 | 3.135 | 5.58 | 3.02 | 2.38 | 1.17 | +1.95 | 2.23 | 0.28 |
| 0.825 | 0.053 | 2.229 | 5.89 | 1.08 | 0.14 | 0.71 | +1.33 | 1.57 | 0.24 |

$C_{\rm brem}(x_f)$ has its maximum $C_{\rm fold}=1.327$ at $x_f\simeq0.33$–$0.35$ ($\Wtot=3.88$–$3.97$,
$\mathcal S=5.6$–$5.8$, i.e. at or below $\QM$), and $\gamma_{\rm th}$ changes sign there, $\dd\mathcal S/\dd\Wtot=\dd\Qedge/\dd\Wtot\simeq2.2$.
Beyond it the fixed points are thermally unstable (the middle root: the ignition separatrix), and at $C_{\rm brem}\to0$
that root sits at $x_f\simeq0.84$ — near the wall, which is why the external drive alone (4.94) cannot ignite.

Comparison of end-of-burn values:

| model | $C_{\rm brem}$ at loss |
|---|---|
| sharp interface (fold) | 1.327 |
| BDF, $\nu_4\times0.25$ (front 0.7 cells) | 1.366–1.369 |
| BDF, $\nu_4\times1$ (production) | 1.309–1.313 |
| BDF, $\nu_4\times4$ | 1.281–1.284 |
| BDF, $\nu_4\times16$ (front 5.6 cells) | 1.228–1.231 |
| Newton in $\Wtot$ (discretised), max of $C_{\rm brem}(\Wtot)$ | 1.3094 at $\Wtot=3.888$ |

Honest caveats.
* The Jacobian eigenvalue of the discretised model does **not** soften at the loss ($-2.85$ at $\nu_4\times1$, $-3.08$
  at $\times0.25$); it starts to at $\nu_4\times16$ ($-1.0$).  The front is about one cell wide and pinned to the
  grid, so the discrete branch is a staircase of pinned states that jumps when the pinning fails rather than turning
  smoothly; the fold is a property of the continuum (sharp) branch, which the BDF loss points bracket.  The trend
  with $\nu_4$ is the finite front width (a wider front puts more of the barrier on the low branch).
* Newton continuation in $\Wtot$ (C the unknown) worked upward (to $\Wtot=5.79$) but only two steps below the last
  burning state ($\Wtot=3.84$, $3.79$, the latter already with $C$ falling: the far side of the fold), then failed at
  $\Wtot=3.74$.  Its two positive "eigenvalues" ($+106$ at $\Wtot=3.99$, $+31.6$ at $4.79$) are not the thermal mode —
  probably front-hopping states between grid pinnings — and are not used.  The unstable branch is taken from the sharp
  model only.

### 1.2 Derivation of the fold condition

Steady state of the sharp model: $\mu=Q_{\rm ss}(x)$ (eq. CHss) and the front rests at $x_M$ with
$Q_{\rm ss}(x_M)=\QM$ (eq. xM), but now $Q_{\rm ss}$ depends on the profile, hence on $x_f$ itself.  Off the fixed point,
the core is lifted uniformly at $\dot p$ (eq. mucore with $\mu(x_f)=\QM$) and the pedestal is quasi-steady, so
$\Qedge=\mathcal S-\dot p\,x_f$ and the energy balance (eq. Wbalance) becomes

$$
\deriv{\Wtot}{t}\;=\;\mathcal S-\Qedge\;=\;\dot p\,x_f\;=\;F(x_f)\;\equiv\;Q_{\rm ss}\big(x_f;\,p[x_f]\big)-\QM .
$$

Fixed points are $F=0$.  The thermal rate of eq. (Wstab) is, along this one-parameter (front-position) family,

$$
\gamma_{\rm th}\;=\;\deriv{\mathcal S}{\Wtot}-\deriv{\Qedge}{\Wtot}\;=\;\frac{F'(x_f)}{\Wtot'(x_f)},\qquad \Wtot'(x_f)<0 ,
$$

and

$$
F'(x_f)\;=\;S(x_f)\;+\;\int_0^{x_f}\pderiv{S}{p}\,\pderiv{p}{x_f}\,\dd x
\;\simeq\;S(x_f)\;-\;\Lambda\,I_{\rm core},
\qquad
I_{\rm core}\equiv\int_0^{x_f}\pderiv{}{p}\big(P_\alpha-P_{\rm brem}\big)\dd x ,
$$

where $\Lambda=-\partial p_{\rm core}/\partial x_f$ is the core lift per unit inward displacement of the front.  At
frozen sources it follows from eq. (lift) and from the pedestal flux rising by $S(x_f)\,\delta$ when the front moves in
by $\delta$ ($\mu=Q_{\rm ss}-\dot p\,x_f$ on $g_+$, slope $1/\chi_{\rm ped}$):

$$
\Lambda_0\;=\;(g_b-g_a)\;+\;\frac{S(x_f)\,(L-x_f)}{\chiRR+\chiMHD}.
$$

The first term is the paper's Stefan lift, the second the pedestal steepening.  Hence:

$$
\boxed{\;\text{stable}\iff S(x_f)>\Lambda\,I_{\rm core};\qquad
\text{fold:}\quad Q_{\rm ss}(x_f)=\QM\ \ \text{and}\ \ S(x_f)=\Lambda\int_0^{x_f}\pderiv{}{p}\big(P_\alpha-P_{\rm brem}\big)\dd x\;}
$$

equivalently $\dd\mathcal S/\dd\Wtot=\dd\Qedge/\dd\Wtot$ with $\mathcal S=\Qedge$.  In words: the source density at
the front, which is what holds the front at $x_M$ (it fixes the restoring rate $1/\tau_f=S(x_M)/x_M(g_b-g_a)$ of eq.
tauf), must beat the thermal gain of the core it encloses.  For $S(x_f)$ large this is the paper's front law; the
thermal feedback weakens the restoring force and the fold is where it vanishes.

Numerical check on the sharp branch ($f=0$; `tables`, "fold criterion"):

| criterion | zero crossing |
|---|---|
| $\gamma_{\rm th}$ of the Stefan family | $x_f=0.328$ |
| $F'(x_f)=\dd Q_{\rm ss}(x_f)/\dd x_f$ | $0.328$ |
| $S(x_f)=\Lambda I_{\rm core}$, $\Lambda$ = self-consistent core lift (numerical, $\approx9.5$–$10$) | $0.328$ |
| $S(x_f)=\Lambda_0 I_{\rm core}$, frozen-source $\Lambda_0$ ($\approx7.6$–$8.2$) | $0.352$ |
| max of $C_{\rm brem}(x_f)$ on the fixed-point branch | $0.35$ (grid of 0.025) |

At the fold $S(x_f)\approx6.2$–$7.4$, $\Lambda\approx9.5$–$10$, $I_{\rm core}\approx0.75$.  The 0.02 between the
derivative crossings and the max of $C(x_f)$ is the finite-difference step ($h=0.004$) against the grid's
core/pedestal staircase ($\Delta x=0.005$); the fold is at $x_f=0.33$–$0.35$.

### 1.3 The transport threshold and a general criterion

With the sink distributed, $Q_{\rm ss}(x)$ is not monotone: it rises through the heated core and falls through the
radiating pedestal.  The existence condition of a barrier is then

$$
\max_x Q_{\rm ss}(x)\;\ge\;\QM ,
$$

not $\mathcal S=Q_{\rm ss}(L)\ge\QM$ (eq. barriercrit, eq. burncrit).  The front rest point is on the rising flank,
$Q_{\rm ss}'(x_M)=S(x_M)>0$; the *transport threshold* is where $x_M$ reaches the maximum of $Q_{\rm ss}$, i.e.
$S(x_M)\to0$, after which the front runs to the wall.  The fold occurs at $S(x_f)=\Lambda I_{\rm core}$.  Therefore:

* if $I_{\rm core}>0$ (core alpha-dominated: $\partial_pP_\alpha>\partial_pP_{\rm brem}$ on average inside the front),
  the fold is reached at a *positive* front source density, before the transport threshold: **the burn ends at a
  thermal fold with $\max Q_{\rm ss}-\QM>0$**, the margin being $\int_{x_f}^{x_{\rm peak}}S\,\dd x$ (0.55 here);
* only if $I_{\rm core}\le0$, or if $S(x_f)\gg\Lambda I_{\rm core}$ all the way to $\max Q_{\rm ss}=\QM$ (front sitting
  in a strong deposition), does the state end at the transport threshold.

Both endings appear in one scan at $\int S_{\rm ext}=7$, $f=0.57$ (`fold_highS_core.npz`): the core burn
($x_f\approx0.35$) is lost by a fold between $C_{\rm brem}=0.523$ and $0.527$ with $\mathcal S=11.5\gg\QM$; the profile
lands on an edge barrier with its front inside the top-hat deposition ($x_f=0.82$–$0.86$, $S(x_f)\approx40$), which
ends between $0.8125$ and $0.8156$ with $\max Q_{\rm ss}=5.856$ (at $x=0.85$) against $\QM=5.851$: the transport
threshold, to $0.1\%$.  (`fold_highS.npz` labels this edge state "cold" once $\Wtot<2$, from $C=0.73$; that label
is a classification artefact, the state is continuous to 0.81.)

Other fold points ($\mathcal S$ at the loss): $f=0$: 5.60 ($<\QM$); $f=0.25$: 8.14; $f=0.5$: 10.46; $f=0.57$: 11.26;
$\int S_{\rm ext}=7$, $f=0$: 3.28 (with $\max Q_{\rm ss}=5.99$).  $\mathcal S\ge\QM$ is neither necessary nor sufficient.

**T3D.** The predictor's fold at $Z_{\rm eff}=6.4$–$6.5$ is the same object: net drive through the barrier tube still
positive (and the hKBM Maxwell flux of that tube zero, so the threshold is at $Z_{\rm eff}\approx16$), $P_\alpha$
falling by a quarter over the last $0.4$ of $Z_{\rm eff}$ while the sink inside the barrier stays at 40 MW per unit:
$I_{\rm core}>0$ large, $S(x_f)$ of the barrier tube modest, fold well before threshold.  The reduced model at $f=0$
is the degenerate case where the fold and the $\mathcal S=\QM$ crossing nearly coincide ($\Delta C\approx0.02$),
which is why the paper's coarse rungs (1.25, 1.5) read it as the threshold.

---------------------------------------------------------------------------------------------------------------------

## 2. Hysteresis window versus edge deposition

Fraction $f$ of $\int S_{\rm ext}=4.94$ moved into a top hat on $x\in[0.75,0.85]$ (`fold_edge.npz`, `fold_sharp_edge.npz`,
`fold_extra.npz`).  The burning root is followed by continuation (in $f$ at $C_{\rm brem}=0.01$, then up in $C$); the cold
root by a down-scan from $C=2$.

| $f$ | burn $x_f$ at $C=0.01$ (BDF) | sharp $x_M$ at $C\to0$ | burn lost (BDF bisection) | sharp $C_{\rm fold}$ | $x_f$ at fold (sharp) | $\mathcal S$ at fold (sharp) | cold root |
|---|---|---|---|---|---|---|---|
| 0 | 0.153 | 0.160 | 1.306–1.313 | 1.327 | 0.35 | 5.57 | all $C\in[0,2]$ |
| 0.10 | – | 0.180 | – | 1.080 | 0.35 | 6.65 | (all) |
| 0.25 | 0.193 | 0.200 | 0.7125–0.7188 | 0.727 | 0.37 | 8.01 | all |
| 0.40 | – | 0.240 | – | 0.393 | 0.38 | 9.58 | (all) |
| 0.50 | 0.284 | 0.290 | 0.170–0.176 | 0.179 | 0.40 | 10.32 | all |
| 0.57 | 0.354 | 0.350 | 0.022–0.028 | 0.034 | 0.40 | 11.02 | all |
| 0.576 / 0.578 | burn / none at $C=0.01$ | | | | | | |
| 0.60–0.75 | none | none | – | none | – | – | all |
| 1.0 | none | – | – | – | – | – | all |

The coexistence window is $0\le C_{\rm brem}\le C_{\rm fold}(f)$: the cold root (rising branch, $\Wtot\approx1.2$–$1.4$)
exists at every $C$ because its drive, $\int S_{\rm ext}+P_\alpha^{\rm cold}\approx5.05<\QM$, never nucleates a barrier
(and the rising branch is metastable up to $\Qmax=10.1$ anyway; the $\int S_{\rm ext}=7$ down-scans never re-ignite
either).  The window closes at $f_c=0.576$–$0.578$ (BDF, bisected at $C_{\rm brem}=0.01$, not 0) / $\approx0.586$ (sharp: the
slope $\dd C_{\rm fold}/\dd f\approx-2.1$ near 0.57 extrapolated to $C_{\rm fold}=0$; no sharp fixed point at $f\ge0.60$),
the BDF running $\sim0.01$ low in $f$ as it does in $C_{\rm fold}$.  Sharp and BDF agree to $\le0.02$ in
$C_{\rm fold}$.

**Why edge heating narrows it.**  The front law places the burning front at $x_M$ with $Q_{\rm ss}(x_M)=\QM$, and the
top hat contributes nothing to $Q_{\rm ss}$ for $x<0.75$, so the front sees only $(1-f)\int S_{\rm ext}$ of external
drive plus the alpha power inside it.  $x_M(C\to0)$ moves out from $0.16$ to $0.35$, the barrier $[x_M,L]$ narrows
(pedestal height $\Delta p=\int_{x_M}^Lg_+(Q_{\rm ss})\,\dd x$, eq. pedheight_theory), the core is cooler
($P_\alpha(C=0.01)$ 9.8 → 6.6, $\Wtot$ 5.86 → 4.51).  The fold, by contrast, sits at an almost fixed front position,
$x_f^{\rm fold}=0.35$–$0.40$ for all $f$, because $S(x_f)=\Lambda I_{\rm core}$ is set by the core-heated alpha profile.
The sink needed to drive the front from $x_M(0)$ out to $x_f^{\rm fold}$ is the window; it vanishes when
$x_M(0)$ reaches $x_f^{\rm fold}$ ($f=0.57$: $0.35$ against $0.40$; $f_c\approx0.58$).  Equivalently: heat deposited
outside the front raises $\mathcal S$ (and $\Qedge$) but not $Q_{\rm ss}(x_f)$, so it helps hold the burn little, and is
outweighed by what it costs: through eq. (pedheight_theory) the top hat raises $g_+$ by $f\cdot4.94/\chi_{\rm ped}$ over
$x>0.85$, lifting the core by $\approx0.36f$ ($0.18$ at $f=0.5$), against $(g_b-g_a)\,\Delta x_M\approx5.7\times0.13\approx0.74$
lost by the outward move of $x_M$.  At the fold $\mathcal S$ rises from 5.6 to 11 with $f$ while $C_{\rm fold}$ falls 40-fold.

**T3D.**  The predictor's window is $5\le Z_{\rm eff}\le6.4$: its upper edge is the thermal fold of §1, the same
mechanism.  Its *lower* edge — the cold root existing only from $Z_{\rm eff}=5$; a collapsed member re-ignites at 3.4 —
is a cold-branch ignition threshold (the hollow 1–2 keV core with the EC-heated cells at 0.75–0.85 re-ignites
unless the sink is large).  The reduced model at these parameters has no such threshold (cold $P_\alpha\approx0.15$–$0.21$)
so its lower edge stays at $C=0$.  So: the reduced model reproduces how edge deposition lowers the *upper* edge (the
fold), not the rising lower edge that t3d.tex attributes to the EC keeping the outer cells warm.  The nearest analogue
in the reduced model is the edge-barrier state of §1.3 ($\int S_{\rm ext}=7$, $f=0.57$): the deposition anchors a
third, warm root with its front at 0.82–0.86, onto which the core burn collapses instead of onto the L state.

---------------------------------------------------------------------------------------------------------------------

## 3. Edge-flux response $\dd\Qedge/\dd P_{\rm rad}$ on the barrier branch

### 3.1 Formula

Add a sink increment $\delta\Sigma$ ($=\delta\!\int P_{\rm brem}\dd x$ on the old profile) and linearise eq. (Wbalance)
about a burning fixed point, with $\mathcal S$ and $\Qedge$ functions of $\Wtot$ along the branch:

$$
\deriv{\,\delta\Wtot}{t}=\gamma_{\rm th}\,\delta\Wtot-\delta\Sigma,\qquad
\delta\Qedge=\deriv{\Qedge}{\Wtot}\,\delta\Wtot
$$

$$
\boxed{\;\frac{\delta\Qedge(t)}{\delta\Sigma}=-\,G\,\big(1-e^{\gamma_{\rm th}t}\big),\qquad
G=\frac{\dd\Qedge/\dd\Wtot}{\dd\Qedge/\dd\Wtot-\dd\mathcal S/\dd\Wtot}=\frac{1}{1-(\dd\mathcal S/\dd\Wtot)/(\dd\Qedge/\dd\Wtot)}\;}
$$

and the plasma gives up $\dd(\delta\Wtot)/\dd t=-\delta\Sigma\,e^{\gamma_{\rm th}t}$: at first the whole sink comes out of
the stored energy, then the edge flux takes it over on $1/|\gamma_{\rm th}|$, and in steady state it takes $G>1$ times the
sink, because the alpha power falls with the core: $\delta\mathcal S=\delta\Qedge=-G\,\delta\Sigma$, i.e.
$\delta P_\alpha=-(G-1)\,\delta\Sigma$.
$G\to\infty$ as $\gamma_{\rm th}\to0$: the divergence of the edge-flux response *is* the fold of §1.  The one-mode
formula alone reproduces the measured response to $\pm0.07$ at all times (below).  The residual (measured faster at
$t\lesssim0.1\tau_E$) is not explained: it is not simply the share of the sink outside the front, which is 0.67–0.81
while the excess is only $\approx0.07$.

### 3.2 Reduced model (`fold_step.npz`, `fold_extra.npz`)

Steady gain, measured against $G$ from the slopes along the Jacobian's leading eigenvector (not the Stefan-family
slopes of §1, which coincide with these only at the fold; hence 5.11 vs 4.35 at the ignited state):

| start state | $\tau_E=\Wtot/\Qedge$ | sink outside front | $\dd\mathcal S/\dd\Wtot$, $\dd\Qedge/\dd\Wtot$ | $G$ predicted | $-\delta\Qedge/\delta\Sigma$ steady |
|---|---|---|---|---|---|
| burn, $C$ 0.01 → 0.06 | 0.40 | 0.81 | 2.16, 5.11 | 1.73 | 1.72 |
| burn, $C$ 0.01 → 0.21 | 0.40 | 0.81 | 2.16, 5.11 | 1.73 | 1.69 (nonlinear step) |
| burn, $C$ 0.80 → 0.85 | 0.52 | 0.74 | 2.14, 4.93 | 1.77 | 1.79 |
| burn, $C$ 1.20 → 1.22 | 0.63 | 0.67 | 2.07, 4.91 | 1.73 | 2.08 (front moved one cell) |
| sharp branch, $x_f=0.27$ ($C=1.20$) | | | 2.11, 3.00 | 3.4 | – |
| cold branch, $C$ 0.8 → 1.0 (fold_step's middle case started cold) | | | −0.12, 16.1 | 0.99 | 0.98 |

Time history at $C=0.8\to0.85$ ($\gamma_{\rm th}=-2.79$):

| $t$ | $t/\tau_E$ | $-\delta\Qedge/\delta\Sigma$ measured | one-mode $G(1-e^{\gamma t})$ |
|---|---|---|---|
| 0.02 | 0.04 | 0.16 | 0.10 |
| 0.05 | 0.10 | 0.30 | 0.23 |
| 0.10 | 0.19 | 0.49 | 0.43 |
| 0.20 | 0.39 | 0.78 | 0.76 |
| 0.50 | 0.97 | 1.30 | 1.33 |
| 1.00 | 1.94 | 1.58 | 1.66 |
| 4.00 | 7.8 | 1.79 | 1.77 |

(The $C=1.2\to1.22$ history is the same to $\pm0.01$ up to $t=0.2$.)  In the discretised model $G$ stays near 1.75
almost to the loss because the pinned front keeps $\gamma_{\rm th}\approx-2.8$ (§1.1); in the continuum it grows to 3.4
at $C=1.2$ and diverges at $C_{\rm fold}$.

### 3.3 T3D

T3D: sink 41 MW per unit $Z_{\rm eff}$ on the starting profile; at 0.1 s $\Qedge$ has fallen by 12 MW per unit, so
$\delta\Qedge/\delta\Sigma=-0.29$ and the plasma gives up $29/41=0.71$ of the sink.  The formula with the predictor's
own slopes (hKBM, $n(0)$: $\dd\mathcal S/\dd\Wtot=0.80$, $\dd\Qedge/\dd\Wtot=7.4$ MW/MJ, so $\gamma_{\rm th}=-6.6\,{\rm s^{-1}}$,
$G=1.12$) gives at $t=0.1$ s $-G(1-e^{-0.66})=-0.54$: the edge 22 MW, the plasma 19 MW of 41 — the right structure,
off by a factor 2 in the split.  Not a clean test: T3D at 0.1 s is relaxing from a non-steady start ($\Qedge=355$ MW
against 83 MW of residual flux), its slopes are the uniform-rescaling proxy, and only $\sim5\%$ of its sink lies
outside the 0.8 barrier (drive through 0.8 falls 128 of 135 MW at $Z_{\rm eff}=3.4$), so the direct pedestal part is
negligible there.  In the reduced model the ratio 0.29 is reached at $t\approx0.1\,\tau_E$.

A branch finite-difference analogue of $G$ along T3D's predicted burning branch, from the predictor's fixed points
($\delta\Qedge=\delta P_\alpha-\delta P_{\rm rad}$, with $\delta P_{\rm rad}$ taken between the end profiles, not
$\delta\Sigma$ on the old profile, so not the formula's $G$ and biased high): $Z_{\rm eff}$ 0 → 3.4: $P_\alpha$ 718 → 568, $P_{\rm rad}$ 0 → 152 MW,
$G=302/152=2.0$; 3.4 → 6: $P_\alpha$ 568 → 348, $P_{\rm rad}$ 152 → 245, $G=313/93=3.4$.  Growing towards the fold,
as in the reduced model (1.73 → 3.4 → ∞).  (The proxy slopes' $G=1.12$ is inconsistent with this — the uniform
rescaling is not the branch direction.)

---------------------------------------------------------------------------------------------------------------------

## What the paper can quote

1. **Correction to Section 5 ("Radiative collapse", fig:brem caption).**  Currently: "the burn ends where $\mathcal S$
   crosses $\QM$".  Numerics: the burning state exists up to $C_{\rm brem}=1.309$ (lost by 1.3125) with
   $\mathcal S=5.60$ ($5.66$ on the trapezoid) $<\QM=5.85$, $\max_xQ_{\rm ss}=6.40>\QM$; the burn ends at a thermal fold
   ($C_{\rm fold}=1.327$ in the sharp-interface limit, at $x_f\simeq0.33$–$0.35$, $\Wtot=3.9$), where
   $\dd\mathcal S/\dd\Wtot=\dd\Qedge/\dd\Wtot\simeq2.2$.  The rungs 1.25 and 1.5 straddle it.  The statement
   "hysteresis in $C_{\rm brem}$ is total whenever $\int S_{\rm ext}<\QM$" stands.
2. **Corrections to eqs. (barriercrit) and (burncrit).**  (barriercrit), existence of a *barrier*: with a
   distributed sink replace $\mathcal S$ by $\max_xQ_{\rm ss}(x)$; $\mathcal S\ge\QM$ is not necessary (burning
   barriers at $\mathcal S=5.60$, and at $3.28$ with $\int S_{\rm ext}=7$).  (burncrit), existence of a *burning* fixed
   point: $\max_xQ_{\rm ss}\ge\QM$ is necessary but not sufficient — the fold adds an independent condition
   $S(x_f)>\Lambda I_{\rm core}$, so no drive-only inequality suffices (the core burn is lost at $\mathcal S=11.5$,
   $\max Q_{\rm ss}=11.6$, by a fold).
3. **Fold condition:** $Q_{\rm ss}(x_f)=\QM$ and $S(x_f)=\Lambda\int_0^{x_f}\partial_p(P_\alpha-P_{\rm brem})\,\dd x$, with
   $\Lambda\simeq(g_b-g_a)+S(x_f)(L-x_f)/(\chiRR+\chiMHD)$ the core lift per unit front displacement; checked to
   0.02 in $x_f$.  General rule: a core with positive thermal gain loses its burn at a fold before the transport
   threshold; the threshold ($\max Q_{\rm ss}=\QM$) ends a state only when the front sits in a strong source (edge barrier
   in a deposition: lost at $\max Q_{\rm ss}=5.856$ vs $\QM=5.851$).  T3D's fold at $Z_{\rm eff}=6.4$–$6.5$ is this.
4. **Edge deposition narrows the window from above:** $C_{\rm fold}=1.33,\,1.08,\,0.73,\,0.39,\,0.18,\,0.03$ for
   $f=0,0.1,0.25,0.4,0.5,0.57$ (sharp; BDF 1.31, 0.71, 0.17, 0.025), no burn for $f\ge0.58$.  Mechanism: edge heat
   moves $x_M$ out (0.16 → 0.35) while the fold stays at $x_f\approx0.35$–$0.40$.  The reduced model does not have
   T3D's raised lower edge ($Z_{\rm eff}=5$); that is a cold-root ignition threshold absent here.
5. **Edge-flux response:** $\delta\Qedge/\delta\Sigma=-G(1-e^{\gamma_{\rm th}t})$, $G=(\dd\Qedge/\dd\Wtot)/(\dd\Qedge/\dd\Wtot-\dd\mathcal S/\dd\Wtot)$;
   reduced model $G=1.73$ predicted, $1.72$ measured at the ignited state, $-0.30$ at $t=0.1\tau_E$, $-0.78$ at
   $0.4\tau_E$; $G$ diverges at the fold.  T3D's $12/41=0.29$ at 0.1 s is reproduced only to a factor 2 with its own
   slopes; the branch finite-difference gain along its burning branch grows towards the fold (2.0, then 3.4), as the
   formula says — quote the trend, not the values.

## Files

* `exp_fold.py` — all stages; new this session: `sharp_edge`, `extra`, `highS_core`; fixed `figs` tag parsing;
  `tables` prints the fold-criterion crossings and the new caches.
* caches: `fold_scan.npz`, `fold_newton.npz`, `fold_sharp.npz`, `fold_nu4.npz`, `fold_edge.npz`, `fold_highS.npz`,
  `fold_step.npz`, `fold_sharp_edge.npz`, `fold_extra.npz` (its `hs57_*` arrays are superseded: started on the edge
  state), `fold_highS_core.npz`.
* `fold_tables.txt`, `fold_figs/*.png` (quick-look, not paper figures).
