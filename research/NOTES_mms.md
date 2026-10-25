# Flux-driven transport as a multiple-scales problem (NOTES_mms.md)

Question (D. Kennedy): *can the theory be done by the method of multiple scales, as an asymptotic theory?*  Yes.  There
are three small parameters, each with its own expansion; all leading-order predictions below were checked against the
reduced model, with the error measured at fixed everything else while one parameter is dialled.

Files: `exp_mms.py` (stages `hot mu mualpha mutraj delta foldd epsstat epsdyn t3d tables figs`), `mms_core.py` (the
discretisation, Newton, eigen-solver, density model), caches `mms_*.npz`, tables `mms_tables.txt`, quick looks
`mms_figs/*.png`, paper figure `transport_equations_paper/scripts/make_mms_figure.py` -> `plots/fig_mms.pdf`.
Local python, one process, this node; every stage runs in seconds to minutes (δ: 10 min).

---------------------------------------------------------------------------------------------------------------------

## 0. Numerical set-up (what differs from the paper's production scheme, and why)

* **Discretisation** (`mms_core.Model`): cell-centred, axis exactly at x = 0, conservative hyperviscous flux that is zero
  on both boundary faces, no binomial smoothing.  Then $\dd W/\dd t=\mathcal S-\Qedge$ holds *exactly* in the discrete
  model (checked to round-off), which is the identity the amplitude equation is built on.
* **Kinks rounded.**  The paper's $Q(g)$ has two kinks ($\max(1-g/\gc,0)$ and $\max(g-\gMHD,0)$).  Inside a resolved
  front the discrete linearisation jumps whenever a face gradient crosses a kink: at N = 500 a spurious eigenvalue of
  +12 appeared at $C_{\rm brem}=1.1$ (stable state), and the thermal eigenvalue was still non-monotone at N = 2000.  The
  inner expansion needs $Q\in C^2$ anyway.  Both kinks are replaced by a softplus of width $\sigma_g=0.05$ (1.25% of
  $\gc$): $\QM=5.8602$ instead of $5.8514$, $g_a=0.7077$, $g_b=6.3708$.  (`sig=0` restores the paper's law: $\QM=5.8514$ reproduced.)
* **Resolution.**  μ and ε runs: $\nu_4=2\times10^{-3}$, N = 1000 (front width $w=1.476\sqrt{\nu_4}=0.066$, 66 cells);
  thermal eigenvalue converged to 4 digits (N = 1000 vs 2000).  At the production $\nu_4=5\times10^{-4}$, N = 2000 is
  needed; at N = 500 (w/h ≈ 4) the front mode is wrong by 20–40%.  The δ runs use w/h ≥ 20 (to 6000 cells); at
  w/h ≈ 21 the front eigenvalue is still off by 5–10% (the steady positions are fine).
* **Physics** as in the radiation scan (NOTES_fold.md): $C_\alpha=2\times10^{-5}$, $T_{\rm ref}=3$ keV,
  $\int S_{\rm ext}=4.94$; hot root at $C_{\rm brem}=0.01$: $W=5.889$, $\gamma_{\rm th}=-2.78$ (paper: $-2.9$).

---------------------------------------------------------------------------------------------------------------------

## 1. The derivation (ready to paste as a paper section; macros as in the paper)

### 1.1 Scales and parameters

Write the model with a density field,
$$
\pderiv{p}{t}=-\pderiv{J}{x}+S(x,p,n),\qquad J=Q(g)-\nu_4\,g_{xx},\qquad g=-p_x,\qquad
S=S_{\rm ext}+P_\alpha(p,n)-P_{\rm brem}(p,n),
$$
$$
\pderiv{n}{t}=-\pderiv{\Gamma}{x}+S_N(x),\qquad \Gamma=-D_n\,n_x ,
$$
with $T=T_{\rm ref}p/n$ in the closures, $p_x=J=0$ and $n_x=0$ on the axis, $p=p_{\rm ped}$, $n=n_{\rm ped}$ and zero
hyperviscous flux at the edge.  The particle model is the minimal one consistent with
Section 6.8 of the paper: a fixed (pellet) source, a diffusive flux, and an inventory time $\tau_N$ long compared
with the thermal time.  Four time scales appear, and the theory is the statement that they separate:
$$
\underbrace{w^2/|Q'|}_{\text{front}}\;\ll\;\underbrace{\tau_E,\ \tau_f}_{\text{profile}}\;\ll\;
\underbrace{\tau_{\rm th}=|\gamma_{\rm th}|^{-1}}_{\text{thermal mode}}\;\ll\;\underbrace{\tau_N}_{\text{density}} .
$$
Three dimensionless ratios measure the separations:
$$
\delta=\frac{w}{L},\quad w=\sqrt{\nu_4}\,\frac{g_b-g_a}{\max_g\sqrt{2[\Omega(g)-\Omega(g_a)]}};\qquad
\mu=1-\frac{\dd\mathcal S/\dd\Wtot}{\dd\Qedge/\dd\Wtot}\Big|_{\Wtot^\star};\qquad
\epsilon=\frac{\tau_{\rm th}}{\tau_N},
$$
with $\Omega=\Phi-\QM g$ the grand potential of Section 4.  $\delta$ is a length ratio and is treated by matched
asymptotics; $\mu$ and $\epsilon$ are rate ratios and are treated by multiple scales.  Since
$\gamma_{\rm th}=-\mu\,\dd\Qedge/\dd\Wtot$, a small margin is the same thing as a slow thermal mode; the
thermal rate is measured against the next profile mode $\lambda_2$, and $|\gamma_{\rm th}/\lambda_2|\propto\mu$.

### 1.2 Sharp interface: the Maxwell rest condition and the Stefan law (δ ≪ 1)

Take $\nu_4=\delta^2\hat\nu$ with $\hat\nu=O(1)$ (lengths in $L$; the tables use $\delta=w/L=1.476\sqrt{\nu_4}$, i.e. $\hat\nu=0.459$; below $\hat\nu=1$ for brevity, so $\xi$ is measured in $\sqrt{\nu_4}$).  **Outer region.**  Away from fronts
$J=Q(g)+O(\delta^2)$, and the profile is slaved to the flux on one of the two rising branches,
$$
g=g_\mp\big(J(x,t)\big)+O(\delta^2),\qquad J=Q_{\rm ss}(x)-\int_0^x\pderiv{p}{t}\,\dd x' .
$$
**Inner region.**  Near a front at $x_f(t)$ put $x=x_f+\delta\xi$, $g=G_0(\xi,t)+\delta G_1+\dots$,
$J=M_0+\delta M_1+\delta^2M_2+\dots$ and $V=\dot x_f$.  The source is smooth in $x$ and $p$ is continuous, so
$\partial_\xi S=O(\delta)$.  The gradient equation $g_t=J_{xx}-S_x$ becomes
$$
-\frac{V}{\delta}\,G_{\xi}+\dots=\frac{1}{\delta^2}\,M_{\xi\xi}+O(1),\qquad M=Q(G)-G_{\xi\xi}.
$$

*Order $\delta^{-2}$.*  $M_{0,\xi\xi}=0$; boundedness gives $M_0=m_0(t)$, and
$$
G_{0,\xi\xi}=Q(G_0)-m_0,\qquad G_0\to g_a\ (\xi\to-\infty),\quad G_0\to g_b\ (\xi\to+\infty).
$$
The first integral $\tfrac12G_{0,\xi}^2=\Omega(G_0)-\Omega(g_a)$ with $\Omega=\Phi-m_0g$ shows that the
heteroclinic exists if and only if $\Omega(g_a)=\Omega(g_b)$ and $Q(g_a)=Q(g_b)=m_0$, i.e.
$$
m_0=\QM\qquad\text{(leading order: the Maxwell rest condition, eq. maxwell).}
$$
The front is the heteroclinic, and to leading order the outer flux at the front is $\QM$.

*Order $\delta^{-1}$.*  $M_{1,\xi\xi}=-V\,G_{0,\xi}$, so
$M_{1,\xi}=J_x^- -V\,(G_0-g_a)$.  Matching $M_{1,\xi}\to J_x^\pm$ as $\xi\to\pm\infty$ gives the jump condition
$$
J_x^+-J_x^-=-V\,(g_b-g_a)\qquad\Longleftrightarrow\qquad V\,[g]=[\,p_t\,],
$$
which is the continuity of $p$ across the moving front.  With the outer core lifted uniformly at $\dot p$ and the
pedestal quasi-steady, this and $J(x_f)=\QM$ give the paper's Stefan law (eq. frontspeed),
$$
\deriv{x_f}{t}=-\frac{Q_{\rm ss}(x_f)-\QM}{x_f\,(g_b-g_a)} .
$$
The quasi-steady outer pedestal is an additional assumption, not part of the δ expansion.  Its error is the ratio of
the pedestal diffusion time $(L-x_f)^2/\pi^2\chi_{\rm ped}$ to the front time $\tau_f$, a fourth small parameter.
This is the 2–3× lag of Section 4.4, which a stiffer cliff halves.
*(Correction, 25 Oct 2026, NOTES_front_lag.md: the Stefan law needs a **frozen** pedestal, $p_t=0$ at its top, and the
lag is the pedestal's own energy. A quasi-steady pedestal adds $S(x_f)(L-x_f)/\chi_{\rm ped}$, 1–2 times $g_b-g_a$ here,
to the latent heat and makes the front slower, not faster; its time is $(L-x_f)^2/3\chi_{\rm ped}$ (flux fixed at the
front), not $/\pi^2$. The small parameters are that ratio and the step's change of the pedestal height, not the time ratio.)*

*Order $\delta^0$ (solvability).*  $\mathcal L G_1=M_1$ with $\mathcal L=Q'(G_0)-\partial_\xi^2$ self-adjoint
and $\mathcal L\,G_{0,\xi}=0$.  The Fredholm condition $\int G_{0,\xi}M_1\,\dd\xi=0$, with
$M_1=m_1+J_x^-\xi-V\,H(\xi)$ and $H=\int_0^\xi(G_0-g_a)$, fixes the $O(\delta)$ correction to the front flux,
$$
m_1=\frac{V\int G_{0,\xi}H\,\dd\xi-J_x^-\int\xi\,G_{0,\xi}\,\dd\xi}{g_b-g_a}.
$$
The second term depends only on where $\xi=0$ is put.  It vanishes if the front position is defined as the **Gibbs
(equal-area) dividing surface** $x_G$, where $\int\xi\,G_{0,\xi}\,\dd\xi=0$, i.e. $x_G$ is the $g$-average of
$x(g)$ across the front.  Then
$$
J(x_G)=\QM+\delta\,\kappa\,V+O(\delta^2),\qquad \kappa=\frac{1}{g_b-g_a}\Big[\int G_{0,\xi}H\,\dd\xi\Big]+\dots,
$$
a kinetic ("undercooling") correction that vanishes for a front at rest.  Consequences:
* the Gibbs surface of a stationary front sits at the Maxwell point to $O(\delta^2)$: $x_G-x_M=O(\delta^2)$;
* any other marker sits a fixed number of inner lengths from it: $x_{\rm foot}=x_G+\sqrt{\nu_4}\,\xi_{\rm foot}$
  and $x_{\rm mid}=x_G+\sqrt{\nu_4}\,\xi_{\rm mid}$.  For the paper's law, $\xi_{\rm foot}=-0.539$ (where $g=\gc/2$)
  and $\xi_{\rm mid}=-0.094$ (where $g=(g_a+g_b)/2$);
* the stored energy, the pedestal height and every threshold built from the steady branch (the fold included)
  converge as $\delta^2$, because the front's excess $\int(g-g_{\rm step})\,\dd x$ vanishes at $x_G$.

The kinetic coefficient κ was not evaluated or tested here (no moving-front runs).

### 1.3 The thermal mode and the amplitude equation (μ ≪ 1)

Let $\mathcal N(p;c)$ be the right-hand side of the pressure equation, with $c$ a control parameter
($C_{\rm brem}$, $C_\alpha$, the cliff stiffness, ...).  Linearise about a steady state:
$$
\mathcal L=\partial_x\big(Q'(g)\,\partial_x\cdot\big)-\nu_4\partial_x^4+\partial_pS .
$$
With $p_x=0$ on the axis, $p$ fixed at the edge and zero hyperviscous flux, the boundary terms cancel and **$\mathcal L$
is self-adjoint**.  Numerically the interior Jacobian is symmetric to $10^{-9}$, and the left and right thermal
eigenvectors have correlation 0.99998.  Its eigenvalues are real.  The least stable one is the thermal mode
$\gamma_{\rm th}$ with eigenfunction $v$; the next one, $\lambda_2$ (the front/pedestal mode), is $O(1/\tau_E)$.  A
small margin means $\gamma_{\rm th}\to0$, which is a saddle-node (fold) of the steady branch, $c\to c_f$.

**Multiple scales.**  Put $c=c_f-\mu^2\sigma$, $t_1=\mu t$, and
$p=p_f+\mu\,a(t_1)\,v_0+\mu^2p_2(x,t,t_1)+\dots$, with $\mathcal L_fv_0=0$ and $\int v_0\,\dd x=1$ (so that
$a=\delta\Wtot/\mu$ at leading order).
* $O(\mu)$: $\mathcal L_fv_0=0$.
* $O(\mu^2)$: $\partial_tp_2=\mathcal L_fp_2-v_0\,\dot a+\tfrac12\mathcal N_{pp}[v_0,v_0]\,a^2-\sigma\,\mathcal N_c$.

$p_2$ stays bounded on the fast time only if the right-hand side has no component along the null vector.  Since
$\mathcal L_f$ is self-adjoint, the adjoint null vector is $v_0$ itself, and
$$
\boxed{\ \deriv{a}{t_1}=\kappa\,a^2-\beta\,\sigma,\qquad
\kappa=\frac{\langle v_0,\mathcal N_{pp}[v_0,v_0]\rangle}{2\langle v_0,v_0\rangle},\qquad
\beta=\frac{\langle v_0,\mathcal N_c\rangle}{\langle v_0,v_0\rangle}\ }
$$
This is the saddle-node normal form.  The hot root $a^\star=+\sqrt{\beta\sigma/\kappa}$ (for $\kappa<0$, $\beta<0$) has
$\gamma_{\rm th}=2\mu\kappa a^\star=-2\sqrt{\kappa\beta\,(c_f-c)}$, so $\mu\propto\sqrt{c_f-c}$, and a quasi-static
scan approaching the fold slows like $(c_f-c)^{-1/2}$.

**The integral identity, and what closes it.**  Integrating the pressure equation over the domain gives, for
*every* profile,
$$
\deriv{\Wtot}{t}=\mathcal S[p]-\Qedge[p]\qquad\text{(exact; discrete version exact to round-off)},
$$
which is eq. Wbalance.  It becomes an amplitude equation $\dot\Wtot=\mathcal S(\Wtot)-\Qedge(\Wtot)$ only under a
closure: the profile must be a function of $\Wtot$ alone, $p=P(x;\Wtot)$.  It is exact when $P$ is the slow
manifold, which the trajectory reaches after the first transient (a few $|\lambda_2|^{-1}$).  The closure therefore holds
**when $\mathcal S$ and $\Qedge$ are measured along the run's own trajectory**, as in Section 6.6.

It does **not** hold, even at leading order, when $P$ is the family of steady states (a quasi-static scan, the
predictor, the Stefan family).  That family is tangent to $v_0$ at the fold, but the balance uses the test function 1,
not the adjoint $v_0$.  Expanding as above:
$$
\deriv{\Wtot}{t}\Big|_{\rm family}=\mu^2\Big[\tfrac12\!\int\!\mathcal N_{pp}[v_0,v_0]\,a^2-\sigma\!\int\!\mathcal N_c\Big]
+\mu^2\!\int\!\mathcal L_fp_2 ,
$$
and $\int\mathcal L_fp_2\neq0$, because the fast part of $p_2$ moves the stored energy too.  The family closure
keeps the zeros of the balance and their order, so the fixed points and the fold location are exact.  It gets the
rate wrong by the projection factor
$$
\Pi=\frac{\int\mathcal N_c\,\dd x}{\langle v_0,\mathcal N_c\rangle/\langle v_0,v_0\rangle}\qquad
\big(\gamma_{\rm th}^{\rm family}\to\Pi\,\gamma_{\rm th}\ \text{as}\ \mu\to0\big),
$$
the integral of the perturbation over its thermal projection.  **Π belongs to the family, i.e. to the parameter
that generates it**, not to the state.  For the radiation sink, 60% of it in the pedestal, $\Pi=1.53$ at the fold and
$1.11$ at the ignited state.  For the alpha power, which sits in the core where the mode lives, $\Pi=1.20$ at the fold
and $0.99$ at its start.  A family generated by a perturbation that is itself the thermal mode has $\Pi=1$.  A sink in the pedestal lowers $\Wtot$ through the fast pedestal response,
not through the thermal mode.  The variationally correct leading-order estimate weights the balance with the mode,
$$
\gamma_{\rm th}\simeq\frac{\langle t,\mathcal Lt\rangle}{\langle t,t\rangle},\qquad t=\partial_cp^\star\ \text{(the steady-family tangent)},
$$
with relative error $O(\mu)$, since $|t-v_0|=O(\mu)$ and a Rayleigh quotient is second order in the error of the
trial function.

**The first transient.**  A perturbation $\delta p_0$ projects onto the slow mode with amplitude
$\langle v_0,\delta p_0\rangle/\langle v_0,v_0\rangle$, not $\int\delta p_0$.  The difference leaves within the first
$|\lambda_2|^{-1}$ through the fast modes: a quarter of a uniform-rescaling perturbation near the fold.  After that,
$\Wtot-\Wtot^\star\propto e^{\gamma_{\rm th}t}$.

### 1.4 The density as a slow parameter: slow manifold, fold and delayed jump (ε ≪ 1)

With $T=\epsilon t$ the density obeys $n_T=O(1)$ and is slaved to nothing, while the pressure relaxes on $t$.  Let
$F(\Wtot,N)=\mathcal S-\Qedge$ on the slow manifold of §1.3 at frozen density, and $\Wtot=\Wtot_0+\epsilon\Wtot_1+\dots$.
* $O(1)$: $F(\Wtot_0,N(T))=0$, i.e. $\Wtot_0=\Wtot^\star(N(T))$ on the stable (hot) branch.  The burn follows its
  frozen-density fixed point.
* $O(\epsilon)$: $\dot\Wtot_0=F_W\Wtot_1$, so
  $$
  \Wtot-\Wtot^\star=\frac{1}{\gamma_{\rm th}}\deriv{\Wtot^\star}{t}+O(\epsilon^2)
  =\tau_{\rm th}\,\big|\dot\Wtot^\star\big|\ \text{(the lag)} .
  $$
  Equivalently, the instantaneous fixed point is $\Wtot^\star\simeq\Wtot+\dot\Wtot/|\gamma_{\rm th}|$, and
  $\epsilon_W=\tau_{\rm th}\,\dd\ln\Wtot^\star/\dd t$ is the relative lag.

The outer expansion fails where $\gamma_{\rm th}\to0$, at the fold $N_f$ of the frozen branch, since $\Wtot_1\propto
1/\gamma_{\rm th}\propto(T_f-T)^{-1/2}$.  **Inner layer.**  Near the fold
$$
\deriv{w}{t}=-A\,w^2-B\,r\,(t-t_f),\qquad w=\Wtot-\Wtot_f,\quad r=\dd s/\dd t ,
$$
where $s$ is any coordinate along the density path ($s=D_nt$ here) and $AB$ follows from the static branch through
$\gamma_{\rm th}^2=4AB\,(s_f-s)$.  The rescaling $w=k\,u$, $t-t_f=c\,\tau$ with $c=(ABr)^{-1/3}$ and $k=1/(Ac)$ gives the
Riccati equation $u_\tau=-u^2-\tau$.  Its solution matching $u\sim\sqrt{-\tau}$ as $\tau\to-\infty$ is
$$
u=-\frac{\mathrm{Ai}'(-\tau)}{\mathrm{Ai}(-\tau)},
$$
which blows up at the first zero of $\mathrm{Ai}(-\tau)$, $\tau=\Omega_0=2.338107$.  The jump is therefore delayed
past the static fold by
$$
\boxed{\ t_{\rm jump}-t_f=\Omega_0\,(AB\,r)^{-1/3},\qquad s_{\rm jump}-s_f=\Omega_0\,r^{2/3}(AB)^{-1/3}\propto\epsilon^{2/3}\ }
$$
in the slow variable, i.e. $\propto\epsilon^{-1/3}$ thermal times.  In terms of measurables: if
$\gamma_{\rm th}=\gamma_0$ a time $T_0$ before the static fold, then $ABr=\gamma_0^2/4T_0$ and
$$
t_{\rm jump}-t_f=4^{1/3}\Omega_0\,(T_0\,\tau_{\rm th,0}^2)^{1/3}=3.71\,(T_0\tau_{\rm th,0}^2)^{1/3}.
$$
The next correction is $O(\epsilon\ln\epsilon^{-1})$ in $s$ (the fold-point theorem of Krupa & Szmolyan 2001,
*SIAM J. Math. Anal.* 33, 286; **citation from memory, to verify**; the numbers in §2.3 are consistent with an
$\epsilon\ln\epsilon$ term).  The $\epsilon^{2/3}$ jump with the Airy constant is Haberman's (1979, *SIAM J. Appl.
Math.* 37, 69; **to verify**).  The exponent and the constant $\Omega_0$ were derived above and checked numerically,
independently of these references.  Note that a saddle-node has no "memory" delay of $O(1)$ in the slow variable: that delay belongs to
the transcritical, pitchfork and Hopf passages of Mandel & Erneux (1987, *J. Stat. Phys.* 48, 1059; **to verify**), where the state
sits on an equilibrium that persists through the bifurcation.  Here the equilibrium disappears, and the delay is only
$O(\epsilon^{2/3})$.

---------------------------------------------------------------------------------------------------------------------

## 2. Verification (all numbers from `mms_tables.txt`)

### 2.1 δ: steady front, fixed source $\int S_{\rm ext}=7.56$, no alpha (`delta`)

Sharp limit: $x_M=0.214561$, $\Wtot=4.424981$.  Inner solution: $\xi_{\rm foot}=-0.5390$, $\xi_{\rm mid}=-0.0937$,
$w=1.4758\sqrt{\nu_4}$.  $\nu_4=2.5\times10^{-5}$ … $6.4\times10^{-3}$ ($\delta=0.0074$ … $0.118$), finest grid per
$\nu_4$ (to N = 6000), each checked at two grids.

| δ | $x_G-x_M$ | $x_{\rm foot}-x_M$ | $x_{\rm foot}-x_M-\sqrt{\nu_4}\xi_{\rm foot}$ | $W-W_{\rm sharp}$ |
|---|---|---|---|---|
| 0.0074 | +6.2e-5 | −2.62e-3 | +7.1e-5 | −1.46e-4 |
| 0.0148 | +2.7e-4 | −5.11e-3 | +2.8e-4 | −5.80e-4 |
| 0.0295 | +1.10e-3 | −9.67e-3 | +1.11e-3 | −2.29e-3 |
| 0.0590 | +4.23e-3 | −1.73e-2 | +4.24e-3 | −8.93e-3 |
| 0.118 | +1.54e-2 | −2.78e-2 | +1.53e-2 | −3.40e-2 |

**Slopes (δ < 0.07):** $x_G$: **2.03**; $x_{\rm foot}$: **0.91** (→1, the $O(\delta)$ inner offset); $x_{\rm foot}$
after subtracting $\sqrt{\nu_4}\xi_{\rm foot}$: **1.97**; $x_{\rm mid}$ likewise: **1.97**; $\Wtot$: **1.98**.

**Fold location against δ** (`foldd`, $C_{\rm brem}$ fold with alpha and the sink self-consistent).  The sharp-interface
fold, from the Maxwell-front branch with $C_{\rm brem}$ solved at each front position, is $C_{\rm fold}=1.32169$ at
$x_f=0.346$, $\Wtot=3.891$.  The discrete model by bordered continuation gives

| $\nu_4$ | δ | $C_{\rm fold}$ | difference |
|---|---|---|---|
| 4e-3 | 0.093 | 1.28938 | −3.23e-2 |
| 2e-3 | 0.066 | 1.30502 | −1.67e-2 |
| 1e-3 | 0.047 | 1.31314 | −8.6e-3 |
| 5e-4 | 0.033 | 1.31731 | −4.4e-3 |
| 2.5e-4 | 0.023 | 1.31943 | −2.3e-3 |
| 1.25e-4 | 0.016 | 1.32051 | −1.2e-3 |

**Slope 1.91.**  (NOTES_fold's 1.327 was on the kinked law with smoothing; the BDF loss points there bracketed it
non-monotonically because the front was 1–5 cells wide.)

### 2.2 μ: thermal rate approaching the fold in $C_{\rm brem}$ and in $C_\alpha$ (`mu`, `mualpha`, `mutraj`)

μ is dialled in two ways.
* **(i) $C_{\rm brem}$** at $C_\alpha=2\times10^{-5}$: the radiation fold of NOTES_fold, verified smooth.  Fold at
  $C_{\rm brem}=1.3046$, $\Wtot=3.888$ ($\nu_4=2\times10^{-3}$).
* **(ii) $C_\alpha$** at $C_{\rm brem}=1.0$, as the task asked.  Fold at $C_\alpha=1.6930\times10^{-5}$,
  $\Wtot=3.853$.

Both are the same thermal fold approached along different directions.  The cliff stiffness was not used: it moves
$\QM$ and the front together, so the margin is not changed alone.

(i) $C_{\rm brem}$:

| $C_{\rm brem}$ | μ | $\gamma_{\rm th}$ | $\lambda_2$ | $\lvert\gamma/\lambda_2\rvert$ | $\gamma^{\rm family}_{W\text{-balance}}/\gamma$ | Π | $\lvert\gamma_{\rm Rayleigh}/\gamma-1\rvert$ | uniform-$p$ proxy |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.565 | −2.783 | −30.7 | 0.091 | 1.108 | 1.115 | 1.9e-3 | −2.37 |
| 1.00 | 0.492 | −2.048 | −19.9 | 0.103 | 1.193 | 1.211 | 8.2e-3 | −2.20 |
| 1.271 | 0.286 | −0.852 | −15.8 | 0.054 | 1.387 | 1.412 | 1.2e-2 | −2.28 |
| 1.300 | 0.127 | −0.315 | −15.3 | 0.021 | 1.489 | 1.502 | 6.1e-3 | −2.35 |
| 1.3044 | 0.051 | −0.117 | −15.3 | 0.0076 | 1.527 | 1.533 | 2.2e-3 | −2.38 |

**Slopes (μ < 0.3):**
* scale separation $|\gamma_{\rm th}/\lambda_2|\sim\mu^{1.13}$;
* Rayleigh quotient on the steady family, relative error $\sim\mu^{1.03}$;
* $W$ balance closed on the steady family: relative error $\sim\mu^{-0.17}$, tending to $\Pi-1=0.53$ and not to zero;
* uniform rescaling of the profile (the paper's eq. Wstab proxy, and the predictor's stability test): it does not see
  the fold at all ($-2.38$ where $\gamma_{\rm th}=-0.12$; error $\sim\mu^{-1.4}$).

Π computed from the adjoint matches the measured $\gamma^{\rm family}/\gamma$ to 0.4–1.5% at every point.

(ii) $C_\alpha$:

| $C_\alpha$ | μ | $\gamma_{\rm th}$ | $\gamma^{\rm family}_{W}/\gamma$ | Π | $\lvert\gamma_{\rm Rayleigh}/\gamma-1\rvert$ |
|---|---|---|---|---|---|
| 2.000e-5 | 0.492 | −2.048 | 0.984 | 0.986 | 7.5e-4 |
| 1.742e-5 | 0.345 | −1.001 | 1.096 | 1.106 | 3.2e-3 |
| 1.704e-5 | 0.206 | −0.491 | 1.153 | 1.161 | 2.4e-3 |
| 1.6936e-5 | 0.055 | −0.109 | 1.194 | 1.196 | 6.6e-4 |
| 1.6930e-5 | 0.006 | −0.012 | 1.204 | 1.204 | 7.2e-5 |

Slopes (μ < 0.3): $|\gamma/\lambda_2|\sim\mu^{1.07}$; Rayleigh relative error $\sim\mu^{1.00}$; $W$-balance
error → $\Pi-1=0.20$.  So the non-asymptotic error of the family closure is real for both families, and its size is
set by the family.

Perturbation runs (`mutraj`; $p\to p^\star+2\times10^{-3}(p^\star-p_{\rm ped})$):

| μ | $\gamma$ (eig) | late-time fit | $\dd\mathcal S/\dd\Wtot$ traj / eig | $\dd\Qedge/\dd\Wtot$ traj / eig | amplitude after transient / $\delta\Wtot_0$ | adjoint prediction |
|---|---|---|---|---|---|---|
| 0.492 | −2.0479 | −2.0457 | 2.114 / 2.114 | 4.172 / 4.162 | 0.949 | 0.954 |
| 0.286 | −0.8517 | −0.8516 | 2.130 / 2.131 | 3.003 / 2.983 | 0.833 | 0.840 |
| 0.127 | −0.3153 | −0.3160 | 2.158 / 2.159 | 2.495 / 2.474 | 0.781 | 0.796 |
| 0.051 | −0.1165 | −0.1174 | 2.170 / 2.171 | 2.309 / 2.288 | 0.748 | 0.781 |

Slopes along the trajectory (the T3D method of §6.6) equal the eigenvector slopes, and the trajectory gives the true
rate.  The first transient removes 5–25% of the perturbation, as the adjoint predicts; the 4% miss at μ = 0.05 is the
nonlinearity of a finite perturbation that close to the fold.

### 2.3 ε: a density ramp through the radiative fold (`epsstat`, `epsdyn`)

Set-up:
* Base state: $C_{\rm brem}=1.0$ (μ = 0.49).
* Density: $n(x,0)=1$, uniform source $s_N$ on $x<0.6$, $n(L)=1$, $n_t=D_n(n_{xx}+s_N)$.  The path $n(x;s)$ with
  $s=D_nt$ is fixed, and $D_n$ sets only its speed.
* $\epsilon=D_n\lambda_1/|\gamma_{\rm th}(0)|$, with $\lambda_1=(\pi/2L)^2$ the slowest density mode.

Rising density raises $P_{\rm brem}\propto n^{3/2}$ faster than it raises $P_\alpha$ (core above 14 keV), and the burn
meets a fold at $s_f=0.37018$ ($N=1.157$, $\Wtot_f=4.029$).  The static branch gives $AB=3.008$ (from
$\gamma^2=4AB(s_f-s)$, μ < 0.12), with $A=2.163$ and $B=1.391$ separately.

| ε | $\Delta s$ measured | Airy $\Omega_0D_n^{2/3}(AB)^{-1/3}$ | ratio |
|---|---|---|---|
| 0.226 | 0.4637 | 0.5306 | 0.874 |
| 0.0565 | 0.1841 | 0.2106 | 0.874 |
| 0.0141 | 0.0751 | 0.0836 | 0.899 |
| 0.0035 | 0.0306 | 0.0332 | 0.924 |
| 0.00088 | 0.01241 | 0.01316 | 0.943 |
| 0.00044 | 0.00788 | 0.00829 | 0.950 |

**Slopes (ε < 0.06):** delay $\sim\epsilon^{0.649}$ (2/3 with a log correction).  $|\Delta s-{\rm Airy}|\sim\epsilon^{0.86}$,
and $|\Delta s-{\rm Airy}|/(\epsilon\ln\epsilon^{-1})=0.16\to0.12$, levelling off: the $O(\epsilon\ln\epsilon)$
correction, negative here.  In inner variables all runs collapse onto $-\mathrm{Ai}'(-\tau)/\mathrm{Ai}(-\tau)$
(`mms_figs/fold_inner.png`).

**Slow manifold (the fixed point $\Wtot^\star(N)$):** the leading-order error $\Wtot-\Wtot^\star(s)$ at $s=0.15$ and
$0.25$ scales as $\epsilon^{0.96}$ and $\epsilon^{0.95}$.  After subtracting the $O(\epsilon)$ lag
$D_n\,\dd\Wtot^\star/\dd s/\gamma_{\rm th}$, the remainder falls as $\epsilon^{1.4}$–$\epsilon^{1.5}$.  It is not yet
$\epsilon^2$: a small $O(\epsilon)$ piece remains because the density tangent $\partial_sp^\star$ is not exactly the
thermal mode; it is a few per cent of the lag.  At $\epsilon=0.11$ the lag prediction is already 50% too large.

### 2.4 Summary table: leading-order prediction, measured error slope (theory)

| prediction | small parameter | slope | theory |
|---|---|---|---|
| front rest position (Gibbs) $x_G=x_M$ | δ | 2.03 | 2 |
| front rest position (foot / mid marker) | δ | 0.91 | 1 (offset $\sqrt{\nu_4}\xi$ predicted; after it, 1.97) |
| stored energy of the barrier state | δ | 1.98 | 2 |
| fold location $C_{\rm fold}$ | δ | 1.91 | 2 |
| thermal rate, Rayleigh on steady family | μ | 1.03 ($C_{\rm brem}$), 1.00 ($C_\alpha$) | 1 |
| thermal rate, $W$ balance on steady family | μ | → const (Π−1 = 0.53 / 0.20) | O(1): not asymptotic |
| thermal rate, $W$ balance on the trajectory | – | exact (≤0.7% incl. nonlinearity) | exact |
| scale separation $\lvert\gamma_{\rm th}/\lambda_2\rvert$ | μ | 1.13 / 1.07 | 1 |
| fixed point $\Wtot^\star(N)$ (slow manifold) | ε | 0.96 | 1 |
| delayed jump $\Delta s$ | ε | 0.649 (prefactor → 0.95 of Airy) | 2/3, Ω0 = 2.338 |
| error of the Airy delay | ε | 0.86 | ε ln ε |

---------------------------------------------------------------------------------------------------------------------

## 3. Where it fails, and the T3D numbers

### 3.1 Limits of the reduced-model theory

* **Moderate parameters.**
  * ε: at ε = 0.11 the $O(\epsilon)$ lag over-predicts by 50%, and the Airy delay is 13% long for 0.05 ≲ ε ≲ 0.5.
    The $\epsilon\ln\epsilon$ correction is still 5% at ε = 4e-4.
  * μ: at μ ≈ 0.5 the thermal mode is only 10× slower than the profile mode ($|\gamma/\lambda_2|=0.09$–0.10).
  * δ: at δ ≳ 0.1 the $x_{\rm foot}$ offset is no longer $\sqrt{\nu_4}\xi_{\rm foot}$ alone (the $\delta^2$ term is ≈ 35% of it at δ = 0.12).
* **The first transient.**  A non-slaved start changes $\Wtot$ through the fast modes before the amplitude equation
  applies: 5–25% of a uniform perturbation in the reduced model.  T3D runs are in this phase for their first
  ~0.3–1 s, so slopes and margins must be fitted after it; the members at $Z_{\rm eff}=6.5$–8, stopped at 0.6–0.8 s,
  are not yet usable.
* **Closure on a steady family.**  Rates from quasi-static scans or from the predictor's fixed points (a steady
  family) are wrong by the family's Π unless the balance is weighted with the mode.  Π was 1.53 (sink) and 1.20
  (alpha) at the fold here, and is 1 only for a family generated along the mode itself; fixed points and the fold
  location are not affected.  The rescaling tested here, $p-p_{\rm ped}\to\lambda(p-p_{\rm ped})$, misses the fold
  entirely ($-2.4$ to $-2.8$ where $\gamma_{\rm th}\to0$).  The paper's eq. Wstab numbers come from a different rescaling
  ($\dd\Qedge/\dd\Wtot=5.1$, close to this mode's 5.04 at the ignited state), so whether *that* proxy sees the fold
  should be checked before the same is said of it.
* **Multiple slow channels.**  The one-mode reduction needs one slow mode.  T3D evolves $T_e$, $T_i$, $n_e$ (and $\psi$),
  and the ion core of the second-stable region is heated on NEO alone, a second slow mode.  For the GS2-QL run the
  amplitude equation's instantaneous fixed point is 1105 MJ, against the predictor's 1435 MJ at the run's late
  density: the residual is the still-heating ion core (predict_endstate.md §6.4).  For the hKBM run the two agree
  (1465 vs 1461 MJ).
* **Discrete fronts.**  The δ expansion needs $w\gg h$.  In the reduced model at w/h ≈ 4 the front mode is wrong by
  20–40%, kinks in $Q$ produce spurious eigenvalues, and at w/h ≈ 20 the front eigenvalue is still 5–10% off.  T3D's
  barrier is one cell wide ($\Delta\rho_N=0.1$; 0.045 on 21 tubes), so δ is set by the grid, $w/h<1$.  This is the
  pinned regime of NOTES_fold §1.1: positions are quantised, the fold is replaced by loss of pinning, and the thermal
  eigenvalue does not soften before the loss.  Only the outer results (branches, Maxwell flux, thresholds) carry over.

### 3.2 μ and ε for the T3D runs (`t3d`; slopes over t > 0.3 s; $N_e=\int n_e\,\dd V$)

| run | t_end [s] | $\Wtot$ [MJ] | $\dd\mathcal S/\dd\Wtot$, $\dd\Qedge/\dd\Wtot$ [1/s] | μ (trajectory) | $\tau_{\rm th}$ [s] | μ fixed-n fit (cond) | $\dd\ln N_e/\dd t$ [1/s] | $\epsilon_N=\tau_{\rm th}\dd\ln N_e/\dd t$ | $\epsilon_W$ (relative lag) | $\Wtot^\star_{\rm inst}$ [MJ] |
|---|---|---|---|---|---|---|---|---|---|---|
| hkbm_F_k1_v2 (ref., CDE on) | 5.34 | 1346 | 0.956, 1.232 | **0.22** | 3.6 | 0.30 (15) | 0.025 | **0.09** | **0.08** | 1465 (predictor, late n: 1461) |
| cde_v127_F_k1 (GS2-QL) | 5.03 | 1050 | 0.821, 1.721 | **0.52** | 1.1 | 0.65 (3) | 0.029 | 0.03 | 0.06 | 1105 (predictor 1435: second slow mode) |
| es frozen Z_eff 0 | 4.68 | 1345 | 1.014, 1.194 | 0.15 | 5.6 | 0.42 (28) | 0.029 | 0.16 | 0.29 | 1888 |
| es brem Z_eff 3.4 | 4.92 | 1221 | 0.823, 0.947 | 0.13 | 8.1 | 0.64 (53) | 0.029 | 0.23 | 0.35 | 1868 |
| es brem Z_eff 6 | 4.42 | 1020 | 0.386, 0.445 | 0.13 | 17 | 0.19 (29) | 0.030 | 0.51 | 0.47 | 1925 |
| es frozen nt21 | 1.69 | 1101 | 1.001, 1.477 | 0.32 | 2.1 | 0.72 (26) | 0.036 | 0.08 | 0.23 | 1399 |
| scenario H (cold, QRAD) | 1.43 | 400 | 0.50, 8.57 | 0.94 | 0.12 | – | −0.027 | ~0 | ~0 | 401 |

Excluded: es Z_eff 6.5/7/8 (t < 0.8 s, first transient: negative or erratic edge slopes); P_aux = 0 (trajectory μ
= −0.04, the density drift dominates); scenario I (W moves only 658 → 690 MJ, edge slope negative: still in its
transient).

Reading:
* **μ.**  The reference burn has μ ≈ 0.2–0.3 (trajectory 0.22; with the density effect regressed out 0.30, but
  ill-conditioned).  GS2-QL has 0.5–0.65.  The frozen ladder members show 0.13–0.15 along the trajectory, but that is
  biased low: the density ramp raises $\mathcal S$ while $\Wtot$ rises, inflating $\dd\mathcal S/\dd\Wtot$.  The
  fixed-density fits give 0.2–0.6 with condition numbers 28–53, so for these μ is only bracketed, ≈0.15–0.6.  The
  leading-order rate has corrections $O(\mu)$ ≈ 20–50%, and the thermal mode is 3–10× slower than the profile
  modes, not 100×.
* **ε.**  The relevant ratio is the drift of the fixed point over the thermal rate, $\epsilon_W=\tau_{\rm th}\,\dd\ln
  \Wtot^\star/\dd t$, which equals the relative lag $(\Wtot^\star-\Wtot)/\Wtot^\star$ at leading order.  It is 0.06–0.08
  for the two current-diffusion runs and 0.23–0.47 for the frozen ladder.  The density inventory itself rises at
  2.5–3.6%/s ($\epsilon_N=0.03$–0.5, with $\tau_{\rm th}$ from the biased trajectory μ).  So the slow-manifold
  statement "the burn sits at $\Wtot^\star(N)$" is good to ~8% for the reference run and to only 25–45% for the
  frozen members.  The latter is the 24–28% gap between the members and the ramp fixed point found in
  predict_endstate.md §6.6, now read as the $O(\epsilon)$ lag: $\Wtot^\star-\Wtot=\tau_{\rm th}\dot\Wtot^\star$.
* **The predicted re-ignition of the Z_eff = 10 member at ≈ 9.5 s** (predict_endstate.md §6.6) is a static-fold
  time.  The fold-passage theory delays it by $3.71\,(T_0\tau_{\rm th,0}^2)^{1/3}$.  With $T_0\approx5$ s to the fold
  and a cold-root relaxation time of 0.1–1 s (scenario H's cold root: 0.12 s), that is ≈1–6 s, comparable to the
  remaining time, at ε that is not small.  Read 9.5 s as a lower bound.
* **δ.**  Set by the grid, $w/h<1$; see §3.1.

What this says about how good leading order should be in T3D: the outer statements (branches, Maxwell rest, the
fixed point at frozen density, the fold's existence) are the robust part.  Rates are good to a factor 1.2–1.5, and
only if measured along the trajectory after its first second.  Times (relaxation, jump) are uncertain by
$O(\epsilon^{1/3})$–$O(1)$ factors at T3D's ε ≈ 0.1–0.5.

---------------------------------------------------------------------------------------------------------------------

## 4. What the paper can quote (proposed changes, not made: no .tex edits)

1. A short section "The asymptotic structure" with §1.1–§1.4 above.  The new results are:
   * the Gibbs-surface statement ($x_G-x_M=O(\delta^2)$, markers offset by $\sqrt{\nu_4}\,\xi$, $\xi_{\rm foot}=-0.54$);
   * the self-adjointness of the profile operator and the projected amplitude equation with the factor Π;
   * the slow-manifold lag;
   * the $\epsilon^{2/3}$ Airy delay at the fold.
2. Section 3.4 (eq. Wstab): say that the rate from slopes is exact along the trajectory (as §6.6 does) and that a
   uniform rescaling is a proxy that fails near a fold.
3. Section 6.6 / 6.8: the 24–28% gap of the burning members to the ramp fixed point is the $O(\epsilon)$ lag; quote
   $\epsilon_W$ and μ from §3.2; and the static re-ignition time is a lower bound.
4. Figure: `plots/fig_mms.pdf` (script `scripts/make_mms_figure.py`).

Caveats: the kinks of Q are rounded (σ_g = 0.05); μ and ε runs use $\nu_4=2\times10^{-3}$; the density model is
one-way coupled (n does not see p).  T3D's NEO particle flux depends on $T$, but its density ramp is set by the
source and the barrier, which is what one-way coupling keeps.
