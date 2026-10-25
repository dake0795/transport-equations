# Front lag (exp_front_lag.py; output front_lag.out), 25 Oct 2026

**Why it never finished before.** Two attempts (23 Oct on cpu-r-29, 9 h then OOM; 25 Oct job 39093208, cancelled after 1 h) hung in
law (b): driven by a flux above Q_M, dx_f/dt = -(Q - Q_M)/(x_f [g]) carries the predicted front to x_f = 0 in finite time, and RK45 at
rtol 1e-8 chatters about the singularity indefinitely. Fix: terminal events at x_f = 0.02 and 0.98. The script now runs in seconds.

**Second fix.** The "measured front flux" used Q(g) alone. Inside the front Q(g) is near the bump top (about 10), and the conservative
hyperviscous flux nu4 p_xxx carries the rest. The total face flux is now used, read 3 cells inside the front.

**Result** (drive 9.45 -> 7.56 and back; Q_M = 5.85):

| transition | measured half-time | law (a): Q_ss, Maxwell jump | law (b): measured flux, Maxwell jump | law (c): measured flux and jump |
|---|---|---|---|---|
| step down (front out, 0.153 -> 0.209) | 0.105 | 0.040 | 0.34 | 0.33 |
| step up (front in, 0.209 -> 0.153) | 0.060 | 0.030 | 0.42 | 0.41 |

- **Law (a) reproduces the paper's numbers** (sec:front: 0.11 vs 0.04, 0.06 vs 0.03): the law is 2-2.6x too fast.
- **Laws (b) and (c) cannot decompose the remainder.** The flux 3 cells inside the front is essentially the local core flux, 0.89-0.93 Q_M throughout,
  so these laws are 3-7x too slow and, after the step up, predict the wrong direction.
- **What decomposing the lag would need:** the jump condition V[g] = [p_t], with the measured core and pedestal heating rates on each side of the front,
  not a flux read at a fixed offset. That has not been done.
- **Paper:** unchanged. Its text already says what this supports: the law gives the direction, the end point and the order of magnitude of the time,
  the lag is partly the pedestal's relaxation, and the remainder is "presumably the readjustment inside the front".
