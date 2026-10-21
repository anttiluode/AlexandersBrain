"""G1 - integer vs continuous memory under noise, in the same ring.

Each ring replica stores a pair: an integer n in {0..4} (by symmetry -n behaves like n) and
a phase level Theta0 = 2 pi j / 9. Disk replicas store only a phase level (no hole possible).
Recall: n read as the winding number; Theta read and snapped to the nearest of 9 levels.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from ab.tdgl import (Medium, annulus, disk, winding_state, winding_radii, ring_phase,
                     _bilinear, grid, wrap, DX)

N = 64
K = 9
Ts = [float(a) for a in sys.argv[1:]] or [0.02, 0.05, 0.1, 0.2, 0.4]
REP = 8
TMAX = 3000.0
CHUNK = 100.0
rng = np.random.default_rng(1)

mu_ring = annulus(N)
mu_disk = disk(N, R=12.0)
_, _, r, ph = grid(N)


def disk_phase(psi):
    w = (r < 11.0)
    return np.angle((psi * w).sum((-1, -2)))


out = {"Ts": Ts, "K": K, "rep": REP, "tmax": TMAX, "per_T": {}}
for T in Ts:
    t0 = time.time()
    ns = np.repeat(np.arange(5), REP)
    lv = rng.integers(0, K, size=ns.size)
    th0 = 2 * np.pi * lv / K
    psi = np.stack([winding_state(mu_ring, n, t) for n, t in zip(ns, th0)])
    lvd = rng.integers(0, K, size=2 * REP)
    thd = 2 * np.pi * lvd / K
    psid = np.stack([winding_state(mu_disk, 0, t) for t in thd])
    # relax at T = 0 first so every replica starts in its exact stationary state
    ring = Medium(mu_ring, psi, seed=int(T * 1000))
    dsk = Medium(mu_disk, psid, seed=int(T * 1000) + 7)
    ring.run(200.0)
    dsk.run(200.0)
    th_start = ring_phase(ring.psi, ns)
    thd_start = disk_phase(dsk.psi)
    I_ring = (np.abs(ring.psi) ** 2).sum((-1, -2)) * DX * DX
    I_disk = (np.abs(dsk.psi) ** 2).sum((-1, -2)) * DX * DX
    times, n_ok, th_var, th_ok, thd_var, thd_ok, n_amb = [], [], [], [], [], [], []
    first_err = {}
    t = 0.0
    while t < TMAX - 1e-9:
        ring.run(CHUNK, T=T)
        dsk.run(CHUNK, T=T)
        t += CHUNK
        wr = winding_radii(ring.psi, mu_ring)
        n_now = wr[:, 1]
        amb = ~np.all(wr == wr[:, :1], axis=1)
        ok = n_now == ns
        for k in np.where(~ok)[0]:
            first_err.setdefault(int(k), (int(ns[k]), t))
        dth = wrap(ring_phase(ring.psi, n_now) - th_start)
        dthd = wrap(disk_phase(dsk.psi) - thd_start)
        snap = np.rint((wrap(th_start + dth - th0) ) / (2 * np.pi / K)) % K == 0
        snapd = np.rint((wrap(thd_start + dthd - thd) ) / (2 * np.pi / K)) % K == 0
        times.append(t)
        n_ok.append(float(ok.mean()))
        n_amb.append(float(amb.mean()))
        th_var.append(float(np.mean(dth[ok] ** 2)) if ok.any() else float("nan"))
        th_ok.append(float(snap.mean()))
        thd_var.append(float(np.mean(dthd ** 2)))
        thd_ok.append(float(snapd.mean()))
    times = np.array(times)
    # unwrapped-variance is only meaningful while sd << pi; fit on early window
    early = np.array(th_var) < 0.5
    pred_ring = T / I_ring[ns == 0].mean()
    pred_disk = T / I_disk.mean()
    fit_ring = float(np.polyfit(times[early], np.array(th_var)[early], 1)[0]) if early.sum() > 2 else float("nan")
    earlyd = np.array(thd_var) < 0.5
    fit_disk = float(np.polyfit(times[earlyd], np.array(thd_var)[earlyd], 1)[0]) if earlyd.sum() > 2 else float("nan")
    def t_half(series, chance):
        s = np.array(series)
        idx = np.where(s < (1 + chance) / 2)[0]
        return float(times[idx[0]]) if idx.size else None
    out["per_T"][str(T)] = {
        "n_recall_final": n_ok[-1],
        "n_recall_by_n_final": {int(n): float((winding_radii(ring.psi, mu_ring)[:, 1] == ns)[ns == n].mean()) for n in range(5)},
        "n_first_errors_(n0,t)": list(first_err.values()),
        "theta_recall_final": th_ok[-1],
        "disk_theta_recall_final": thd_ok[-1],
        "t_half_theta_ring": t_half(th_ok, 1 / K),
        "t_half_theta_disk": t_half(thd_ok, 1 / K),
        "t_half_n": t_half(n_ok, 1 / K),
        "theta_var_slope_ring_measured": fit_ring,
        "theta_var_slope_ring_predicted_T_over_int_psi2_n0": float(pred_ring),
        "theta_var_slope_disk_measured": fit_disk,
        "theta_var_slope_disk_predicted": float(pred_disk),
        "ambiguous_fraction_max": float(max(n_amb)),
        "series": {"t": times.tolist(), "n_recall": n_ok, "theta_recall": th_ok,
                   "disk_theta_recall": thd_ok, "theta_var": th_var, "disk_theta_var": thd_var},
        "wall_s": round(time.time() - t0, 1),
    }
    s = out["per_T"][str(T)]
    print(T, {k: v for k, v in s.items() if k not in ("series",)}, flush=True)

tag = "_".join(str(T) for T in Ts)
json.dump(out, open(f"results/g1_noise_T{tag}.json", "w"), indent=1)
if len(Ts) == 1:
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, len(Ts), figsize=(4 * len(Ts), 3.3), sharey=True)
for a, T in zip(ax, Ts):
    s = out["per_T"][str(T)]["series"]
    a.plot(s["t"], s["n_recall"], label="integer n (ring)", lw=2)
    a.plot(s["t"], s["theta_recall"], label="phase Theta (ring), 9 levels")
    a.plot(s["t"], s["disk_theta_recall"], "--", label="phase (disk), 9 levels")
    a.axhline(1 / K, color="grey", lw=0.6)
    a.set_title(f"T = {T}")
    a.set_xlabel("time")
ax[0].set_ylabel("recall")
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig("results/g1_noise.png", dpi=110)
