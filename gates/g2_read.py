"""G2 - reading the memories.

a. pinch read of n: lower mu to mu_p on a patch of the ring (full width, arc length 2) for 30 t.
   Signal = phase drop across the patch, between two points of the field itself:
   arg< psi(b) psi(a)* > averaged over the last 10 t of the pinch. No external reference.
   100 reads per ring; n checked after every read.
b. absolute read of Theta: inject a reference wave eps e^{i phi_r} on a patch for 30 t,
   phi_r alternating 0, pi/2 (a full two-quadrature read per pair). Measure the
   permanent shift of Theta after each read (T = 0, so any shift is read damage).
c. local damage: make a disk of radius r on the ring strongly normal for 20 t, release,
   check n.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from ab.tdgl import (Medium, annulus, winding_state, winding, winding_radii, ring_phase,
                     _bilinear, grid, wrap, DX)

N = 64
R = 10.0
mu0 = annulus(N)
X, Y, r, ph = grid(N)
out = {}


def patch(angle, length=2.0, W=4.0):
    d = np.abs(wrap(ph - angle))
    return (d < (length / 2) / R) & (np.abs(r - R) < W / 2 + 0.01)


# ------------------------------------------------------------- a. pinch reads
PATCH = patch(np.pi)
a_ang = np.pi - (1.0 + 0.75) / R
b_ang = np.pi + (1.0 + 0.75) / R
pa = (R * np.cos(a_ang), R * np.sin(a_ang))
pb = (R * np.cos(b_ang), R * np.sin(b_ang))
mups = [0.6, 0.3, 0.15, 0.05, 0.0, -0.3]
ns = np.arange(-4, 5)
NREAD = 100
T_A = 0.05
t0 = time.time()
combos = [(mp, n) for mp in mups for n in ns]
psi = np.stack([winding_state(mu0, n) for _, n in combos])
m = Medium(mu0, psi, seed=11)
m.run(200.0)
base_mu = m.mu.copy()
pinch_mu = base_mu.copy()
for k, (mp, n) in enumerate(combos):
    pinch_mu[k][PATCH] = mp
sig = np.zeros((len(combos), NREAD))
nafter = np.zeros((len(combos), NREAD), int)
for rd in range(NREAD):
    m.mu[:] = pinch_mu
    m.run(20.0, T=T_A)
    acc = np.zeros(len(combos), complex)
    for _ in range(5):
        m.run(2.0, T=T_A)
        va = _bilinear(m.psi, np.array([pa[0]]), np.array([pa[1]]))[:, 0]
        vb = _bilinear(m.psi, np.array([pb[0]]), np.array([pb[1]]))[:, 0]
        acc += vb * np.conj(va)
    sig[:, rd] = np.angle(acc)
    m.mu[:] = base_mu
    m.run(30.0, T=T_A)
    nafter[:, rd] = winding(m.psi, mu0)
res = {}
for mp in mups:
    idx = [k for k, c in enumerate(combos) if c[0] == mp]
    nn = np.array([combos[k][1] for k in idx])
    s = sig[idx]
    changed = (nafter[idx] != nn[:, None])
    first = [int(np.argmax(c)) + 1 if c.any() else None for c in changed]
    # read signal of reads before any change
    means = {int(n): float(s[i][~np.cumsum(changed[i]).astype(bool)].mean()) if (~changed[i]).any() else None
             for i, n in enumerate(nn)}
    sds = {int(n): float(s[i][~np.cumsum(changed[i]).astype(bool)].std()) if (~changed[i]).any() else None
           for i, n in enumerate(nn)}
    ok_n = [n for n in nn if means[int(n)] is not None]
    mvals = np.array([means[int(n)] for n in ok_n])
    gaps = np.diff(mvals) if len(mvals) > 1 else np.array([np.nan])
    sd_max = max(sds[int(n)] for n in ok_n) if ok_n else None
    res[str(mp)] = {
        "rings_with_n_changed_after_100_reads": {int(n): bool(c.any()) for n, c in zip(nn, changed)},
        "first_destructive_read": {int(n): f for n, f in zip(nn, first)},
        "signal_mean_by_n": means, "signal_sd_by_n": sds,
        "min_adjacent_gap_over_max_sd": float(np.min(np.abs(gaps)) / sd_max) if sd_max else None,
        "signal_monotone_in_n": bool(np.all(gaps > 0) or np.all(gaps < 0)),
    }
    print("pinch", mp, res[str(mp)]["rings_with_n_changed_after_100_reads"],
          "sep", res[str(mp)]["min_adjacent_gap_over_max_sd"], res[str(mp)]["signal_monotone_in_n"], flush=True)
out["a_pinch"] = {"T": T_A, "reads": NREAD, "per_mu_p": res, "wall_s": round(time.time() - t0, 1)}

# ------------------------------------------------------------- b. absolute reads of Theta
t0 = time.time()
SRC = patch(0.0)
epss = [0.01, 0.03, 0.1, 0.3]
nb = [0, 1, 3]
combos = [(e, n) for e in epss for n in nb]
th_init = np.linspace(0.3, 5.9, len(combos))
psi = np.stack([winding_state(mu0, n, t) for (e, n), t in zip(combos, th_init)])
mb = Medium(mu0, psi, seed=12)
mb.run(200.0)
nn = np.array([c[1] for c in combos])
th_start = ring_phase(mb.psi, nn)
drift = np.zeros((len(combos), NREAD))
nbad = np.zeros(len(combos), int)
quad = np.zeros((len(combos), NREAD))
th_before = np.zeros((len(combos), NREAD))
for rd in range(NREAD):
    phr = 0.0 if rd % 2 == 0 else np.pi / 2
    th_before[:, rd] = ring_phase(mb.psi, nn)
    # one source per replica would need per-replica src; run each eps group separately
    for e in epss:
        idx = [k for k, c in enumerate(combos) if c[0] == e]
        sub = Medium(mu0, mb.psi[idx], seed=1000 + rd)
        sub.run(30.0, src=np.where(SRC, e * np.exp(1j * phr), 0))
        # interference quadrature with the reference: Re(psi e^{-i phi_r}) on the patch
        quad[idx, rd] = np.real((sub.psi * SRC).sum((-1, -2)) * np.exp(-1j * phr)) / SRC.sum()
        sub.run(30.0)
        mb.pr[idx] = sub.pr
        mb.pi[idx] = sub.pi
    nw = winding(mb.psi, mu0)
    nbad += (nw != nn)
    drift[:, rd] = wrap(ring_phase(mb.psi, nn) - th_start)
resb = {}
for e in epss:
    idx = [k for k, c in enumerate(combos) if c[0] == e]
    resb[str(e)] = {
        "theta_shift_after_1_read": [float(drift[k, 0]) for k in idx],
        "theta_shift_after_2_reads": [float(drift[k, 1]) for k in idx],
        "theta_shift_after_100_reads": [float(drift[k, -1]) for k in idx],
        "n_changed_any": [bool(nbad[k]) for k in idx],
        "decode_error_first_pair": [float(wrap(np.arctan2(quad[k, 1], quad[k, 0]) - th_before[k, 0])) for k in idx],
        "decode_error_last_pair": [float(wrap(np.arctan2(quad[k, -1], quad[k, -2]) - th_before[k, -2])) for k in idx],
        "n": [int(nn[k]) for k in idx],
    }
    print("abs", e, resb[str(e)], flush=True)
out["b_absolute_theta_read"] = {"T": 0.0, "reads": NREAD, "per_eps": resb, "wall_s": round(time.time() - t0, 1)}

# ------------------------------------------------------------- c. local damage
t0 = time.time()
rads = [0.5, 1.0, 1.5, 1.8, 2.2, 3.0]
nc = [1, 2, 3, 4]
REPc = 4
combos = [(rr, n) for rr in rads for n in nc for _ in range(REPc)]
psi = np.stack([winding_state(mu0, n) for _, n in combos])
mc = Medium(mu0, psi, seed=13)
mc.run(200.0)
cx, cy = 0.0, R  # point on the ring centre line at phi = pi/2
for k, (rr, n) in enumerate(combos):
    mc.mu[k][np.hypot(X - cx, Y - cy) < rr] = -5.0
mc.run(20.0, T=0.05)
mc.mu[:] = mu0
mc.run(100.0, T=0.05)
nw = winding(mc.psi, mu0)
resc = {}
for rr in rads:
    idx = [k for k, c in enumerate(combos) if c[0] == rr]
    resc[str(rr)] = {"fraction_n_changed": float(np.mean([nw[k] != combos[k][1] for k in idx])),
                     "final_n_by_initial": {int(n): [int(nw[k]) for k in idx if combos[k][1] == n] for n in nc}}
    print("hole r", rr, resc[str(rr)], flush=True)
out["c_local_damage"] = {"ring_width": 4.0, "hole_radius": resc, "wall_s": round(time.time() - t0, 1)}

json.dump(out, open("results/g2_read.json", "w"), indent=1)
