"""Which windings are truly stationary at T = 0?  (replaces the 400-t calibration check,
which was shorter than an open arc's own phase-relaxation time, ~ L^2 / pi^2 ~ 400 t)

All geometries in one batch, 4000 t of deterministic relaxation. At 400, 1000, 2000, 4000 t
records the winding and the arc current k = mean phase gradient along the centre line
in ordered material (a stationary trapped winding keeps k constant; a draining one decays).
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from ab.tdgl import Medium, constricted_annulus, c_ring, winding_state, winding, _bilinear, loop

N = 64
cfg = [("w", w, n) for w in [0.6, 1.0, 1.5, 2.0, 2.5, 3.0] for n in [1, 2, 3]] + \
      [("g", g, n) for g in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0] for n in [1, 2, 3]]
mus = np.stack([constricted_annulus(N, v) if k == "w" else c_ring(N, v) for k, v, n in cfg])
psi = np.stack([winding_state(mu, n) for mu, (_, _, n) in zip(mus, cfg)])
m = Medium(mus, psi)


def arc_k(psi, mu, R=10.0, M=256):
    x, y, _ = loop(R, M)
    keep = _bilinear(mu, x, y) > 0
    v = _bilinear(psi, x, y)
    d = np.angle(np.roll(v, -1) * np.conj(v))
    good = keep & np.roll(keep, -1)  # both ends in ordered material: skip the gap jump
    return float(d[good].sum() / (good.sum() * 2 * np.pi * R / M))


rec = {f"{k}{v}_n{n}": {} for k, v, n in cfg}
t = 0.0
for target in [400.0, 1000.0, 2000.0, 4000.0]:
    m.run(target - t)
    t = target
    for b, (k, v, n) in enumerate(cfg):
        rec[f"{k}{v}_n{n}"][str(int(t))] = {"winding": int(winding(m.psi[b:b + 1], mus[b])[0]),
                                            "arc_k": round(arc_k(m.psi[b], mus[b]), 5)}
    print("t", t, flush=True)
for key, r in rec.items():
    print(key, r)
json.dump(rec, open("results/g34_t0_long.json", "w"), indent=1)
