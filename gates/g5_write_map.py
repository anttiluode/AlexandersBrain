"""G5 - write map of one ring in 2-D: written winding vs global flux f, gate open time, start n0.

Schedule (all replicas share the field, which is global):
  t = 0         field on (flux f per ring); gated replicas open their gap (mu = -1, arc length 3)
  t = t_w       gated replicas with this open time close their gap (field still on)
  t = 1800+200  field off
  +300          read the winding (full-field oracle)
Closed-arm replicas never open: they show crosstalk from the field alone.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from ab.tdgl import annulus, winding_state
from ab.tdgl2 import Medium2, winding_at, gate_mask

N = 64
R = 10.0
T = 0.05
fs = [float(a) for a in sys.argv[1:]] or [-2.0, -1.5, -1.0, -0.6, -0.25, 0.0, 0.25, 0.6, 1.0, 1.5, 2.0]
TWS = [200.0, 600.0, 1800.0]
N0 = [-1, 0, 1]
mu = annulus(N)
GM = gate_mask(N, (0.0, 0.0), R, 4.0, 3.0)
out = {}
for f in fs:
    t0 = time.time()
    Bf = 2 * f / R ** 2  # flux B * pi R^2 = 2 pi f through the centre line
    cfg = [(arm, tw, n0) for arm in ["gated", "closed"] for tw in TWS for n0 in N0]
    m = Medium2(mu, np.stack([winding_state(mu, n0) for _, _, n0 in cfg]), seed=int(1000 * (f + 3)))
    m.run(100.0, T=0.0)  # settle the start state
    for k, (arm, tw, n0) in enumerate(cfg):
        if arm == "gated":
            m.mu[k][GM] = -1.0
    t = 0.0
    for tw in TWS:
        m.run(tw - t, Bfield=Bf, T=T)
        t = tw
        for k, (arm, tw2, n0) in enumerate(cfg):
            if arm == "gated" and tw2 == tw:
                m.mu[k][GM] = 1.0
    m.run(200.0, Bfield=Bf)
    m.run(300.0, Bfield=0.0)
    n = winding_at(m.psi, mu, R)
    res = {f"{arm}_tw{int(tw)}_n0{n0}": int(n[k]) for k, (arm, tw, n0) in enumerate(cfg)}
    out[str(f)] = res
    print(f, res, round(time.time() - t0, 1), flush=True)
tag = "_".join(str(f) for f in fs)
json.dump({"T": T, "R": R, "gap": 3.0, "open_times": TWS, "per_f": out},
          open(f"results/g5_write_map_{tag}.json", "w"), indent=1)
