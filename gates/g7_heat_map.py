"""G7a - heat as the gate, one ring.

The ring has a constriction (w = 2, arc length 2) at angle 0; no material is ever switched.
A patch of arc length 3 over the constriction is heated to T_hot for t_h while the global
field is on (flux f), then cools to the background T = 0.05 with the field still on (200 t),
field off, wait 300 t, read. Cold arm: field on, no heat.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from ab.tdgl import constricted_annulus, winding_state
from ab.tdgl2 import Medium2, winding_at, gate_mask

N = 64
R = 10.0
T_BG = 0.05
mu = constricted_annulus(N, 2.0)
HEAT = gate_mask(N, (0.0, 0.0), R, 4.0, 3.0)
THOT = [0.1, 0.3, 1.0, 2.0]
THS = [200.0, 1000.0]
N0 = [-1, 0, 1]
out = {}
for f in [-1.0, 0.0, 1.0]:
    t0 = time.time()
    Bf = 2 * f / R ** 2
    cfg = [("heat", th, tlen, n0) for th in THOT for tlen in THS for n0 in N0] + \
          [("cold", T_BG, max(THS), n0) for n0 in N0]
    m = Medium2(mu, np.stack([winding_state(mu, n0) for *_, n0 in cfg]), seed=int(100 * (f + 5)))
    m.run(100.0, T=0.0)
    start = winding_at(m.psi, mu, R)
    Tmap = np.full((len(cfg), N, N), T_BG)
    for k, (arm, th, tlen, n0) in enumerate(cfg):
        if arm == "heat":
            Tmap[k][HEAT] = th
    t = 0.0
    for tlen in THS:
        m.run(tlen - t, Bfield=Bf, T=Tmap)
        t = tlen
        for k, (arm, th, tl2, n0) in enumerate(cfg):
            if tl2 == tlen:
                Tmap[k][HEAT] = T_BG
    m.run(200.0, Bfield=Bf, T=T_BG)
    m.run(300.0, Bfield=0.0)
    n = winding_at(m.psi, mu, R)
    res = {f"{arm}_T{th}_t{int(tl)}_n0{n0}": int(n[k]) for k, (arm, th, tl, n0) in enumerate(cfg)}
    res["start_ok"] = bool(np.all(start == np.array([c[3] for c in cfg])))
    out[str(f)] = res
    print(f, res, round(time.time() - t0, 1), flush=True)
json.dump({"T_bg": T_BG, "constriction_w": 2.0, "per_f": out}, open("results/g7_heat_map.json", "w"), indent=1)
