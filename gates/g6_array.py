"""G6 - an addressed array: 4 rings, ONE global write field, a local gate per ring.

usage: python gates/g6_array.py t_open ops reps [heat T_hot]

Each op: global flux f = value (shared by all replicas: the field is global), each replica
opens the gate of its own randomly chosen target ring for t_open, closes it with the
field still on (200 t), field off, wait 100 t. After every op all 4 windings are read
(full-field oracle). Alphabet {-1, 0, 1}. Background noise T = 0.05.

Gate kind:
  material (default): a gap of normal material (mu = -1, arc length 3) in the target ring.
  heat (G7b):        no material change; the target ring has a constriction w = 2 at its
                     gate and a patch of arc length 3 over it is heated to T_hot.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from ab.tdgl import grid
from ab.tdgl2 import Medium2, winding_at, ring_mu, gate_mask, winding_state_multi

t_open = float(sys.argv[1])
OPS = int(sys.argv[2])
REPS = int(sys.argv[3])
kind = sys.argv[4] if len(sys.argv) > 4 else "material"
T_hot = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
T_BG = 0.05
N = 96
R, W = 8.0, 4.0
C = [(-12.0, -12.0), (-12.0, 12.0), (12.0, -12.0), (12.0, 12.0)]
GATE_ANG = [np.arctan2(-c[1], -c[0]) for c in C]  # each gate faces the box centre
X, Y, _, _ = grid(N)
base = ring_mu(N, C, R, W)
gates = [gate_mask(N, c, R, W, 3.0, a) for c, a in zip(C, GATE_ANG)]
if kind == "heat":
    for c, a in zip(C, GATE_ANG):  # constriction w = 2 at each gate, as in G3
        r = np.hypot(X - c[0], Y - c[1])
        ph = np.arctan2(Y - c[1], X - c[0])
        win = np.abs((ph - a + np.pi) % (2 * np.pi) - np.pi) < (1.0 / R)
        base[win & (np.abs(r - R) < W / 2 + 0.01) & (np.abs(r - R) >= 1.0)] = -1.0

rng = np.random.default_rng(int(t_open) + 17 * OPS + (7 if kind == "heat" else 0))
state0 = rng.integers(-1, 2, size=(REPS, 4))
psi = np.stack([winding_state_multi(base, C, s) for s in state0])
m = Medium2(base, psi, seed=int(t_open) + (99 if kind == "heat" else 0))
m.run(200.0, T=T_BG)


def read_all():
    return np.stack([winding_at(m.psi, base, R, c) for c in C], 1)


cur = read_all()
log = {"start": cur.tolist(), "start_requested": state0.tolist(), "ops": []}
tgt_ok = tgt_n = xt = xt_n = 0
real_ok = real_n = 0
t0 = time.time()
for op in range(OPS):
    v = int(rng.integers(-1, 2))
    targets = rng.integers(0, 4, size=REPS)
    Bf = 2 * v / R ** 2
    Tmap = np.full((REPS, N, N), T_BG)
    for b, k in enumerate(targets):
        if kind == "material":
            m.mu[b][gates[k]] = -1.0
        else:
            Tmap[b][gates[k]] = T_hot
    m.run(t_open, Bfield=Bf, T=Tmap)
    for b, k in enumerate(targets):
        if kind == "material":
            m.mu[b][gates[k]] = 1.0
    m.run(200.0, Bfield=Bf, T=T_BG)
    m.run(100.0, Bfield=0.0)
    new = read_all()
    for b, k in enumerate(targets):
        ok = new[b, k] == v
        tgt_ok += ok
        tgt_n += 1
        if cur[b, k] != v:
            real_ok += ok
            real_n += 1
        for j in range(4):
            if j != k:
                xt_n += 1
                xt += new[b, j] != cur[b, j]
    log["ops"].append({"value": v, "targets": targets.tolist(), "before": cur.tolist(), "after": new.tolist()})
    cur = new
    print(f"op {op} v={v} target_ok={tgt_ok}/{tgt_n} real_change_ok={real_ok}/{real_n} "
          f"crosstalk={xt}/{xt_n} t={time.time() - t0:.0f}s", flush=True)
res = {"kind": kind, "T_hot": T_hot, "t_open": t_open, "ops": OPS, "reps": REPS, "T_bg": T_BG,
       "target_correct": tgt_ok / tgt_n, "real_change_correct": real_ok / max(real_n, 1),
       "real_changes": real_n, "crosstalk_rate": xt / xt_n, "crosstalk_events": xt,
       "ring_ops_checked": xt_n, "log": log}
name = f"results/g6_array_{kind}_tw{int(t_open)}" + (f"_Th{T_hot}" if kind == "heat" else "") + ".json"
json.dump(res, open(name, "w"), indent=1)
print({k: v for k, v in res.items() if k != "log"})
