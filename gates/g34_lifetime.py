"""G3 / G4 - how long does a stored winding live?

usage: python gates/g34_lifetime.py constriction n0 T reps tmax w1 w2 ...
       python gates/g34_lifetime.py gap          n0 T reps tmax g1 g2 ...

constriction: the ring narrowed to width w over an arc of length 2 (ordered material).
gap:          the ring cut through its full width by normal material of arc length g
              (a loop that does not quite close: the 2-D stand-in for one clasp).
Lifetime = exponential MLE with right censoring at tmax.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from ab.tdgl import constricted_annulus, c_ring
from ab.escape import escape_times, tau_mle

kind = sys.argv[1]
n0 = int(sys.argv[2])
T = float(sys.argv[3])
reps = int(sys.argv[4])
tmax = float(sys.argv[5])
vals = [float(v) for v in sys.argv[6:]]
N = 64
geo = constricted_annulus if kind == "constriction" else c_ring
out = {"kind": kind, "n0": n0, "T": T, "reps": reps, "tmax": tmax, "per_value": {}}
for v in vals:
    t0 = time.time()
    times, esc, start = escape_times(geo(N, v), n0, T, reps, tmax, seed=int(1000 * v + 100 * T) + n0)
    ok = start == n0  # replicas that held n0 after T = 0 relaxation
    r = tau_mle(times[ok], esc[ok]) if ok.any() else {"tau": 0.0, "events": 0, "not_metastable_at_T0": True}
    r.update({"start_windings": start.tolist(), "times": times.tolist(), "escaped": esc.tolist(),
              "wall_s": round(time.time() - t0, 1)})
    out["per_value"][str(v)] = r
    print(kind, "n0", n0, "T", T, "value", v, {k: r[k] for k in r if k not in ("times", "escaped", "start_windings")},
          "started_in_n0", int((start == n0).sum()), flush=True)
json.dump(out, open(f"results/g34_{kind}_n{n0}_T{T}.json", "w"), indent=1)
