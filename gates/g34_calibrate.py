"""Calibration for G3/G4: which windings survive at T = 0 in each geometry, and rough
escape times at a few T. Not a gate; it only picks n0 and the T range."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from ab.tdgl import constricted_annulus, c_ring
from ab.escape import relax_survives, escape_times, tau_mle

N = 64
out = {"constriction_T0": {}, "gap_T0": {}, "quick": {}}
for w in [0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    out["constriction_T0"][str(w)] = {n: relax_survives(constricted_annulus(N, w), n) for n in range(1, 6)}
    print("w", w, out["constriction_T0"][str(w)], flush=True)
for g in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    out["gap_T0"][str(g)] = {n: relax_survives(c_ring(N, g), n) for n in range(1, 5)}
    print("g", g, out["gap_T0"][str(g)], flush=True)
for name, mu in [("w1.5", constricted_annulus(N, 1.5)), ("w2.5", constricted_annulus(N, 2.5)),
                 ("g1.0", c_ring(N, 1.0)), ("g2.0", c_ring(N, 2.0))]:
    for T in [0.1, 0.3]:
        t, e, s = escape_times(mu, 1, T, reps=8, tmax=2000.0, seed=5)
        out["quick"][f"{name}_T{T}"] = tau_mle(t, e)
        print(name, T, out["quick"][f"{name}_T{T}"], flush=True)
json.dump(out, open("results/g34_calibrate.json", "w"), indent=1)
