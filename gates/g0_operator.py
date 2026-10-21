"""G0 - the medium is what it says it is.

  a. discrete Laplacian has the right sign and value (the bug Grok's PhiWorld code had)
  b. T = 0 dynamics never increases the free energy (gradient flow)
  c. winding states n = 0..8 on the ring: which survive deterministic relaxation
     (the depairing limit, predicted n_c ~ R / sqrt(3) = 5.8 for R = 10)
  d. in a surviving state the bulk obeys |psi|^2 = 1 - k^2 with k = n / R
"""
import json, sys, time
import numpy as np
sys.path.insert(0, ".")
from ab.tdgl import (Medium, annulus, winding_state, winding, winding_radii, free_energy,
                     grid, _bilinear, DX)

N = 64
out = {}

# a. Laplacian on a smooth wave
X, Y, r, ph = grid(N)
kx = 2 * np.pi * 3 / (N * DX)
f = np.cos(kx * X)
lap = (np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f) / DX**2
exact = -kx**2 * f
out["a_laplacian_corr"] = float(np.corrcoef(lap.ravel(), exact.ravel())[0, 1])
out["a_laplacian_rel_err"] = float(np.abs(lap - exact).max() / np.abs(exact).max())
cb = (-1.0) ** (np.arange(N)[:, None] + np.arange(N)[None, :])
lcb = (np.roll(cb, 1, 0) + np.roll(cb, -1, 0) + np.roll(cb, 1, 1) + np.roll(cb, -1, 1) - 4 * cb)
out["a_checkerboard_lap_over_value"] = float((lcb / cb).mean())  # should be -8

# b, c: relax winding states at T = 0
mu = annulus(N)
ns = np.arange(0, 9)
psi0 = np.stack([winding_state(mu, n) for n in ns])
m = Medium(mu, psi0)
t0 = time.time()
F = [free_energy(m.psi, mu)]
for _ in range(40):
    m.run(10.0)
    F.append(free_energy(m.psi, mu))
F = np.array(F)
out["speed_s_per_1000_steps_batch9"] = (time.time() - t0) / (400 * 10 / 0.04 / 1000) * 1.0
out["b_max_energy_increase"] = float(np.max(np.diff(F, axis=0)))
wr = winding_radii(m.psi, mu)
out["c_initial_n"] = ns.tolist()
out["c_final_n_radii_9_10_11"] = wr.tolist()
surv = [int(n) for n, w in zip(ns, wr) if np.all(w == n)]
out["c_surviving_n"] = surv

# d: bulk |psi|^2 vs 1 - k^2 on the centre line
x = 10.0 * np.cos(np.linspace(0, 2 * np.pi, 64, endpoint=False))
y = 10.0 * np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))
a2 = np.abs(_bilinear(m.psi, x, y)).mean(-1) ** 2
out["d_centre_abs2"] = {int(n): round(float(a), 4) for n, a in zip(ns, a2) if n in surv}
out["d_pred_1_minus_k2"] = {int(n): round(1 - (n / 10.0) ** 2, 4) for n in surv}
print(json.dumps(out, indent=1))
json.dump(out, open("results/g0_operator.json", "w"), indent=1)
