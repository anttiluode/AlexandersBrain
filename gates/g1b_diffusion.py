"""G1b - the exact phase-diffusion law, with a proper estimator.

Theta = arg S, S = sum_x psi(x) over ordered material (full field, not a centre-line sample),
n = 0, 32 replicas per geometry and T. Prediction (no free parameter):
    d Var(Theta) / dt = T n_W / (dx^2 <|S|^2>)  ~  T / integral |psi|^2 dA
(the noise in S has variance n_W T dt / dx^2; the deterministic flow does not rotate S)
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from ab.tdgl import Medium, annulus, disk, winding_state, wrap, DX

N = 64
REP = 32
out = {}
for name, mu in [("ring", annulus(N)), ("disk", disk(N, R=12.0))]:
    W = mu > 0
    for T in [0.02, 0.05, 0.1]:
        m = Medium(mu, np.stack([winding_state(mu, 0)] * REP), seed=int(T * 1e4) + len(name))
        m.run(100.0)
        m.run(100.0, T=T)  # thermalise amplitudes first
        th0 = np.angle((m.psi * W).sum((-1, -2)))
        ts, vs, Is, Ss = [], [], [], []
        acc = np.zeros(REP)
        prev = th0.copy()
        for k in range(30):
            m.run(20.0, T=T)
            th = np.angle((m.psi * W).sum((-1, -2)))
            acc += wrap(th - prev)  # unwrap in time
            prev = th
            ts.append(20.0 * (k + 1))
            vs.append(float(np.mean(acc ** 2)))
            Is.append(float((np.abs(m.psi * W) ** 2).sum((-1, -2)).mean() * DX * DX))
            Ss.append(float(np.mean(np.abs((m.psi * W).sum((-1, -2))) ** 2)))
        ts = np.array(ts)
        slope = float(np.sum(ts * np.array(vs)) / np.sum(ts ** 2))  # fit through origin
        se = float(slope * np.sqrt(2.0 / REP))  # chi-square scatter of a variance estimate
        pred = T / float(np.mean(Is))  # T / integral |psi|^2 over the material
        # exact for this estimator: noise in S = sum psi has variance nW T dt / dx^2
        pred_est = T * W.sum() / (DX * DX * float(np.mean(Ss)))
        out[f"{name}_T{T}"] = {"slope_measured": slope, "slope_sd_approx": se,
                               "slope_predicted_T_over_int_psi2": pred,
                               "slope_predicted_estimator_exact": pred_est,
                               "ratio_to_estimator_exact": slope / pred_est,
                               "ratio_to_T_over_int_psi2": slope / pred,
                               "int_psi2_thermal": float(np.mean(Is))}
        print(name, T, out[f"{name}_T{T}"], flush=True)
json.dump(out, open("results/g1b_diffusion.json", "w"), indent=1)
