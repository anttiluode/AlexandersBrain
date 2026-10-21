"""Combine per-T G1 results into one table and figure."""
import glob, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = {}
for f in sorted(glob.glob("results/g1_noise_T*.json")):
    d = json.load(open(f))
    for T, s in d["per_T"].items():
        runs[float(T)] = s
Ts = sorted(runs)
K = 9
fig, ax = plt.subplots(1, len(Ts), figsize=(3.6 * len(Ts), 3.3), sharey=True)
ax = [ax] if len(Ts) == 1 else ax
for a, T in zip(ax, Ts):
    s = runs[T]["series"]
    a.plot(s["t"], s["n_recall"], label="integer n (ring)", lw=2, color="#2a78d6")
    a.plot(s["t"], s["theta_recall"], label="phase (ring), 9 levels", color="#eb6834")
    a.plot(s["t"], s["disk_theta_recall"], "--", label="phase (disk), 9 levels", color="#1baf7a")
    a.axhline(1 / K, color="grey", lw=0.6)
    a.set_title(f"T = {T}")
    a.set_xlabel("time")
ax[0].set_ylabel("recall (chance 1/9)")
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig("results/g1_noise.png", dpi=110)
rows = []
for T in Ts:
    s = runs[T]
    rows.append({"T": T, "n_recall_by_n": s["n_recall_by_n_final"],
                 "theta_recall_ring": s["theta_recall_final"], "theta_recall_disk": s["disk_theta_recall_final"],
                 "t_half_theta_ring": s["t_half_theta_ring"], "t_half_theta_disk": s["t_half_theta_disk"],
                 "var_slope_ring_meas": s["theta_var_slope_ring_measured"],
                 "var_slope_ring_pred": s["theta_var_slope_ring_predicted_T_over_int_psi2_n0"],
                 "var_slope_disk_meas": s["theta_var_slope_disk_measured"],
                 "var_slope_disk_pred": s["theta_var_slope_disk_predicted"]})
json.dump(rows, open("results/g1_summary.json", "w"), indent=1)
for r in rows:
    print(r)
