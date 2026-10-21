"""Written winding vs applied flux, for three gate-open times and the closed arm."""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

per = {}
for f in glob.glob("results/g5_write_map_*.json"):
    per.update(json.load(open(f))["per_f"])
fs = sorted(per, key=float)
COL = {-1: "#2a78d6", 0: "#eb6834", 1: "#1baf7a"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
panels = [("gated", 200, "gate open 200 t"), ("gated", 600, "gate open 600 t"),
          ("gated", 1800, "gate open 1800 t"), ("closed", 200, "gate closed (field only)")]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
x = np.array([float(f) for f in fs])
for ax, (arm, tw, title) in zip(axes, panels):
    ax.plot(x, np.round(x), color="#c3c2b7", lw=6, solid_capstyle="round", zorder=0,
            label="round(f)" if arm == "gated" else None)
    for j, n0 in enumerate([-1, 0, 1]):
        y = [per[f][f"{arm}_tw{tw}_n0{n0}"] for f in fs]
        ax.scatter(x + (j - 1) * 0.06, y, s=30, color=COL[n0], edgecolor="white", linewidth=1.2,
                   zorder=3, label=f"start n0 = {n0}")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("applied flux f (quanta per ring)")
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
axes[0].set_ylabel("winding read after field off")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig("results/g5_write_map.png", dpi=120)
