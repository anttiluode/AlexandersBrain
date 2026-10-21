"""Lifetime of a stored n = 1 vs constriction width w (G3) and clasp gap g (G4)."""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {0.07: "#2a78d6", 0.1: "#eb6834", 0.14: "#1baf7a"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
summary = {}
for ax, kind, xlabel, extra in [
        (axes[0], "constriction", "constriction width w (coherence lengths; ring width 4)", None),
        (axes[1], "gap", "clasp gap g of normal material (coherence lengths)", 1.75)]:
    for f in sorted(glob.glob(f"results/g34_{kind}_n1_T*.json")):
        d = json.load(open(f))
        T = d["T"]
        xs, ys, cens = [], [], []
        for v, r in d["per_value"].items():
            xs.append(float(v))
            if r.get("tau") is None:
                ys.append(r["tau_lower_bound"])
                cens.append(True)
            else:
                ys.append(r["tau"])
                cens.append(False)
            summary[f"{kind}_{v}_T{T}"] = {"tau": r.get("tau"), "lower_bound": r.get("tau_lower_bound"),
                                           "events": r["events"]}
        o = np.argsort(xs)
        xs, ys, cens = np.array(xs)[o], np.array(ys)[o], np.array(cens)[o]
        c = COL[T]
        ax.plot(xs, ys, "-", color=c, lw=2, label=f"T = {T}")
        ax.scatter(xs[~cens], ys[~cens], s=40, color=c, zorder=3, edgecolor="white", linewidth=1.5)
        ax.scatter(xs[cens], ys[cens], s=60, marker="^", facecolor="white", edgecolor=c, linewidth=2,
                   zorder=3)
    if extra:
        ax.axvspan(extra, 2.6, color="#d9d8d3", alpha=0.6, lw=0)
        ax.text(2.17, 2500, "no stationary\nn = 1 at T = 0\n(drains in ~1000 t)", ha="center", fontsize=8,
                color="#52514e")
        ax.set_xlim(-0.1, 2.6)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
axes[0].set_ylabel("mean lifetime of n = 1 (time units)")
axes[0].set_title("G3  narrowed ring", loc="left", fontsize=10)
axes[1].set_title("G4  ring cut by a gap (one clasp)", loc="left", fontsize=10)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.text(0.01, 0.005, "hollow triangles: no escape in 12 replicas x 3000 t (lower bound = 36000)",
         fontsize=7.5, color="#52514e")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("results/g34_lifetime.png", dpi=120)
json.dump(summary, open("results/g34_summary.json", "w"), indent=1)
