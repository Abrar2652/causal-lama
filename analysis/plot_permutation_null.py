#!/usr/bin/env python3
"""Figure: observed RCS profiles against the permuted-label null envelope.

Takeaway: the permuted-label null is flat at ~0.02 and every observed profile sits far
above it and is measurably more concentrated, so the reported breadth is not an artifact
of RCS being nonnegative.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import null_core as nc

ND = os.path.join(REPO, "results", "permutation_null")
res = json.load(open(os.path.join(ND, "results_corrected.json")))
dists = np.load(os.path.join(ND, "permutation_distribution_corrected.npz"))

plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.8, "axes.edgecolor": "black",
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

# ---- Panel grid: per-model profile vs null envelope (log y, since the gap is ~50x)
fig, axes = plt.subplots(3, 3, figsize=(7.4, 5.6))
for ax, r in zip(axes.ravel(), res):
    alias = r["alias"]
    d = json.load(open(os.path.join(REPO, "logs", f"rcs_analysis_{alias}.json")))
    layers = sorted(int(k) for k in d["rcs_scores"])
    obs = np.array([d["rcs_scores"][str(l)] for l in layers])
    null = dists[alias]                                  # [n_perm, L]
    lo = np.percentile(null, 2.5, axis=0); hi = np.percentile(null, 97.5, axis=0)
    med = np.percentile(null, 50, axis=0)

    ax.fill_between(layers, lo, hi, color="0.75", edgecolor="none", zorder=1,
                    label="null 95% band")
    ax.plot(layers, med, color="0.35", lw=0.9, ls="--", zorder=2, label="null median")
    ax.plot(layers, obs, color="#1f4e79", lw=1.3, zorder=3, label="observed RCS")
    ax.set_yscale("log")
    ax.set_ylim(6e-3, max(3.0, obs.max() * 1.6))
    ax.set_xlim(min(layers), max(layers))
    ax.tick_params(labelsize=7, width=0.8)
    for s in ax.spines.values(): s.set_edgecolor("black")
    # both annotations live in the empty strip BELOW the null envelope, so no curve is crossed
    ax.text(0.03, 0.04, r["model"], transform=ax.transAxes, fontsize=7.2,
            va="bottom", ha="left", fontweight="bold")
    # annotation parked in the empty band BELOW the null envelope, never over a curve
    ax.text(0.97, 0.045,
            f"obs/null {r['observed']['mean_rcs']/r['null_per_layer_rcs']['mean']:.0f}x",
            transform=ax.transAxes, fontsize=6.5, va="bottom", ha="right", color="0.25")
for ax in axes[-1, :]: ax.set_xlabel("layer index", fontsize=7.5)
for ax in axes[:, 0]:  ax.set_ylabel("RCS (log)", fontsize=7.5)
axes[0, 0].legend(loc="center right", bbox_to_anchor=(1.0, 0.46), fontsize=6.2,
                  handlelength=1.4, borderpad=0.2)
fig.tight_layout(pad=0.5, h_pad=0.7, w_pad=0.7)
p1 = os.path.join(ND, "plots", "null_vs_observed_profiles")
fig.savefig(p1 + ".pdf"); fig.savefig(p1 + ".png", dpi=220); plt.close(fig)

# ---- Concentration statistics: observed vs null, all nine models
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
names = [r["model"] for r in res]
y = np.arange(len(res))
for ax, key, obskey, lab, flip in [
        (axes[0], "gini", "gini", "Gini coefficient", False),
        (axes[1], "n80_over_L", "n80_over_L", r"$n_{80}/L$", True)]:
    nl = [r["null_profile_stats"][key] for r in res]
    ob = [r["observed"][obskey] for r in res]
    for i, (n, o) in enumerate(zip(nl, ob)):
        ax.plot([n["q2.5"], n["q97.5"]], [i, i], color="0.6", lw=4, solid_capstyle="butt",
                zorder=1, label="null 95% band" if i == 0 else None)
        ax.plot(n["mean"], i, "o", ms=3.4, mfc="0.3", mec="black", mew=0.5, zorder=2,
                label="null mean" if i == 0 else None)
        ax.plot(o, i, "D", ms=4.2, mfc="#c0392b", mec="black", mew=0.6, zorder=3,
                label="observed" if i == 0 else None)
    ax.set_yticks(y); ax.set_yticklabels(names if ax is axes[0] else [], fontsize=7)
    ax.set_xlabel(lab, fontsize=7.5); ax.tick_params(labelsize=7, width=0.8)
    ax.set_ylim(-0.7, len(res) - 0.3); ax.invert_yaxis()
    for s in ax.spines.values(): s.set_edgecolor("black")
    ax.grid(axis="x", color="0.9", lw=0.6, zorder=0)
axes[0].legend(loc="lower right", fontsize=6.2, handlelength=1.2, borderpad=0.25)
fig.tight_layout(pad=0.5, w_pad=0.6)
p2 = os.path.join(ND, "plots", "null_vs_observed_concentration")
fig.savefig(p2 + ".pdf"); fig.savefig(p2 + ".png", dpi=220); plt.close(fig)

# one-takeaway captions + the numbers behind each panel, stored beside the figures
cap = """null_vs_observed_profiles
  TAKEAWAY: every observed RCS profile lies 12x-48x above the permuted-label null, so the
  breadth of RCS mass is a real harmful-harmless signal and not a consequence of RCS being
  nonnegative. Gray band = 2.5-97.5%% of 200 relabelings; dashed = null median; blue = observed.
  y-axis is logarithmic because observed and null differ by nearly two orders of magnitude.

null_vs_observed_concentration
  TAKEAWAY: on both concentration statistics all nine observed profiles fall outside the
  null's 95%% band, in the direction of MORE concentration than chance (higher Gini, lower
  n80/L), so "distributed" does not mean "indistinguishable from flat".
  Empirical tail probability was 0.000 for every model at 200 relabelings.

Numbers behind the panels
"""
for r in res:
    o = r["observed"]; ns = r["null_profile_stats"]; pl = r["null_per_layer_rcs"]
    cap += (f"  {r['model']:<15} obs_mean_RCS={o['mean_rcs']:.4f} null={pl['mean']:.5f} "
            f"ratio={o['mean_rcs']/pl['mean']:.1f}x | Gini obs={o['gini']:.4f} "
            f"null={ns['gini']['mean']:.4f} [{ns['gini']['q2.5']:.4f},{ns['gini']['q97.5']:.4f}] | "
            f"n80/L obs={o['n80_over_L']:.3f} null={ns['n80_over_L']['mean']:.3f} "
            f"[{ns['n80_over_L']['q2.5']:.3f},{ns['n80_over_L']['q97.5']:.3f}]\n")
open(os.path.join(ND, "plots", "FIGURE_NOTES.txt"), "w").write(cap)
print("wrote", p1 + ".{pdf,png}"); print("wrote", p2 + ".{pdf,png}")
print("wrote", os.path.join(ND, "plots", "FIGURE_NOTES.txt"))
