"""Regenerate figures/fig_auroc_by_category.png from the three k*_per_category.csv files.

The original figure was authored at a 22-inch figsize, so when the PNG was
scaled to the 16 cm content width of the report its text fell to roughly 2 pt
and the category labels were unreadable in print. This rebuilds it at a size
that survives that downscale.
"""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

WS = pathlib.Path(__file__).resolve().parent.parent
PAPER_MEAN = 0.971

data = {}
for k in (1, 2, 4):
    with open(WS / f"k{k}_per_category.csv", newline="", encoding="utf8") as f:
        for row in csv.DictReader(f):
            data.setdefault(row["Category"], {})[k] = float(row["Image AUROC"])

cats = sorted(data)
assert len(cats) == 15, f"expected 15 categories, found {len(cats)}"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=300)
x = np.arange(len(cats))
w = 0.27
colors = ["#4C78A8", "#F58518", "#54A24B"]

for i, k in enumerate((1, 2, 4)):
    ax.bar(x + (i - 1) * w, [data[c][k] for c in cats], w,
           label=f"k = {k}", color=colors[i],
           edgecolor="white", linewidth=0.4, zorder=3)

ax.axhline(PAPER_MEAN, ls="--", lw=1.0, color="#C0392B", zorder=4)

ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=42, ha="right", fontsize=7.6)
ax.set_ylabel("Image AUROC", fontsize=8.2)
ax.set_ylim(0.60, 1.035)
ax.set_yticks(np.arange(0.60, 1.01, 0.05))
ax.tick_params(axis="y", labelsize=7.6)
ax.grid(axis="y", ls=":", lw=0.6, color="#999999", zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(3)]
handles.append(Line2D([0], [0], ls="--", lw=1.0, color="#C0392B"))
ax.legend(handles,
          ["k = 1", "k = 2", "k = 4", f"paper 1-shot mean ({PAPER_MEAN:.3f})"],
          loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4,
          frameon=False, fontsize=8)

fig.tight_layout()
out = WS / "figures" / "fig_auroc_by_category.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"wrote {out}  {out.stat().st_size} bytes")
