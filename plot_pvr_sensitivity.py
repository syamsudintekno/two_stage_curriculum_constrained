from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "extended_results"
K = 10

data = pd.read_csv(RESULTS / "E1_E3_main_by_beta.csv")
hard = data.loc[data["variant"].eq("hard_c1_c2")].sort_values("beta")
hard_c1_only = hard.loc[hard["beta"].eq(1.0)].iloc[0]
pvr_bound = (K - hard_c1_only["len"]) / K

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.45, 2.35), layout="constrained")

for variant, label, color, marker, linestyle in [
    ("soft_frac_c1", "Soft fractional C1 + C2", "#2166AC", "o", "-"),
    ("soft_bin_c1", "Soft binary C1 + C2", "#B35806", "s", "--"),
]:
    rows = data.loc[data["variant"].eq(variant)].sort_values("beta")
    ax.errorbar(
        rows["beta"], rows["pvr"], yerr=rows["pvr_sd"],
        label=label, color=color, marker=marker, linestyle=linestyle,
        markersize=3, linewidth=1, capsize=2,
    )

ax.axhline(
    pvr_bound, color="0.25", linestyle=":", linewidth=1.1,
    label=f"Fixed-$K$ lower bound ({pvr_bound:.2%})",
)
ax.plot(
    hard["beta"], hard["pvr"], color="0.15", marker="^",
    markersize=3, linewidth=0.9, label="Hard C1 (variable length)",
)
ax.annotate(
    f"Full@10 = {hard_c1_only['full']:.1%}",
    xy=(0.7, 0), xytext=(0, 7), textcoords="offset points",
    ha="center", fontsize=7,
)

ax.set(
    xlabel=r"Relevance weight $\beta$",
    ylabel="PVR@10",
    xlim=(0, 1.04),
    ylim=(-0.018, 0.64),
)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax.grid(axis="y", color="0.90", linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=7)

for extension in ("pdf", "png"):
    fig.savefig(RESULTS / f"pvr_sensitivity_ieee.{extension}", dpi=600)
plt.close(fig)