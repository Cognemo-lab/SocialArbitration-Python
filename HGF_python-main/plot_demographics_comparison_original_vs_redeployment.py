from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_CSV = BASE / "hitop_batch2" / "processed_data" / "demographics_comparison_original_vs_redeployment.csv"
OUT_DIR = BASE / "hitop_batch2" / "processed_data"
OUT_FIG = OUT_DIR / "figure_demographics_comparison_original_vs_redeployment.png"

CONTINUOUS = [
    ("age_mean", "Age"),
    ("education_years_mean", "Education years"),
]

PROPORTIONS = [
    ("woman_prop", "Women"),
    ("man_prop", "Men"),
    ("nonbinary_prop", "Nonbinary"),
    ("trans_prop", "Trans"),
    ("english_primary_prop", "English primary"),
    ("current_treatment_prop", "Current treatment"),
    ("previous_treatment_prop", "Previous treatment"),
    ("inpatient_history_prop", "Inpatient history"),
    ("emergency_history_prop", "Emergency history"),
    ("left_handed_prop", "Left-handed"),
]

COLORS = {"Original": "#4C78A8", "Redeployed": "#E76F51"}


def main() -> None:
    df = pd.read_csv(IN_CSV)
    wide = df.pivot(index="metric", columns="dataset", values="value")

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.4])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Continuous panel
    x = np.arange(len(CONTINUOUS))
    width = 0.36
    orig_vals = [wide.loc[m, "Original"] for m, _ in CONTINUOUS]
    red_vals = [wide.loc[m, "Redeployed"] for m, _ in CONTINUOUS]
    ax1.bar(x - width / 2, orig_vals, width, label="Original", color=COLORS["Original"], alpha=0.9)
    ax1.bar(x + width / 2, red_vals, width, label="Redeployed", color=COLORS["Redeployed"], alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([label for _, label in CONTINUOUS], rotation=0)
    ax1.set_title("Continuous demographics", fontsize=13, pad=10)
    ax1.grid(axis="y", alpha=0.2)
    for xi, val in zip(x - width / 2, orig_vals):
        ax1.text(xi, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    for xi, val in zip(x + width / 2, red_vals):
        ax1.text(xi, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # Proportion panel
    ypos = np.arange(len(PROPORTIONS))
    orig_p = np.array([wide.loc[m, "Original"] for m, _ in PROPORTIONS]) * 100
    red_p = np.array([wide.loc[m, "Redeployed"] for m, _ in PROPORTIONS]) * 100
    h = 0.36
    ax2.barh(ypos + h / 2, orig_p, height=h, color=COLORS["Original"], alpha=0.9, label="Original")
    ax2.barh(ypos - h / 2, red_p, height=h, color=COLORS["Redeployed"], alpha=0.9, label="Redeployed")
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([label for _, label in PROPORTIONS])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_title("Sample composition (%)", fontsize=13, pad=10)
    ax2.grid(axis="x", alpha=0.2)
    for y, val in zip(ypos + h / 2, orig_p):
        ax2.text(val + 1.2, y, f"{val:.1f}", va="center", fontsize=8)
    for y, val in zip(ypos - h / 2, red_p):
        ax2.text(val + 1.2, y, f"{val:.1f}", va="center", fontsize=8)

    n_orig = int(wide.loc["n", "Original"])
    n_red = int(wide.loc["n", "Redeployed"])
    fig.suptitle(
        f"Demographics comparison: original vs redeployed HiTOP samples\nOriginal n={n_orig}, Redeployed n={n_red}",
        fontsize=16,
        y=0.98,
    )
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.92))
    fig.tight_layout(rect=[0, 0, 1, 0.9], w_pad=2.5)
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    print(OUT_FIG)


if __name__ == "__main__":
    main()
