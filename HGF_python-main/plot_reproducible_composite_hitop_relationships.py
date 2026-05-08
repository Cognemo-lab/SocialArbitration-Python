from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_PATH = BASE / "composite_vs_hitop_inq_t1_t2" / "composites_vs_hitop_inq_correlations.csv"
OUT_DIR = BASE / "composite_vs_hitop_inq_t1_t2" / "reproducible_relationships"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(IN_PATH)

    piv = df.pivot_table(
        index=["timepoint", "composite", "composite_label", "target", "target_label"],
        columns="dataset",
        values=["pearson_r", "p_value", "q_value", "n"],
    )
    piv.columns = ["_".join(col) for col in piv.columns]
    piv = piv.reset_index()
    piv["same_sign"] = piv["pearson_r_Original"] * piv["pearson_r_Redeployed"] > 0
    piv["both_p_lt_05"] = (piv["p_value_Original"] < 0.05) & (piv["p_value_Redeployed"] < 0.05)
    piv["orig_p_lt_05_red_p_lt_10"] = (piv["p_value_Original"] < 0.05) & (piv["p_value_Redeployed"] < 0.10)
    piv["min_abs_r"] = piv[["pearson_r_Original", "pearson_r_Redeployed"]].abs().min(axis=1)

    strict = piv.loc[piv["same_sign"] & piv["both_p_lt_05"]].copy()
    lenient = piv.loc[piv["same_sign"] & piv["orig_p_lt_05_red_p_lt_10"]].copy()
    strict = strict.sort_values(["timepoint", "min_abs_r"], ascending=[True, False])
    lenient = lenient.sort_values(["timepoint", "min_abs_r"], ascending=[True, False])

    strict.to_csv(OUT_DIR / "strict_reproducible_relationships.csv", index=False)
    lenient.to_csv(OUT_DIR / "lenient_reproducible_relationships.csv", index=False)

    plot_df = lenient.copy()
    plot_df["label"] = plot_df["composite_label"] + "\nvs " + plot_df["target_label"]
    plot_df = plot_df.sort_values(["timepoint", "min_abs_r"], ascending=[True, False]).reset_index(drop=True)

    colors = {"Original": "#4C78A8", "Redeployed": "#F58518"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, tp, title in zip(axes, ["t1", "t2"], ["T1", "T2"]):
        sub = plot_df.loc[plot_df["timepoint"] == tp].copy()
        ypos = np.arange(len(sub))
        ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
        for dataset, offset in [("Original", -0.10), ("Redeployed", 0.10)]:
            ax.scatter(
                sub[f"pearson_r_{dataset}"],
                ypos + offset,
                s=65,
                color=colors[dataset],
                edgecolor="white",
                linewidth=0.7,
                label=dataset,
                zorder=3,
            )
            for y, x0, x1 in zip(ypos, sub["pearson_r_Original"], sub["pearson_r_Redeployed"]):
                ax.plot([x0, x1], [y, y], color="#BBBBBB", alpha=0.8, linewidth=1, zorder=1)
        ax.set_title(title)
        ax.set_xlabel("Pearson r")
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlim(-0.3, 0.05)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["label"])
        ax.invert_yaxis()

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Original"], markersize=8, label="Original"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Redeployed"], markersize=8, label="Redeployed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Composite-measure relationships reproducible across deployments", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(OUT_DIR / "figure_reproducible_composite_relationships.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
