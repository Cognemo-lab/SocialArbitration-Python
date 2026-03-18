from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_DIR = BASE / "hgf_baselinefixed_full" / "construct_validity_behavior"
OUT_DIR = IN_DIR / "figures"

BEHAVIOR_LABELS = {
    "accuracy_rate": "Accuracy",
    "advice_taking_rate": "Advice-taking",
    "win_stay_rate": "Win-stay",
    "lose_switch_rate": "Lose-switch",
}


def fdr_stars(q: float) -> str:
    if pd.isna(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    assoc = pd.read_csv(IN_DIR / "construct_validity_parameter_behavior.csv")
    top = pd.read_csv(IN_DIR / "top_construct_validity_hits.csv")

    pooled = assoc.loc[assoc["analysis_scope"] == "pooled_t1_t2"].copy()
    pooled["behavior_label"] = pooled["behavior_metric"].map(BEHAVIOR_LABELS)

    heat = pooled.pivot(index="parameter", columns="behavior_label", values="pearson_r")
    heat = heat.loc[heat.abs().max(axis=1).sort_values(ascending=False).index]

    qvals = pooled.pivot(
        index="parameter",
        columns="behavior_label",
        values="fdr_q_pearson_within_behavior_scope",
    ).reindex(index=heat.index, columns=heat.columns)

    fig_h = max(7, 0.3 * len(heat))
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    im = ax.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("Construct Validity: Parameter-Behavior Correlations")
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            val = heat.iloc[i, j]
            star = fdr_stars(qvals.iloc[i, j])
            text = f"{val:.2f}{star}" if pd.notna(val) else ""
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_construct_validity_heatmap.png", dpi=220)
    plt.close(fig)

    focus = top.loc[top["analysis_scope"] == "pooled_t1_t2"].copy()
    focus["behavior_label"] = focus["behavior_metric"].map(BEHAVIOR_LABELS)
    focus = (
        focus.sort_values(["behavior_label", "pearson_r"])
        .groupby("behavior_label", sort=False)
        .head(5)
        .copy()
    )

    behaviors = list(BEHAVIOR_LABELS.values())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes = axes.ravel()
    color_pos = "#2a9d8f"
    color_neg = "#e76f51"
    for ax, behavior in zip(axes, behaviors):
        sub = focus.loc[focus["behavior_label"] == behavior].sort_values("pearson_r")
        colors = [color_pos if r >= 0 else color_neg for r in sub["pearson_r"]]
        ax.barh(sub["parameter"], sub["pearson_r"], color=colors, alpha=0.9)
        ax.axvline(0, color="0.3", lw=1)
        ax.set_title(behavior)
        ax.set_xlabel("Pearson r")
        lim = max(0.15, np.nanmax(np.abs(sub["pearson_r"])) * 1.2)
        ax.set_xlim(-lim, lim)
        for i, (_, row) in enumerate(sub.iterrows()):
            star = fdr_stars(row["fdr_q_pearson_within_behavior_scope"])
            xpos = row["pearson_r"] + (0.015 if row["pearson_r"] >= 0 else -0.015)
            ha = "left" if row["pearson_r"] >= 0 else "right"
            ax.text(xpos, i, star, va="center", ha=ha, fontsize=9)
    fig.suptitle("Top Parameter Correlates of Behavioral Performance", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "figure_construct_validity_top_hits.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
