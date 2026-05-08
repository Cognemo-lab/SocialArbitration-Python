from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_DIR = BASE / "hgf_batch2_t1_baselinefixed_fullfidelity" / "construct_validity_behavior"
OUT_DIR = IN_DIR / "figures"

BEHAVIOR_LABELS = {
    "accuracy_rate": "Accuracy",
    "advice_taking_rate": "Advice-taking",
    "lose_switch_rate": "Lose-switch",
    "win_stay_rate": "Win-stay",
    "mean_wager": "Mean wager",
}

PARAMETER_LABELS = {
    "ka_a": "Kappa-Advice",
    "ka_r": "Kappa-Reward",
    "m_a": "Equilibrium-Advice",
    "m_r": "Equilibrium-Reward",
    "om_a": "Omega-Advice",
    "om_r": "Omega-Reward",
    "th_a": "Theta-Advice",
    "th_r": "Theta-Reward",
    "mu3r_0": "Initial volatility-Reward",
    "sa2r_0": "Prior uncertainty-Reward L2",
    "sa3a_0": "Prior uncertainty-Advice",
    "sa3r_0": "Prior uncertainty-Reward",
    "be0": "Intercept",
    "be1": "Surprise",
    "be2": "Arbitration",
    "be3": "Informational uncertainty-Advice",
    "be4": "Informational uncertainty-Reward",
    "be5": "Volatility-Advice",
    "be6": "Volatility-Reward",
    "be_ch": "Choice noise",
    "be_wager": "Wager noise",
    "ze": "Social bias",
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
    pooled["parameter_label"] = pooled["parameter"].map(PARAMETER_LABELS).fillna(pooled["parameter"])
    pooled["behavior_label"] = pooled["behavior_metric"].map(BEHAVIOR_LABELS)
    pooled["abs_r"] = pooled["pearson_r"].abs()
    ordered_params = (
        pooled.groupby("parameter_label", as_index=False)["abs_r"].max()
        .sort_values("abs_r", ascending=False)["parameter_label"]
        .tolist()
    )

    heat = pooled.pivot(index="parameter_label", columns="behavior_label", values="pearson_r")
    heat = heat.reindex(index=ordered_params, columns=list(BEHAVIOR_LABELS.values()))
    qvals = pooled.pivot(
        index="parameter_label",
        columns="behavior_label",
        values="fdr_q_pearson_within_behavior_scope",
    ).reindex(index=ordered_params, columns=list(BEHAVIOR_LABELS.values()))

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8.5, max(6, 0.42 * len(ordered_params))))
    sns.heatmap(
        heat,
        cmap="RdBu_r",
        center=0.0,
        vmin=-0.7,
        vmax=0.7,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Batch 2 Parameters vs Behavior")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    for i, row_name in enumerate(heat.index):
        for j, col_name in enumerate(heat.columns):
            val = heat.loc[row_name, col_name]
            if pd.notna(val):
                star = fdr_stars(qvals.loc[row_name, col_name])
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}{star}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_construct_validity_heatmap.png", dpi=220)
    plt.close(fig)

    focus = top.loc[top["analysis_scope"] == "pooled_t1_t2"].copy()
    focus["parameter_label"] = focus["parameter"].map(PARAMETER_LABELS).fillna(focus["parameter"])
    focus["behavior_label"] = focus["behavior_metric"].map(BEHAVIOR_LABELS)
    focus = (
        focus.sort_values(["behavior_label", "pearson_r"])
        .groupby("behavior_label", sort=False)
        .head(5)
        .copy()
    )

    behaviors = list(BEHAVIOR_LABELS.values())
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 8))
    axes = axes.ravel()
    for ax, behavior in zip(axes, behaviors):
        sub = focus.loc[focus["behavior_label"] == behavior].sort_values("pearson_r")
        sns.barplot(data=sub, x="pearson_r", y="parameter_label", color="#4C78A8", ax=ax)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(behavior)
        ax.set_xlabel("Pearson r")
        ax.set_ylabel("")
        for patch, (_, row) in zip(ax.patches, sub.iterrows()):
            x = row["pearson_r"]
            star = fdr_stars(row["fdr_q_pearson_within_behavior_scope"])
            ax.text(
                x + (0.02 if x >= 0 else -0.02),
                patch.get_y() + patch.get_height() / 2,
                star,
                va="center",
                ha="left" if x >= 0 else "right",
                fontsize=11,
            )
    for ax in axes[len(behaviors):]:
        ax.axis("off")
    fig.suptitle("Batch 2 Top Parameter-Behavior Associations", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "figure_construct_validity_top_hits.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
