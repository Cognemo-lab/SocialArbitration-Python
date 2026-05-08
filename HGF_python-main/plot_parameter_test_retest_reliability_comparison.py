from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
ORIG_REL = BASE / "hgf_baselinefixed_full" / "test_retest_reliability_by_parameter.csv"
REDEPLOY_REL = BASE / "hgf_batch2_t2_parameter_reliability" / "test_retest_reliability_by_parameter.csv"
OUT_DIR = BASE / "hgf_batch2_t2_parameter_reliability"

PARAM_LABELS = {
    "ka_a": "Kappa-Advice",
    "ka_r": "Kappa-Reward",
    "m_a": "Equilibrium-Advice",
    "om_a": "Omega-Advice",
    "sa3a_0": "Prior Uncertainty-Advice",
    "sa3r_0": "Prior Uncertainty-Reward",
    "th_a": "Theta-Advice",
    "th_r": "Theta-Reward",
    "be0": "Intercept",
    "be1": "Surprise",
    "be2": "Arbitration",
    "be3": "Informational uncertainty - advice",
    "be4": "Informational uncertainty - reward location",
    "be5": "Volatility advice",
    "be6": "Volatility reward location",
    "ze": "Social bias",
    "be_ch": "Choice noise",
    "be_wager": "Wager noise",
}

ORDER = [
    "ze", "be_wager", "be_ch", "be6", "be5", "be4", "be3", "be2", "be1", "be0",
    "th_r", "th_a", "sa3r_0", "sa3a_0", "om_a", "m_a", "ka_r", "ka_a",
]

GROUP_SPLIT_INDEX = 10


def build_plot_df() -> pd.DataFrame:
    orig = pd.read_csv(ORIG_REL)
    orig["dataset"] = "Original"
    redeploy = pd.read_csv(REDEPLOY_REL)
    redeploy["dataset"] = "Redeployment"

    keep = [p for p in ORDER if p in orig["parameter"].values and p in redeploy["parameter"].values]
    df = pd.concat([orig, redeploy], ignore_index=True)
    df = df[(df["parameter"].isin(keep)) & (~df["is_fixed"])].copy()
    df["label"] = df["parameter"].map(PARAM_LABELS)
    df["order"] = df["parameter"].map({p: i for i, p in enumerate(keep)})
    return df.sort_values(["order", "dataset"]).reset_index(drop=True)


def plot_metric(ax, df: pd.DataFrame, metric: str, title: str) -> None:
    keep = [p for p in ORDER if p in df["parameter"].values]
    y_positions = np.arange(len(keep))[::-1]
    y_map = {p: y for p, y in zip(keep, y_positions)}

    ax.axvline(0.4, color="#999999", linestyle="--", linewidth=1)
    ax.axhspan(y_positions[GROUP_SPLIT_INDEX] - 0.5, y_positions[0] + 0.5, color="#f1e6d4", alpha=0.6, zorder=0)
    ax.axhspan(y_positions[-1] - 0.5, y_positions[GROUP_SPLIT_INDEX] - 0.5, color="#dce7f6", alpha=0.65, zorder=0)

    colors = {"Original": "#4C78A8", "Redeployment": "#F58518"}
    for p in keep:
        sub = df[df["parameter"] == p].copy()
        if len(sub) != 2:
            continue
        y = y_map[p]
        x1 = float(sub.loc[sub["dataset"] == "Original", metric].iloc[0])
        x2 = float(sub.loc[sub["dataset"] == "Redeployment", metric].iloc[0])
        ax.plot([x1, x2], [y, y], color="#BBBBBB", linewidth=1.2, zorder=1)
        ax.scatter(x1, y, s=55, color=colors["Original"], edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(x2, y, s=55, color=colors["Redeployment"], edgecolor="white", linewidth=0.8, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([PARAM_LABELS[p] for p in keep], fontsize=10.5)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(title, fontsize=14, pad=12)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_plot_df()
    df.to_csv(OUT_DIR / "parameter_test_retest_reliability_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 11), sharey=True)
    plot_metric(axes[0], df, "pearson_r_t1_t2", "Parameter Pearson r")
    plot_metric(axes[1], df, "icc3_1_t1_t2", "Parameter ICC(3,1)")
    axes[0].set_ylabel("")
    axes[0].text(
        -0.55,
        0.83,
        "Observation parameters",
        transform=axes[0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        color="#c47a28",
    )
    axes[0].text(
        -0.55,
        0.18,
        "Perceptual parameters",
        transform=axes[0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        color="#4C78A8",
    )
    for ax in axes:
        ax.set_xlabel("Reliability", fontsize=12)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C78A8", markeredgecolor="white", markersize=8, label="Original"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#F58518", markeredgecolor="white", markersize=8, label="Redeployment"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Parameter Test-Retest Reliability Comparison: Original vs Redeployment", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96), w_pad=1.0)
    fig.savefig(OUT_DIR / "figure_parameter_test_retest_reliability_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
