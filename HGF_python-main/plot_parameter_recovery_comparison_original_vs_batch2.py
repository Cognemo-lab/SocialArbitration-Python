from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
ORIG = BASE / "hgf_baselinefixed_full" / "parameter_recovery_report.csv"
BATCH2 = BASE / "hgf_batch2_t1_baselinefixed_fullfidelity" / "parameter_recovery_report.csv"
OUT_DIR = BASE / "hgf_batch2_t1_baselinefixed_fullfidelity" / "comparison_to_original"

PRC_ORDER = [
    "ka_a",
    "ka_r",
    "m_a",
    "om_a",
    "sa3a_0",
    "sa3r_0",
    "th_a",
    "th_r",
]

OBS_ORDER = [
    "be0",
    "be1",
    "be2",
    "be3",
    "be4",
    "be5",
    "be6",
    "be_ch",
    "be_wager",
    "ze",
]

LABELS = {
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
    "be_ch": "Choice noise",
    "be_wager": "Wager noise",
    "ze": "Social bias",
}

DATASET_COLORS = {"Original": "#4C78A8", "Batch 2": "#F58518"}
GROUP_BG = {"prc": "#EEF4FB", "obs": "#FAF2E8"}


def load_recovery(path: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(~df["is_fixed"]) & (df["parameter"] != "m_r")].copy()
    df["dataset"] = dataset
    return df


def build_plot_df() -> pd.DataFrame:
    orig = load_recovery(ORIG, "Original")
    batch2 = load_recovery(BATCH2, "Batch 2")
    keep = PRC_ORDER + OBS_ORDER
    long = pd.concat([orig, batch2], ignore_index=True)
    long = long[long["parameter"].isin(keep)].copy()
    long["label"] = long["parameter"].map(LABELS)
    order_map = {param: i for i, param in enumerate(keep)}
    long["order"] = long["parameter"].map(order_map)
    long = long.sort_values(["order", "dataset"]).reset_index(drop=True)
    return long


def add_group_background(ax: plt.Axes, plot_df: pd.DataFrame, y_map: dict[str, float]) -> None:
    prc_labels = [LABELS[p] for p in PRC_ORDER if p in plot_df["parameter"].unique()]
    obs_labels = [LABELS[p] for p in OBS_ORDER if p in plot_df["parameter"].unique()]
    if prc_labels:
        ys = [y_map[label] for label in prc_labels]
        ax.axhspan(min(ys) - 0.5, max(ys) + 0.5, color=GROUP_BG["prc"], zorder=0)
    if obs_labels:
        ys = [y_map[label] for label in obs_labels]
        ax.axhspan(min(ys) - 0.5, max(ys) + 0.5, color=GROUP_BG["obs"], zorder=0)


def make_figure(plot_df: pd.DataFrame, out_name: str, title: str) -> Path:
    labels = [LABELS[p] for p in PRC_ORDER + OBS_ORDER if p in plot_df["parameter"].unique()]
    y_map = {label: idx for idx, label in enumerate(labels)}

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    metric_specs = [
        ("pearson_r", "Recovery Pearson r"),
        ("icc3_1", "Recovery ICC(3,1)"),
    ]

    for ax, (metric, title) in zip(axes, metric_specs):
        add_group_background(ax, plot_df, y_map)
        metric_df = plot_df.copy()
        metric_df["y"] = metric_df["label"].map(y_map)

        for label in labels:
            sub = metric_df[metric_df["label"] == label].sort_values("dataset")
            if len(sub) == 2:
                ax.plot(
                    sub[metric].to_numpy(),
                    sub["y"].to_numpy(),
                    color="#B8B8B8",
                    linewidth=1.2,
                    zorder=1,
                )

        for dataset in ["Original", "Batch 2"]:
            sub = metric_df[metric_df["dataset"] == dataset]
            ax.scatter(
                sub[metric],
                sub["y"],
                s=58,
                color=DATASET_COLORS[dataset],
                edgecolor="white",
                linewidth=0.8,
                label=dataset,
                zorder=3,
            )

        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("")
        ax.set_xlim(-0.05, 1.05)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#D0D0D0", linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.axvline(0.4, linestyle="--", linewidth=1.0, color="#808080", alpha=0.9)

    axes[0].set_ylabel("")
    axes[1].set_ylabel("")
    handles, labels_legend = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[:2], labels_legend[:2], loc="lower right", frameon=True, title="")

    fig.suptitle(title, fontsize=16, y=0.98)
    fig.text(0.015, 0.76, "Observation parameters", rotation=90, fontsize=10, color="#B7783B", va="center")
    fig.text(0.015, 0.33, "Perceptual parameters", rotation=90, fontsize=10, color="#5A7BA6", va="center")
    fig.tight_layout(rect=(0.04, 0.03, 1, 0.96))

    out = OUT_DIR / out_name
    fig.savefig(out, dpi=260)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = build_plot_df()
    plot_df.to_csv(OUT_DIR / "parameter_recovery_comparison_original_vs_batch2.csv", index=False)
    make_figure(
        plot_df,
        "figure_parameter_recovery_comparison_original_vs_batch2.png",
        "Parameter Recovery Comparison: Original vs Batch 2",
    )
    make_figure(
        plot_df,
        "figure_parameter_recovery_comparison_original_vs_batch2_paper_subset.png",
        "Parameter Recovery Comparison: Paper Recovery Panel Subset",
    )


if __name__ == "__main__":
    main()
