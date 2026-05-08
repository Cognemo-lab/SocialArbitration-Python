from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
OUTDIR = ROOT / "reliability_comparison_behavior_parameters"
OUTDIR.mkdir(parents=True, exist_ok=True)

BEHAV_PATH = ROOT / "behavior_test_retest_comparison" / "behavior_test_retest_reliability_comparison.csv"
ORIG_PARAM_PATH = ROOT / "hgf_baselinefixed_full" / "test_retest_reliability_by_parameter.csv"
REDEP_PARAM_PATH = ROOT / "hgf_batch2_t2_parameter_reliability" / "test_retest_reliability_by_parameter.csv"
BEHAV_COMP_PATH = ROOT / "behavior_test_retest_comparison" / "two_axis_behavior_model" / "behavior_two_axis_reliability.csv"
PARAM_COMP_PATH = ROOT / "hgf_parameter_two_axis_model" / "parameter_two_axis_reliability.csv"

PARAM_LABELS = {
    "be0": "Intercept",
    "be1": "Surprise",
    "be2": "Arbitration",
    "be3": "Info uncertainty - advice",
    "be4": "Info uncertainty - reward",
    "be5": "Volatility advice",
    "be6": "Volatility reward",
    "be_ch": "Choice noise",
    "be_wager": "Wager noise",
    "ze": "Social bias",
    "ka_a": "Kappa-Advice",
    "ka_r": "Kappa-Reward",
    "m_a": "Equilibrium-Advice",
    "mu3r_0": "Mu3r_0",
    "om_a": "Omega-Advice",
    "sa2r_0": "Sa2r_0",
    "sa3a_0": "Prior Uncertainty-Advice",
    "sa3r_0": "Prior Uncertainty-Reward",
    "th_a": "Theta-Advice",
    "th_r": "Theta-Reward",
}

BEHAV_LABELS = {
    "Accuracy": "Accuracy",
    "Accuracy (first 30)": "Accuracy first 30",
    "Accuracy (last 20)": "Accuracy last 20",
    "Advice-taking": "Advice-taking",
    "Win-stay": "Win-stay",
    "Lose-switch": "Lose-switch",
    "Mean wager": "Mean wager",
}

DISPLAY_LABELS = {
    "Info uncertainty - advice": "Info uncertainty\n- advice",
    "Info uncertainty - reward": "Info uncertainty\n- reward",
    "Prior Uncertainty-Advice": "Prior Uncertainty\n- Advice",
    "Prior Uncertainty-Reward": "Prior Uncertainty\n- Reward",
    "Task-performance / policy axis": "Task-performance /\npolicy axis",
    "Observation / response axis": "Observation /\nresponse axis",
    "Advice-learning / uncertainty axis": "Advice-learning /\nuncertainty axis",
    "Wager / confidence axis": "Wager /\nconfidence axis",
}


def load_behavior() -> pd.DataFrame:
    df = pd.read_csv(BEHAV_PATH).copy()
    df["domain"] = "Behavior"
    df["item"] = df["label"].map(BEHAV_LABELS)
    df["dataset"] = df["dataset"].replace({"Redeployment": "Redeployed"})
    df = df.rename(columns={"pearson_r": "pearson_r_t1_t2", "icc3_1": "icc3_1_t1_t2"})
    return df[["dataset", "domain", "item", "pearson_r_t1_t2", "icc3_1_t1_t2"]]


def load_parameters(path: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df = df.loc[df["is_fixed"] == False].copy()
    df = df.loc[df["parameter"] != "m_r"].copy()
    df["domain"] = df["group"].map({"obs": "Observation parameters", "prc": "Perceptual parameters"})
    df["item"] = df["parameter"].map(PARAM_LABELS).fillna(df["parameter"])
    df["dataset"] = dataset
    return df[["dataset", "domain", "item", "pearson_r_t1_t2", "icc3_1_t1_t2"]]


def load_composites() -> pd.DataFrame:
    beh = pd.read_csv(BEHAV_COMP_PATH).copy()
    beh["domain"] = "Composite measures"
    beh["item"] = beh["axis_label"]
    beh["dataset"] = beh["dataset"]
    beh = beh.rename(columns={"pearson_r": "pearson_r_t1_t2", "icc3_1": "icc3_1_t1_t2"})
    beh = beh[["dataset", "domain", "item", "pearson_r_t1_t2", "icc3_1_t1_t2"]]

    param = pd.read_csv(PARAM_COMP_PATH).copy()
    param["domain"] = "Composite measures"
    param["item"] = param["axis_label"]
    param["dataset"] = param["dataset"]
    param = param.rename(columns={"pearson_r": "pearson_r_t1_t2", "icc3_1": "icc3_1_t1_t2"})
    param = param[["dataset", "domain", "item", "pearson_r_t1_t2", "icc3_1_t1_t2"]]

    return pd.concat([beh, param], ignore_index=True)


def build_summary() -> pd.DataFrame:
    beh = load_behavior()
    orig = load_parameters(ORIG_PARAM_PATH, "Original")
    redep = load_parameters(REDEP_PARAM_PATH, "Redeployed")
    comps = load_composites()
    all_df = pd.concat([beh, orig, redep, comps], ignore_index=True)
    domain_order = {
        "Behavior": 0,
        "Observation parameters": 1,
        "Perceptual parameters": 2,
        "Composite measures": 3,
    }
    all_df["domain_order"] = all_df["domain"].map(domain_order)
    sort_metric = all_df.groupby(["domain", "item"])["pearson_r_t1_t2"].mean().rename("sort_metric").reset_index()
    all_df = all_df.merge(sort_metric, on=["domain", "item"], how="left")
    all_df = all_df.sort_values(["domain_order", "sort_metric"], ascending=[True, False]).reset_index(drop=True)
    return all_df


def plot_summary(df: pd.DataFrame) -> None:
    colors = {"Original": "#4C78A8", "Redeployed": "#F58518"}
    domain_fill = {
        "Behavior": "#eef6ff",
        "Observation parameters": "#fff3e8",
        "Perceptual parameters": "#eef7ef",
        "Composite measures": "#f7efff",
    }

    item_order = df[["domain", "item", "domain_order", "sort_metric"]].drop_duplicates()
    item_order = item_order.sort_values(["domain_order", "sort_metric"], ascending=[True, False]).reset_index(drop=True)
    item_order["y"] = np.arange(len(item_order))
    item_order["display_item"] = item_order["item"].map(DISPLAY_LABELS).fillna(item_order["item"])

    merged = df.merge(item_order[["domain", "item", "y"]], on=["domain", "item"], how="left")

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 12.5), sharey=True)
    metrics = [("pearson_r_t1_t2", "Pearson r"), ("icc3_1_t1_t2", "ICC(3,1)")]

    domain_ranges = (
        item_order.groupby("domain")
        .agg(ymin=("y", "min"), ymax=("y", "max"))
        .reset_index()
    )

    for ax, (metric, title) in zip(axes, metrics):
        for _, row in domain_ranges.iterrows():
            ax.axhspan(row["ymin"] - 0.5, row["ymax"] + 0.5, color=domain_fill[row["domain"]], zorder=0)

        for dataset, offset in [("Original", -0.12), ("Redeployed", 0.12)]:
            sub = merged.loc[merged["dataset"] == dataset].copy()
            ax.scatter(
                sub[metric],
                sub["y"] + offset,
                s=58,
                color=colors[dataset],
                alpha=0.95,
                edgecolor="white",
                linewidth=0.6,
                label=dataset,
                zorder=3,
            )

        ax.axvline(0.4, color="#8f8f8f", linestyle="--", linewidth=1)
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlim(-0.05, 0.85)
        ax.set_xlabel("Reliability")
        ax.set_title(title, fontsize=13)

    axes[0].set_yticks(item_order["y"])
    axes[0].set_yticklabels(item_order["display_item"], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", pad=6)

    for _, row in domain_ranges.iterrows():
        center = (row["ymin"] + row["ymax"]) / 2
        axes[0].text(
            -0.28,
            center,
            row["domain"],
            transform=axes[0].get_yaxis_transform(),
            rotation=90,
            va="center",
            ha="right",
            fontsize=11,
            fontweight="medium",
            color="#555555",
        )

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Original"], markersize=8, label="Original"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Redeployed"], markersize=8, label="Redeployed"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Behavioral and parameter test-retest reliability: original vs redeployed", fontsize=16, y=0.985)
    fig.tight_layout(rect=[0.23, 0.04, 1, 0.96])
    fig.savefig(OUTDIR / "figure_behavior_parameter_reliability_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = build_summary()
    df.to_csv(OUTDIR / "behavior_parameter_reliability_comparison.csv", index=False)
    plot_summary(df)


if __name__ == "__main__":
    main()
