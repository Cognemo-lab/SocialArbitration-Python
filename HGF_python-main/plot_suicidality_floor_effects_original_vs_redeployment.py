from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
OUT_DIR = BASE / "behavior_fod_moderation_comparison" / "state_and_prevalence_assessment"
OUT = OUT_DIR / "figure_suicidality_floor_effects_original_vs_redeployment.png"


def load_original() -> pd.DataFrame:
    t1 = pd.read_csv(BASE / "hitop" / "processed_data" / "hitop_scales_T1.csv")[["prolific_id", "hitop_suicidality"]].copy()
    t2 = pd.read_csv(BASE / "hitop" / "processed_data" / "hitop_scales_T2.csv")[["prolific_id", "hitop_suicidality"]].copy()
    t1["timepoint"] = "t1"
    t2["timepoint"] = "t2"
    df = pd.concat([t1, t2], ignore_index=True)
    df["dataset"] = "Original"
    return df


def load_redeployment() -> pd.DataFrame:
    t1 = pd.read_csv(BASE / "hitop_batch2" / "processed_data" / "psych_scales_T1_extended.csv")[["prolific_id", "hitop_suicidality"]].copy()
    t2 = pd.read_csv(BASE / "hitop_batch2" / "processed_data" / "psych_scales_T2_extended.csv")[["prolific_id", "hitop_suicidality"]].copy()
    t1["timepoint"] = "t1"
    t2["timepoint"] = "t2"
    df = pd.concat([t1, t2], ignore_index=True)
    df["dataset"] = "Redeployed"
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([load_original(), load_redeployment()], ignore_index=True)
    df["hitop_suicidality"] = pd.to_numeric(df["hitop_suicidality"], errors="coerce")
    df = df.dropna(subset=["hitop_suicidality"]).copy()
    df.to_csv(OUT_DIR / "suicidality_distribution_long.csv", index=False)

    summary = (
        df.groupby("dataset")["hitop_suicidality"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    floor = df.groupby("dataset").apply(lambda x: (x["hitop_suicidality"] == 4).mean()).rename("prop_floor_4").reset_index()
    summary = summary.merge(floor, on="dataset", how="left")
    summary.to_csv(OUT_DIR / "suicidality_distribution_summary.csv", index=False)

    palette = {"Original": "#4C78A8", "Redeployed": "#E76F51"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    score_levels = sorted(df["hitop_suicidality"].dropna().astype(int).unique())
    x = range(len(score_levels))
    orig_counts = (
        df.loc[df["dataset"] == "Original", "hitop_suicidality"]
        .astype(int)
        .value_counts()
        .reindex(score_levels, fill_value=0)
        .to_numpy()
    )
    red_counts = (
        df.loc[df["dataset"] == "Redeployed", "hitop_suicidality"]
        .astype(int)
        .value_counts()
        .reindex(score_levels, fill_value=0)
        .to_numpy()
    )
    width = 0.38
    ax.bar([i - width / 2 for i in x], orig_counts, width=width, color=palette["Original"], label="Original")
    ax.bar([i + width / 2 for i in x], red_counts, width=width, color=palette["Redeployed"], label="Redeployed")
    if 4 in score_levels:
        floor_idx = score_levels.index(4)
        ax.axvline(floor_idx, color="#666", linestyle="--", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(score_levels)
    ax.set_title("HiTOP suicidality distribution")
    ax.set_xlabel("HiTOP suicidality score")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)

    ax2 = axes[1]
    floor_df = summary.copy()
    ax2.bar(floor_df["dataset"], floor_df["prop_floor_4"], color=[palette[d] for d in floor_df["dataset"]])
    for i, row in floor_df.reset_index(drop=True).iterrows():
        ax2.text(i, row["prop_floor_4"] + 0.02, f"{row['prop_floor_4']*100:.1f}%", ha="center", va="bottom", fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Floor prevalence (score = 4)")
    ax2.set_xlabel("")
    ax2.set_ylabel("Proportion at floor")

    fig.suptitle("Suicidality floor effects: original vs redeployed sample", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
