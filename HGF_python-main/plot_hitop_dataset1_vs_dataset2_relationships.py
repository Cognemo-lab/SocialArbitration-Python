from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
ORIG_ASSOC = BASE / "hitop" / "model_associations_reliable" / "hitop_reliable_quantity_associations.csv"
BATCH2_ASSOC = BASE / "hitop_batch2" / "model_associations_reliable" / "hitop_reliable_quantity_associations.csv"
OUT_DIR = BASE / "hitop_cross_dataset_comparison"

OUTCOME_CONFIG = {
    "hitop_mistrust_suspiciousness": {
        "title": "A. Suspiciousness",
        "predictors": ["inferv_a_mean", "om_a", "abs_eps2_a_mean", "abs_eps3_a_mean", "eps3_a_mean"],
    },
    "hitop_reality_distortion": {
        "title": "B. Overall Reality Distortion",
        "predictors": ["be_wager", "be_ch"],
    },
    "hitop_reality_distortion_delusions": {
        "title": "C. Delusions",
        "predictors": ["be_ch", "be_wager"],
    },
}

LABELS = {
    "inferv_a_mean": "Advice inferential variance",
    "om_a": "Omega-Advice",
    "abs_eps2_a_mean": "Abs advice epsilon2",
    "abs_eps3_a_mean": "Abs advice epsilon3",
    "eps3_a_mean": "Advice epsilon3",
    "be_wager": "Wager noise",
    "be_ch": "Choice noise",
}

COLORS = {"Dataset 1": "#4C78A8", "Dataset 2": "#F58518"}


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if pd.isna(r) or n <= 3 or abs(r) >= 1:
        return np.nan, np.nan
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = 1.959963984540054
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return float(lo), float(hi)


def fdr_star(q: float) -> str:
    if pd.isna(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def load_effects() -> pd.DataFrame:
    orig = pd.read_csv(ORIG_ASSOC)
    orig = orig[orig["analysis_scope"] == "t1"].copy()
    orig["dataset"] = "Dataset 1"
    orig["scope_name"] = "t1"

    batch2 = pd.read_csv(BATCH2_ASSOC)
    batch2 = batch2[batch2["analysis_scope"] == "t1_only"].copy()
    batch2["dataset"] = "Dataset 2"
    batch2["scope_name"] = "t1"

    keep_rows = []
    for outcome, cfg in OUTCOME_CONFIG.items():
        wanted = set(cfg["predictors"])
        for df in [orig, batch2]:
            sub = df[(df["hitop_measure"] == outcome) & (df["quantity_name"].isin(wanted))].copy()
            keep_rows.append(sub)

    comp = pd.concat(keep_rows, ignore_index=True)
    comp["outcome_title"] = comp["hitop_measure"].map({k: v["title"] for k, v in OUTCOME_CONFIG.items()})
    comp["predictor_label"] = comp["quantity_name"].map(LABELS)
    ci = comp.apply(lambda row: fisher_ci(row["pearson_r"], int(row["n"])), axis=1, result_type="expand")
    comp["ci_low"] = ci[0]
    comp["ci_high"] = ci[1]
    comp["star"] = comp["fdr_q_pearson_within_scope"].map(fdr_star)
    return comp


def make_figure(comp: pd.DataFrame) -> Path:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True)
    outcomes = list(OUTCOME_CONFIG.keys())

    for ax, outcome in zip(axes, outcomes):
        cfg = OUTCOME_CONFIG[outcome]
        sub = comp[comp["hitop_measure"] == outcome].copy()
        pred_order = [LABELS[p] for p in cfg["predictors"]]
        y_positions = np.arange(len(pred_order))
        y_map = {label: pos for pos, label in enumerate(pred_order)}
        offsets = {"Dataset 1": -0.12, "Dataset 2": 0.12}

        for dataset in ["Dataset 1", "Dataset 2"]:
            dsub = sub[sub["dataset"] == dataset].copy()
            dsub["y"] = dsub["predictor_label"].map(y_map) + offsets[dataset]
            ax.hlines(dsub["y"], dsub["ci_low"], dsub["ci_high"], color=COLORS[dataset], linewidth=2, alpha=0.9)
            ax.scatter(dsub["pearson_r"], dsub["y"], color=COLORS[dataset], s=52, edgecolor="white", linewidth=0.8, zorder=3, label=dataset)
            for _, row in dsub.iterrows():
                if row["star"]:
                    ax.text(row["pearson_r"] + (0.02 if row["pearson_r"] >= 0 else -0.02), row["y"], row["star"], va="center", ha="left" if row["pearson_r"] >= 0 else "right", fontsize=10, color=COLORS[dataset])

        ax.axvline(0.0, color="black", linewidth=0.9)
        ax.set_title(cfg["title"], fontsize=12)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(pred_order, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(-0.35, 0.2)
        ax.set_xlabel("Pearson r")
        ax.grid(axis="y", visible=False)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Cross-Dataset Comparison of HiTOP Associations at T1", fontsize=15, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    out = OUT_DIR / "figure_hitop_dataset1_vs_dataset2_relationships.png"
    fig.savefig(out, dpi=260)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comp = load_effects()
    comp.to_csv(OUT_DIR / "hitop_dataset1_vs_dataset2_relationships.csv", index=False)
    make_figure(comp)


if __name__ == "__main__":
    main()
