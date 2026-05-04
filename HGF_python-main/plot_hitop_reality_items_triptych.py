from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_DIR = BASE / "hitop" / "consistent_domain_analysis"
OUT_DIR = IN_DIR

OUTCOMES = [
    "hitop_reality_distortion",
    "hitop_reality_distortion_delusions",
    "hitop_reality_distortion_hallucinations",
]

OUTCOME_LABELS = {
    "hitop_reality_distortion": "B. Reality distortion",
    "hitop_reality_distortion_delusions": "C. Delusions",
    "hitop_reality_distortion_hallucinations": "D. Hallucinations",
}

PREDICTOR_LABELS = {
    "be_wager": "Wager noise",
    "be_ch": "Choice noise",
}


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main() -> None:
    selected = pd.read_csv(IN_DIR / "selected_predictors_by_outcome.csv")
    pca = pd.read_csv(IN_DIR / "pca_model_summary.csv")
    loadings = pd.read_csv(IN_DIR / "pca_loadings.csv")
    joint = pd.read_csv(IN_DIR / "multivariate_joint_wald_tests.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))

    for ax, outcome in zip(axes, OUTCOMES):
        pooled_load = loadings[
            (loadings["outcome"] == outcome) & (loadings["analysis_scope"] == "pooled_clustered")
        ].copy()
        pooled_sel = selected[
            (selected["outcome"] == outcome) & (selected["analysis_scope"] == "pooled_t1_t2")
        ].copy()
        pooled_pca = pca[
            (pca["outcome"] == outcome) & (pca["analysis_scope"] == "pooled_clustered")
        ].iloc[0]
        pooled_joint = joint[
            (joint["outcome"] == outcome) & (joint["analysis_scope"] == "pooled_clustered")
        ].iloc[0]

        pooled_load["label"] = pooled_load["predictor"].map(PREDICTOR_LABELS).fillna(pooled_load["predictor"])
        pooled_load = pooled_load.sort_values("pc1_loading")
        colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in pooled_load["pc1_loading"]]

        ax.barh(pooled_load["label"], pooled_load["pc1_loading"], color=colors, alpha=0.9)
        ax.axvline(0, color="0.35", lw=1)
        ax.set_title(
            f"{OUTCOME_LABELS[outcome]}\nPC1 variance = {pooled_pca['explained_variance_ratio_pc1']:.3f}",
            fontsize=12,
        )
        ax.set_xlabel("PC1 loading / standardized effect")

        predictor_line = "Predictor: " + ", ".join(
            [PREDICTOR_LABELS.get(p, p) for p in pooled_sel["predictor"].tolist()]
        )
        summary_text = (
            f"{predictor_line}\n\n"
            f"beta = {pooled_pca['beta_pc1']:.3f}{_stars(pooled_pca['p_pc1'])}\n"
            f"p = {pooled_pca['p_pc1']:.3g}\n"
            f"R² = {pooled_pca['r_squared']:.3f}\n"
            f"Joint p = {pooled_joint['p_value']:.3g}"
        )
        ax.text(
            0.03,
            0.04,
            summary_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.8", alpha=0.95),
        )

    fig.suptitle("Reality Distortion Item Analyses", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.0)
    fig.savefig(OUT_DIR / "figure_hitop_reality_items_triptych.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
