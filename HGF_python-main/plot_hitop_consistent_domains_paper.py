from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_DIR = BASE / "hitop" / "consistent_domain_analysis"
OUT_DIR = IN_DIR

OUTCOME_LABELS = {
    "hitop_mistrust_suspiciousness": "Suspiciousness",
    "hitop_reality_distortion": "Reality distortion",
    "hitop_reality_distortion_delusions": "Delusions",
    "hitop_reality_distortion_hallucinations": "Hallucinations",
}

PREDICTOR_LABELS = {
    "inferv_a_mean": "Advice inferential variance",
    "om_a": "Omega-Advice",
    "abs_eps2_a_mean": "Abs advice epsilon2",
    "abs_eps3_a_mean": "Abs advice epsilon3",
    "eps3_a_mean": "Advice epsilon3",
    "ka_a": "Kappa-Advice",
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


def _effect_text(pca_df: pd.DataFrame, joint_df: pd.DataFrame, outcome: str) -> str:
    parts = []
    pooled = pca_df[(pca_df["outcome"] == outcome) & (pca_df["analysis_scope"] == "pooled_clustered")]
    if not pooled.empty:
        row = pooled.iloc[0]
        parts.append(
            f"Pooled: beta={row['beta_pc1']:.3f}{_stars(row['p_pc1'])}, "
            f"p={row['p_pc1']:.3g}, R²={row['r_squared']:.3f}"
        )
        joint = joint_df[(joint_df["outcome"] == outcome) & (joint_df["analysis_scope"] == "pooled_clustered")]
        if not joint.empty:
            j = joint.iloc[0]
            parts.append(f"Joint test: p={j['p_value']:.3g}")

    t2 = pca_df[(pca_df["outcome"] == outcome) & (pca_df["analysis_scope"] == "t2")]
    if not t2.empty:
        row = t2.iloc[0]
        parts.append(
            f"T2: beta={row['beta_pc1']:.3f}{_stars(row['p_pc1'])}, "
            f"p={row['p_pc1']:.3g}, R²={row['r_squared']:.3f}"
        )
        joint = joint_df[(joint_df["outcome"] == outcome) & (joint_df["analysis_scope"] == "t2")]
        if not joint.empty:
            j = joint.iloc[0]
            parts.append(f"T2 joint: p={j['p_value']:.3g}")

    return "\n".join(parts)


def main() -> None:
    selected = pd.read_csv(IN_DIR / "selected_predictors_by_outcome.csv")
    pca = pd.read_csv(IN_DIR / "pca_model_summary.csv")
    loadings = pd.read_csv(IN_DIR / "pca_loadings.csv")
    joint = pd.read_csv(IN_DIR / "multivariate_joint_wald_tests.csv")

    outcomes = list(OUTCOME_LABELS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11))
    axes = axes.ravel()

    for idx, outcome in enumerate(outcomes):
        ax = axes[idx]
        pooled_sel = selected[
            (selected["outcome"] == outcome) & (selected["analysis_scope"] == "pooled_t1_t2")
        ].copy()
        t2_sel = selected[
            (selected["outcome"] == outcome) & (selected["analysis_scope"] == "t2")
        ].copy()

        pooled_load = loadings[
            (loadings["outcome"] == outcome) & (loadings["analysis_scope"] == "pooled_clustered")
        ].copy()

        if pooled_load.empty:
            ax.axis("off")
            continue

        pooled_load["label"] = pooled_load["predictor"].map(PREDICTOR_LABELS).fillna(pooled_load["predictor"])
        pooled_load = pooled_load.sort_values("pc1_loading")
        colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in pooled_load["pc1_loading"]]
        ax.barh(pooled_load["label"], pooled_load["pc1_loading"], color=colors, alpha=0.9)
        ax.axvline(0, color="0.35", lw=1)
        ax.set_xlabel("PC1 loading / standardized effect")
        ev = pooled_load["explained_variance_ratio"].iloc[0]
        ax.set_title(f"{chr(65 + idx)}. {OUTCOME_LABELS[outcome]}\nPooled predictor composite (PC1 var={ev:.3f})", fontsize=12)

        # Add right-side text summary in panel coordinates.
        text = _effect_text(pca, joint, outcome)
        predictor_line = "Predictors: " + ", ".join(
            [PREDICTOR_LABELS.get(p, p) for p in pooled_sel["predictor"].tolist()]
        )
        if not t2_sel.empty:
            predictor_line += "\nT2 predictors: " + ", ".join(
                [PREDICTOR_LABELS.get(p, p) for p in t2_sel["predictor"].tolist()]
            )
        ax.text(
            0.02,
            0.02,
            predictor_line + "\n\n" + text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.8", alpha=0.95),
        )

    fig.suptitle("Harmonized HiTOP Domain Analyses", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0, w_pad=2.0)
    fig.savefig(OUT_DIR / "figure_hitop_consistent_domains_paper.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
