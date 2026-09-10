from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
    "inferv_a_mean": "Inferential variance",
    "om_a": "Omega-advice",
    "abs_eps2_a_mean": "|epsilon2|",
    "abs_eps3_a_mean": "|epsilon3|",
    "eps3_a_mean": "epsilon3",
    "ka_a": "Kappa-advice",
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


def _panel_summary_text(pca_df: pd.DataFrame, joint_df: pd.DataFrame, outcome: str) -> str:
    lines: list[str] = []
    pooled = pca_df[(pca_df["outcome"] == outcome) & (pca_df["analysis_scope"] == "pooled_clustered")]
    if not pooled.empty:
        row = pooled.iloc[0]
        lines.append(
            f"Pooled: β={row['beta_pc1']:.3f}{_stars(row['p_pc1'])}, "
            f"p={row['p_pc1']:.3g}, R²={row['r_squared']:.3f}"
        )
        joint = joint_df[(joint_df["outcome"] == outcome) & (joint_df["analysis_scope"] == "pooled_clustered")]
        if not joint.empty:
            lines.append(f"Joint Wald p={joint.iloc[0]['p_value']:.3g}")
    t2 = pca_df[(pca_df["outcome"] == outcome) & (pca_df["analysis_scope"] == "t2")]
    if not t2.empty:
        row = t2.iloc[0]
        lines.append(
            f"T2: β={row['beta_pc1']:.3f}{_stars(row['p_pc1'])}, "
            f"p={row['p_pc1']:.3g}, R²={row['r_squared']:.3f}"
        )
        joint = joint_df[(joint_df["outcome"] == outcome) & (joint_df["analysis_scope"] == "t2")]
        if not joint.empty:
            lines.append(f"T2 joint Wald p={joint.iloc[0]['p_value']:.3g}")
    return "\n".join(lines)


def main() -> None:
    selected = pd.read_csv(IN_DIR / "selected_predictors_by_outcome.csv")
    pca = pd.read_csv(IN_DIR / "pca_model_summary.csv")
    loadings = pd.read_csv(IN_DIR / "pca_loadings.csv")
    joint = pd.read_csv(IN_DIR / "multivariate_joint_wald_tests.csv")

    outcomes = list(OUTCOME_LABELS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0))
    axes = axes.ravel()

    for idx, outcome in enumerate(outcomes):
        ax = axes[idx]
        pooled_load = loadings[
            (loadings["outcome"] == outcome) & (loadings["analysis_scope"] == "pooled_clustered")
        ].copy()
        pooled_sel = selected[
            (selected["outcome"] == outcome) & (selected["analysis_scope"] == "pooled_t1_t2")
        ].copy()
        t2_sel = selected[
            (selected["outcome"] == outcome) & (selected["analysis_scope"] == "t2")
        ].copy()

        if pooled_load.empty:
            ax.axis("off")
            continue

        pooled_load["label"] = pooled_load["predictor"].map(PREDICTOR_LABELS).fillna(pooled_load["predictor"])
        pooled_load = pooled_load.sort_values("pc1_loading")
        colors = ["#2f7f77" if v >= 0 else "#bf6a4a" for v in pooled_load["pc1_loading"]]

        ax.barh(pooled_load["label"], pooled_load["pc1_loading"], color=colors, alpha=0.95, height=0.8)
        ax.axvline(0, color="#666666", lw=1.1)
        ax.set_xlabel("PC1 loading", fontsize=11)
        ax.set_title(
            f"{chr(65 + idx)}. {OUTCOME_LABELS[outcome]}",
            loc="left",
            fontsize=14,
            fontweight="bold",
            pad=8,
        )
        ev = pooled_load["explained_variance_ratio"].iloc[0]
        ax.text(
            0.01,
            1.01,
            f"Pooled predictor composite, PC1 variance = {ev:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="#555555",
        )
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="x", alpha=0.18, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Labels at bar ends
        xpad = 0.015
        for row in pooled_load.itertuples(index=False):
            ha = "left" if row.pc1_loading >= 0 else "right"
            xpos = row.pc1_loading + xpad if row.pc1_loading >= 0 else row.pc1_loading - xpad
            ax.text(xpos, row.label, f"{row.pc1_loading:.2f}", va="center", ha=ha, fontsize=9, color="#333333")

        predictor_text = ", ".join(PREDICTOR_LABELS.get(p, p) for p in pooled_sel["predictor"].tolist())
        stats_text = _panel_summary_text(pca, joint, outcome)
        bottom_text = f"Predictors: {predictor_text}"
        if not t2_sel.empty:
            bottom_text += "\nT2 predictors: " + ", ".join(PREDICTOR_LABELS.get(p, p) for p in t2_sel["predictor"].tolist())
        bottom_text += "\n" + stats_text
        ax.text(
            0.02,
            0.02,
            bottom_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.8,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#dddddd", alpha=0.96),
        )

    fig.suptitle("Harmonized HiTOP Domain Analyses", fontsize=18, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.2, w_pad=2.0)

    out_png = OUT_DIR / "figure_hitop_consistent_domains_clear.png"
    out_svg = OUT_DIR / "figure_hitop_consistent_domains_clear.svg"
    fig.savefig(out_png, dpi=260)
    fig.savefig(out_svg)
    plt.close(fig)

    print(out_png)
    print(out_svg)


if __name__ == "__main__":
    main()
