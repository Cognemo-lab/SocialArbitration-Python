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
OUTCOME_ORDER = list(OUTCOME_LABELS.keys())

PREDICTOR_LABELS = {
    "inferv_a_mean_z": "Inferential variance",
    "om_a_z": "Omega-advice",
    "abs_eps2_a_mean_z": "|epsilon2|",
    "eps3_a_mean_z": "epsilon3",
    "ka_a_z": "Kappa-advice",
    "be_wager_z": "Wager noise",
    "be_ch_z": "Choice noise",
}

COLORS = {
    "neg": "#355f9c",
    "pos": "#b55a4a",
    "neutral": "#5f6f82",
    "accent": "#2f7f77",
    "grid": "#d9d9d9",
}


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], alpha=0.35, linewidth=0.8)
    ax.axvline(0, color="#666666", lw=1)
    ax.tick_params(labelsize=10)


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "coef": pd.read_csv(IN_DIR / "multivariate_coefficients.csv"),
        "joint": pd.read_csv(IN_DIR / "multivariate_joint_wald_tests.csv"),
        "model": pd.read_csv(IN_DIR / "multivariate_model_summary.csv"),
        "pca": pd.read_csv(IN_DIR / "pca_model_summary.csv"),
    }


def pooled_model_info(model_df: pd.DataFrame, outcome: str) -> pd.Series:
    return model_df[(model_df["outcome"] == outcome) & (model_df["analysis_scope"] == "pooled_clustered")].iloc[0]


def pooled_joint_info(joint_df: pd.DataFrame, outcome: str) -> pd.Series:
    return joint_df[(joint_df["outcome"] == outcome) & (joint_df["analysis_scope"] == "pooled_clustered")].iloc[0]


def pooled_coef_rows(coef_df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    sub = coef_df[(coef_df["outcome"] == outcome) & (coef_df["analysis_scope"] == "pooled_clustered")].copy()
    sub = sub[~sub["term"].isin(["Intercept", "C(timepoint)[T.t2]"])].copy()
    sub["label"] = sub["term"].map(PREDICTOR_LABELS).fillna(sub["term"])
    return sub


def make_coefficient_only_figure(data: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    coef_df = data["coef"]
    joint_df = data["joint"]
    model_df = data["model"]

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0))
    axes = axes.ravel()

    for ax, outcome in zip(axes, OUTCOME_ORDER):
        sub = pooled_coef_rows(coef_df, outcome).sort_values("coef")
        model_row = pooled_model_info(model_df, outcome)
        joint_row = pooled_joint_info(joint_df, outcome)

        y = np.arange(len(sub))
        colors = [COLORS["neg"] if v < 0 else COLORS["pos"] for v in sub["coef"]]
        ax.barh(y, sub["coef"], color=colors, height=0.76)
        ax.errorbar(
            sub["coef"],
            y,
            xerr=1.96 * sub["se"],
            fmt="none",
            ecolor="black",
            elinewidth=1.1,
            capsize=2.5,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"])
        ax.set_xlabel("Standardized beta", fontsize=11)
        ax.set_title(
            f"{OUTCOME_LABELS[outcome]}\nJoint Wald p={joint_row['p_value']:.3g}, R²={model_row['r_squared']:.3f}",
            loc="left",
            fontsize=13,
            fontweight="bold",
            pad=8,
        )
        lim = max(0.25, float(np.max(np.abs(sub["coef"]) + 1.96 * sub["se"])) * 1.2)
        ax.set_xlim(-lim, lim)
        for i, row in enumerate(sub.itertuples(index=False)):
            x = row.coef - 0.01 if row.coef < 0 else row.coef + 0.01
            ha = "right" if row.coef < 0 else "left"
            ax.text(x, i, f"{row.coef:.2f}{stars(row.p_value)}", va="center", ha=ha, fontsize=9)
        style_axis(ax)

    fig.suptitle("HiTOP domain analyses: pooled multivariate coefficients", fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02, "* p < .05, ** p < .01, *** p < .001", ha="center", fontsize=9, color="#666666")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    out_png = OUT_DIR / "figure_hitop_consistent_domains_coefficients_only.png"
    out_svg = OUT_DIR / "figure_hitop_consistent_domains_coefficients_only.svg"
    fig.savefig(out_png, dpi=260)
    fig.savefig(out_svg)
    plt.close(fig)
    return out_png, out_svg


def make_pca_only_figure(data: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    pca_df = data["pca"]

    pooled = pca_df[pca_df["analysis_scope"] == "pooled_clustered"].copy()
    pooled["outcome_label"] = pooled["outcome"].map(OUTCOME_LABELS)
    pooled = pooled.set_index("outcome").loc[OUTCOME_ORDER].reset_index()

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    y = np.arange(len(pooled))
    colors = [COLORS["neg"] if v < 0 else COLORS["pos"] for v in pooled["beta_pc1"]]
    ax.barh(y, pooled["beta_pc1"], color=colors, height=0.74)
    ax.errorbar(
        pooled["beta_pc1"],
        y,
        xerr=1.96 * pooled["se_pc1"],
        fmt="none",
        ecolor="black",
        elinewidth=1.1,
        capsize=2.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(pooled["outcome_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Standardized beta for pooled predictor PC1", fontsize=11)
    ax.set_title("HiTOP domain analyses: pooled PCA-composite effects", loc="left", fontsize=15, fontweight="bold", pad=8)
    lim = max(0.25, float(np.max(np.abs(pooled["beta_pc1"]) + 1.96 * pooled["se_pc1"])) * 1.2)
    ax.set_xlim(-lim, lim)
    for i, row in enumerate(pooled.itertuples(index=False)):
        x = row.beta_pc1 - 0.012 if row.beta_pc1 < 0 else row.beta_pc1 + 0.012
        ha = "right" if row.beta_pc1 < 0 else "left"
        ax.text(
            x,
            i,
            f"β={row.beta_pc1:.2f}{stars(row.p_pc1)}\nR²={row.r_squared:.3f}",
            va="center",
            ha=ha,
            fontsize=9,
        )
    style_axis(ax)
    fig.text(0.5, 0.02, "* p < .05, ** p < .01, *** p < .001", ha="center", fontsize=9, color="#666666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_png = OUT_DIR / "figure_hitop_consistent_domains_pca_only.png"
    out_svg = OUT_DIR / "figure_hitop_consistent_domains_pca_only.svg"
    fig.savefig(out_png, dpi=260)
    fig.savefig(out_svg)
    plt.close(fig)
    return out_png, out_svg


def make_retained_multivariate_figure(data: dict[str, pd.DataFrame]) -> tuple[Path, Path]:
    coef_df = data["coef"]
    joint_df = data["joint"]
    model_df = data["model"]

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4))
    axes = axes.ravel()

    for ax, outcome in zip(axes, OUTCOME_ORDER):
        sub = pooled_coef_rows(coef_df, outcome)
        retained = sub[sub["p_value"] < 0.05].sort_values("coef").copy()
        model_row = pooled_model_info(model_df, outcome)
        joint_row = pooled_joint_info(joint_df, outcome)

        ax.set_title(
            f"{OUTCOME_LABELS[outcome]}\nJoint Wald p={joint_row['p_value']:.3g}, R²={model_row['r_squared']:.3f}",
            loc="left",
            fontsize=13,
            fontweight="bold",
            pad=8,
        )

        if retained.empty:
            ax.text(
                0.5,
                0.5,
                "No retained pooled effect",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="#666666",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(False)
            continue

        y = np.arange(len(retained))
        colors = [COLORS["neg"] if v < 0 else COLORS["pos"] for v in retained["coef"]]
        ax.barh(y, retained["coef"], color=colors, height=0.76)
        ax.errorbar(
            retained["coef"],
            y,
            xerr=1.96 * retained["se"],
            fmt="none",
            ecolor="black",
            elinewidth=1.1,
            capsize=2.5,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(retained["label"])
        ax.set_xlabel("Standardized beta", fontsize=11)
        lim = max(0.25, float(np.max(np.abs(retained["coef"]) + 1.96 * retained["se"])) * 1.25)
        ax.set_xlim(-lim, lim)
        for i, row in enumerate(retained.itertuples(index=False)):
            x = row.coef - 0.01 if row.coef < 0 else row.coef + 0.01
            ha = "right" if row.coef < 0 else "left"
            ax.text(x, i, f"β={row.coef:.2f}{stars(row.p_value)}", va="center", ha=ha, fontsize=9)
        style_axis(ax)

    fig.suptitle("HiTOP domain analyses: retained pooled multivariate effects", fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02, "* p < .05, ** p < .01, *** p < .001", ha="center", fontsize=9, color="#666666")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    out_png = OUT_DIR / "figure_hitop_consistent_domains_retained_multivariate.png"
    out_svg = OUT_DIR / "figure_hitop_consistent_domains_retained_multivariate.svg"
    fig.savefig(out_png, dpi=260)
    fig.savefig(out_svg)
    plt.close(fig)
    return out_png, out_svg


def main() -> None:
    data = load_data()
    outs = []
    outs.extend(make_coefficient_only_figure(data))
    outs.extend(make_pca_only_figure(data))
    outs.extend(make_retained_multivariate_figure(data))
    for path in outs:
        print(path)


if __name__ == "__main__":
    main()
