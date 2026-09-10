from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
OUT_DIR = BASE / "hitop" / "paper_joint_summary"

SUSP_ASSOC = (
    BASE
    / "hitop"
    / "model_associations_reliable"
    / "hitop_reliable_quantity_associations_subject_mean_ranked.csv"
)
SUSP_PCA = BASE / "hitop" / "results_suspiciousness" / "pca_model_summary.csv"
RD_COEF = BASE / "hitop" / "results_reality_distortion" / "coefficients.csv"


BLUE = "#355f9c"
RED = "#b55a4a"
SLATE = "#4f6275"
GRID = "#d9d9d9"


def star(p: float) -> str:
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
    ax.grid(axis="x", color=GRID, alpha=0.35, linewidth=0.8)
    ax.axvline(0, color="#666666", lw=1)
    ax.tick_params(labelsize=10)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    susp_assoc = pd.read_csv(SUSP_ASSOC)
    susp_assoc = susp_assoc[
        (susp_assoc["analysis_scope"] == "subject_mean_t1_t2")
        & (susp_assoc["hitop_measure"] == "hitop_mistrust_suspiciousness")
        & (
            susp_assoc["quantity_name"].isin(
                ["inferv_a_mean", "om_a", "abs_eps2_a_mean", "abs_eps3_a_mean", "eps3_a_mean"]
            )
        )
    ].copy()
    label_map = {
        "inferv_a_mean": "Inferential variance",
        "om_a": "Omega-advice",
        "abs_eps2_a_mean": "|epsilon2|",
        "abs_eps3_a_mean": "|epsilon3|",
        "eps3_a_mean": "epsilon3",
    }
    order = ["inferv_a_mean", "om_a", "abs_eps2_a_mean", "abs_eps3_a_mean", "eps3_a_mean"]
    susp_assoc["label"] = susp_assoc["quantity_name"].map(label_map)
    susp_assoc["order"] = susp_assoc["quantity_name"].map({k: i for i, k in enumerate(order)})
    susp_assoc = susp_assoc.sort_values("order")

    susp_pca = pd.read_csv(SUSP_PCA)
    susp_pca["label"] = susp_pca["analysis_scope"].map({"pooled_clustered": "Pooled", "t2": "T2"})

    rd_coef = pd.read_csv(RD_COEF)
    rd_raw = rd_coef[rd_coef["analysis_scope"] == "raw_predictors"].copy()
    be_wager = rd_raw.loc[rd_raw["term"] == "be_wager_z"].iloc[0]
    be_ch = rd_raw.loc[rd_raw["term"] == "be_ch_z"].iloc[0]
    pred_pc1 = rd_coef[
        (rd_coef["analysis_scope"] == "predictor_pc1") & (rd_coef["term"] == "pred_pc1_z")
    ].iloc[0]

    reality_rows = pd.DataFrame(
        [
            ["Overall RD", -0.13305747897218007, 0.04049592100495278, 0.0010172901447605728],
            ["Delusions", 0.12619183366980968, 0.09330449219073457, 0.1762239572260237],
            ["Hallucinations", -0.12779874018957096, 0.040798029124567224, 0.0017334014550820487],
        ],
        columns=["label", "beta", "se", "p"],
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8))
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    y = np.arange(len(susp_assoc))
    vals = susp_assoc["pearson_r"].to_numpy()
    colors = [BLUE if v < 0 else RED for v in vals]
    ax.barh(y, vals, color=colors, height=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(susp_assoc["label"])
    ax.invert_yaxis()
    ax.set_xlim(-0.22, 0.22)
    ax.set_xlabel("Pearson r", fontsize=11)
    ax.set_title("A. Suspiciousness predictors", loc="left", fontsize=13, fontweight="bold")
    for i, row in enumerate(susp_assoc.itertuples(index=False)):
        x = row.pearson_r - 0.008 if row.pearson_r < 0 else row.pearson_r + 0.008
        ax.text(x, i, f"{row.pearson_r:.2f}{star(row.pearson_p)}", ha="right" if row.pearson_r < 0 else "left", va="center", fontsize=9)
    style_axis(ax)

    ax = axes[0, 1]
    y = np.arange(len(susp_pca))
    ax.barh(y, susp_pca["beta"], color=[BLUE, "#3f8f73"], height=0.78)
    ax.errorbar(
        susp_pca["beta"],
        y,
        xerr=1.96 * susp_pca["se"],
        fmt="none",
        ecolor="black",
        elinewidth=1.1,
        capsize=2.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(susp_pca["label"])
    ax.invert_yaxis()
    ax.set_xlim(-0.32, 0.12)
    ax.set_xlabel("Standardized beta", fontsize=11)
    ax.set_title("B. Suspiciousness composite", loc="left", fontsize=13, fontweight="bold")
    for i, row in enumerate(susp_pca.itertuples(index=False)):
        ax.text(row.beta - 0.012, i, f"{row.beta:.2f}{star(row.p_value)}", ha="right", va="center", fontsize=9)
    style_axis(ax)

    ax = axes[1, 0]
    y = np.arange(len(reality_rows))
    ax.barh(y, reality_rows["beta"], color=[BLUE if v < 0 else RED for v in reality_rows["beta"]], height=0.78)
    ax.errorbar(
        reality_rows["beta"],
        y,
        xerr=1.96 * reality_rows["se"],
        fmt="none",
        ecolor="black",
        elinewidth=1.1,
        capsize=2.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(reality_rows["label"])
    ax.invert_yaxis()
    ax.set_xlim(-0.28, 0.28)
    ax.set_xlabel("Standardized beta", fontsize=11)
    ax.set_title("C. Reality distortion outcomes", loc="left", fontsize=13, fontweight="bold")
    for i, row in enumerate(reality_rows.itertuples(index=False)):
        x = row.beta - 0.01 if row.beta < 0 else row.beta + 0.01
        ax.text(x, i, f"{row.beta:.2f}{star(row.p)}", ha="right" if row.beta < 0 else "left", va="center", fontsize=9)
    style_axis(ax)

    ax = axes[1, 1]
    rd_plot = pd.DataFrame(
        [
            ["be_wager", be_wager["coef"], be_wager["se"], be_wager["p_value"]],
            ["be_ch", be_ch["coef"], be_ch["se"], be_ch["p_value"]],
            ["Predictor PC1", pred_pc1["coef"], pred_pc1["se"], pred_pc1["p_value"]],
        ],
        columns=["label", "beta", "se", "p"],
    )
    y = np.arange(len(rd_plot))
    ax.barh(y, rd_plot["beta"], color=[BLUE, RED, SLATE], height=0.78)
    ax.errorbar(
        rd_plot["beta"],
        y,
        xerr=1.96 * rd_plot["se"],
        fmt="none",
        ecolor="black",
        elinewidth=1.1,
        capsize=2.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(rd_plot["label"])
    ax.invert_yaxis()
    ax.set_xlim(-0.28, 0.28)
    ax.set_xlabel("Standardized beta", fontsize=11)
    ax.set_title("D. Reality distortion model terms", loc="left", fontsize=13, fontweight="bold")
    for i, row in enumerate(rd_plot.itertuples(index=False)):
        x = row.beta - 0.01 if row.beta < 0 else row.beta + 0.01
        ax.text(x, i, f"{row.beta:.2f}{star(row.p)}", ha="right" if row.beta < 0 else "left", va="center", fontsize=9)
    style_axis(ax)

    fig.suptitle("HiTOP suspiciousness and reality distortion", fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02, "* p < .05, ** p < .01, *** p < .001", ha="center", fontsize=9, color="#666666")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    out_png = OUT_DIR / "figure_hitop_suspiciousness_reality_summary_minimal.png"
    out_svg = OUT_DIR / "figure_hitop_suspiciousness_reality_summary_minimal.svg"
    fig.savefig(out_png, dpi=240)
    fig.savefig(out_svg)
    plt.close(fig)

    print(out_png)
    print(out_svg)


if __name__ == "__main__":
    main()
