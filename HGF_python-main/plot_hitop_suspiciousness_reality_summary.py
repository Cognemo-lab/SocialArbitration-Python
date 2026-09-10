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
RD_SUMMARY = BASE / "hitop" / "results_reality_distortion" / "model_summary.csv"


SUSP_LABELS = {
    "inferv_a_mean": "Inferential variance",
    "om_a": "Omega-advice",
    "abs_eps2_a_mean": "|epsilon2|",
    "abs_eps3_a_mean": "|epsilon3|",
    "eps3_a_mean": "epsilon3",
}

RD_LABELS = {
    "be_wager_z": "Overall: wager noise",
    "be_ch_z": "Overall: choice noise",
    "pred_pc1_z": "Composite predictor PC1",
}

COLORS = {
    "neg": "#1f5aa6",
    "pos": "#b14a3a",
    "pooled": "#244b8f",
    "t2": "#2c8c69",
    "ns": "#c7c7c7",
}


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


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
    susp_assoc["label"] = susp_assoc["quantity_name"].map(SUSP_LABELS)
    order = ["inferv_a_mean", "om_a", "abs_eps2_a_mean", "abs_eps3_a_mean", "eps3_a_mean"]
    susp_assoc["order"] = susp_assoc["quantity_name"].map({name: i for i, name in enumerate(order)})
    susp_assoc = susp_assoc.sort_values("order")

    susp_pca = pd.read_csv(SUSP_PCA)
    susp_pca["analysis_label"] = susp_pca["analysis_scope"].map(
        {"pooled_clustered": "Pooled PCA", "t2": "T2 PCA"}
    )

    rd_coef = pd.read_csv(RD_COEF)
    rd_summary = pd.read_csv(RD_SUMMARY)

    rd_raw = rd_coef[rd_coef["analysis_scope"] == "raw_predictors"].copy()
    rd_raw = rd_raw[rd_raw["term"].isin(["be_wager_z", "be_ch_z"])].copy()
    rd_raw["label"] = rd_raw["term"].map(
        {"be_wager_z": "be_wager", "be_ch_z": "be_ch"}
    )

    pred_pc1 = rd_coef[
        (rd_coef["analysis_scope"] == "predictor_pc1") & (rd_coef["term"] == "pred_pc1_z")
    ].iloc[0]
    raw_r2 = float(rd_summary.loc[rd_summary["model"] == "raw_predictors", "r_squared"].iloc[0])
    pc1_r2 = float(rd_summary.loc[rd_summary["model"] == "predictor_pc1", "r_squared"].iloc[0])

    reality_rows = pd.DataFrame(
        [
            {
                "label": "Overall RD: be_wager",
                "beta": -0.13305747897218007,
                "se": 0.04049592100495278,
                "p": 0.0010172901447605728,
                "r2": 0.018772808838260158,
            },
            {
                "label": "Delusions: be_ch",
                "beta": 0.12619183366980968,
                "se": 0.09330449219073457,
                "p": 0.1762239572260237,
                "r2": 0.015988555239458635,
            },
            {
                "label": "Hallucinations: be_wager",
                "beta": -0.12779874018957096,
                "se": 0.040798029124567224,
                "p": 0.0017334014550820487,
                "r2": 0.018416875642927488,
            },
        ]
    )

    fig = plt.figure(figsize=(14, 8.8))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    y = np.arange(len(susp_assoc))
    vals = susp_assoc["pearson_r"].to_numpy()
    cols = [COLORS["neg"] if v < 0 else COLORS["pos"] for v in vals]
    ax1.barh(y, vals, color=cols, alpha=0.95)
    ax1.axvline(0, color="0.35", lw=1)
    ax1.set_yticks(y)
    ax1.set_yticklabels(susp_assoc["label"], fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlabel("Pearson r")
    ax1.set_title("A. Suspiciousness: pooled predictor-level effects", loc="left", fontsize=13, fontweight="bold")
    ax1.set_xlim(-0.24, 0.24)
    for i, row in enumerate(susp_assoc.itertuples(index=False)):
        ha = "right" if row.pearson_r < 0 else "left"
        x = row.pearson_r - 0.008 if row.pearson_r < 0 else row.pearson_r + 0.008
        ax1.text(
            x,
            i,
            f"r={row.pearson_r:.3f}{stars(row.pearson_p)}",
            va="center",
            ha=ha,
            fontsize=9,
        )

    ax2 = fig.add_subplot(gs[0, 1])
    y2 = np.arange(len(susp_pca))
    colors2 = [COLORS["pooled"], COLORS["t2"]]
    ax2.barh(y2, susp_pca["beta"], color=colors2, alpha=0.95)
    ax2.errorbar(
        susp_pca["beta"],
        y2,
        xerr=1.96 * susp_pca["se"],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
    )
    ax2.axvline(0, color="0.35", lw=1)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(susp_pca["analysis_label"], fontsize=11)
    ax2.invert_yaxis()
    ax2.set_xlabel("Standardized beta")
    ax2.set_title("B. Suspiciousness: PCA composite effects", loc="left", fontsize=13, fontweight="bold")
    lim2 = max(float(np.max(np.abs(susp_pca["beta"]) + 1.96 * susp_pca["se"])) * 1.25, 0.3)
    ax2.set_xlim(-lim2, lim2)
    for i, row in enumerate(susp_pca.itertuples(index=False)):
        x = row.beta - 0.01 if row.beta < 0 else row.beta + 0.01
        ha = "right" if row.beta < 0 else "left"
        ax2.text(
            x,
            i,
            f"p={row.p_value:.3g}{stars(row.p_value)}\nR²={row.r_squared:.3f}",
            va="center",
            ha=ha,
            fontsize=9,
        )

    ax3 = fig.add_subplot(gs[1, 0])
    y3 = np.arange(len(reality_rows))
    cols3 = [COLORS["neg"] if v < 0 else COLORS["pos"] for v in reality_rows["beta"]]
    ax3.barh(y3, reality_rows["beta"], color=cols3, alpha=0.95)
    ax3.errorbar(
        reality_rows["beta"],
        y3,
        xerr=1.96 * reality_rows["se"],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
    )
    ax3.axvline(0, color="0.35", lw=1)
    ax3.set_yticks(y3)
    ax3.set_yticklabels(reality_rows["label"], fontsize=11)
    ax3.invert_yaxis()
    ax3.set_xlabel("Standardized beta")
    ax3.set_title("C. Reality distortion: individual pooled effects", loc="left", fontsize=13, fontweight="bold")
    ax3.set_xlim(-0.34, 0.34)
    for i, row in enumerate(reality_rows.itertuples(index=False)):
        x = row.beta - 0.01 if row.beta < 0 else row.beta + 0.01
        ha = "right" if row.beta < 0 else "left"
        ax3.text(x, i, f"p={row.p:.3g}{stars(row.p)}\nR²={row.r2:.3f}", va="center", ha=ha, fontsize=9)

    ax4 = fig.add_subplot(gs[1, 1])
    rd_plot = pd.DataFrame(
        [
            {
                "label": "be_wager",
                "beta": float(rd_raw.loc[rd_raw["term"] == "be_wager_z", "coef"].iloc[0]),
                "se": float(rd_raw.loc[rd_raw["term"] == "be_wager_z", "se"].iloc[0]),
                "p": float(rd_raw.loc[rd_raw["term"] == "be_wager_z", "p_value"].iloc[0]),
                "r2": raw_r2,
            },
            {
                "label": "be_ch",
                "beta": float(rd_raw.loc[rd_raw["term"] == "be_ch_z", "coef"].iloc[0]),
                "se": float(rd_raw.loc[rd_raw["term"] == "be_ch_z", "se"].iloc[0]),
                "p": float(rd_raw.loc[rd_raw["term"] == "be_ch_z", "p_value"].iloc[0]),
                "r2": raw_r2,
            },
            {
                "label": "Predictor PC1",
                "beta": float(pred_pc1["coef"]),
                "se": float(pred_pc1["se"]),
                "p": float(pred_pc1["p_value"]),
                "r2": pc1_r2,
            },
        ]
    )
    y4 = np.arange(len(rd_plot))
    cols4 = [COLORS["neg"], COLORS["pos"], "#3a4c5f"]
    ax4.barh(y4, rd_plot["beta"], color=cols4, alpha=0.95)
    ax4.errorbar(
        rd_plot["beta"],
        y4,
        xerr=1.96 * rd_plot["se"],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
    )
    ax4.axvline(0, color="0.35", lw=1)
    ax4.set_yticks(y4)
    ax4.set_yticklabels(rd_plot["label"], fontsize=11)
    ax4.invert_yaxis()
    ax4.set_xlabel("Standardized beta")
    ax4.set_title("D. Reality distortion: raw vs PCA model", loc="left", fontsize=13, fontweight="bold")
    ax4.set_xlim(-0.34, 0.34)
    for i, row in enumerate(rd_plot.itertuples(index=False)):
        x = row.beta - 0.01 if row.beta < 0 else row.beta + 0.01
        ha = "right" if row.beta < 0 else "left"
        ax4.text(x, i, f"p={row.p:.3g}{stars(row.p)}\nR²={row.r2:.3f}", va="center", ha=ha, fontsize=9)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.16)

    fig.suptitle("HiTOP Suspiciousness and Reality Distortion Effects", fontsize=18, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        "Negative effects indicate higher symptom scores with lower model-derived values.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    out_png = OUT_DIR / "figure_hitop_suspiciousness_reality_summary.png"
    out_svg = OUT_DIR / "figure_hitop_suspiciousness_reality_summary.svg"
    fig.savefig(out_png, dpi=240)
    fig.savefig(out_svg)
    plt.close(fig)

    print(out_png)
    print(out_svg)


if __name__ == "__main__":
    main()
