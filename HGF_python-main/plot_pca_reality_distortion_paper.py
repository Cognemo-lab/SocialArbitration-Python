from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_DIR = BASE / "hitop" / "pca_reality_distortion"

LABELS = {
    "hitop_reality_distortion": "Overall reality distortion",
    "hitop_reality_distortion_delusions": "Delusions",
    "hitop_reality_distortion_hallucinations": "Hallucinations",
    "be_wager": "Wager slope (be_wager)",
    "be_ch": "Choice inverse temp. (be_ch)",
}


def main() -> None:
    out_load = pd.read_csv(IN_DIR / "reality_distortion_pca_loadings.csv")
    pred_load = pd.read_csv(IN_DIR / "predictor_pca_loadings.csv")
    coef = pd.read_csv(IN_DIR / "coefficients.csv")
    summary = pd.read_csv(IN_DIR / "model_summary.csv")

    out_load["label"] = out_load["variable"].map(LABELS).fillna(out_load["variable"])
    pred_load["label"] = pred_load["variable"].map(LABELS).fillna(pred_load["variable"])

    pred_pc1 = coef.loc[(coef["analysis_scope"] == "predictor_pc1") & (coef["term"] == "pred_pc1_z")].iloc[0]
    raw = coef.loc[(coef["analysis_scope"] == "raw_predictors") & (coef["term"].isin(["be_wager_z", "be_ch_z"]))].copy()
    raw["label"] = raw["term"].str.replace("_z", "", regex=False).map(LABELS)

    fig = plt.figure(figsize=(13, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.95, 1.1], wspace=0.45)

    ax1 = fig.add_subplot(gs[0, 0])
    out_plot = out_load.sort_values("pc1_loading")
    colors1 = ["#2a9d8f" if v >= 0 else "#e76f51" for v in out_plot["pc1_loading"]]
    ax1.barh(out_plot["label"], out_plot["pc1_loading"], color=colors1, alpha=0.9)
    ax1.axvline(0, color="0.3", lw=1)
    ax1.set_title(
        f"Reality Distortion PC1\nExplained variance = {out_load['explained_variance_ratio'].iloc[0]:.3f}"
    )
    ax1.set_xlabel("PC1 loading")

    ax2 = fig.add_subplot(gs[0, 1])
    pred_plot = pred_load.sort_values("pc1_loading")
    colors2 = ["#2a9d8f" if v >= 0 else "#e76f51" for v in pred_plot["pc1_loading"]]
    ax2.barh(pred_plot["label"], pred_plot["pc1_loading"], color=colors2, alpha=0.9)
    ax2.axvline(0, color="0.3", lw=1)
    ax2.set_title(
        f"Predictor PC1\nExplained variance = {pred_load['explained_variance_ratio'].iloc[0]:.3f}"
    )
    ax2.set_xlabel("PC1 loading")

    ax3 = fig.add_subplot(gs[0, 2])
    y = np.arange(3)
    betas = [raw.loc[raw["term"] == "be_wager_z", "coef"].iloc[0], raw.loc[raw["term"] == "be_ch_z", "coef"].iloc[0], pred_pc1["coef"]]
    ses = [raw.loc[raw["term"] == "be_wager_z", "se"].iloc[0], raw.loc[raw["term"] == "be_ch_z", "se"].iloc[0], pred_pc1["se"]]
    labels = ["be_wager", "be_ch", "Predictor PC1"]
    colors3 = ["#e76f51", "#2a9d8f", "#264653"]
    ax3.barh(y, betas, color=colors3, alpha=0.9)
    ax3.errorbar(betas, y, xerr=1.96 * np.array(ses), fmt="none", ecolor="black", elinewidth=1.2, capsize=3)
    ax3.axvline(0, color="0.3", lw=1)
    ax3.set_yticks(y)
    ax3.set_yticklabels(labels)
    ax3.set_xlabel("Standardized beta on reality distortion PC1")
    ax3.set_title("Regression Effects")
    lim = max(0.3, np.max(np.abs(np.array(betas) + 1.96 * np.array(ses)))) * 1.15
    ax3.set_xlim(-lim, lim)
    pvals = [
        raw.loc[raw["term"] == "be_wager_z", "p_value"].iloc[0],
        raw.loc[raw["term"] == "be_ch_z", "p_value"].iloc[0],
        pred_pc1["p_value"],
    ]
    r2_raw = summary.loc[summary["model"] == "raw_predictors", "r_squared"].iloc[0]
    r2_pc1 = summary.loc[summary["model"] == "predictor_pc1", "r_squared"].iloc[0]
    for i, (b, p) in enumerate(zip(betas, pvals)):
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        r2 = r2_pc1 if i == 2 else r2_raw
        xpos = b - 0.02 if b < 0 else b + 0.02
        ha = "right" if b < 0 else "left"
        ax3.text(xpos, i, f"p={p:.3g}{star}\nR²={r2:.3f}", va="center", ha=ha, fontsize=9)

    fig.suptitle("Reality Distortion PCA Summary", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(IN_DIR / "figure_pca_reality_distortion_paper.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
