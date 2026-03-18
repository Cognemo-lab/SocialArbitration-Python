from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_DIR = BASE / "hitop" / "pca_suspiciousness"

LABELS = {
    "inferv_a_mean": "Advice inferential variance",
    "om_a": "Advice volatility (om_a)",
    "abs_eps2_a_mean": "Absolute advice epsilon2",
    "abs_eps3_a_mean": "Absolute advice epsilon3",
    "eps3_a_mean": "Advice epsilon3",
    "ka_a": "Advice kappa (ka_a)",
}


def main() -> None:
    pooled_load = pd.read_csv(IN_DIR / "pooled_pca_loadings.csv")
    t2_load = pd.read_csv(IN_DIR / "t2_pca_loadings.csv")
    summary = pd.read_csv(IN_DIR / "pca_model_summary.csv")

    pooled_load["label"] = pooled_load["predictor"].map(LABELS).fillna(pooled_load["predictor"])
    t2_load["label"] = t2_load["predictor"].map(LABELS).fillna(t2_load["predictor"])

    pooled = summary.loc[summary["analysis_scope"] == "pooled_clustered"].iloc[0]
    t2 = summary.loc[summary["analysis_scope"] == "t2"].iloc[0]

    fig = plt.figure(figsize=(13, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.05, 0.95], wspace=0.45)

    ax1 = fig.add_subplot(gs[0, 0])
    pooled_plot = pooled_load.sort_values("pc1_loading")
    colors1 = ["#2a9d8f" if v >= 0 else "#e76f51" for v in pooled_plot["pc1_loading"]]
    ax1.barh(pooled_plot["label"], pooled_plot["pc1_loading"], color=colors1, alpha=0.9)
    ax1.axvline(0, color="0.3", lw=1)
    ax1.set_title(
        f"Pooled PC1 Loadings\nExplained variance = {pooled_load['explained_variance_ratio'].iloc[0]:.3f}"
    )
    ax1.set_xlabel("PC1 loading")

    ax2 = fig.add_subplot(gs[0, 1])
    t2_plot = t2_load.sort_values("pc1_loading")
    colors2 = ["#2a9d8f" if v >= 0 else "#e76f51" for v in t2_plot["pc1_loading"]]
    ax2.barh(t2_plot["label"], t2_plot["pc1_loading"], color=colors2, alpha=0.9)
    ax2.axvline(0, color="0.3", lw=1)
    ax2.set_title(
        f"T2 PC1 Loadings\nExplained variance = {t2_load['explained_variance_ratio'].iloc[0]:.3f}"
    )
    ax2.set_xlabel("PC1 loading")

    ax3 = fig.add_subplot(gs[0, 2])
    effects = pd.DataFrame(
        {
            "analysis": ["Pooled PC1", "T2 PC1"],
            "beta": [pooled["beta"], t2["beta"]],
            "se": [pooled["se"], t2["se"]],
            "p_value": [pooled["p_value"], t2["p_value"]],
            "r_squared": [pooled["r_squared"], t2["r_squared"]],
        }
    )
    y = np.arange(len(effects))
    ax3.barh(y, effects["beta"], color="#264653", alpha=0.9)
    ax3.errorbar(
        effects["beta"],
        y,
        xerr=1.96 * effects["se"],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
    )
    ax3.axvline(0, color="0.3", lw=1)
    ax3.set_yticks(y)
    ax3.set_yticklabels(effects["analysis"])
    ax3.set_xlabel("Standardized beta on suspiciousness")
    ax3.set_title("Composite Association with HiTOP Suspiciousness")
    lim = max(0.3, np.max(np.abs(effects["beta"]) + 1.96 * effects["se"])) * 1.1
    ax3.set_xlim(-lim, lim)
    for i, row in effects.iterrows():
        star = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else ""))
        txt = f"p={row['p_value']:.3g}{star}\nR²={row['r_squared']:.3f}"
        xpos = row["beta"] - 0.02 if row["beta"] < 0 else row["beta"] + 0.02
        ha = "right" if row["beta"] < 0 else "left"
        ax3.text(xpos, i, txt, va="center", ha=ha, fontsize=9)

    fig.suptitle("Suspiciousness PCA Composite", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(IN_DIR / "figure_pca_suspiciousness_paper.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
