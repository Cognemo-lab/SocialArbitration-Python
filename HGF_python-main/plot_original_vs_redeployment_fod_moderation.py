from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
ORIG_STATE = BASE / "hitop" / "results-suicidality" / "table_fod_moderation_summary.csv"
ORIG_HIER = BASE / "hitop" / "hierarchical_fod_moderation_features" / "hierarchical_moderation_interactions.csv"
REDEPLOY_STATE = BASE / "hitop_batch2" / "fod_moderation_redeployment_matched_hitop" / "moderation_summary.csv"
REDEPLOY_HIER = BASE / "hitop_batch2" / "fod_moderation_redeployment_matched_hitop" / "hierarchical_moderation_interactions.csv"
OUT = BASE / "hitop_batch2" / "fod_moderation_redeployment_matched_hitop" / "figure_original_vs_redeployment_fod_moderation.png"

LABELS = {
    "om_a": "Omega-Advice",
    "inferv_a_mean": "Advice inferential variance",
    "abs_eps2_a_mean": "Abs advice epsilon2",
    "pc1_composite": "Advice-uncertainty PC1",
    "abs_eps2_a_mean_z:hitop_suicidality_z": "Abs advice epsilon2 x suicidality",
    "inferv_a_mean_z:hitop_suicidality_z": "Advice inferential variance x suicidality",
    "ka_a_z:hitop_suicidality_z": "Kappa-Advice x suicidality",
    "om_a_z:hitop_suicidality_z": "Omega-Advice x suicidality",
    "eps3_a_mean_z:hitop_suicidality_z": "Advice epsilon3 x suicidality",
    "abs_eps2_a_mean_z:hitop_suicidality_proxy_z": "Abs advice epsilon2 x suicidality",
    "inferv_a_mean_z:hitop_suicidality_proxy_z": "Advice inferential variance x suicidality",
    "ka_a_z:hitop_suicidality_proxy_z": "Kappa-Advice x suicidality",
    "om_a_z:hitop_suicidality_proxy_z": "Omega-Advice x suicidality",
    "eps3_a_mean_z:hitop_suicidality_proxy_z": "Advice epsilon3 x suicidality",
}


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main() -> None:
    orig = pd.read_csv(ORIG_STATE).rename(columns={"analysis_scope": "scope"})
    orig = orig.loc[orig["scope"].isin(["pooled_clustered", "t2"])].copy()
    orig["scope"] = orig["scope"].replace({"pooled_clustered": "Pooled", "t2": "T2"})
    orig["dataset"] = "Original"

    redeploy = pd.read_csv(REDEPLOY_STATE)
    redeploy = redeploy.loc[redeploy["scope"].isin(["pooled", "t2"])].copy()
    redeploy["scope"] = redeploy["scope"].replace({"pooled": "Pooled", "t2": "T2"})
    redeploy["dataset"] = "Redeployed"

    comb = pd.concat(
        [
            orig[["dataset", "scope", "predictor", "interaction_beta", "interaction_se", "interaction_p"]],
            redeploy[["dataset", "scope", "predictor", "interaction_beta", "interaction_se", "interaction_p"]],
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), gridspec_kw={"width_ratios": [1.1, 1.1, 1.25]})
    colors = {"Original": "#1d3557", "Redeployed": "#e76f51"}
    predictors = ["om_a", "inferv_a_mean", "abs_eps2_a_mean", "pc1_composite"]

    for ax, scope in zip(axes[:2], ["Pooled", "T2"]):
        sub = comb.loc[comb["scope"] == scope].copy()
        y = np.arange(len(predictors))[::-1]
        for off, dataset in [(-0.12, "Original"), (0.12, "Redeployed")]:
            cur = sub.loc[sub["dataset"] == dataset].set_index("predictor").reindex(predictors)
            ax.errorbar(
                cur["interaction_beta"],
                y + off,
                xerr=1.96 * cur["interaction_se"],
                fmt="o",
                color=colors[dataset],
                ms=6,
                capsize=3,
                label=dataset if scope == "Pooled" else None,
            )
            for x, yy, p in zip(cur["interaction_beta"], y + off, cur["interaction_p"]):
                s = stars(float(p))
                if s:
                    ax.text(x, yy + 0.08, s, ha="center", va="bottom", fontsize=10, color=colors[dataset])
        ax.axvline(0, color="#888", ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[p] for p in predictors], fontsize=10)
        ax.set_title(f"{scope} univariate models")
        ax.set_xlabel("Interaction beta")
        if scope == "Pooled":
            ax.legend(frameon=False, fontsize=9, loc="lower left")

    orig_h = pd.read_csv(ORIG_HIER).rename(columns={"coef": "beta", "p_value": "p"})
    red_h = pd.read_csv(REDEPLOY_HIER)
    orig_h["dataset"] = "Original"
    red_h["dataset"] = "Redeployed"
    hier = pd.concat([orig_h[["dataset", "term", "beta", "se", "p"]], red_h[["dataset", "term", "beta", "se", "p"]]], ignore_index=True)
    terms = [
        "abs_eps2_a_mean_z:hitop_suicidality_z",
        "inferv_a_mean_z:hitop_suicidality_z",
        "ka_a_z:hitop_suicidality_z",
        "om_a_z:hitop_suicidality_z",
        "eps3_a_mean_z:hitop_suicidality_z",
    ]
    # Map redeployment term names onto original names for plotting.
    hier["term"] = hier["term"].replace(
        {
            "abs_eps2_a_mean_z:hitop_suicidality_proxy_z": "abs_eps2_a_mean_z:hitop_suicidality_z",
            "inferv_a_mean_z:hitop_suicidality_proxy_z": "inferv_a_mean_z:hitop_suicidality_z",
            "ka_a_z:hitop_suicidality_proxy_z": "ka_a_z:hitop_suicidality_z",
            "om_a_z:hitop_suicidality_proxy_z": "om_a_z:hitop_suicidality_z",
            "eps3_a_mean_z:hitop_suicidality_proxy_z": "eps3_a_mean_z:hitop_suicidality_z",
        }
    )
    ax = axes[2]
    y = np.arange(len(terms))[::-1]
    for off, dataset in [(-0.12, "Original"), (0.12, "Redeployed")]:
        cur = hier.loc[hier["dataset"] == dataset].set_index("term").reindex(terms)
        ax.errorbar(
            cur["beta"],
            y + off,
            xerr=1.96 * cur["se"],
            fmt="o",
            color=colors[dataset],
            ms=6,
            capsize=3,
        )
        for x, yy, p in zip(cur["beta"], y + off, cur["p"]):
            s = stars(float(p))
            if s:
                ax.text(x, yy + 0.08, s, ha="center", va="bottom", fontsize=10, color=colors[dataset])
    ax.axvline(0, color="#888", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[t] for t in terms], fontsize=10)
    ax.set_title("Hierarchical joint model")
    ax.set_xlabel("Interaction beta")

    fig.suptitle("FoD x HiTOP suicidality moderation\nOriginal vs redeployed", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
