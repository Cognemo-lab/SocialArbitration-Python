from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
STATE_DIR = BASE / "hitop" / "reliable_state_fod_moderation"
PARAM_DIR = BASE / "hitop" / "parameter_fod_moderation"
OUT_DIR = BASE / "hitop" / "paper_fod_moderation_summary"

LABELS = {
    "om_a": "Advice volatility (om_a)",
    "inferv_a_mean": "Advice inferential variance",
    "abs_eps2_a_mean": "Absolute advice epsilon2",
}

SCOPE_LABELS = {
    "pooled_clustered": "Pooled",
    "subject_mean": "Subject mean",
    "t1": "T1",
    "t2": "T2",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    state_summary = pd.read_csv(STATE_DIR / "moderation_summary_all_scopes.csv")
    state_slopes = pd.read_csv(STATE_DIR / "moderation_simple_slopes_all_scopes.csv")
    param_summary = pd.read_csv(PARAM_DIR / "moderation_summary_all_scopes.csv")
    param_slopes = pd.read_csv(PARAM_DIR / "moderation_simple_slopes_all_scopes.csv")

    state_summary = state_summary.loc[
        state_summary["state_metric"].isin(["inferv_a_mean", "abs_eps2_a_mean"])
    ].copy()
    state_summary = state_summary.rename(columns={"state_metric": "predictor"})
    state_slopes = state_slopes.loc[
        state_slopes["state_metric"].isin(["inferv_a_mean", "abs_eps2_a_mean"])
    ].copy()
    state_slopes = state_slopes.rename(
        columns={"state_metric": "predictor", "simple_slope_state": "simple_slope"}
    )

    param_summary = param_summary.loc[param_summary["parameter"] == "om_a"].copy()
    param_summary = param_summary.rename(columns={"parameter": "predictor"})
    param_slopes = param_slopes.loc[param_slopes["parameter"] == "om_a"].copy()
    param_slopes = param_slopes.rename(
        columns={"parameter": "predictor", "simple_slope_parameter": "simple_slope"}
    )

    summary = pd.concat([param_summary, state_summary], ignore_index=True)
    slopes = pd.concat([param_slopes, state_slopes], ignore_index=True)
    summary["predictor_label"] = summary["predictor"].map(LABELS)
    summary["scope_label"] = summary["analysis_scope"].map(SCOPE_LABELS)
    slopes["predictor_label"] = slopes["predictor"].map(LABELS)
    slopes["scope_label"] = slopes["analysis_scope"].map(SCOPE_LABELS)

    summary = summary[
        [
            "predictor",
            "predictor_label",
            "analysis_scope",
            "scope_label",
            "n",
            "interaction_beta",
            "interaction_se",
            "interaction_p",
            "main_suicidality_beta",
            "main_suicidality_p",
            "r_squared",
        ]
    ].sort_values(["predictor_label", "analysis_scope"])
    summary.to_csv(OUT_DIR / "table_fod_moderation_summary.csv", index=False)

    slopes = slopes[
        [
            "predictor",
            "predictor_label",
            "analysis_scope",
            "scope_label",
            "moderator_level",
            "suicidality_z",
            "simple_slope",
        ]
    ].sort_values(["predictor_label", "analysis_scope", "suicidality_z"])
    slopes.to_csv(OUT_DIR / "table_fod_moderation_simple_slopes.csv", index=False)

    order = ["pooled_clustered", "subject_mean", "t1", "t2"]
    predictors = ["om_a", "inferv_a_mean", "abs_eps2_a_mean"]

    heat = (
        summary.pivot(index="predictor_label", columns="scope_label", values="interaction_beta")
        .reindex(index=[LABELS[p] for p in predictors], columns=[SCOPE_LABELS[s] for s in order])
    )
    pvals = (
        summary.pivot(index="predictor_label", columns="scope_label", values="interaction_p")
        .reindex(index=heat.index, columns=heat.columns)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    ax = axes[0, 0]
    im = ax.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-0.28, vmax=0.28, aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("Interaction Betas by Analysis Scope")
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            beta = heat.iloc[i, j]
            p = pvals.iloc[i, j]
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            ax.text(j, i, f"{beta:.2f}{star}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Interaction beta")

    colors = {"low": "#2a9d8f", "mean": "#264653", "high": "#e76f51"}
    for panel, scope in zip([axes[0, 1], axes[1, 0], axes[1, 1]], ["pooled_clustered", "t1", "t2"]):
        sub = slopes.loc[slopes["analysis_scope"] == scope].copy()
        pivot = sub.pivot(index="predictor_label", columns="moderator_level", values="simple_slope")
        pivot = pivot.reindex(index=[LABELS[p] for p in predictors], columns=["low", "mean", "high"])
        y = np.arange(len(pivot.index))
        width = 0.23
        for offset, level in zip([-width, 0, width], ["low", "mean", "high"]):
            panel.barh(y + offset, pivot[level], height=0.22, color=colors[level], label=level if scope == "pooled_clustered" else None)
        panel.axvline(0, color="0.3", lw=1)
        panel.set_yticks(y)
        panel.set_yticklabels(pivot.index)
        panel.set_title(f"Simple Slopes: {SCOPE_LABELS[scope]}")
        panel.set_xlabel("Slope on fearlessness of death")
    axes[0, 1].legend(frameon=False, title="Suicidality")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_fod_moderation_summary.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
