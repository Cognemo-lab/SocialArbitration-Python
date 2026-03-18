from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
PARAM_DIR = BASE / "hgf_baselinefixed_full" / "construct_validity_behavior"
STATE_DIR = BASE / "hgf_baselinefixed_full" / "construct_validity_states"
OUT_DIR = BASE / "hgf_baselinefixed_full" / "paper_construct_validity"

BEHAVIOR_LABELS = {
    "accuracy_rate": "Accuracy",
    "advice_taking_rate": "Advice-taking",
    "win_stay_rate": "Win-stay",
    "lose_switch_rate": "Lose-switch",
}

STATE_LABELS = {
    "abs_eps2_a_mean": "Abs advice epsilon2",
    "abs_eps3_a_mean": "Abs advice epsilon3",
    "eps3_a_mean": "Advice epsilon3",
    "inferv_a_mean": "Advice inferential variance",
    "wager_pred_mean": "Predicted wager mean",
}


def fdr_stars(q: float) -> str:
    if pd.isna(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def prepare_top_hits(df: pd.DataFrame, entity_col: str, label_map: dict[str, str] | None = None) -> pd.DataFrame:
    pooled = df.loc[df["analysis_scope"] == "pooled_t1_t2"].copy()
    pooled["behavior_label"] = pooled["behavior_metric"].map(BEHAVIOR_LABELS)
    pooled["abs_r"] = pooled["pearson_r"].abs()
    pooled["entity_label"] = pooled[entity_col]
    if label_map is not None:
        pooled["entity_label"] = pooled[entity_col].map(label_map).fillna(pooled[entity_col])
    top = (
        pooled.sort_values(["behavior_metric", "abs_r"], ascending=[True, False])
        .groupby("behavior_metric", as_index=False)
        .head(4)
        .copy()
    )
    return top


def prepare_heat(df: pd.DataFrame, entity_col: str, label_map: dict[str, str] | None = None) -> pd.DataFrame:
    pooled = df.loc[df["analysis_scope"] == "pooled_t1_t2"].copy()
    pooled["behavior_label"] = pooled["behavior_metric"].map(BEHAVIOR_LABELS)
    pooled["entity_label"] = pooled[entity_col]
    if label_map is not None:
        pooled["entity_label"] = pooled[entity_col].map(label_map).fillna(pooled[entity_col])
    heat = pooled.pivot(index="entity_label", columns="behavior_label", values="pearson_r")
    heat = heat.loc[heat.abs().max(axis=1).sort_values(ascending=False).index]
    qvals = pooled.pivot(
        index="entity_label",
        columns="behavior_label",
        values="fdr_q_pearson_within_behavior_scope",
    ).reindex(index=heat.index, columns=heat.columns)
    return heat, qvals


def draw_heatmap(ax, heat: pd.DataFrame, qvals: pd.DataFrame, title: str, vlim: float) -> None:
    im = ax.imshow(heat.to_numpy(), cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title(title)
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            ax.text(
                j,
                i,
                f"{heat.iloc[i, j]:.2f}{fdr_stars(qvals.iloc[i, j])}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    return im


def draw_top_panel(ax, sub: pd.DataFrame, title: str) -> None:
    sub = sub.sort_values("pearson_r").copy()
    colors = ["#2a9d8f" if r >= 0 else "#e76f51" for r in sub["pearson_r"]]
    ax.barh(sub["entity_label"], sub["pearson_r"], color=colors, alpha=0.9)
    ax.axvline(0, color="0.3", lw=1)
    ax.set_title(title)
    ax.set_xlabel("Pearson r")
    lim = max(0.12, np.nanmax(np.abs(sub["pearson_r"])) * 1.18)
    ax.set_xlim(-lim, lim)
    for i, (_, row) in enumerate(sub.iterrows()):
        xpos = row["pearson_r"] + (0.012 if row["pearson_r"] >= 0 else -0.012)
        ax.text(xpos, i, fdr_stars(row["fdr_q_pearson_within_behavior_scope"]), va="center",
                ha="left" if row["pearson_r"] >= 0 else "right", fontsize=9)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    param = pd.read_csv(PARAM_DIR / "construct_validity_parameter_behavior.csv")
    state = pd.read_csv(STATE_DIR / "construct_validity_state_behavior.csv")

    param_top = prepare_top_hits(param, "parameter")
    state_top = prepare_top_hits(state, "state_metric", STATE_LABELS)
    param_heat, param_q = prepare_heat(param, "parameter")
    state_heat, state_q = prepare_heat(state, "state_metric", STATE_LABELS)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.4, wspace=0.35)

    ax_param_heat = fig.add_subplot(gs[0, 0])
    ax_state_heat = fig.add_subplot(gs[0, 1])
    im1 = draw_heatmap(ax_param_heat, param_heat, param_q, "Parameters vs Behavior", 0.7)
    im2 = draw_heatmap(ax_state_heat, state_heat, state_q, "Reliable States vs Behavior", 0.5)
    cbar1 = fig.colorbar(im1, ax=ax_param_heat, fraction=0.046, pad=0.03)
    cbar1.set_label("Pearson r")
    cbar2 = fig.colorbar(im2, ax=ax_state_heat, fraction=0.046, pad=0.03)
    cbar2.set_label("Pearson r")

    behaviors = list(BEHAVIOR_LABELS.values())
    for idx, behavior in enumerate(behaviors):
        ax_left = fig.add_subplot(gs[1 + idx // 2, idx % 2])
        sub_p = param_top.loc[param_top["behavior_label"] == behavior].copy()
        sub_p["entity_label"] = sub_p["entity_label"]
        draw_top_panel(ax_left, sub_p, f"Top Parameter Hits: {behavior}")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_construct_validity_paper_panel.png", dpi=240)
    plt.close(fig)

    # Separate focused top-hit panel comparing parameters and states by behavior.
    fig, axes = plt.subplots(4, 2, figsize=(13, 15))
    for row, behavior in enumerate(behaviors):
        psub = param_top.loc[param_top["behavior_label"] == behavior].copy()
        ssub = state_top.loc[state_top["behavior_label"] == behavior].copy()
        draw_top_panel(axes[row, 0], psub, f"{behavior}: Parameters")
        draw_top_panel(axes[row, 1], ssub, f"{behavior}: Reliable States")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_construct_validity_top_comparison.png", dpi=240)
    plt.close(fig)

    # Paper summary table.
    param_top["entity_type"] = "parameter"
    state_top["entity_type"] = "state"
    combined = pd.concat([param_top, state_top], ignore_index=True)
    combined = combined[
        [
            "entity_type",
            "behavior_metric",
            "behavior_label",
            "entity_label",
            "pearson_r",
            "pearson_p",
            "fdr_q_pearson_within_behavior_scope",
            "spearman_rho",
        ]
    ].sort_values(["behavior_metric", "entity_type", "pearson_r"], ascending=[True, True, False])
    combined.to_csv(OUT_DIR / "table_construct_validity_top_hits_combined.csv", index=False)


if __name__ == "__main__":
    main()
