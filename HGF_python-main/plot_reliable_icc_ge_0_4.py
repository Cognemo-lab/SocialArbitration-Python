from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
PARAM_CSV = BASE / "hgf_baselinefixed_full" / "test_retest_reliability_by_parameter.csv"
STATE_CSV = BASE / "hgf_baselinefixed_full" / "state_reliability" / "state_test_retest_reliability.csv"
OUT_DIR = BASE / "hgf_baselinefixed_full" / "figures_reliable_icc"

STATE_LABELS = {
    "abs_eps2_a_mean": "Abs advice epsilon2",
    "abs_eps3_a_mean": "Abs advice epsilon3",
    "eps3_a_mean": "Advice epsilon3",
    "inferv_a_mean": "Advice inferential variance",
    "wager_pred_mean": "Predicted wager mean",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    params = pd.read_csv(PARAM_CSV)
    states = pd.read_csv(STATE_CSV)

    params = params.loc[(~params["is_fixed"]) & (params["icc3_1_t1_t2"] >= 0.4)].copy()
    params = params.loc[params["parameter"] != "be0"].copy()
    params["label"] = params["parameter"]
    params["kind"] = "Parameter"

    states = states.loc[states["icc3_1_t1_t2"] >= 0.4].copy()
    states = states.loc[states["state_metric"] != "abs_eps3_a_mean"].copy()
    states["label"] = states["state_metric"].map(STATE_LABELS).fillna(states["state_metric"])
    states["kind"] = "State"

    combined = pd.concat(
        [
            params[["label", "kind", "pearson_r_t1_t2", "icc3_1_t1_t2"]],
            states[["label", "kind", "pearson_r_t1_t2", "icc3_1_t1_t2"]],
        ],
        ignore_index=True,
    ).sort_values(["kind", "icc3_1_t1_t2", "pearson_r_t1_t2"], ascending=[True, False, False])

    combined.to_csv(OUT_DIR / "reliable_icc_ge_0_4_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    palette = {"Parameter": "#3a7ca5", "State": "#e07a5f"}
    plot_df = combined.sort_values(["kind", "icc3_1_t1_t2", "pearson_r_t1_t2"], ascending=[True, True, True]).reset_index(drop=True)
    y = list(range(len(plot_df)))
    colors = [palette[k] for k in plot_df["kind"]]

    for ax, metric, title in [
        (axes[0], "icc3_1_t1_t2", "ICC(3,1)"),
        (axes[1], "pearson_r_t1_t2", "Pearson r"),
    ]:
        ax.barh(list(y), plot_df[metric], color=colors, alpha=0.9)
        ax.axvline(0.4, color="#444444", linestyle="--", linewidth=1)
        ax.set_yticks(list(y))
        ax.set_yticklabels(plot_df["label"])
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_xlim(0, max(0.7, plot_df[metric].max() * 1.1))
        for i, v in enumerate(plot_df[metric]):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", ha="left", fontsize=9)

    handles = [
        plt.Line2D([0], [0], color=palette["Parameter"], lw=8),
        plt.Line2D([0], [0], color=palette["State"], lw=8),
    ]
    fig.legend(handles, ["Parameter", "State"], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Test-Retest Reliable Model Quantities (ICC ≥ 0.4)", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(OUT_DIR / "figure_reliable_icc_ge_0_4.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
